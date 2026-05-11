"""
VOX Daemon WebSocket proxy — M4/M5.

Sits between the Chrome browser and WhisperLiveKit (wlk). Three responsibilities:

  1. Forward audio frames  : browser → wlk (transparent binary proxy, int16 LE PCM)
  2. Forward Stage A events: wlk → browser as {"type":"stage_a", lines, buffer, uid}
  3. Gemma Stage B events  : when a line's text has been stable for STABLE_S
                             OR a later line index has appeared, fire Gemma on
                             that line's text and emit {"type":"stage_b", ...}

Trigger design — why a stability/debounce timer instead of `len(lines) > prev`:
  Empirically wlk's lines[] is NOT a clean "appends one per utterance" stream:
    - empty placeholder entries (`text=''`) appear and disappear
    - line[i].text grows over time WITHIN one utterance as more words commit
      (e.g. `' irl'` → `' irl you play some games yes'` at the same index,
      with a 4 s gap between updates and stable text in between)
  So we wait STABLE_S after each line-text change before calling Gemma, AND we
  re-fire Gemma when the line text changes substantively after a previous
  correction landed (LocalAgreement-2 can produce two stable plateaus for one
  utterance; we want Stage B to reflect the latest committed text). The
  browser keeps only the most recent Stage B per line, so re-firing is
  harmless. Once a later line index appears with non-empty text, the earlier
  line is conclusively final and we fire immediately, skipping the wait.
  Empty entries are always ignored — Gemma on `""` produces a "please provide
  the transcript" hallucination.

Gemma is single-subprocess (not re-entrant); concurrent corrections serialize
behind an asyncio.Lock.
"""

import asyncio
import json
import logging
import time

import aiohttp
from aiohttp import web

from scripts.vox_daemon.gemma import GemmaWorker

log = logging.getLogger(__name__)

# Seconds a line's text must remain unchanged before we accept it as final and
# fire Gemma. Tuned so the trailing pause that wlk's VAC already needs to call
# an endpoint is enough to satisfy stability without adding much extra latency.
STABLE_S = 1.0
# Background poll interval for the stability check. wlk only sends new
# messages when content changes; once a line is stable wlk goes silent, so we
# need our own ticker to notice and fire.
STABILITY_TICK_S = 0.2


class VoxDaemonProxy:
    """aiohttp WebSocket handler: browser ↔ WhisperLiveKit ↔ Gemma."""

    def __init__(self, wlk_url: str, gemma: GemmaWorker, gemma_timeout: float = 1.5) -> None:
        self._wlk_url = wlk_url
        self._gemma = gemma
        self._gemma_timeout = gemma_timeout
        self._gemma_lock = asyncio.Lock()
        self._uid = 0

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        client_ws = web.WebSocketResponse()
        await client_ws.prepare(request)
        log.info("Browser client connected")

        session = aiohttp.ClientSession()
        try:
            async with session.ws_connect(self._wlk_url) as wlk_ws:
                log.info("Connected to wlk at %s", self._wlk_url)
                # Per line-index state. fired_text records the text we last
                # dispatched to Gemma; if `text` later differs we re-fire so
                # Stage B reflects the latest committed wording.
                lines_state: dict[int, dict] = {}

                def fire_gemma(i: int, text: str) -> None:
                    state = lines_state.get(i)
                    if state is not None:
                        state["fired_text"] = text
                    self._uid += 1
                    asyncio.create_task(self._correct_and_emit(client_ws, self._uid, i, text))

                async def _client_to_wlk() -> None:
                    async for msg in client_ws:
                        if msg.type == aiohttp.WSMsgType.BINARY:
                            await wlk_ws.send_bytes(msg.data)
                        elif msg.type == aiohttp.WSMsgType.TEXT:
                            await wlk_ws.send_str(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break

                async def _wlk_to_client() -> None:
                    async for msg in wlk_ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            break
                        try:
                            data = json.loads(msg.data)
                        except json.JSONDecodeError:
                            log.warning("Non-JSON from wlk: %s", msg.data[:80])
                            continue

                        # One-shot config from wlk on connect.
                        if data.get("type") == "config":
                            try:
                                await client_ws.send_json(data)
                            except Exception:
                                break
                            continue

                        lines = data.get("lines", [])
                        buffer = data.get("buffer_transcription", "")

                        # Forward Stage A immediately.
                        try:
                            await client_ws.send_json(
                                {
                                    "type": "stage_a",
                                    "lines": lines,
                                    "buffer": buffer,
                                    "uid": self._uid,
                                }
                            )
                        except Exception:
                            break

                        # Update per-line state. On text change, reset the
                        # stability timer; the ticker decides when to fire.
                        now = time.monotonic()
                        max_nonempty_idx = -1
                        for i, line in enumerate(lines):
                            text = (line.get("text") or "").strip()
                            if not text:
                                continue
                            max_nonempty_idx = max(max_nonempty_idx, i)
                            prev = lines_state.get(i)
                            if prev is None:
                                lines_state[i] = {
                                    "text": text,
                                    "changed_at": now,
                                    "fired_text": None,
                                }
                            elif prev["text"] != text:
                                prev["text"] = text
                                prev["changed_at"] = now

                        # A later non-empty line proves earlier lines are
                        # conclusively done; fire those immediately if their
                        # current text hasn't been dispatched.
                        for i, state in list(lines_state.items()):
                            if i < max_nonempty_idx and state["fired_text"] != state["text"]:
                                fire_gemma(i, state["text"])

                async def _stability_ticker() -> None:
                    # wlk goes silent once a line stabilises (no message means
                    # no content change), so we need our own clock to notice
                    # the line has been quiet long enough.
                    while True:
                        await asyncio.sleep(STABILITY_TICK_S)
                        now = time.monotonic()
                        for i, state in list(lines_state.items()):
                            if state["fired_text"] == state["text"]:
                                continue
                            if (now - state["changed_at"]) >= STABLE_S:
                                fire_gemma(i, state["text"])

                ticker_task = asyncio.create_task(_stability_ticker())
                try:
                    await asyncio.gather(_client_to_wlk(), _wlk_to_client())
                finally:
                    ticker_task.cancel()
                    try:
                        await ticker_task
                    except asyncio.CancelledError:
                        pass
        finally:
            await session.close()
            log.info("Browser client disconnected")

        return client_ws

    async def _correct_and_emit(
        self,
        client_ws: web.WebSocketResponse,
        uid: int,
        line_index: int,
        stage_a_text: str,
    ) -> None:
        loop = asyncio.get_event_loop()
        async with self._gemma_lock:
            corrected = await loop.run_in_executor(
                None, self._gemma.correct, stage_a_text, self._gemma_timeout
            )
        stage_b = {
            "type": "stage_b",
            "uid": uid,
            "line_index": line_index,
            "text": corrected,
        }
        try:
            await client_ws.send_json(stage_b)
            log.debug("Stage B uid=%d line=%d: %s", uid, line_index, corrected[:60])
        except Exception:
            pass  # Client disconnected before Stage B arrived


def create_app(
    wlk_url: str,
    gemma: GemmaWorker,
    static_dir: str | None = None,
    gemma_timeout: float = 1.5,
) -> web.Application:
    """Build and return the aiohttp Application.

    The caller is responsible for starting and stopping the app (via
    web.AppRunner or web.run_app). GemmaWorker lifetime is managed by the
    caller; proxy.create_app does not close it.
    """
    proxy = VoxDaemonProxy(wlk_url=wlk_url, gemma=gemma, gemma_timeout=gemma_timeout)
    app = web.Application()
    app.router.add_get("/asr", proxy.handle)

    if static_dir:
        # Serve index.html at "/" instead of an autoindex directory listing.
        index_path = f"{static_dir}/index.html"

        async def _index(_: web.Request) -> web.FileResponse:
            return web.FileResponse(index_path)

        app.router.add_get("/", _index)
        app.router.add_static("/static/", path=static_dir, name="static")
        # The worklet is fetched as a same-origin script from the page;
        # expose it at the root so `addModule("worklet.js")` works without
        # a path prefix.
        worklet_path = f"{static_dir}/worklet.js"

        async def _worklet(_: web.Request) -> web.FileResponse:
            return web.FileResponse(worklet_path)

        app.router.add_get("/worklet.js", _worklet)

    return app
