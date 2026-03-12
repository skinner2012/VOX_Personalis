"""FastAPI application — routes and WebSocket handler for MVS.

Endpoints:
  WS  /ws/transcribe  — streaming transcription
  GET /metrics        — SLA metrics JSON
  GET /health         — readiness check (200 ready, 503 starting)
  GET /demo           — browser demo UI
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from serving import metrics as metrics_module
from serving import model as model_module
from serving.vad import VADSegmenter

log = logging.getLogger("serving.api")

app = FastAPI(title="VOX Personalis MVS", version="0.1.0")

# Populated by cli.py before uvicorn starts
_collector: metrics_module.MetricsCollector | None = None
_vad_silence_ms: int = 1000
_vad_max_utterance_sec: float = 30.0

_STATIC_DIR = Path(__file__).parent / "static"
_INFERENCE_TIMEOUT_SEC = 30.0


def configure(
    collector: metrics_module.MetricsCollector,
    silence_ms: int,
    max_utterance_sec: float,
) -> None:
    """Called by cli.py after model load, before uvicorn.run()."""
    global _collector, _vad_silence_ms, _vad_max_utterance_sec
    _collector = collector
    _vad_silence_ms = silence_ms
    _vad_max_utterance_sec = max_utterance_sec


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    if model_module.model_loaded:
        return JSONResponse(
            {"status": "ok", "model_loaded": True, "model_version": "v2"},
            status_code=200,
        )
    return JSONResponse(
        {"status": "starting", "model_loaded": False, "message": "Loading checkpoint..."},
        status_code=503,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@app.get("/metrics")
async def metrics() -> JSONResponse:
    if _collector is None:
        return JSONResponse({"error": "metrics not configured"}, status_code=503)
    return JSONResponse(_collector.get_metrics())


# ---------------------------------------------------------------------------
# Demo UI
# ---------------------------------------------------------------------------


@app.get("/demo")
async def demo() -> FileResponse:
    return FileResponse(_STATIC_DIR / "demo.html", media_type="text/html")


# ---------------------------------------------------------------------------
# WebSocket transcription
# ---------------------------------------------------------------------------


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()
    log.debug("WebSocket accepted")

    segmenter = VADSegmenter(
        silence_ms=_vad_silence_ms,
        max_utterance_sec=_vad_max_utterance_sec,
    )

    frame_count = 0
    try:
        while True:
            message = await websocket.receive()

            # Text control message: {"type": "stop"} or similar
            if "text" in message:
                try:
                    ctrl = json.loads(message["text"])
                except (json.JSONDecodeError, ValueError):
                    ctrl = {}

                if ctrl.get("type") == "stop":
                    log.debug("stop received — flushing (frames=%d)", frame_count)
                    remainder = segmenter.flush()
                    if remainder:
                        log.debug("flush produced %d bytes", len(remainder))
                        await _handle_utterance(websocket, remainder, segmenter)
                    else:
                        log.debug("flush produced nothing")
                    break
                continue

            # Binary audio frame
            frame: bytes = message.get("bytes", b"")
            if not frame:
                log.debug("empty binary frame, skipping")
                continue

            frame_count += 1
            if frame_count <= 5 or frame_count % 100 == 0:
                log.debug("frame #%d, len=%d bytes", frame_count, len(frame))

            # Feed the frame; VADSegmenter expects exactly FRAME_BYTES per call.
            # The client sends full 30ms frames, but guard against partials.
            frame_bytes = VADSegmenter.FRAME_BYTES
            for offset in range(0, len(frame), frame_bytes):
                chunk = frame[offset : offset + frame_bytes]
                if len(chunk) < frame_bytes:
                    break  # discard partial trailing frame

                try:
                    utterance = segmenter.feed(chunk)
                except ValueError:
                    continue  # wrong size — skip

                if utterance is not None:
                    log.debug(
                        "utterance detected: %d bytes (%.1fs)",
                        len(utterance),
                        len(utterance) / (VADSegmenter.SAMPLE_RATE * VADSegmenter.BYTES_PER_SAMPLE),
                    )
                    # Max utterance exceeded — notify client then transcribe
                    max_frames = int(_vad_max_utterance_sec * 1000 / VADSegmenter.FRAME_MS)
                    max_bytes = VADSegmenter.FRAME_BYTES * max_frames
                    if segmenter.buffer_duration_ms() == 0 and len(utterance) >= max_bytes:
                        msg = (
                            f"Buffer exceeded {_vad_max_utterance_sec:.0f}s"
                            " — transcribing partial segment"
                        )
                        await _send_error(websocket, "max_utterance_exceeded", msg)
                    await _handle_utterance(websocket, utterance, segmenter)

    except WebSocketDisconnect:
        log.debug("client disconnected (total frames=%d)", frame_count)
    except Exception as exc:
        log.debug("ws handler error: %s (total frames=%d)", exc, frame_count)
        try:
            await _send_error(websocket, "inference_error", "Unexpected server error")
        except Exception:
            pass


async def _handle_utterance(
    websocket: WebSocket,
    audio_bytes: bytes,
    segmenter: VADSegmenter,
) -> None:
    """Transcribe one utterance and send the JSON segment to the client."""
    assert _collector is not None

    duration_sec = len(audio_bytes) / (VADSegmenter.SAMPLE_RATE * VADSegmenter.BYTES_PER_SAMPLE)
    segment_id = _collector.next_segment_id()
    t_start = time.monotonic()

    try:
        loop = asyncio.get_event_loop()
        text = await asyncio.wait_for(
            loop.run_in_executor(None, model_module.transcribe, audio_bytes),
            timeout=_INFERENCE_TIMEOUT_SEC,
        )
        latency_ms = (time.monotonic() - t_start) * 1000

        _collector.record_segment(
            segment_id=segment_id,
            latency_ms=latency_ms,
            duration_sec=duration_sec,
            status="ok",
        )

        await websocket.send_json(
            {
                "type": "transcript",
                "segment_id": segment_id,
                "transcript": text,
                "duration_sec": round(duration_sec, 3),
                "latency_ms": round(latency_ms, 1),
                "model_version": "v2",
            }
        )

    except TimeoutError:
        latency_ms = (time.monotonic() - t_start) * 1000
        _collector.record_segment(
            segment_id=segment_id,
            latency_ms=latency_ms,
            duration_sec=duration_sec,
            status="timeout",
        )
        segmenter._reset()  # Clear buffer to prevent carrying over on next frames
        await _send_error(
            websocket,
            "inference_timeout",
            "Model inference exceeded time limit (> 30s)",
        )

    except Exception as exc:
        latency_ms = (time.monotonic() - t_start) * 1000
        _collector.record_segment(
            segment_id=segment_id,
            latency_ms=latency_ms,
            duration_sec=duration_sec,
            status="error",
        )
        segmenter._reset()  # Clear buffer to prevent carrying over on next frames
        await _send_error(websocket, "inference_error", f"Inference failed: {exc}")


async def _send_error(websocket: WebSocket, code: str, message: str) -> None:
    try:
        await websocket.send_json({"type": "error", "code": code, "message": message})
    except Exception:
        pass
