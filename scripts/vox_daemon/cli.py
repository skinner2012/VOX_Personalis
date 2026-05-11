"""
VOX Daemon CLI (M5) — orchestrates the three pieces of the live demo:

  1. WhisperLiveKit subprocess on port 8001 (faster-whisper backend serving the
     merged S1-M7 CT2 model, LocalAgreement-2 streaming policy, VAC enabled).
  2. GemmaWorker — persistent llama-cli subprocess holding the 26B Q4_K_M GGUF
     in memory.
  3. aiohttp app on port 8000 that serves the Chrome frontend and proxies the
     browser <-> wlk WebSocket (with Gemma corrections layered on top).

Startup order matters: wlk must be ready before the proxy app starts accepting
client connections, otherwise the proxy's ws_connect to wlk fails for the
first browser load. We tail wlk's stderr for uvicorn's
"Application startup complete." sentinel before bringing the proxy up.

Shutdown order is the reverse: stop aiohttp -> close Gemma -> terminate wlk.

Run:
  python -m scripts.vox_daemon --open-browser
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

from aiohttp import web

from scripts.vox_daemon.gemma import GemmaWorker
from scripts.vox_daemon.proxy import create_app


def _find_wlk() -> str:
    """Locate the wlk CLI. Prefers the venv's binary so we use the same install."""
    venv_wlk = Path(sys.executable).parent / "wlk"
    if venv_wlk.is_file():
        return str(venv_wlk)
    which = shutil.which("wlk")
    if which:
        return which
    raise FileNotFoundError(
        "Cannot locate 'wlk' binary. Did you install whisperlivekit in the active venv?"
    )


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _spawn_wlk(args: argparse.Namespace) -> subprocess.Popen[str]:
    cmd = [
        _find_wlk(),
        "serve",
        "--backend",
        "faster-whisper",
        "--model_dir",
        args.whisper_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.wlk_port),
        "--min-chunk-size",
        "0.5",
        "--backend-policy",
        "localagreement",
        "--warmup-file",
        "",
        "--pcm-input",
        "-l",
        "INFO",
    ]
    print(f"[daemon] spawning wlk: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
    )


def _wait_for_wlk(wlk_proc: subprocess.Popen[str], timeout: float) -> None:
    """Block until wlk prints uvicorn's ready sentinel on stdout or stderr.

    wlk inherits uvicorn's logger which emits "Application startup complete."
    once the websocket route is mounted. We tee both streams to our own
    stderr so demo operators can see model-load progress.
    """
    ready = threading.Event()
    failed = threading.Event()

    def reader(stream: IO[str], label: str) -> None:
        for line in iter(stream.readline, ""):
            sys.stderr.write(f"[wlk:{label}] {line}")
            sys.stderr.flush()
            if "Application startup complete" in line or "Uvicorn running on" in line:
                ready.set()
        # Stream closed unexpectedly — wlk died.
        if not ready.is_set():
            failed.set()

    threading.Thread(target=reader, args=(wlk_proc.stdout, "out"), daemon=True).start()
    threading.Thread(target=reader, args=(wlk_proc.stderr, "err"), daemon=True).start()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready.is_set():
            return
        if failed.is_set() or wlk_proc.poll() is not None:
            raise RuntimeError(
                f"wlk exited with code {wlk_proc.returncode} before becoming ready. "
                "See [wlk:err] lines above for the failure mode."
            )
        time.sleep(0.1)
    raise TimeoutError(f"wlk did not become ready within {timeout}s")


def _open_browser(port: int) -> None:
    if sys.platform == "darwin":
        subprocess.Popen(["open", f"http://localhost:{port}"])
    elif sys.platform == "linux":
        subprocess.Popen(["xdg-open", f"http://localhost:{port}"])
    elif sys.platform == "win32":
        os.startfile(f"http://localhost:{port}")  # type: ignore[attr-defined]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--whisper-model",
        default="./out/whisper_small_en_s1m7_merged_ct2",
        help="Path to the merged Whisper CT2 model directory (M1 output)",
    )
    p.add_argument(
        "--gemma-model",
        default="./models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    )
    p.add_argument(
        "--llama-cli",
        default="/Users/skinnercheng/llama.cpp/build/bin/llama-cli",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the daemon HTTP+WS server (the browser connects here)",
    )
    p.add_argument(
        "--wlk-port",
        type=int,
        default=8001,
        dest="wlk_port",
        help="Port wlk subprocess listens on",
    )
    p.add_argument(
        "--wlk-startup-timeout",
        type=float,
        default=60.0,
        dest="wlk_startup_timeout",
    )
    p.add_argument(
        "--gemma-startup-timeout",
        type=float,
        default=120.0,
        dest="gemma_startup_timeout",
    )
    p.add_argument(
        "--gemma-correction-timeout",
        type=float,
        default=1.5,
        dest="gemma_timeout",
        help="Per-line Gemma correction budget in seconds (live demo)",
    )
    p.add_argument("--open-browser", action="store_true", dest="open_browser")
    args = p.parse_args()

    if not _port_free("127.0.0.1", args.port):
        print(f"ERROR: port {args.port} is in use — pick another with --port", file=sys.stderr)
        return 1
    if not _port_free("127.0.0.1", args.wlk_port):
        print(
            f"ERROR: port {args.wlk_port} is in use — pick another with --wlk-port",
            file=sys.stderr,
        )
        return 1

    wlk_proc = _spawn_wlk(args)
    gemma: GemmaWorker | None = None
    try:
        _wait_for_wlk(wlk_proc, timeout=args.wlk_startup_timeout)
        print("[daemon] wlk is ready")

        print("[daemon] starting Gemma worker (this can take ~3-50s)…")
        t0 = time.monotonic()
        gemma = GemmaWorker(
            model_path=args.gemma_model,
            llama_cli=args.llama_cli,
            startup_timeout=args.gemma_startup_timeout,
        )
        print(f"[daemon] Gemma worker ready in {time.monotonic() - t0:.1f}s")

        static_dir = Path(__file__).resolve().parent / "static"
        app = create_app(
            wlk_url=f"ws://127.0.0.1:{args.wlk_port}/asr",
            gemma=gemma,
            static_dir=str(static_dir),
            gemma_timeout=args.gemma_timeout,
        )

        if args.open_browser:
            _open_browser(args.port)

        print(f"[daemon] serving at http://localhost:{args.port}")
        # web.run_app installs its own SIGINT handler that shuts down cleanly,
        # which then unwinds through this try/finally.
        web.run_app(app, host="127.0.0.1", port=args.port, print=lambda *_: None)
    finally:
        if gemma is not None:
            print("[daemon] closing Gemma worker")
            gemma.close()
        print("[daemon] terminating wlk")
        wlk_proc.terminate()
        try:
            wlk_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            wlk_proc.kill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
