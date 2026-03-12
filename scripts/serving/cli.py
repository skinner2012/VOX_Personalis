"""CLI parsing and startup orchestration for the MVS serving layer."""

import argparse
import logging
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.serving",
        description="VOX Personalis S1-M6: Minimal Viable Serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m scripts.serving \\
    --checkpoint ./out/capacity_scaling/20260304-103236/checkpoint \\
    --decode_config ./configs/DECODE_V1.json \\
    --host 127.0.0.1 --port 8000 \\
    --silence_ms 1000 --max_utterance_sec 30 \\
    --metrics_out ./out/serving/metrics.jsonl \\
    --verbose
""",
    )

    parser.add_argument("--checkpoint", required=True, help="Path to Model v2 LoRA checkpoint")
    parser.add_argument("--decode_config", required=True, help="Path to DECODE_V1.json")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--silence_ms",
        type=int,
        default=1000,
        metavar="MS",
        help="VAD silence threshold in ms, range 300–2000 (default: 1000)",
    )
    parser.add_argument(
        "--max_utterance_sec",
        type=float,
        default=30.0,
        metavar="SEC",
        help="Max utterance buffer before forced transcription (default: 30)",
    )
    parser.add_argument(
        "--metrics_out",
        default="./out/serving/metrics.jsonl",
        help="Path for rolling JSONL metrics log (default: ./out/serving/metrics.jsonl)",
    )
    parser.add_argument(
        "--no_warmup",
        action="store_true",
        help="Skip warm-up inference pass (testing only)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed logging")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate required files upfront — fast fail before any heavy imports
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[error] Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    decode_config_path = Path(args.decode_config)
    if not decode_config_path.exists():
        print(f"[error] Decode config not found: {decode_config_path}", file=sys.stderr)
        sys.exit(1)

    if not (300 <= args.silence_ms <= 2000):
        print(
            f"[error] --silence_ms must be 300–2000 (got {args.silence_ms})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Configure logging — DEBUG for serving.* when --verbose, suppress noisy libs
    logging.basicConfig(
        format="[%(name)s] %(message)s",
        level=logging.WARNING,
        stream=sys.stdout,
    )
    if args.verbose:
        logging.getLogger("serving").setLevel(logging.DEBUG)

    # Heavy imports deferred until after argument validation
    import uvicorn  # type: ignore[import-untyped]

    from serving import model as model_module
    from serving.api import app, configure
    from serving.metrics import MetricsCollector

    # Set up metrics collector
    collector = MetricsCollector(Path(args.metrics_out))

    # Configure the API app before uvicorn starts accepting connections
    configure(
        collector=collector,
        silence_ms=args.silence_ms,
        max_utterance_sec=args.max_utterance_sec,
    )

    # -----------------------------------------------------------------------
    # Startup sequence
    # -----------------------------------------------------------------------
    try:
        print(f"[startup]  Loading Model v2 from checkpoint: {checkpoint_path}")
        model_module.load_for_serving(
            checkpoint_path=str(checkpoint_path),
            decode_config_path=str(decode_config_path),
            device="cpu",
        )
        print("[startup]  Checkpoint loaded  ✓")

        if not args.no_warmup:
            print("[startup]  Running warm-up inference...")
            model_module.warm_up()
            print("[startup]  Warm-up complete  ✓")
        else:
            # Skip warm-up but still mark model as ready (testing only)
            model_module.model_loaded = True

    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[error] Model load failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[startup]  Model ready. Serving on ws://{args.host}:{args.port}")

    # -----------------------------------------------------------------------
    # Start server
    # -----------------------------------------------------------------------
    log_level = "info" if args.verbose else "warning"
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level=log_level,
        )
    except KeyboardInterrupt:
        print("\n[shutdown] Interrupted — clean exit.")
        sys.exit(130)
