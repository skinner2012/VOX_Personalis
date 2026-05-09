"""CLI for Zipformer WER evaluation (M0 stock baseline, M2 post-Gemma)."""

import argparse
import sys

from scripts.zipformer_eval.eval import run_eval


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a streaming Zipformer model (via sherpa-onnx) on a manifest CSV."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV with columns: file_name, transcript_raw (or reference_raw), optionally split.",
    )
    parser.add_argument(
        "--audio-root",
        default="~/Downloads/takeout-E407",
        help="Directory containing the .wav files (default: ~/Downloads/takeout-E407).",
    )
    parser.add_argument(
        "--model-dir",
        default="./models/sherpa-onnx-streaming-zipformer-en-2023-06-21",
        help="Directory with encoder-*.onnx, decoder-*.onnx, joiner-*.onnx, tokens.txt.",
    )
    parser.add_argument(
        "--provider",
        default="cpu",
        choices=["cpu", "coreml"],
        help="ONNX Runtime provider (default: cpu).",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Evaluate only this split (val, test, train). Omit to evaluate all rows.",
    )
    parser.add_argument(
        "--norm-version",
        type=int,
        default=2,
        help="Text normalizer version (default: 2 = create_normalizer(version=2)).",
    )
    parser.add_argument(
        "--output",
        default="./results/S2-M0_zipformer_baseline",
        help="Output directory for predictions.csv and metrics.json.",
    )

    args = parser.parse_args()

    try:
        run_eval(
            manifest_csv=args.manifest,
            audio_root=args.audio_root,
            model_dir=args.model_dir,
            provider=args.provider,
            split=args.split,
            output_dir=args.output,
            norm_version=args.norm_version,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise

    return 0
