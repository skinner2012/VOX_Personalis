"""Command-line interface and main pipeline for baseline evaluation."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from .error_analysis import extract_error_patterns
from .inference import load_whisper_model, transcribe_samples
from .metrics import compute_aggregate_metrics, compute_sample_metrics
from .normalization import create_normalizer
from .reporting import (
    generate_baseline_errors_csv,
    generate_baseline_metrics_json,
    generate_baseline_predictions_csv,
    generate_baseline_report_md,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VOX Personalis S1-M2 Baseline Evaluation - Whisper ASR baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m scripts.baseline_eval \\
    --manifest_path "./out/dataset_v1/20260206-142756/dataset_v1_manifest.csv" \\
    --out_dir "./out/baseline_eval" \\
    --model_size small.en \\
    --device mps \\
    --splits test,val \\
    --verbose
""",
    )

    # Required arguments
    parser.add_argument(
        "--manifest_path",
        required=True,
        help="Path to Dataset v1 manifest CSV",
    )

    # Optional arguments
    parser.add_argument(
        "--out_dir",
        default="./out/baseline_eval",
        help="Output directory (default: ./out/baseline_eval)",
    )
    parser.add_argument(
        "--model_size",
        default="small.en",
        choices=["tiny.en", "base.en", "small.en", "medium.en", "large"],
        help="Whisper model size (default: small.en)",
    )
    parser.add_argument(
        "--device",
        default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Device for inference (default: mps)",
    )
    parser.add_argument(
        "--splits",
        default="test,val",
        help="Comma-separated splits to evaluate (default: test,val)",
    )

    # Display / Logging
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable progress bar and detailed logging",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress all output except errors",
    )

    return parser.parse_args()


def run_baseline_eval(args: argparse.Namespace) -> int:
    """
    Main baseline evaluation pipeline.

    Returns:
        Exit code (0=success, 1=fatal error, 2=validation failed)
    """
    # Parse and validate splits
    splits = [s.strip() for s in args.splits.split(",")]
    valid_splits = {"train", "val", "test"}
    invalid_splits = set(splits) - valid_splits
    if invalid_splits:
        print(f"Error: Invalid splits: {invalid_splits}", file=sys.stderr)
        return 1

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Output directory: {out_dir}")

    # Load manifest
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"Fatal error: Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest_df = pd.read_csv(manifest_path)
    except Exception as e:
        print(f"Fatal error: Cannot read manifest CSV: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Loaded manifest: {len(manifest_df)} rows")

    # Validate required columns
    required_cols = [
        "file_name",
        "audio_path_resolved",
        "duration_sec",
        "duration_bin",
        "transcript_raw",
        "pair_sha256",
        "split",
    ]
    missing = set(required_cols) - set(manifest_df.columns)
    if missing:
        print(f"Fatal error: Missing required columns: {missing}", file=sys.stderr)
        return 1

    # Filter by requested splits
    eval_df = manifest_df[manifest_df["split"].isin(splits)].copy()
    if len(eval_df) == 0:
        print(f"Fatal error: No samples in splits: {splits}", file=sys.stderr)
        return 2

    if not args.quiet:
        for split in splits:
            count = len(eval_df[eval_df["split"] == split])
            print(f"  {split}: {count} samples")

    # Load Whisper model
    if not args.quiet:
        print(f"\nLoading Whisper model: {args.model_size} on {args.device}...")

    try:
        model, actual_device = load_whisper_model(args.model_size, args.device)
    except Exception as e:
        print(f"Fatal error: Model load failed: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Model loaded on device: {actual_device}")

    # Create text normalizer
    normalizer = create_normalizer()

    # Run inference and compute per-sample metrics
    if not args.quiet:
        print("\nRunning inference...")

    predictions_df, skipped_samples = transcribe_samples(
        eval_df, model, normalizer, verbose=args.verbose
    )

    if len(predictions_df) == 0:
        print("Fatal error: All samples failed, no predictions generated", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"\nProcessed {len(predictions_df)} samples")
        if skipped_samples["count"] > 0:
            print(f"  Skipped: {skipped_samples['count']} samples")

    # Compute per-sample WER/CER metrics
    if not args.quiet:
        print("Computing metrics...")

    predictions_df = compute_sample_metrics(predictions_df)

    # Compute aggregate metrics
    aggregate_metrics = compute_aggregate_metrics(predictions_df, splits)

    # Extract error patterns
    if not args.quiet:
        print("Analyzing error patterns...")

    error_patterns = extract_error_patterns(predictions_df, top_n=50)

    # Generate outputs
    if not args.quiet:
        print("Generating outputs...")

    generate_baseline_predictions_csv(predictions_df, out_dir / "baseline_predictions.csv")

    generate_baseline_metrics_json(
        aggregate_metrics,
        args.model_size,
        actual_device,
        skipped_samples,
        out_dir / "baseline_metrics.json",
    )

    generate_baseline_errors_csv(error_patterns, out_dir / "baseline_errors.csv")

    generate_baseline_report_md(
        aggregate_metrics,
        error_patterns,
        args.model_size,
        actual_device,
        skipped_samples,
        out_dir / "baseline_report.md",
    )

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("BASELINE EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Model: {args.model_size}")
        print(f"Device: {actual_device}")
        print(f"Samples evaluated: {len(predictions_df)}")
        if skipped_samples["count"] > 0:
            print(f"Samples skipped: {skipped_samples['count']}")
        print("")
        for split in splits:
            if split in aggregate_metrics:
                wer = aggregate_metrics[split]["wer"]
                cer = aggregate_metrics[split]["cer"]
                count = aggregate_metrics[split]["sample_count"]
                print(f"{split.upper()}: WER={wer:.1%}, CER={cer:.1%} ({count} samples)")
        print("=" * 60)
        print(f"\nOutputs written to: {out_dir}")
        print("  - baseline_predictions.csv")
        print("  - baseline_metrics.json")
        print("  - baseline_errors.csv")
        print("  - baseline_report.md")

    return 0


def main() -> int:
    """Entry point."""
    args = parse_args()

    # Validate flags
    if args.quiet and args.verbose:
        print("Error: Cannot use --quiet and --verbose together", file=sys.stderr)
        return 1

    try:
        return run_baseline_eval(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
