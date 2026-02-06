"""Command-line interface and main pipeline for Dataset v1 creation."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from .cleaning import (
    apply_cleaning_rules,
    compute_hashes_for_dataframe,
    detect_duplicate_audio_different_transcript,
)
from .reporting import (
    generate_excluded_csv,
    generate_frozen_test_csv,
    generate_manifest_csv,
    generate_report_md,
    generate_summary_json,
)
from .splitting import assign_duration_bins, get_split_statistics, stratified_split
from .temporal import temporal_leakage_report
from .validation import check_distribution_balance, validate_split_sizes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VOX Personalis S1-M1 Dataset Versioning - Create Dataset v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m scripts.dataset_versioning \\
    --inventory_dir "./out/inventory/20260205-142601" \\
    --out_dir "./out/dataset_v1" \\
    --seed 42 \\
    --verbose
""",
    )

    # Required arguments
    parser.add_argument(
        "--inventory_dir",
        required=True,
        help="Path to S1-M0 inventory output directory (must contain inventory_files.csv)",
    )

    # Optional arguments
    parser.add_argument(
        "--out_dir",
        default="./out/dataset_v1",
        help="Output directory (default: ./out/dataset_v1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting (default: 42)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split proportion (default: 0.8)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split proportion (default: 0.1)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test split proportion (default: 0.1)",
    )
    parser.add_argument(
        "--duration_bins",
        default="1,3,10,30",
        help="Duration bin edges in seconds, comma-separated (default: 1,3,10,30)",
    )
    parser.add_argument(
        "--skip_temporal_check",
        action="store_true",
        help="Skip temporal clustering analysis",
    )
    parser.add_argument(
        "--allow_small_splits",
        action="store_true",
        help="Allow splits below minimum sample/duration thresholds with WARNING",
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


def run_versioning(args: argparse.Namespace) -> int:
    """
    Main dataset versioning pipeline.

    Returns:
        Exit code (0=success, 1=fatal error, 2=validation failed)
    """
    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.999 <= total_ratio <= 1.001):
        print(
            f"Error: Split ratios must sum to 1.0, got {total_ratio} "
            f"({args.train_ratio} + {args.val_ratio} + {args.test_ratio})",
            file=sys.stderr,
        )
        return 1

    # Validate duration_bins format
    try:
        bins = [float(x) for x in args.duration_bins.split(",")]
        if len(bins) < 2:
            print("Error: --duration_bins must have at least 2 values", file=sys.stderr)
            return 1
        if not all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)):
            print("Error: --duration_bins must be strictly increasing", file=sys.stderr)
            return 1
    except ValueError:
        print("Error: --duration_bins must be comma-separated numbers", file=sys.stderr)
        return 1

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Output directory: {out_dir}")

    # Load inventory
    inventory_dir = Path(args.inventory_dir)
    inventory_csv = inventory_dir / "inventory_files.csv"

    if not inventory_csv.exists():
        print(f"Fatal error: Inventory file not found: {inventory_csv}", file=sys.stderr)
        return 1

    try:
        df = pd.read_csv(inventory_csv)
    except Exception as e:
        print(f"Fatal error: Cannot read inventory CSV: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Loaded inventory: {len(df)} rows")

    # Validate required columns
    required_cols = [
        "file_name",
        "audio_path_resolved",
        "duration_sec",
        "transcript_raw",
        "audio_read_ok",
        "transcript_is_blank",
        "manifest_row_index",
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"Fatal error: Missing required columns: {missing}", file=sys.stderr)
        return 1

    # Phase 1: Compute hashes
    if not args.quiet:
        print("Phase 1: Computing hashes...")

    df = compute_hashes_for_dataframe(df, verbose=args.verbose)

    # Validate hash columns were added
    hash_cols = ["audio_sha256", "transcript_sha256", "pair_sha256"]
    missing_hashes = set(hash_cols) - set(df.columns)
    if missing_hashes:
        print(
            f"Fatal error: Hash computation failed, missing columns: {missing_hashes}",
            file=sys.stderr,
        )
        return 1

    # Phase 2: Apply cleaning rules
    if not args.quiet:
        print("Phase 2: Applying cleaning rules...")

    included_df, excluded_df = apply_cleaning_rules(df, verbose=args.verbose)

    if len(included_df) == 0:
        print("Fatal error: All samples excluded, cannot create dataset", file=sys.stderr)
        return 2

    if not args.quiet:
        print(f"  Included: {len(included_df)}, Excluded: {len(excluded_df)}")

    # Flag duplicate audio with different transcripts
    included_df = detect_duplicate_audio_different_transcript(included_df)
    dup_count = included_df["duplicate_audio_flag"].sum()
    if dup_count > 0 and not args.quiet:
        print(f"  WARNING: {dup_count} samples have duplicate audio with different transcripts")

    # Phase 3: Assign duration bins
    if not args.quiet:
        print("Phase 3: Assigning duration bins...")

    bin_edges = [0] + [float(x) for x in args.duration_bins.split(",")] + [float("inf")]
    included_df = assign_duration_bins(included_df, bin_edges)

    # Phase 4: Stratified splitting
    if not args.quiet:
        print("Phase 4: Performing stratified split...")

    included_df = stratified_split(
        included_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_stats = get_split_statistics(included_df)
    if not args.quiet:
        for split, count in split_stats["split_counts"].items():
            hours = split_stats["split_durations_hours"][split]
            print(f"  {split}: {count} samples, {hours:.2f} hours")

    # Phase 5: Temporal check
    if not args.quiet:
        print("Phase 5: Temporal clustering check...")

    if args.skip_temporal_check:
        temporal_report = {
            "temporal_check_status": "skipped_by_user",
            "temporal_clusters_crossing_splits": None,
            "total_clusters": None,
            "timestamp_coverage_pct": 0.0,
        }
        if not args.quiet:
            print("  Skipped by user request")
    else:
        temporal_report = temporal_leakage_report(included_df, verbose=args.verbose)

    # Phase 6: Validation
    if not args.quiet:
        print("Phase 6: Validating splits...")

    validation_result = validate_split_sizes(included_df, args.allow_small_splits)
    distribution_warnings = check_distribution_balance(included_df)

    if distribution_warnings:
        validation_result.warnings.extend(distribution_warnings)

    if not validation_result.passed:
        for err in validation_result.errors:
            print(f"Validation error: {err}", file=sys.stderr)
        if not args.allow_small_splits:
            print(
                "Use --allow_small_splits to override minimum thresholds",
                file=sys.stderr,
            )
            return 2

    if validation_result.warnings and not args.quiet:
        for warning in validation_result.warnings:
            print(f"  WARNING: {warning}")

    # Phase 7: Add metadata columns and generate outputs
    if not args.quiet:
        print("Phase 7: Generating outputs...")

    # Add constant columns per spec
    included_df["dataset_version"] = "v1"
    included_df["source"] = "euphonia"
    included_df["recording_device"] = "macbook_pro"

    # Ensure transcript metadata columns exist
    if "transcript_len_chars" not in included_df.columns:
        included_df["transcript_len_chars"] = included_df["transcript_raw"].str.len()
    if "transcript_len_words" not in included_df.columns:
        included_df["transcript_len_words"] = included_df["transcript_raw"].str.split().str.len()

    # Config dict for reporting
    config = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "duration_bins": args.duration_bins,
    }

    # Generate all outputs
    generate_manifest_csv(included_df, out_dir / "dataset_v1_manifest.csv", out_dir)
    generate_excluded_csv(excluded_df, out_dir / "dataset_v1_excluded.csv")
    generate_frozen_test_csv(included_df, out_dir / "test_set_v1_frozen.csv")

    summary = generate_summary_json(
        included_df,
        excluded_df,
        validation_result,
        temporal_report,
        config,
        out_dir / "dataset_v1_summary.json",
    )

    generate_report_md(
        summary,
        config,
        validation_result,
        distribution_warnings,
        out_dir / "dataset_v1_report.md",
    )

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("DATASET V1 SUMMARY")
        print("=" * 60)
        print(f"Input samples: {summary['input_manifest_rows']:,}")
        print(f"Excluded samples: {summary['excluded_count']:,}")
        print(f"Final dataset size: {summary['included_count']:,}")
        print(f"Total duration: {sum(summary['split_durations_hours'].values()):.2f} hours")
        print("")
        print("Split distribution:")
        for split in ["train", "val", "test"]:
            count = summary["split_counts"].get(split, 0)
            hours = summary["split_durations_hours"].get(split, 0)
            print(f"  {split}: {count:,} samples, {hours:.2f} hours")
        print("=" * 60)
        print(f"\nOutputs written to: {out_dir}")
        print("  - dataset_v1_manifest.csv")
        print("  - dataset_v1_summary.json")
        print("  - dataset_v1_excluded.csv")
        print("  - test_set_v1_frozen.csv")
        print("  - dataset_v1_report.md")

    return 0


def main() -> int:
    """Entry point."""
    args = parse_args()

    # Validate flags
    if args.quiet and args.verbose:
        print("Error: Cannot use --quiet and --verbose together", file=sys.stderr)
        return 1

    try:
        return run_versioning(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
