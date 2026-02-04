"""Command-line interface and main pipeline"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from .integrity import check_file_existence, check_manifest_validity
from .metadata import compute_rms_db, extract_audio_metadata
from .reporting import (
    generate_files_csv,
    generate_report_md,
    generate_samples_csv,
    generate_summary_json,
)
from .sampling import stratified_sample_by_duration
from .transcript import analyze_transcript
from .vad import detect_silence_vad


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VOX Personalis S1-M0 Data Inventory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m scripts.data_inventory \\
    --dataset_name "vox-personalis-v1" \\
    --data_dir "./data/audio" \\
    --manifest_csv "./data/labels.csv" \\
    --out_dir "./out" \\
    --seed 42 \\
    --sample_n 100 \\
    --verbose
""",
    )

    # Required arguments
    parser.add_argument("--dataset_name", required=True, help="Dataset name for reporting")
    parser.add_argument("--data_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--manifest_csv", required=True, help="Path to manifest CSV file")

    # Optional arguments
    parser.add_argument("--out_dir", default="./out", help="Output directory (default: ./out)")
    parser.add_argument(
        "--audio_glob", default="**/*", help="Audio file glob pattern (default: **/*)"
    )
    parser.add_argument(
        "--file_col", default="file_name", help="File name column in manifest (default: file_name)"
    )
    parser.add_argument(
        "--text_col",
        default="transcript",
        help="Transcript column in manifest (default: transcript)",
    )
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding (default: utf-8)")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=100,
        help="Number of samples for manual review (default: 100)",
    )
    parser.add_argument(
        "--no-stratify",
        dest="stratify",
        action="store_false",
        help="Disable stratified sampling by duration (enabled by default)",
    )
    parser.add_argument(
        "--stratify_bins",
        default="1,3,10,30",
        help="Duration bins for stratified sampling, comma-separated (default: 1,3,10,30)",
    )
    parser.set_defaults(stratify=True)

    # Display / Logging
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable progress bar and detailed logging"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress all output except errors"
    )

    return parser.parse_args()


def process_file(
    row: pd.Series, data_dir: Path, file_col: str, text_col: str, verbose: bool = False
) -> dict:
    """
    Process a single manifest row and extract all metadata.

    Args:
        row: Pandas Series representing a manifest row
        data_dir: Path to audio directory
        file_col: Name of file column
        text_col: Name of transcript column
        verbose: Whether to print errors

    Returns:
        Dictionary with all per-file metadata
    """
    file_name = row[file_col]
    transcript = row.get(text_col, "")

    # Initialize result
    result = {
        "file_name": file_name,
        "manifest_row_index": row.name,
    }

    # Analyze transcript
    transcript_metrics = analyze_transcript(transcript)
    result.update(transcript_metrics)
    result["transcript_raw"] = transcript

    # Resolve audio path
    audio_path = data_dir / file_name
    result["audio_path_resolved"] = str(audio_path)

    # Check if file exists
    audio_exists = audio_path.exists() and audio_path.is_file()
    result["audio_exists"] = audio_exists

    if not audio_exists:
        # File doesn't exist - fill with nulls
        result.update(
            {
                "audio_read_ok": False,
                "duration_sec": None,
                "sample_rate_hz": None,
                "channels": None,
                "format": None,
                "bit_depth": None,
                "silence_ratio_est": None,
                "longest_silence_sec_est": None,
                "rms_db_est": None,
            }
        )
        return result

    # Extract audio metadata
    try:
        audio_meta = extract_audio_metadata(audio_path)
        result.update(audio_meta)

        if audio_meta["audio_read_ok"]:
            # Compute RMS
            try:
                audio_data, sr = sf.read(audio_path)
                rms = compute_rms_db(audio_data)
                result["rms_db_est"] = rms
            except Exception:
                result["rms_db_est"] = None

            # Detect silence using VAD
            vad_metrics = detect_silence_vad(audio_path)
            result.update(vad_metrics)
        else:
            # Audio read failed
            result["silence_ratio_est"] = None
            result["longest_silence_sec_est"] = None
            result["rms_db_est"] = None

    except Exception as e:
        if verbose:
            print(f"Error processing {file_name}: {e}", file=sys.stderr)
        # Fill with nulls on error
        result.update(
            {
                "audio_read_ok": False,
                "duration_sec": None,
                "sample_rate_hz": None,
                "channels": None,
                "format": None,
                "bit_depth": None,
                "silence_ratio_est": None,
                "longest_silence_sec_est": None,
                "rms_db_est": None,
            }
        )

    return result


def run_inventory(args: argparse.Namespace) -> int:
    """
    Main inventory pipeline.

    Returns:
        Exit code (0=success, 1=fatal error)
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / "inventory" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Output directory: {out_dir}")

    # Load manifest CSV
    try:
        manifest_df = pd.read_csv(args.manifest_csv, encoding=args.encoding)
    except Exception as e:
        print(f"Fatal error: Cannot read manifest CSV: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"Loaded manifest: {len(manifest_df)} rows")

    # Data integrity checks
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Fatal error: Data directory does not exist: {data_dir}", file=sys.stderr)
        return 1

    manifest_stats = check_manifest_validity(manifest_df, args.file_col, args.text_col)

    if not args.quiet:
        print("Checking file existence...")

    missing_files, extra_files = check_file_existence(
        manifest_df, data_dir, args.file_col, args.audio_glob
    )

    if not args.quiet:
        print(f"  Missing files: {len(missing_files)}")
        print(f"  Extra files: {len(extra_files)}")

    # Process each file
    if not args.quiet:
        print(f"Processing {len(manifest_df)} files...")

    results = []
    iterator = manifest_df.iterrows()

    if args.verbose:
        iterator = tqdm(iterator, total=len(manifest_df), desc="Processing files", unit="file")

    for idx, row in iterator:
        try:
            result = process_file(row, data_dir, args.file_col, args.text_col, args.verbose)
            results.append(result)
        except Exception as e:
            if args.verbose:
                tqdm.write(f"Error processing row {idx}: {e}")
            # Continue processing remaining files

    # Convert to DataFrame
    file_metadata = pd.DataFrame(results)

    # Generate aggregate summary
    if not args.quiet:
        print("Generating summary statistics...")

    summary = generate_summary_json(
        manifest_stats, file_metadata, out_dir / "inventory_summary.json"
    )

    # Write files CSV
    if not args.quiet:
        print("Writing per-file metadata CSV...")

    generate_files_csv(file_metadata, out_dir / "inventory_files.csv")

    # Sampling
    if not args.quiet:
        sampling_method = "stratified" if args.stratify else "random"
        print(f"Performing {sampling_method} sampling (n={args.sample_n})...")

    if args.stratify:
        stratify_bins = [float(x) for x in args.stratify_bins.split(",")]
        sampled_df = stratified_sample_by_duration(
            file_metadata, args.sample_n, args.seed, stratify_bins
        )
    else:
        # Simple random sampling from readable files
        readable = file_metadata[file_metadata["audio_read_ok"]]
        sample_size = min(args.sample_n, len(readable))
        sampled_df = readable.sample(n=sample_size, random_state=args.seed)

    if not args.quiet:
        print(f"  Sampled {len(sampled_df)} files")

    generate_samples_csv(sampled_df, out_dir / "inventory_samples.csv")

    # Generate markdown report
    if not args.quiet:
        print("Generating markdown report...")

    # Get tool versions
    tool_versions = {
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "soundfile": sf.__version__,
    }

    generate_report_md(
        dataset_name=args.dataset_name,
        summary=summary,
        data_dir=data_dir,
        manifest_csv=Path(args.manifest_csv),
        output_path=out_dir / "inventory_report.md",
        run_timestamp=timestamp,
        tool_versions=tool_versions,
    )

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("INVENTORY SUMMARY")
        print("=" * 60)
        print(f"Dataset: {args.dataset_name}")
        print(f"Total files: {summary['dataset_size']['num_manifest_rows']:,}")
        print(f"Total duration: {summary['dataset_size']['total_duration_hours']:.2f} hours")
        print(f"Read failures: {summary['audio_distributions']['read_failure_count']:,}")
        print(f"Missing files: {summary['audio_distributions']['missing_file_count']:,}")
        print(f"Blank transcripts: {summary['transcript_sanity']['blank_transcript_count']:,}")
        print("=" * 60)
        print(f"\nOutputs written to: {out_dir}")
        print("  - inventory_summary.json")
        print("  - inventory_files.csv")
        print("  - inventory_samples.csv")
        print("  - inventory_report.md")

    return 0


def main() -> int:
    """Entry point."""
    args = parse_args()

    # Validate flags
    if args.quiet and args.verbose:
        print("Error: Cannot use --quiet and --verbose together", file=sys.stderr)
        return 3

    try:
        return run_inventory(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
