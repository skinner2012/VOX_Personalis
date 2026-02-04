"""Report generation functions (JSON, CSV, Markdown)"""

import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]


def compute_histogram(values: list[float | None], bins: list[float]) -> dict[str, int]:
    """
    Compute histogram with specified bins by grouping values into ranges.

    Example:
        >>> compute_histogram([2.5, 7.3, 15.2], bins=[0, 5, 10, 20])
        {'(0.0, 5.0]': 1, '(5.0, 10.0]': 1, '(10.0, 20.0]': 1}

    Args:
        values: List of numeric values (None and NaN are filtered out)
        bins: Bin edges defining ranges (e.g., [0, 5, 10] creates bins 0-5 and 5-10)

    Returns:
        Dictionary mapping bin interval strings to counts.
        Returns empty dict if no valid values, invalid bins, or on error.
    """
    try:
        # Validate bins
        if not bins or len(bins) < 2:
            return {}

        # Filter out None and NaN values
        values = [v for v in values if v is not None and not pd.isna(v)]

        if not values:
            return {}

        # Use pandas cut to bin values
        binned = pd.cut(values, bins=bins, include_lowest=True)
        counts = binned.value_counts().sort_index()

        # Convert to dict with string keys
        result = {str(interval): int(count) for interval, count in counts.items()}

        return result

    except Exception:
        return {}


def generate_summary_json(
    manifest_stats: dict, file_metadata: pd.DataFrame, output_path: Path
) -> dict:
    """
    Generate aggregate summary statistics and write to JSON.

    Args:
        manifest_stats: Manifest integrity check results
        file_metadata: Per-file metadata DataFrame
        output_path: Path to write inventory_summary.json

    Returns:
        Summary dictionary (also written to file)
    """
    # Dataset size
    num_manifest_rows = manifest_stats["num_rows"]
    num_unique_files = file_metadata["file_name"].nunique()

    # Total duration (only readable files)
    readable = file_metadata[file_metadata["audio_read_ok"]]
    total_duration_sec = readable["duration_sec"].sum()

    # Duration histogram
    duration_bins = [0, 1, 3, 10, 30, 60, float("inf")]
    duration_histogram = compute_histogram(readable["duration_sec"].tolist(), duration_bins)

    # Transcript length histogram
    transcript_len_bins = [0, 10, 50, 100, 200, float("inf")]
    transcript_len_histogram = compute_histogram(
        file_metadata["transcript_len_chars"].tolist(), transcript_len_bins
    )

    # Audio distributions
    sample_rate_distribution = file_metadata["sample_rate_hz"].value_counts().to_dict()
    sample_rate_distribution = {
        str(k): int(v) for k, v in sample_rate_distribution.items() if pd.notna(k)
    }

    channels_distribution = file_metadata["channels"].value_counts().to_dict()
    channels_distribution = {
        str(k): int(v) for k, v in channels_distribution.items() if pd.notna(k)
    }

    format_distribution = file_metadata["format"].value_counts().to_dict()
    format_distribution = {str(k): int(v) for k, v in format_distribution.items() if pd.notna(k)}

    read_failure_count = int((~file_metadata["audio_read_ok"]).sum())
    missing_file_count = int((~file_metadata["audio_exists"]).sum())

    # Transcript sanity
    blank_transcript_count = int((file_metadata["transcript_is_blank"]).sum())
    very_short_transcript_count = int((file_metadata["transcript_len_words"] <= 2).sum())

    # Duplicate transcripts (optional)
    duplicate_transcript_count = int(file_metadata["transcript_raw"].duplicated().sum())

    # Silence/Noise distributions
    silence_ratio_bins = [0, 0.1, 0.2, 0.4, 0.6, 1.0]
    silence_ratio_distribution = compute_histogram(
        file_metadata["silence_ratio_est"].tolist(), silence_ratio_bins
    )

    longest_silence_bins = [0, 0.5, 1, 2, 5, float("inf")]
    longest_silence_distribution = compute_histogram(
        file_metadata["longest_silence_sec_est"].tolist(), longest_silence_bins
    )

    rms_db_bins = [float("-inf"), -60, -40, -20, -10, float("inf")]
    rms_db_distribution = compute_histogram(file_metadata["rms_db_est"].tolist(), rms_db_bins)

    # Build summary
    summary = {
        "dataset_size": {
            "num_manifest_rows": num_manifest_rows,
            "num_unique_files": num_unique_files,
            "total_duration_sec": round(float(total_duration_sec), 2),
            "total_duration_hours": round(float(total_duration_sec) / 3600, 2),
            "duration_histogram": duration_histogram,
            "transcript_len_histogram": transcript_len_histogram,
        },
        "audio_distributions": {
            "sample_rate_distribution": sample_rate_distribution,
            "channels_distribution": channels_distribution,
            "format_distribution": format_distribution,
            "read_failure_count": read_failure_count,
            "missing_file_count": missing_file_count,
        },
        "transcript_sanity": {
            "blank_transcript_count": blank_transcript_count,
            "very_short_transcript_count": very_short_transcript_count,
            "duplicate_transcript_count": duplicate_transcript_count,
        },
        "silence_noise_coarse": {
            "silence_ratio_distribution": silence_ratio_distribution,
            "longest_silence_distribution": longest_silence_distribution,
            "rms_db_distribution": rms_db_distribution,
        },
        "manifest_integrity": manifest_stats,
    }

    # Write to JSON
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_files_csv(file_metadata: pd.DataFrame, output_path: Path) -> None:
    """
    Write per-file metadata to CSV.

    Args:
        file_metadata: DataFrame with all per-file columns
        output_path: Path to write inventory_files.csv
    """
    # Sort by file_name for determinism
    file_metadata_sorted = file_metadata.sort_values("file_name")

    # Write to CSV
    file_metadata_sorted.to_csv(output_path, index=False)


def generate_samples_csv(sampled_df: pd.DataFrame, output_path: Path) -> None:
    """
    Write sampled files for manual review to CSV.

    Args:
        sampled_df: Sampled DataFrame
        output_path: Path to write inventory_samples.csv
    """
    # Select columns for manual review
    review_cols = [
        "file_name",
        "duration_sec",
        "transcript_raw",
        "audio_path_resolved",
    ]

    # Add empty columns for manual annotation
    samples = sampled_df[review_cols].copy()
    samples["manual_obvious_error"] = ""
    samples["manual_blank_or_garbled"] = ""
    samples["manual_mismatch_signal"] = ""
    samples["notes"] = ""

    # Write to CSV
    samples.to_csv(output_path, index=False)


def generate_report_md(
    dataset_name: str,
    summary: dict,
    data_dir: Path,
    manifest_csv: Path,
    output_path: Path,
    run_timestamp: str,
    tool_versions: dict,
) -> None:
    """
    Generate human-readable markdown report (1-2 pages).

    Args:
        dataset_name: Name of dataset
        summary: Summary statistics from generate_summary_json (includes manifest_integrity)
        data_dir: Path to audio directory
        manifest_csv: Path to manifest CSV
        output_path: Path to write inventory_report.md
        run_timestamp: Timestamp of run
        tool_versions: Dictionary of tool versions
    """
    lines = []

    # Header
    lines.append("# Data Inventory Report")
    lines.append(f"## {dataset_name}")
    lines.append("")

    # 1. Overview
    lines.append("## 1. Overview")
    lines.append("")
    lines.append(f"- **Dataset**: {dataset_name}")
    lines.append(f"- **Audio Directory**: `{data_dir}`")
    lines.append(f"- **Manifest CSV**: `{manifest_csv}`")
    lines.append(f"- **Run Timestamp**: {run_timestamp}")
    lines.append("")
    lines.append("**Tool Versions**:")
    for tool, version in tool_versions.items():
        lines.append(f"- {tool}: {version}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. Manifest Integrity
    lines.append("## 2. Manifest Integrity")
    lines.append("")

    manifest = summary["manifest_integrity"]
    lines.append(f"- **Total Rows**: {manifest['num_rows']:,}")
    lines.append(f"- **Duplicate file_name Entries**: {manifest['duplicate_file_count']:,}")
    lines.append(f"- **Empty/Null Transcripts**: {manifest['empty_transcript_count']:,}")
    lines.append(f"- **Empty/Null Filenames**: {manifest['empty_filename_count']:,}")
    lines.append("")

    # Add warnings if issues found
    issues = []
    if manifest["duplicate_file_count"] > 0:
        issues.append(f"⚠️  Found {manifest['duplicate_file_count']} duplicate filenames")
    if manifest["empty_transcript_count"] > 0:
        issues.append(f"⚠️  Found {manifest['empty_transcript_count']} empty transcripts")
    if manifest["empty_filename_count"] > 0:
        issues.append(f"⚠️  Found {manifest['empty_filename_count']} empty filenames")

    if issues:
        lines.append("**Issues Detected**:")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    lines.append("---")
    lines.append("")

    # 3. Inventory Summary
    lines.append("## 3. Inventory Summary")
    lines.append("")

    ds = summary["dataset_size"]
    lines.append(f"- **Total Manifest Rows**: {ds['num_manifest_rows']:,}")
    lines.append(f"- **Unique Files**: {ds['num_unique_files']:,}")
    lines.append(
        f"- **Total Duration**: {ds['total_duration_hours']:.2f} hours "
        f"({ds['total_duration_sec']:.1f} seconds)"
    )
    lines.append("")

    audio = summary["audio_distributions"]
    lines.append(f"- **Read Failures**: {audio['read_failure_count']:,}")
    lines.append(f"- **Missing Files**: {audio['missing_file_count']:,}")
    lines.append("")

    lines.append("**File Format Distribution**:")
    for fmt, count in audio["format_distribution"].items():
        lines.append(f"- {fmt}: {count:,}")
    lines.append("")

    lines.append("**Sample Rate Distribution**:")
    for sr, count in audio["sample_rate_distribution"].items():
        lines.append(f"- {sr} Hz: {count:,}")
    lines.append("")

    lines.append("**Channel Distribution**:")
    for ch, count in audio["channels_distribution"].items():
        lines.append(f"- {ch} channel(s): {count:,}")
    lines.append("")

    lines.append("**Duration Distribution**:")
    for bin_label, count in ds["duration_histogram"].items():
        lines.append(f"- {bin_label} seconds: {count:,}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. Transcript Sanity
    lines.append("## 4. Transcript Sanity")
    lines.append("")

    ts = summary["transcript_sanity"]
    total = ds["num_manifest_rows"]

    blank_pct = (ts["blank_transcript_count"] / total * 100) if total > 0 else 0
    short_pct = (ts["very_short_transcript_count"] / total * 100) if total > 0 else 0
    dup_pct = (ts["duplicate_transcript_count"] / total * 100) if total > 0 else 0

    lines.append(f"- **Blank Transcripts**: {ts['blank_transcript_count']:,} ({blank_pct:.2f}%)")
    lines.append(
        f"- **Very Short Transcripts** (≤2 words): {ts['very_short_transcript_count']:,} "
        f"({short_pct:.2f}%)"
    )
    lines.append(
        f"- **Duplicate Transcripts**: {ts['duplicate_transcript_count']:,} ({dup_pct:.2f}%)"
    )
    lines.append("")

    lines.append("**Transcript Length Distribution** (characters):")
    for bin_label, count in ds["transcript_len_histogram"].items():
        lines.append(f"- {bin_label} chars: {count:,}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. Coarse Silence / Noise
    lines.append("## 5. Coarse Silence / Noise (VAD-based)")
    lines.append("")

    sn = summary["silence_noise_coarse"]

    lines.append("**Silence Ratio Distribution** (% non-speech frames):")
    for bin_label, count in sn["silence_ratio_distribution"].items():
        lines.append(f"- {bin_label}: {count:,}")
    lines.append("")

    lines.append("**Longest Silence Distribution** (seconds):")
    for bin_label, count in sn["longest_silence_distribution"].items():
        lines.append(f"- {bin_label}: {count:,}")
    lines.append("")

    lines.append("**RMS dB Distribution**:")
    for bin_label, count in sn["rms_db_distribution"].items():
        lines.append(f"- {bin_label}: {count:,}")
    lines.append("")

    lines.append("**Red Flags**:")
    # Count files with high silence ratio (>0.4)
    # Count files with long silence (>2s)
    # Note: We'd need file_metadata to compute these, so we'll provide placeholders
    lines.append("- Files with silence_ratio > 0.4: *see inventory_files.csv for details*")
    lines.append("- Files with longest_silence > 2.0s: *see inventory_files.csv for details*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. Initial Conclusion
    lines.append("## 6. Initial Conclusion")
    lines.append("")

    # Determine major cleanup needed
    major_issues = []
    if audio["read_failure_count"] > 0:
        major_issues.append(f"Audio read failures ({audio['read_failure_count']} files)")
    if audio["missing_file_count"] > 0:
        major_issues.append(f"Missing audio files ({audio['missing_file_count']} files)")
    if blank_pct > 5:
        major_issues.append(f"High blank transcript rate ({blank_pct:.1f}%)")
    if short_pct > 10:
        major_issues.append(f"High very-short transcript rate ({short_pct:.1f}%)")

    if major_issues:
        lines.append("**Major Cleanup Required**: Yes")
        lines.append("")
        lines.append("**Dominant Failure Modes**:")
        for i, issue in enumerate(major_issues[:3], 1):
            lines.append(f"{i}. {issue}")
    else:
        lines.append("**Major Cleanup Required**: No")
        lines.append("")
        lines.append("Dataset appears to be in good condition with minimal issues.")

    lines.append("")
    lines.append("**Recommended Next Steps**:")
    if major_issues:
        lines.append("1. Review `inventory_files.csv` to identify specific problematic files")
        lines.append("2. Manually inspect sampled files in `inventory_samples.csv`")
        lines.append("3. Design cleanup policy (S1-M1) based on failure modes")
    else:
        lines.append("1. Manually verify sampled files in `inventory_samples.csv`")
        lines.append("2. Proceed to evaluation or model training phase")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*End of Report*")

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
