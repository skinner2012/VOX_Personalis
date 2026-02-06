"""Report generation for Dataset v1 outputs."""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from .validation import ValidationResult

# Manifest column order per spec
MANIFEST_COLUMNS = [
    "dataset_version",
    "file_name",
    "source",
    "manifest_row_index",
    "audio_path_resolved",
    "duration_sec",
    "duration_bin",
    "transcript_raw",
    "transcript_len_chars",
    "transcript_len_words",
    "timestamp_ms",
    "recording_device",
    "audio_sha256",
    "transcript_sha256",
    "pair_sha256",
    "split",
    "duplicate_audio_flag",
]


def generate_manifest_csv(df: pd.DataFrame, output_path: Path, manifest_base_dir: Path) -> None:
    """
    Generate dataset_v1_manifest.csv with relative audio paths.

    Args:
        df: Final DataFrame with all required columns
        output_path: Path to write CSV
        manifest_base_dir: Base directory for computing relative paths
    """
    result = df.copy()

    # Convert audio paths to relative from manifest directory
    def make_relative(abs_path: str) -> str:
        try:
            abs_p = Path(abs_path)
            rel_p = abs_p.relative_to(manifest_base_dir)
            return str(rel_p)
        except ValueError:
            # If not relative, compute proper relative path
            try:
                return str(Path(abs_path).resolve().relative_to(manifest_base_dir.resolve()))
            except ValueError:
                # Fall back to relative path from manifest location
                return str(Path(abs_path))

    result["audio_path_resolved"] = result["audio_path_resolved"].apply(make_relative)

    # Ensure column order per spec
    columns_to_write = [c for c in MANIFEST_COLUMNS if c in result.columns]
    result = result[columns_to_write]

    result.to_csv(output_path, index=False)


def generate_summary_json(
    df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    validation_result: ValidationResult,
    temporal_report: dict,
    config: dict,
    output_path: Path,
) -> dict:
    """
    Generate dataset_v1_summary.json.

    Args:
        df: Final included DataFrame
        excluded_df: Excluded samples DataFrame
        validation_result: Result of validation checks
        temporal_report: Temporal clustering analysis results
        config: Configuration dict with seed, ratios, etc.
        output_path: Path to write JSON

    Returns:
        Summary dict (also written to file)
    """
    # Cleaning summary
    input_count = len(df) + len(excluded_df)
    excluded_count = len(excluded_df)
    included_count = len(df)

    excluded_breakdown = {}
    if len(excluded_df) > 0 and "excluded_reason" in excluded_df.columns:
        excluded_breakdown = excluded_df["excluded_reason"].value_counts().to_dict()

    # Split summary
    split_counts = df["split"].value_counts().to_dict() if "split" in df.columns else {}
    split_durations_sec = (
        df.groupby("split")["duration_sec"].sum().to_dict() if "split" in df.columns else {}
    )
    split_durations_hours = {k: v / 3600 for k, v in split_durations_sec.items()}

    # Duration distributions per split
    split_duration_distributions = {}
    if "split" in df.columns and "duration_bin" in df.columns:
        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split]
            if len(split_df) > 0:
                bin_counts = split_df["duration_bin"].value_counts().to_dict()
                split_duration_distributions[split] = {
                    str(k): int(v) for k, v in bin_counts.items()
                }

    # Quality flags
    duplicate_audio_count = 0
    if "duplicate_audio_flag" in df.columns:
        duplicate_audio_count = int(df["duplicate_audio_flag"].sum())

    # Build summary
    summary = {
        # Cleaning
        "input_manifest_rows": input_count,
        "excluded_count": excluded_count,
        "excluded_breakdown": excluded_breakdown,
        "included_count": included_count,
        # Splits
        "split_counts": split_counts,
        "split_durations_sec": split_durations_sec,
        "split_durations_hours": split_durations_hours,
        "split_duration_distributions": split_duration_distributions,
        # Quality
        "duplicate_audio_different_transcript_count": duplicate_audio_count,
        "temporal_clusters_crossing_splits": temporal_report.get(
            "temporal_clusters_crossing_splits"
        ),
        "temporal_check_status": temporal_report.get("temporal_check_status"),
        # Validation
        "min_sample_validation_passed": validation_result.sample_validation_passed,
        "min_duration_validation_passed": validation_result.duration_validation_passed,
        "split_quality_warnings": validation_result.warnings,
        # Metadata
        "dataset_version": "v1",
        "created_timestamp": datetime.now().isoformat(),
        "spec_version": None,  # Could be git SHA if available
        "seed": config.get("seed", 42),
        "tool_versions": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_excluded_csv(excluded_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate dataset_v1_excluded.csv.

    Args:
        excluded_df: DataFrame with excluded samples
        output_path: Path to write CSV
    """
    columns = [
        "file_name",
        "manifest_row_index",
        "excluded_reason",
        "audio_sha256",
        "transcript_sha256",
    ]
    columns_to_write = [c for c in columns if c in excluded_df.columns]

    if len(excluded_df) > 0:
        excluded_df[columns_to_write].to_csv(output_path, index=False)
    else:
        # Write empty CSV with headers
        pd.DataFrame(columns=columns_to_write).to_csv(output_path, index=False)


def generate_frozen_test_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate test_set_v1_frozen.csv for version locking.

    Contains: file_name, pair_sha256, audio_sha256, transcript_sha256

    Args:
        df: Full dataset DataFrame
        output_path: Path to write CSV
    """
    test_df = df[df["split"] == "test"].copy()

    columns = ["file_name", "pair_sha256", "audio_sha256", "transcript_sha256"]
    columns_to_write = [c for c in columns if c in test_df.columns]

    test_df[columns_to_write].to_csv(output_path, index=False)


def generate_report_md(
    summary: dict,
    config: dict,
    validation_result: ValidationResult,
    distribution_warnings: list[str],
    output_path: Path,
) -> None:
    """
    Generate human-readable dataset_v1_report.md.

    Args:
        summary: Summary dict from generate_summary_json
        config: Configuration dict
        validation_result: Validation result
        distribution_warnings: Warnings from distribution balance check
        output_path: Path to write markdown
    """
    lines = []

    # Header
    lines.append("# Dataset v1 Report")
    lines.append("")
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("- **Dataset version:** v1")
    lines.append("- **Source:** Euphonia recordings via web upload on Macbook Pro")
    lines.append(f"- **Created:** {summary['created_timestamp']}")
    lines.append(f"- **Seed:** {summary['seed']}")
    lines.append("")

    # Cleaning Summary
    lines.append("## 2. Cleaning Summary")
    lines.append("")
    lines.append(f"- **Input samples:** {summary['input_manifest_rows']:,}")
    lines.append(
        f"- **Excluded samples:** {summary['excluded_count']:,} "
        f"({summary['excluded_count'] / summary['input_manifest_rows'] * 100:.1f}%)"
    )
    lines.append("")

    if summary["excluded_breakdown"]:
        lines.append("| Exclusion Reason | Count |")
        lines.append("|-----------------|-------|")
        for reason, count in sorted(summary["excluded_breakdown"].items()):
            lines.append(f"| {reason} | {count:,} |")
        lines.append("")

    lines.append(f"- **Final dataset size:** {summary['included_count']:,}")
    total_hours = sum(summary["split_durations_hours"].values())
    lines.append(f"- **Total duration:** {total_hours:.2f} hours")
    lines.append("")

    # Split Summary
    lines.append("## 3. Split Summary")
    lines.append("")
    lines.append("| Split | Count | Duration (hours) | Percentage |")
    lines.append("|-------|-------|------------------|------------|")

    total_count = summary["included_count"]
    for split in ["train", "val", "test"]:
        count = summary["split_counts"].get(split, 0)
        hours = summary["split_durations_hours"].get(split, 0)
        pct = count / total_count * 100 if total_count > 0 else 0
        lines.append(f"| {split} | {count:,} | {hours:.2f} | {pct:.1f}% |")
    lines.append("")

    # Duration Distribution
    lines.append("## 4. Duration Distribution")
    lines.append("")

    if summary["split_duration_distributions"]:
        # Get all bins
        all_bins_set = set()
        for split_dist in summary["split_duration_distributions"].values():
            all_bins_set.update(split_dist.keys())
        all_bins = sorted(all_bins_set)

        lines.append("| Duration Bin | Train | Val | Test |")
        lines.append("|--------------|-------|-----|------|")

        for bin_label in all_bins:
            train_count = summary["split_duration_distributions"].get("train", {}).get(bin_label, 0)
            val_count = summary["split_duration_distributions"].get("val", {}).get(bin_label, 0)
            test_count = summary["split_duration_distributions"].get("test", {}).get(bin_label, 0)
            lines.append(f"| {bin_label} | {train_count} | {val_count} | {test_count} |")
        lines.append("")

    # Quality Checks
    lines.append("## 5. Quality Checks")
    lines.append("")
    lines.append(
        f"- **Duplicate audio with different transcripts:** "
        f"{summary['duplicate_audio_different_transcript_count']}"
    )

    temporal_status = summary.get("temporal_check_status", "unknown")
    temporal_crossing = summary.get("temporal_clusters_crossing_splits")
    if temporal_status == "completed":
        lines.append(
            f"- **Temporal session leakage:** {temporal_crossing} clusters crossing splits"
        )
    else:
        lines.append(f"- **Temporal session leakage:** Check skipped ({temporal_status})")

    sample_status = "PASS" if validation_result.sample_validation_passed else "FAIL"
    duration_status = "PASS" if validation_result.duration_validation_passed else "FAIL"
    lines.append(f"- **Minimum sample validation:** {sample_status}")
    lines.append(f"- **Minimum duration validation:** {duration_status}")
    lines.append("")

    # Split Quality Assessment
    lines.append("## 6. Split Quality Assessment")
    lines.append("")

    if distribution_warnings:
        lines.append("**Distribution balance warnings:**")
        lines.append("")
        for warning in distribution_warnings:
            lines.append(f"- {warning}")
        lines.append("")
    else:
        lines.append("Duration distributions are balanced across splits.")
        lines.append("")

    if validation_result.warnings:
        lines.append("**Validation warnings:**")
        lines.append("")
        for warning in validation_result.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    if validation_result.errors:
        lines.append("**Validation errors:**")
        lines.append("")
        for error in validation_result.errors:
            lines.append(f"- {error}")
        lines.append("")

    # Recommendation
    if validation_result.passed and not distribution_warnings:
        lines.append("**Recommendation:** READY FOR TRAINING")
    else:
        lines.append("**Recommendation:** NEEDS REVIEW")
    lines.append("")

    # Test Set Lock
    lines.append("## 7. Test Set Lock")
    lines.append("")
    lines.append("- **Test set frozen:** `test_set_v1_frozen.csv`")
    lines.append(f"- **Test samples count:** {summary['split_counts'].get('test', 0)}")
    lines.append("")
    lines.append("**Instructions for future dataset versions:**")
    lines.append("")
    lines.append("1. Load `test_set_v1_frozen.csv`")
    lines.append("2. Preserve all v1 test samples in test split (match by `pair_sha256`)")
    lines.append("3. MAY add new samples to test")
    lines.append("4. MUST NOT move v1 test samples to train/val")
    lines.append("")

    # Next Steps
    lines.append("## 8. Next Steps")
    lines.append("")
    if validation_result.passed:
        lines.append("- Proceed to S1-M2 (Audio Preprocessing)")
    else:
        lines.append("- Review and address validation issues before proceeding")
    lines.append("")

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
