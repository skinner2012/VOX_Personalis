"""Report generation for baseline evaluation outputs."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

# Predictions CSV column order
PREDICTIONS_COLUMNS = [
    "file_name",
    "pair_sha256",
    "split",
    "duration_sec",
    "duration_bin",
    "reference_raw",
    "hypothesis_raw",
    "reference",
    "hypothesis",
    "wer",
    "cer",
    "word_insertions",
    "word_deletions",
    "word_substitutions",
]


def generate_baseline_predictions_csv(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate baseline_predictions.csv with per-sample results.

    Args:
        df: DataFrame with predictions and metrics
        output_path: Path to write CSV
    """
    # Select and order columns
    columns_to_write = [c for c in PREDICTIONS_COLUMNS if c in df.columns]
    df[columns_to_write].to_csv(output_path, index=False)


def generate_baseline_metrics_json(
    aggregate_metrics: dict[str, Any],
    model_size: str,
    device: str,
    skipped_samples: dict,
    output_path: Path,
) -> None:
    """
    Generate baseline_metrics.json with aggregate metrics.

    Args:
        aggregate_metrics: Aggregate metrics by split
        model_size: Whisper model size used
        device: Device used for inference
        skipped_samples: Skipped sample tracking
        output_path: Path to write JSON
    """
    # Get library versions
    import jiwer
    import torch
    import whisper  # type: ignore[import-untyped]

    whisper_version = getattr(whisper, "__version__", "unknown")
    jiwer_version = getattr(jiwer, "__version__", "unknown")

    # Restructure for output
    output: dict[str, Any] = {
        "dataset_version": "v1",
        "baseline_model": {
            "name": "whisper",
            "size": model_size,
            "library": "openai-whisper",
            "library_version": whisper_version,
        },
        "evaluation_config": {
            "temperature": 0,
            "language": "en",
            "device": device,
            "normalization": "jiwer_standard",
        },
        "splits_evaluated": list(aggregate_metrics.keys()),
        "aggregate": {},
        "by_duration_bin": {},
    }

    # Populate per-split metrics
    for split, metrics in aggregate_metrics.items():
        output["aggregate"][split] = {
            "sample_count": metrics["sample_count"],
            "total_duration_sec": metrics["total_duration_sec"],
            "total_words": metrics["total_words"],
            "total_chars": metrics["total_chars"],
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "insertions": metrics["insertions"],
            "deletions": metrics["deletions"],
            "substitutions": metrics["substitutions"],
        }

        output["by_duration_bin"][split] = metrics.get("by_duration_bin", {})

    output["skipped_samples"] = skipped_samples
    output["created_timestamp"] = datetime.now(UTC).isoformat()
    output["tool_versions"] = {
        "python": sys.version.split()[0],
        "whisper": whisper_version,
        "jiwer": jiwer_version,
        "torch": torch.__version__,
        "pandas": pd.__version__,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def generate_baseline_errors_csv(error_patterns: dict[str, list[dict]], output_path: Path) -> None:
    """
    Generate baseline_errors.csv with error pattern analysis.

    Args:
        error_patterns: Dict with 'substitutions', 'deletions', 'insertions' lists
        output_path: Path to write CSV
    """
    rows = []

    # Define error type configurations: (key, error_type, ref_field, hyp_field)
    error_configs = [
        ("substitutions", "substitution", "reference_token", "hypothesis_token"),
        ("deletions", "deletion", "reference_token", None),
        ("insertions", "insertion", None, "hypothesis_token"),
    ]

    for key, error_type, ref_field, hyp_field in error_configs:
        for pattern in error_patterns.get(key, []):
            rows.append(
                {
                    "error_type": error_type,
                    "reference_token": pattern.get(ref_field, "") if ref_field else "",
                    "hypothesis_token": pattern.get(hyp_field, "") if hyp_field else "",
                    "count": pattern["count"],
                    "example_files": pattern["example_files"],
                }
            )

    errors_df = pd.DataFrame(rows)
    errors_df.to_csv(output_path, index=False)


def generate_baseline_report_md(
    aggregate_metrics: dict[str, Any],
    error_patterns: dict[str, list[dict]],
    model_size: str,
    device: str,
    skipped_samples: dict,
    output_path: Path,
) -> None:
    """
    Generate baseline_report.md with human-readable summary.

    Args:
        aggregate_metrics: Aggregate metrics by split
        error_patterns: Error pattern analysis
        model_size: Whisper model size
        device: Device used
        skipped_samples: Skipped sample tracking
        output_path: Path to write markdown
    """
    lines = []

    # Header
    lines.append("# Baseline Evaluation Report")
    lines.append("")

    # 1. Overview
    lines.append("## 1. Overview")
    lines.append("")
    lines.append("- **Dataset version:** v1")
    lines.append(f"- **Baseline model:** Whisper {model_size}")
    lines.append(f"- **Device:** {device}")
    lines.append(f"- **Evaluation splits:** {', '.join(aggregate_metrics.keys())}")
    lines.append(f"- **Created:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("")

    # 2. Aggregate Metrics
    lines.append("## 2. Aggregate Metrics")
    lines.append("")
    lines.append("| Split | Samples | Duration | Words | WER | CER |")
    lines.append("|-------|---------|----------|-------|-----|-----|")

    for split in ["test", "val", "train"]:
        if split not in aggregate_metrics:
            continue
        m = aggregate_metrics[split]
        duration_min = m["total_duration_sec"] / 60
        lines.append(
            f"| {split} | {m['sample_count']:,} | {duration_min:.1f} min | "
            f"{m['total_words']:,} | {m['wer']:.1%} | {m['cer']:.1%} |"
        )
    lines.append("")

    # 3. Error Breakdown
    lines.append("## 3. Error Breakdown")
    lines.append("")

    for split in ["test", "val"]:
        if split not in aggregate_metrics:
            continue
        m = aggregate_metrics[split]
        total_errors = m["insertions"] + m["deletions"] + m["substitutions"]

        lines.append(f"### {split.upper()} Split")
        lines.append("")
        lines.append("| Error Type | Count | Proportion |")
        lines.append("|------------|-------|------------|")

        if total_errors > 0:
            lines.append(
                f"| Substitutions | {m['substitutions']:,} | "
                f"{m['substitutions'] / total_errors:.1%} |"
            )
            lines.append(
                f"| Deletions | {m['deletions']:,} | {m['deletions'] / total_errors:.1%} |"
            )
            lines.append(
                f"| Insertions | {m['insertions']:,} | {m['insertions'] / total_errors:.1%} |"
            )
            lines.append(f"| **Total** | **{total_errors:,}** | 100% |")
        else:
            lines.append("| (no errors) | 0 | - |")
        lines.append("")

    # 4. Evaluation Slices (Duration Bins)
    lines.append("## 4. Evaluation Slices")
    lines.append("")
    lines.append("Performance by duration bin:")
    lines.append("")

    for split in ["test", "val"]:
        if split not in aggregate_metrics:
            continue

        by_bin = aggregate_metrics[split].get("by_duration_bin", {})
        if not by_bin:
            continue

        lines.append(f"### {split.upper()} Split")
        lines.append("")
        lines.append("| Duration Bin | Samples | WER | CER | vs Aggregate |")
        lines.append("|--------------|---------|-----|-----|--------------|")

        split_wer = aggregate_metrics[split]["wer"]

        for bin_name in sorted(by_bin.keys()):
            bin_metrics = by_bin[bin_name]
            bin_wer = bin_metrics["wer"]
            bin_cer = bin_metrics["cer"]

            # Compare to aggregate
            if split_wer > 0:
                diff_pct = (bin_wer - split_wer) / split_wer * 100
                if abs(diff_pct) < 5:
                    comparison = "similar"
                elif diff_pct > 0:
                    comparison = f"+{diff_pct:.0f}% higher"
                else:
                    comparison = f"{diff_pct:.0f}% lower"
            else:
                comparison = "-"

            lines.append(
                f"| {bin_name} | {bin_metrics['sample_count']} | "
                f"{bin_wer:.1%} | {bin_cer:.1%} | {comparison} |"
            )
        lines.append("")

    # 5. Error Pattern Analysis
    lines.append("## 5. Error Pattern Analysis")
    lines.append("")

    # Top 10 substitutions
    lines.append("### Top 10 Substitutions")
    lines.append("")
    substitutions = error_patterns.get("substitutions", [])[:10]
    if substitutions:
        lines.append("| Reference | Hypothesis | Count |")
        lines.append("|-----------|------------|-------|")
        for pattern in substitutions:
            lines.append(
                f"| {pattern['reference_token']} | {pattern['hypothesis_token']} | "
                f"{pattern['count']} |"
            )
    else:
        lines.append("(no substitutions)")
    lines.append("")

    # Top 10 deletions
    lines.append("### Top 10 Deletions")
    lines.append("")
    deletions = error_patterns.get("deletions", [])[:10]
    if deletions:
        lines.append("| Deleted Word | Count |")
        lines.append("|--------------|-------|")
        for pattern in deletions:
            lines.append(f"| {pattern['reference_token']} | {pattern['count']} |")
    else:
        lines.append("(no deletions)")
    lines.append("")

    # Top 10 insertions
    lines.append("### Top 10 Insertions")
    lines.append("")
    insertions = error_patterns.get("insertions", [])[:10]
    if insertions:
        lines.append("| Inserted Word | Count |")
        lines.append("|---------------|-------|")
        for pattern in insertions:
            lines.append(f"| {pattern['hypothesis_token']} | {pattern['count']} |")
    else:
        lines.append("(no insertions)")
    lines.append("")

    # 6. Key Takeaways
    lines.append("## 6. Key Takeaways")
    lines.append("")

    # Auto-generate insights based on metrics
    if "test" in aggregate_metrics:
        test_wer = aggregate_metrics["test"]["wer"]
        test_m = aggregate_metrics["test"]

        lines.append("### What the baseline does poorly")
        lines.append("")
        if test_wer > 0.3:
            lines.append(
                f"- High overall WER ({test_wer:.1%}) indicates significant recognition challenges"
            )
        if test_m["deletions"] > test_m["substitutions"]:
            lines.append("- Model tends to miss words (deletions > substitutions)")
        if test_m["insertions"] > test_m["deletions"]:
            lines.append("- Model tends to hallucinate words (insertions > deletions)")

        # Check duration bin patterns
        by_bin = test_m.get("by_duration_bin", {})
        short_bin = by_bin.get("(1.0, 3]", {})
        medium_bin = by_bin.get("(3.0, 10]", {})
        if short_bin and medium_bin:
            if short_bin.get("wer", 0) > medium_bin.get("wer", 0) * 1.2:
                lines.append("- Short utterances have notably higher error rates")

        lines.append("")
        lines.append("### What the baseline does reasonably well")
        lines.append("")
        if test_wer < 0.5:
            lines.append("- Recognizes majority of spoken content")
        if test_m["cer"] < test_m["wer"] * 0.5:
            lines.append("- Character-level accuracy is relatively good (errors are often partial)")
        lines.append("")

        lines.append("### Errors likely addressable via personalization")
        lines.append("")
        lines.append("- Speaker-specific vocabulary and proper nouns")
        lines.append("- Acoustic patterns unique to this speaker")
        if substitutions:
            lines.append("- Consistent substitution patterns (see error analysis)")
        lines.append("")

    # 7. Limitations
    lines.append("## 7. Limitations")
    lines.append("")
    lines.append("- **Single-speaker bias:** Results reflect one speaker's speech patterns")
    lines.append("- **Limited linguistic diversity:** Dataset contains specific utterance types")
    lines.append(
        "- **Baseline model constraints:** Whisper optimized for general speech, not accessibility"
    )
    lines.append("- **Text normalization:** Punctuation removal may affect some comparisons")
    lines.append("")

    # Skipped samples note
    if skipped_samples["count"] > 0:
        lines.append("## Notes")
        lines.append("")
        lines.append(f"**Skipped samples:** {skipped_samples['count']}")
        if skipped_samples["reasons"]:
            for reason, count in skipped_samples["reasons"].items():
                lines.append(f"- {reason}: {count}")
        lines.append("")

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
