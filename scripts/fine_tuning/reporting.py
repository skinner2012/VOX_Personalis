"""Output generation for fine-tuning results."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def generate_predictions_csv(
    predictions_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Generate predictions CSV file.

    Args:
        predictions_df: DataFrame with predictions and metrics
        output_path: Output file path

    Returns:
        Path to generated file
    """
    output_path = Path(output_path)

    # Define column order
    columns = [
        "file_name",
        "pair_sha256",
        "split",
        "duration_sec",
        "duration_bin",
        "baseline_error_type",
        "reference_raw",
        "hypothesis_raw",
        "reference",
        "hypothesis",
        "wer",
        "cer",
        "word_insertions",
        "word_deletions",
        "word_substitutions",
        "baseline_wer",
    ]

    # Only include columns that exist
    available_cols = [c for c in columns if c in predictions_df.columns]

    predictions_df[available_cols].to_csv(output_path, index=False)

    return output_path


def generate_metrics_json(
    model: str,
    approach: str,
    lora_rank: int,
    training_time_sec: float,
    device: str,
    baseline_reference: dict,
    eval_results: dict,
    slice_metrics: dict,
    wer_history: list[dict] | None,
    output_path: str | Path,
) -> Path:
    """
    Generate metrics JSON file.

    Args:
        model: Model name
        approach: Experiment approach label
        lora_rank: LoRA rank used
        training_time_sec: Training time in seconds
        device: Device used
        baseline_reference: Baseline metrics reference
        eval_results: Evaluation results
        slice_metrics: Slice-based metrics
        wer_history: Training WER history
        output_path: Output file path

    Returns:
        Path to generated file
    """
    output_path = Path(output_path)

    metrics = {
        "model": model,
        "approach": approach,
        "method": "lora",
        "lora_rank": lora_rank,
        "epochs_trained": eval_results.get("epochs_trained", 3),
        "training_time_sec": training_time_sec,
        "device": device,
        "baseline_reference": baseline_reference,
        "created_timestamp": datetime.now(UTC).isoformat(),
    }

    # Add evaluation results
    if "aggregate" in eval_results:
        split = eval_results.get("split", "val")
        metrics[f"{split}_results"] = {
            **eval_results["aggregate"],
            **(eval_results.get("comparison", {})),
        }

    # Add slice metrics
    if slice_metrics:
        metrics["by_duration_bin"] = slice_metrics.get("by_duration_bin", {})
        metrics["by_error_type"] = slice_metrics.get("by_error_type", {})

    # Add training history
    if wer_history:
        metrics["training_history"] = wer_history

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    return output_path


def update_experiment_log(
    log_path: str | Path,
    experiment_id: str,
    model: str,
    approach: str,
    lora_rank: int,
    epochs: int,
    device: str,
    training_time_sec: float,
    baseline_wer: float,
    val_wer: float,
    relative_improvement_pct: float,
    checkpoint_path: str,
) -> Path:
    """
    Update experiment log CSV.

    Args:
        log_path: Path to experiment_log.csv
        experiment_id: Unique experiment identifier
        model: Model name
        approach: Experiment label
        lora_rank: LoRA rank
        epochs: Epochs trained
        device: Device used
        training_time_sec: Training time
        baseline_wer: Baseline WER
        val_wer: Validation WER
        relative_improvement_pct: Relative improvement
        checkpoint_path: Path to checkpoint

    Returns:
        Path to log file
    """
    log_path = Path(log_path)

    entry = {
        "experiment_id": experiment_id,
        "model": model,
        "approach": approach,
        "lora_rank": lora_rank,
        "epochs": epochs,
        "device": device,
        "training_time_sec": int(training_time_sec),
        "baseline_wer": baseline_wer,
        "val_wer": val_wer,
        "relative_improvement_pct": relative_improvement_pct,
        "checkpoint_path": str(checkpoint_path),
        "created_timestamp": datetime.now(UTC).isoformat(),
    }

    if log_path.exists():
        df = pd.read_csv(log_path)
        new_row = pd.DataFrame([entry])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(log_path, index=False)

    return log_path


def generate_report_md(
    model: str,
    approach: str,
    lora_rank: int,
    training_time_sec: float,
    device: str,
    baseline_reference: dict,
    eval_results: dict,
    slice_metrics: dict,
    wer_history: list[dict] | None,
    output_path: str | Path,
) -> Path:
    """
    Generate human-readable Markdown report.

    Args:
        model: Model name
        approach: Experiment label
        lora_rank: LoRA rank
        training_time_sec: Training time
        device: Device used
        baseline_reference: Baseline metrics
        eval_results: Evaluation results
        slice_metrics: Slice metrics
        wer_history: Training WER history
        output_path: Output file path

    Returns:
        Path to generated file
    """
    output_path = Path(output_path)

    lines = [
        "# Fine-Tuning Report",
        "",
        "## 1. Overview",
        "",
        f"- **Model:** {model}",
        f"- **Approach:** {approach}",
        f"- **Method:** LoRA (rank={lora_rank})",
        f"- **Device:** {device}",
        f"- **Training time:** {training_time_sec / 60:.1f} minutes",
        f"- **Created:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]

    # Baseline reference
    baseline_file = baseline_reference.get("metrics_file", "N/A")
    baseline_wer = baseline_reference.get("baseline_wer", 0)
    norm_version = baseline_reference.get("normalization_version", "textnorm_v1")
    lines.extend(
        [
            "## 2. Baseline Reference",
            "",
            f"- **Baseline file:** `{baseline_file}`",
            f"- **Baseline WER:** {baseline_wer:.2%}",
            f"- **Normalization:** {norm_version}",
            "",
        ]
    )

    # Evaluation results
    split = eval_results.get("split", "val")
    aggregate = eval_results.get("aggregate", {})
    comparison = eval_results.get("comparison", {})

    sample_count = aggregate.get("sample_count", 0)
    agg_wer = aggregate.get("wer", 0)
    agg_cer = aggregate.get("cer", 0)
    comp_baseline_wer = comparison.get("baseline_wer", 0)
    abs_improvement = comparison.get("absolute_improvement", 0)
    rel_improvement = comparison.get("relative_improvement_pct", 0)
    lines.extend(
        [
            f"## 3. {split.upper()} Results",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Samples | {sample_count} |",
            f"| WER | {agg_wer:.2%} |",
            f"| CER | {agg_cer:.2%} |",
            f"| Baseline WER | {comp_baseline_wer:.2%} |",
            f"| Absolute Improvement | {abs_improvement:.2%} |",
            f"| Relative Improvement | {rel_improvement:.1f}% |",
            "",
        ]
    )

    # Error breakdown
    lines.extend(
        [
            "### Error Breakdown",
            "",
            "| Error Type | Count |",
            "|------------|-------|",
            f"| Insertions | {aggregate.get('insertions', 0)} |",
            f"| Deletions | {aggregate.get('deletions', 0)} |",
            f"| Substitutions | {aggregate.get('substitutions', 0)} |",
            "",
        ]
    )

    # Duration bin slices
    if slice_metrics.get("by_duration_bin"):
        lines.extend(
            [
                "## 4. Performance by Duration",
                "",
                "| Duration Bin | Samples | WER | CER |",
                "|--------------|---------|-----|-----|",
            ]
        )

        for bin_name, m in sorted(slice_metrics["by_duration_bin"].items()):
            count = m.get("sample_count", 0)
            wer = m.get("wer", 0)
            cer = m.get("cer", 0)
            lines.append(f"| {bin_name} | {count} | {wer:.2%} | {cer:.2%} |")
        lines.append("")

    # Error type slices
    if slice_metrics.get("by_error_type"):
        lines.extend(
            [
                "## 5. Performance by Error Type",
                "",
                "| Error Type | Samples | Fine-tuned WER | Baseline WER | Improvement |",
                "|------------|---------|----------------|--------------|-------------|",
            ]
        )

        for error_type, m in sorted(slice_metrics["by_error_type"].items()):
            count = m.get("sample_count", 0)
            wer = m.get("wer", 0)
            bl_wer = m.get("baseline_wer", 0)
            improvement = m.get("improvement", 0)
            lines.append(
                f"| {error_type} | {count} | {wer:.2%} | {bl_wer:.2%} | {improvement:+.2%} |"
            )
        lines.append("")

    # Training history
    if wer_history:
        lines.extend(
            [
                "## 6. Training History",
                "",
                "| Epoch | Val WER |",
                "|-------|---------|",
            ]
        )

        for entry in wer_history:
            lines.append(f"| {entry.get('epoch', 0):.1f} | {entry.get('wer', 0):.2%} |")
        lines.append("")

    # Key takeaways
    improved = comparison.get("improved", False)

    lines.extend(
        [
            "## 7. Key Takeaways",
            "",
        ]
    )

    if improved:
        lines.append(
            f"- ✅ **Improvement achieved:** {rel_improvement:.1f}% relative WER reduction"
        )
    else:
        lines.append("- ❌ **No improvement:** WER increased or unchanged")

    lines.extend(
        [
            f"- Model trained on {device.upper()} in {training_time_sec / 60:.1f} minutes",
            f"- LoRA rank {lora_rank} with ~1-3% trainable parameters",
            "",
        ]
    )

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path
