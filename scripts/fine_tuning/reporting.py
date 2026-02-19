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


# ---------------------------------------------------------------------------
# Controlled experiment output writers
# ---------------------------------------------------------------------------


def write_frozen_config(out_dir: str | Path, config: dict) -> Path:
    """
    Write per-experiment frozen_config.json for reproducibility.

    Args:
        out_dir: Experiment output directory
        config: Full reproducible state of the run

    Returns:
        Path to written file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "frozen_config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    return path


def generate_v1_1_predictions_csv(
    out_dir: str | Path,
    v1_pred_df: pd.DataFrame,
    v1_1_pred_df: pd.DataFrame,
) -> Path:
    """
    Generate side-by-side val predictions CSV for Model v1 vs v1.1.

    Columns per spec: file_name, pair_sha256, duration_sec, duration_bin,
    ref_norm, hyp_v1_norm, hyp_v1_1_norm, wer_v1, wer, cer, improvement.

    Args:
        v1_pred_df: Model v1 val predictions (must include pair_sha256, hypothesis, wer)
        v1_1_pred_df: Model v1.1 val predictions (same schema)

    Returns:
        Path to written file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Merge on pair_sha256
    merged = v1_1_pred_df.copy().reset_index(drop=True)
    v1_lookup = v1_pred_df.set_index("pair_sha256")[["hypothesis", "wer"]]

    merged["hyp_v1_norm"] = merged["pair_sha256"].map(v1_lookup["hypothesis"])
    merged["wer_v1"] = merged["pair_sha256"].map(v1_lookup["wer"])
    merged["improvement"] = (merged["wer_v1"] - merged["wer"]).round(4)

    output_cols = [
        "file_name",
        "pair_sha256",
        "duration_sec",
        "duration_bin",
        "reference",  # ref_norm
        "hyp_v1_norm",
        "hypothesis",  # hyp_v1_1_norm
        "wer_v1",
        "wer",
        "cer",
        "improvement",
    ]
    # Rename to match spec column names
    result = merged[[c for c in output_cols if c in merged.columns]].rename(
        columns={"reference": "ref_norm", "hypothesis": "hyp_v1_1_norm"}
    )

    path = out_dir / "model_v1.1_val_predictions.csv"
    result.to_csv(path, index=False)
    return path


def generate_v1_1_metrics_json(
    out_dir: str | Path,
    v1_1_metrics: dict,
    included_experiments: list[str],
    model_v1_val_wer: float,
    model_v1_checkpoint: str,
    lora_rank: int,
) -> Path:
    """
    Generate model_v1.1_metrics.json per spec schema.

    Returns:
        Path to written file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = v1_1_metrics.get("aggregate", {})
    wer = agg.get("wer", 0.0)
    abs_improvement = model_v1_val_wer - wer
    rel_improvement = (abs_improvement / model_v1_val_wer * 100) if model_v1_val_wer > 0 else 0.0

    metrics = {
        "model_version": "v1.1",
        "base_model": "whisper-base.en",
        "experiments_included": included_experiments,
        "model_v1_reference": {
            "checkpoint_path": model_v1_checkpoint,
            "val_wer": round(model_v1_val_wer, 4),
            "lora_rank": lora_rank,
        },
        "val_results": {
            "wer": round(wer, 4),
            "cer": round(agg.get("cer", 0.0), 4),
            "absolute_improvement_pts": round(abs_improvement, 4),
            "relative_improvement_pct": round(rel_improvement, 2),
        },
        "by_duration_bin": v1_1_metrics.get("by_duration_bin", {}),
        "by_error_type": v1_1_metrics.get("by_error_type", {}),
        "created_timestamp": datetime.now(UTC).isoformat(),
    }

    path = out_dir / "model_v1.1_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    return path


def write_included_experiments(
    out_dir: str | Path,
    experiments: list[str],
) -> Path:
    """
    Write included_experiments.json listing which experiments are in v1.1.

    Returns:
        Path to written file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "included_experiments.json"
    with open(path, "w") as f:
        json.dump(experiments, f, indent=2)
    return path


def generate_improvement_analysis_report(
    out_dir: str | Path,
    experiment_log_entries: list[dict],
    model_v1_val_wer: float,
    v1_1_val_wer: float,
    included_experiments: list[str],
) -> Path:
    """
    Generate improvement_analysis_report.md per spec sections.

    Returns:
        Path to written file
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abs_improvement = model_v1_val_wer - v1_1_val_wer
    rel_improvement = (abs_improvement / model_v1_val_wer * 100) if model_v1_val_wer > 0 else 0.0

    lines = [
        "# Improvement Analysis Report",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- Model v1 val WER: **{model_v1_val_wer:.2%}**",
        f"- Model v1.1 val WER: **{v1_1_val_wer:.2%}**",
        f"- Absolute improvement: **{abs_improvement:.4f} pts**",
        f"- Relative improvement: **{rel_improvement:.1f}%**",
        f"- Experiments included in v1.1: {included_experiments}",
        "",
        "---",
        "",
        "## 2. Controlled Experiment Summary",
        "",
        "| ID | Category | Variable | Baseline | New Value | Val WER | Delta | Decision |",
        "| -- | -------- | -------- | -------- | --------- | ------- | ----- | -------- |",
    ]

    for e in experiment_log_entries:
        lines.append(
            f"| {e.get('experiment_id')} "
            f"| {e.get('category')} "
            f"| {e.get('variable_name')} "
            f"| {e.get('baseline_value')} "
            f"| {e.get('experiment_value')} "
            f"| {float(e.get('experiment_val_wer', 0)):.4f} "
            f"| {float(e.get('val_wer_delta', 0)):+.4f} "
            f"| **{e.get('decision')}** |"
        )

    lines += [
        "",
        "---",
        "",
        "## 3. WER/CER Comparison (val only)",
        "",
        "| Model | Val WER | Abs Improvement |",
        "| ----- | ------- | --------------- |",
        f"| Model v1 | {model_v1_val_wer:.2%} | — |",
        f"| Model v1.1 | {v1_1_val_wer:.2%} | {abs_improvement:.4f} pts |",
        "",
        "---",
        "",
        "## 4. Recommendations for Future Milestones",
        "",
        "- Review REJECT experiments for potential combination opportunities",
        "- Consider Whisper small.en upgrade (A1) if further gains are needed",
        "",
    ]

    path = out_dir / "improvement_analysis_report.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
