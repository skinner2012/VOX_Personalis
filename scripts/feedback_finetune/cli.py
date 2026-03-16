"""CLI for S1-M7 feedback-driven fine-tuning pipeline."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m feedback_finetune",
        description="VOX Personalis S1-M7: Feedback Loop Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m feedback_finetune \\
    --feedback_dir ./out/feedback \\
    --original_manifest ./out/dataset_v1/.../dataset_v1_manifest.csv \\
    --output_dir ./out/feedback_finetune \\
    --model small.en --lora_rank 16 \\
    --device cpu --verbose
""",
    )

    # Required
    parser.add_argument(
        "--feedback_dir", required=True, help="Directory containing correction subdirectories"
    )
    parser.add_argument(
        "--original_manifest", required=True, help="Path to original dataset_v1_manifest.csv"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Root output directory for batch results"
    )

    # Model
    parser.add_argument(
        "--model", default="small.en", help="Base model shorthand (default: small.en)"
    )
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    parser.add_argument("--device", default="cpu", help="Device: cpu, mps, cuda (default: cpu)")

    # Training
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Per-device batch size (default: 4)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Max training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")

    # Misc
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feedback_dir = Path(args.feedback_dir)
    original_manifest_path = Path(args.original_manifest)
    output_root = Path(args.output_dir)

    # Validate inputs before heavy imports
    if not original_manifest_path.exists():
        print(f"[error] Original manifest not found: {original_manifest_path}", file=sys.stderr)
        sys.exit(1)

    # Heavy imports deferred until after validation
    from feedback_finetune.manifest import (
        generate_manifest_rows,
        mark_consumed,
        merge_manifests,
        scan_pending_corrections,
    )

    # ------------------------------------------------------------------
    # Step 1: Scan pending corrections
    # ------------------------------------------------------------------
    print(f"[M7]  Scanning {feedback_dir} for pending corrections...")
    pending = scan_pending_corrections(feedback_dir)
    if not pending:
        print("[M7]  No pending corrections found — exiting.")
        sys.exit(0)
    print(f"[M7]  Found {len(pending)} pending correction(s)")

    # ------------------------------------------------------------------
    # Step 2: Set up batch output directory
    # ------------------------------------------------------------------
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = output_root / f"batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[M7]  Batch directory: {batch_dir}")

    # ------------------------------------------------------------------
    # Step 3: Build merged manifest
    # ------------------------------------------------------------------
    from fine_tuning.data import load_manifest

    print("[M7]  Loading original training and validation manifests...")
    original_train_df = load_manifest(original_manifest_path, "train")
    original_val_df = load_manifest(original_manifest_path, "val")
    print(f"[M7]  Original train: {len(original_train_df)} samples")
    print(f"[M7]  Original val:   {len(original_val_df)} samples")

    print("[M7]  Generating manifest rows from corrections...")
    corrections_df = generate_manifest_rows(pending)
    print(f"[M7]  Corrections:    {len(corrections_df)} rows")

    merged_df = merge_manifests(original_train_df, corrections_df)
    print(f"[M7]  Merged train:   {len(merged_df)} samples")

    merged_csv = batch_dir / "merged_manifest.csv"
    merged_df.to_csv(merged_csv, index=False)
    print(f"[M7]  Saved merged manifest → {merged_csv}")

    # ------------------------------------------------------------------
    # Step 4: Set up model
    # ------------------------------------------------------------------
    from fine_tuning.models import setup_model_and_processor

    print(f"[M7]  Loading {args.model} + LoRA r={args.lora_rank}...")
    model, processor, actual_device = setup_model_and_processor(
        model_name=args.model,
        lora_rank=args.lora_rank,
        device=args.device,
        lora_dropout=args.dropout,
    )

    # ------------------------------------------------------------------
    # Step 5: Prepare datasets
    # ------------------------------------------------------------------
    from baseline_eval.normalization import create_normalizer
    from fine_tuning.data import create_hf_dataset, prepare_dataset

    normalizer = create_normalizer(version=2)

    print("[M7]  Preparing training dataset...")
    train_hf = create_hf_dataset(merged_df)
    train_ds = prepare_dataset(train_hf, processor, normalizer, verbose=args.verbose)

    print("[M7]  Preparing validation dataset...")
    val_hf = create_hf_dataset(original_val_df)
    val_ds = prepare_dataset(val_hf, processor, normalizer, verbose=args.verbose)

    # ------------------------------------------------------------------
    # Step 6: Train
    # ------------------------------------------------------------------
    from fine_tuning.training import TrainingConfig, save_checkpoint, train_model

    config = TrainingConfig(
        output_dir=str(batch_dir / "trainer"),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=actual_device,
        disable_tqdm=not args.verbose,
    )

    print("[M7]  Starting fine-tuning...")
    model, training_metrics = train_model(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        normalizer=normalizer,
        config=config,
        verbose=args.verbose,
    )

    training_metrics_path = batch_dir / "training_metrics.json"
    with training_metrics_path.open("w") as f:
        json.dump(training_metrics, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Step 7: Save checkpoint
    # ------------------------------------------------------------------
    print("[M7]  Saving checkpoint...")
    checkpoint_dir = save_checkpoint(
        model=model,
        processor=processor,
        output_dir=batch_dir,
        training_config={
            "model": args.model,
            "lora_rank": args.lora_rank,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "n_corrections": len(corrections_df),
            "n_original": len(original_train_df),
            "batch_id": batch_id,
        },
    )
    print(f"[M7]  Checkpoint saved → {checkpoint_dir}")

    # ------------------------------------------------------------------
    # Step 8: Evaluate on val set
    # ------------------------------------------------------------------
    from fine_tuning.evaluation import run_full_evaluation

    print("[M7]  Running evaluation on val set...")
    # v2 baseline: Val WER 44.02% (from S1-M5 capacity scaling run)
    baseline_metrics = {"aggregate": {"val": {"wer": 0.4402}}}
    predictions_df, eval_metrics = run_full_evaluation(
        model=model,
        processor=processor,
        manifest_df=original_val_df,
        prepared_dataset=val_ds,
        normalizer=normalizer,
        baseline_metrics=baseline_metrics,
        split="val",
        device=actual_device,
        verbose=args.verbose,
    )

    predictions_csv = batch_dir / "predictions.csv"
    predictions_df.to_csv(predictions_csv, index=False)

    eval_metrics_path = batch_dir / "metrics.json"
    with eval_metrics_path.open("w") as f:
        json.dump(eval_metrics, f, indent=2, default=str)

    new_wer = eval_metrics.get("aggregate", {}).get("wer", 0)
    print(f"[M7]  Val WER: {new_wer:.2%}  (v2 baseline: 44.02%)")

    # ------------------------------------------------------------------
    # Step 9: Mark corrections consumed
    # ------------------------------------------------------------------
    improved = eval_metrics.get("comparison", {}).get("improved", False)
    eval_status = "improved" if improved else "not_improved"
    mark_consumed(
        correction_dirs=pending,
        batch_id=batch_id,
        output_checkpoint=str(checkpoint_dir),
        eval_status=eval_status,
    )
    print(f"[M7]  Marked {len(pending)} correction(s) as consumed (eval_status={eval_status})")

    # ------------------------------------------------------------------
    # Step 10: Comparison report
    # ------------------------------------------------------------------
    _write_comparison_report(batch_dir, eval_metrics, len(corrections_df), len(original_train_df))

    print(f"\n[M7]  Done. Results in {batch_dir}")


def _write_comparison_report(
    batch_dir: Path,
    eval_metrics: dict,
    n_corrections: int,
    n_original: int,
) -> None:
    """Generate a markdown comparison report."""
    agg = eval_metrics.get("aggregate", {})
    cmp = eval_metrics.get("comparison", {})

    v2_wer = float(cmp.get("baseline_wer", 0.4402))
    new_wer = float(agg.get("wer", 0))
    abs_improvement = v2_wer - new_wer
    rel_improvement = float(cmp.get("relative_improvement_pct", 0))
    improved = bool(cmp.get("improved", False))

    report = f"""# S1-M7 Feedback Fine-Tuning — Comparison Report

## Training Data

| Source | Samples |
| --- | --- |
| Original train | {n_original} |
| Corrections | {n_corrections} |
| **Total** | **{n_original + n_corrections}** |

## Val WER Comparison

| Model | Val WER |
| --- | --- |
| v2 (baseline) | {v2_wer:.2%} |
| v2 + corrections | {new_wer:.2%} |

**Absolute improvement:** {abs_improvement:+.2%}
**Relative improvement:** {rel_improvement:+.1f}%
**Result:** {"Improved ✓" if improved else "Not improved ✗"}

## Notes

- Fresh LoRA from base model (not continued from v2 checkpoint)
- Merged original train + corrections to prevent catastrophic forgetting
- textnorm\\_v2 applied to all transcripts during training and evaluation
"""

    report_path = batch_dir / "comparison_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"[M7]  Comparison report → {report_path}")
