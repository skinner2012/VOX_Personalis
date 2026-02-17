"""CLI and pipeline orchestration for fine-tuning."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

# Reuse S1-M2 normalization
from baseline_eval.normalization import create_normalizer

from fine_tuning.audit import (
    append_test_audit_log,
    create_audit_entry,
    enforce_test_policy,
)
from fine_tuning.data import (
    create_hf_dataset,
    load_manifest,
    prepare_dataset,
)
from fine_tuning.evaluation import run_full_evaluation
from fine_tuning.models import (
    load_checkpoint,
    setup_model_and_processor,
)
from fine_tuning.reporting import (
    generate_metrics_json,
    generate_predictions_csv,
    generate_report_md,
    update_experiment_log,
)
from fine_tuning.slices import (
    add_error_type_classification,
    compute_all_slice_metrics,
    load_baseline_predictions,
)
from fine_tuning.training import (
    TrainingConfig,
    save_checkpoint,
    train_model,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VOX Personalis S1-M3: Fine-Tuning with LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training mode (train + eval on val)
  python -m scripts.fine_tuning \\
    --manifest_path "./out/dataset_v1/YYYYMMDD/dataset_v1_manifest.csv" \\
    --baseline_metrics "./out/baseline_eval/YYYYMMDD/baseline_metrics.json"

  # Eval-only mode (test evaluation)
  python -m scripts.fine_tuning \\
    --manifest_path "./out/dataset_v1/YYYYMMDD/dataset_v1_manifest.csv" \\
    --baseline_metrics "./out/baseline_eval/YYYYMMDD/baseline_metrics.json" \\
    --checkpoint_path "./out/fine_tuning/YYYYMMDD/checkpoint" \\
    --eval_split test \\
    --eval_only \\
    --justification "Best val WER among r=8,16 ablations"
        """,
    )

    # Required arguments
    parser.add_argument(
        "--manifest_path",
        required=True,
        help="Path to Dataset v1 manifest CSV",
    )
    parser.add_argument(
        "--baseline_metrics",
        required=True,
        help="Path to S1-M2 baseline_metrics.json",
    )

    # Optional arguments
    parser.add_argument(
        "--out_dir",
        default="./out/fine_tuning",
        help="Output directory (default: ./out/fine_tuning)",
    )
    parser.add_argument(
        "--model",
        default="whisper-base.en",
        choices=["whisper-base.en", "whisper-small.en", "base.en", "small.en"],
        help="Model to fine-tune (default: whisper-base.en)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        choices=[8, 16],
        help="LoRA rank (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--eval_split",
        default="val",
        choices=["val", "test"],
        help="Evaluation split (default: val)",
    )

    # Eval-only mode
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluation only mode (skip training)",
    )
    parser.add_argument(
        "--checkpoint_path",
        help="Path to checkpoint for eval-only mode",
    )
    parser.add_argument(
        "--justification",
        help="Justification for test evaluation (required for --eval_split test)",
    )

    # Baseline predictions for error-type slicing
    parser.add_argument(
        "--baseline_predictions",
        help="Path to baseline_predictions.csv for error-type slicing",
    )

    # Decoding parameters
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size for decoding (1=greedy, default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0=deterministic, default: 0.0)",
    )

    # Logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check required files exist
    if not Path(args.manifest_path).exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest_path}")

    if not Path(args.baseline_metrics).exists():
        raise FileNotFoundError(f"Baseline metrics not found: {args.baseline_metrics}")

    # Eval-only mode requires checkpoint
    if args.eval_only and not args.checkpoint_path:
        raise ValueError("--checkpoint_path required for --eval_only mode")

    # Test evaluation requires justification
    if args.eval_split == "test" and not args.justification:
        raise ValueError("--justification required for test evaluation")

    # Check checkpoint exists in eval-only mode
    if args.eval_only and not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")


def load_baseline_metrics_json(path: str | Path) -> dict:
    """Load baseline metrics JSON."""
    with open(path) as f:
        result: dict = json.load(f)
        return result


def run_training_pipeline(args: argparse.Namespace) -> int:
    """
    Run training + evaluation pipeline.

    Returns exit code.
    """
    verbose = args.verbose and not args.quiet

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Output directory: {out_dir}")

    # Create approach label
    model_short = args.model.replace("whisper-", "").replace(".en", "")
    approach = f"{model_short}_r{args.lora_rank}"

    # Load baseline metrics
    try:
        baseline_metrics = load_baseline_metrics_json(args.baseline_metrics)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Loading baseline metrics: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Loading baseline metrics: invalid JSON - {e}") from e

    # Create normalizer (same as S1-M2)
    normalizer = create_normalizer()

    if verbose:
        print("\n=== Phase 1: Loading Data ===")

    # Load manifest for train and val
    try:
        train_df = load_manifest(args.manifest_path, "train")
    except (FileNotFoundError, ValueError) as e:
        raise type(e)(f"Loading train split: {e}") from e

    try:
        val_df = load_manifest(args.manifest_path, "val")
    except (FileNotFoundError, ValueError) as e:
        raise type(e)(f"Loading val split: {e}") from e

    if verbose:
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")

    # Setup model
    if verbose:
        print("\n=== Phase 2: Loading Model ===")

    model, processor, actual_device = setup_model_and_processor(
        model_name=args.model,
        lora_rank=args.lora_rank,
        device=args.device,
    )

    # Prepare datasets
    if verbose:
        print("\n=== Phase 3: Preparing Datasets ===")

    train_hf = create_hf_dataset(train_df)
    val_hf = create_hf_dataset(val_df)

    train_prepared = prepare_dataset(train_hf, processor, normalizer, verbose=verbose)
    val_prepared = prepare_dataset(val_hf, processor, normalizer, verbose=verbose)

    # Training config
    training_config = TrainingConfig(
        output_dir=str(out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=actual_device,
        disable_tqdm=not verbose,  # Show progress bars when verbose
    )

    # Train
    if verbose:
        print("\n=== Phase 4: Training ===")

    model, training_metrics = train_model(
        model=model,
        processor=processor,
        train_dataset=train_prepared,
        eval_dataset=val_prepared,
        normalizer=normalizer,
        config=training_config,
        verbose=verbose,
    )

    # Save checkpoint
    checkpoint_dir = save_checkpoint(
        model=model,
        processor=processor,
        output_dir=out_dir,
        training_config={
            "model": args.model,
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
    )

    # Evaluate on val
    if verbose:
        print("\n=== Phase 5: Evaluation ===")

    # Ensure model is on correct device (Trainer may have moved it)
    model = model.to(actual_device)

    predictions_df, eval_results = run_full_evaluation(
        model=model,
        processor=processor,
        manifest_df=val_df,
        prepared_dataset=val_prepared,
        normalizer=normalizer,
        baseline_metrics=baseline_metrics,
        split="val",
        device=actual_device,
        verbose=verbose,
        beam_size=args.beam_size,
        temperature=args.temperature,
    )

    # Determine baseline predictions path for error-type slicing
    baseline_pred_path = args.baseline_predictions
    if not baseline_pred_path:
        # Try to find it automatically
        baseline_dir = Path(args.baseline_metrics).parent
        auto_path = baseline_dir / "baseline_predictions.csv"
        if auto_path.exists():
            baseline_pred_path = str(auto_path)

    # Compute slice metrics (this handles error-type classification internally)
    slice_metrics = compute_all_slice_metrics(
        predictions_df,
        baseline_pred_path,
    )

    # If error-type slices were computed, add the classification to predictions_df for CSV output
    if baseline_pred_path and slice_metrics.get("by_error_type"):
        try:
            baseline_pred_df = load_baseline_predictions(baseline_pred_path)
            predictions_df = add_error_type_classification(predictions_df, baseline_pred_df)
        except Exception as e:
            if verbose:
                print(f"WARNING: Could not add error types to predictions: {e}")

    # Generate outputs
    if verbose:
        print("\n=== Phase 6: Generating Outputs ===")

    # Predictions CSV
    pred_path = out_dir / f"{approach}_predictions.csv"
    generate_predictions_csv(predictions_df, pred_path)
    if verbose:
        print(f"Generated: {pred_path}")

    # Metrics JSON
    baseline_ref = {
        "metrics_file": str(args.baseline_metrics),
        "baseline_wer": baseline_metrics.get("aggregate", {}).get("val", {}).get("wer", 0),
        "normalization_version": "textnorm_v1",
        "jiwer_version": baseline_metrics.get("tool_versions", {}).get("jiwer", "unknown"),
    }

    eval_results["epochs_trained"] = training_metrics.get("epochs_trained", args.epochs)

    metrics_path = out_dir / f"{approach}_metrics.json"
    generate_metrics_json(
        model=args.model,
        approach=approach,
        lora_rank=args.lora_rank,
        training_time_sec=training_metrics["training_time_sec"],
        device=actual_device,
        baseline_reference=baseline_ref,
        eval_results=eval_results,
        slice_metrics=slice_metrics,
        wer_history=training_metrics.get("wer_history"),
        output_path=metrics_path,
    )
    if verbose:
        print(f"Generated: {metrics_path}")

    # Report
    report_path = out_dir / "fine_tuning_report.md"
    generate_report_md(
        model=args.model,
        approach=approach,
        lora_rank=args.lora_rank,
        training_time_sec=training_metrics["training_time_sec"],
        device=actual_device,
        baseline_reference=baseline_ref,
        eval_results=eval_results,
        slice_metrics=slice_metrics,
        wer_history=training_metrics.get("wer_history"),
        output_path=report_path,
    )
    if verbose:
        print(f"Generated: {report_path}")

    # Update experiment log
    exp_log_path = Path(args.out_dir) / "experiment_log.csv"
    comparison = eval_results.get("comparison", {})
    update_experiment_log(
        log_path=exp_log_path,
        experiment_id=timestamp,
        model=args.model,
        approach=approach,
        lora_rank=args.lora_rank,
        epochs=training_metrics.get("epochs_trained", args.epochs),
        device=actual_device,
        training_time_sec=training_metrics["training_time_sec"],
        baseline_wer=comparison.get("baseline_wer", 0),
        val_wer=comparison.get("finetuned_wer", 0),
        relative_improvement_pct=comparison.get("relative_improvement_pct", 0),
        checkpoint_path=str(checkpoint_dir),
    )
    if verbose:
        print(f"Updated: {exp_log_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Training time: {training_metrics['training_time_sec'] / 60:.1f} minutes")
    print(f"Val WER: {comparison.get('finetuned_wer', 0):.2%}")
    print(f"Baseline WER: {comparison.get('baseline_wer', 0):.2%}")
    print(f"Improvement: {comparison.get('relative_improvement_pct', 0):.1f}%")
    print(f"Checkpoint: {checkpoint_dir}")
    print("=" * 60)

    return 0


def run_eval_only_pipeline(args: argparse.Namespace) -> int:
    """
    Run evaluation-only pipeline.

    Returns exit code.
    """
    verbose = args.verbose and not args.quiet

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / f"eval_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create approach label
    model_short = args.model.replace("whisper-", "").replace(".en", "")
    # Try to infer LoRA rank from checkpoint
    checkpoint_path = Path(args.checkpoint_path)
    approach = f"{model_short}_eval"

    # Load baseline metrics
    try:
        baseline_metrics = load_baseline_metrics_json(args.baseline_metrics)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Loading baseline metrics: {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Loading baseline metrics: invalid JSON - {e}") from e

    # Create normalizer
    normalizer = create_normalizer()

    # Enforce test policy if evaluating on test
    if args.eval_split == "test":
        config = {
            "model": args.model,
            "checkpoint_path": str(args.checkpoint_path),
        }
        audit_log_path = Path(args.out_dir) / "test_audit_log.csv"
        enforce_test_policy(audit_log_path, config, verbose)

    # Load manifest
    try:
        eval_df = load_manifest(args.manifest_path, args.eval_split)
    except (FileNotFoundError, ValueError) as e:
        raise type(e)(f"Loading {args.eval_split} split: {e}") from e

    if verbose:
        print(f"Eval samples ({args.eval_split}): {len(eval_df)}")

    # Load checkpoint
    if verbose:
        print("\nLoading checkpoint...")

    try:
        model, processor, actual_device = load_checkpoint(
            checkpoint_path=str(args.checkpoint_path),
            base_model_name=args.model,
            device=args.device,
        )
    except (FileNotFoundError, RuntimeError) as e:
        raise type(e)(f"Loading checkpoint: {e}") from e

    # Prepare dataset
    eval_hf = create_hf_dataset(eval_df)
    eval_prepared = prepare_dataset(eval_hf, processor, normalizer, verbose=verbose)

    # Evaluate
    predictions_df, eval_results = run_full_evaluation(
        model=model,
        processor=processor,
        manifest_df=eval_df,
        prepared_dataset=eval_prepared,
        normalizer=normalizer,
        baseline_metrics=baseline_metrics,
        split=args.eval_split,
        device=actual_device,
        verbose=verbose,
        beam_size=args.beam_size,
        temperature=args.temperature,
    )

    # Generate predictions CSV
    pred_path = out_dir / f"{approach}_{args.eval_split}_predictions.csv"
    generate_predictions_csv(predictions_df, pred_path)
    if verbose:
        print(f"Generated: {pred_path}")

    # If test evaluation, update audit log
    if args.eval_split == "test":
        comparison = eval_results.get("comparison", {})

        # Get val WER from previous experiments
        exp_log_path = Path(args.out_dir) / "experiment_log.csv"
        val_wer = 0
        if exp_log_path.exists():
            exp_df = pd.read_csv(exp_log_path)
            # Find matching checkpoint
            matches = exp_df[
                exp_df["checkpoint_path"].str.contains(str(checkpoint_path.name), na=False)
            ]
            if len(matches) > 0:
                val_wer = matches.iloc[-1]["val_wer"]

        audit_entry = create_audit_entry(
            run_id=timestamp,
            model=args.model,
            val_wer=val_wer,
            test_wer=comparison.get("finetuned_wer", 0),
            config=config,
            justification=args.justification,
        )

        audit_log_path = Path(args.out_dir) / "test_audit_log.csv"
        append_test_audit_log(audit_log_path, audit_entry)

        if verbose:
            print(f"Updated: {audit_log_path}")

    # Summary
    comparison = eval_results.get("comparison", {})
    print("\n" + "=" * 60)
    print(f"{args.eval_split.upper()} EVALUATION COMPLETE")
    print("=" * 60)
    print(f"WER: {comparison.get('finetuned_wer', 0):.2%}")
    print(f"Baseline WER: {comparison.get('baseline_wer', 0):.2%}")
    print(f"Improvement: {comparison.get('relative_improvement_pct', 0):.1f}%")
    print("=" * 60)

    return 0


def main() -> int:
    """Main entry point."""
    try:
        args = parse_args()
        validate_args(args)

        if args.eval_only:
            return run_eval_only_pipeline(args)
        else:
            return run_training_pipeline(args)

    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
