"""Controlled experiment pipelines for single-variable ablations."""

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from baseline_eval.normalization import create_normalizer

from fine_tuning.data import create_hf_dataset, load_manifest, prepare_dataset
from fine_tuning.evaluation import run_full_evaluation
from fine_tuning.experiment_log import (
    append_controlled_experiment_log,
    create_b_category_entry,
    create_c_category_entry,
    get_adopt_experiments,
    get_best_b_experiment,
    read_controlled_experiment_log,
)
from fine_tuning.models import load_checkpoint, setup_model_and_processor
from fine_tuning.reporting import (
    generate_improvement_analysis_report,
    generate_predictions_csv,
    generate_v1_1_metrics_json,
    generate_v1_1_predictions_csv,
    write_frozen_config,
    write_included_experiments,
)
from fine_tuning.training import TrainingConfig, save_checkpoint, set_seeds, train_model

# Model v1 locked reference from S1-M3
# Measured with DECODE_V1.json (beam=5, temp=0.0) on val split
# Source: out/fine_tuning/decoding_ablation/eval_20260217-121521/
_MODEL_V1_VAL_WER = 0.6478


def _load_baseline_metrics(path: str | Path) -> dict:
    with open(path) as f:
        result: dict = json.load(f)
    return result


def _load_decode_config(path: str) -> dict:
    """Load and return DECODE_V1.json."""
    with open(path) as f:
        result: dict = json.load(f)
    return result


def _run_c_category(args: argparse.Namespace, out_dir: Path, verbose: bool) -> int:
    """
    C1/C2: Inference hygiene — run val eval × n runs, check WER variance.

    C1 (--explicit_attn_mask): explicit attention mask passed to model.generate()
    C2 (no flag):              decode params loaded from DECODE_V1.json
    """
    exp_id = args.experiment_id
    n_runs = args.n_reproducibility_runs
    exp_dir = out_dir / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    descriptions = {
        "C1": "explicit attention mask (all ones) passed to model.generate()",
        "C2": "decode params loaded from DECODE_V1.json instead of per-call args",
    }
    description = descriptions.get(exp_id, exp_id)

    if verbose:
        print(f"\n=== {exp_id}: {description} ===")
        print(f"  Runs: {n_runs} (variance check < 0.2 abs pts)")

    decode_cfg = _load_decode_config(args.decode_config)
    beam_size = decode_cfg["beam_size"]
    temperature = float(decode_cfg["temperature"])

    model, processor, actual_device = load_checkpoint(
        checkpoint_path=args.model_v1_checkpoint,
        base_model_name=args.model,
        device=args.device,
    )

    normalizer = create_normalizer()
    val_df = load_manifest(args.manifest_path, "val")
    val_hf = create_hf_dataset(val_df)
    val_prepared = prepare_dataset(val_hf, processor, normalizer, verbose=verbose)
    baseline_metrics = _load_baseline_metrics(args.baseline_metrics)

    wer_runs: list[float] = []
    for run_idx in range(1, n_runs + 1):
        if verbose:
            print(f"\n  Run {run_idx}/{n_runs}...")

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
            beam_size=beam_size,
            temperature=temperature,
            explicit_attn_mask=(exp_id == "C1"),
        )

        run_wer = eval_results["aggregate"]["wer"]
        wer_runs.append(run_wer)

        pred_path = exp_dir / f"val_predictions_run{run_idx}.csv"
        generate_predictions_csv(predictions_df, pred_path)

        if verbose:
            print(f"  Run {run_idx} WER: {run_wer:.4f}")

    variance = max(wer_runs) - min(wer_runs)
    mean_wer = sum(wer_runs) / len(wer_runs)
    passed = variance < 0.2

    print(
        f"\n  {exp_id} Reproducibility: variance={variance:.4f}, mean={mean_wer:.4f} "
        f"({'PASS' if passed else 'FAIL'} < 0.2 threshold)"
    )

    frozen = {
        "run_id": exp_id,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "model_v1_checkpoint": args.model_v1_checkpoint,
        "decode_config": args.decode_config,
        "decode_config_version": decode_cfg.get("config_version", "DECODE_V1"),
        "explicit_attn_mask": (exp_id == "C1"),
        "n_runs": n_runs,
        "wer_runs": [round(w, 4) for w in wer_runs],
        "variance": round(variance, 4),
        "textnorm": "textnorm_v1",
    }
    write_frozen_config(exp_dir, frozen)

    entry = create_c_category_entry(
        experiment_id=exp_id,
        timestamp=datetime.now().isoformat(),
        description=description,
        variable_name="attention_mask" if exp_id == "C1" else "decode_config_source",
        baseline_value="implicit" if exp_id == "C1" else "per-call args",
        experiment_value="explicit (all ones)" if exp_id == "C1" else "DECODE_V1.json",
        model_v1_val_wer=_MODEL_V1_VAL_WER,
        wer_runs=wer_runs,
    )
    log_path = out_dir / "controlled_experiment_log.csv"
    append_controlled_experiment_log(log_path, entry)

    if verbose:
        print(f"\n  Written: {exp_dir}/frozen_config.json")
        print(f"  Updated: {log_path}")

    return 0


def _run_b_category(args: argparse.Namespace, out_dir: Path, verbose: bool) -> int:
    """
    B1/B2/B3: Training regularization — retrain from scratch with one changed hyperparam.

    B1: learning_rate 1e-4 → 5e-5    (lower LR, reduces overfitting)
    B2: dropout      0.1  → 0.15     (higher LoRA dropout, regularization)
    B3: weight_decay 0.0  → 0.01     (L2 regularization)
    """
    exp_id = args.experiment_id
    exp_dir = out_dir / "experiments" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    experiment_meta = {
        "B1": {
            "description": "lower learning rate 1e-4 → 5e-5",
            "variable_name": "learning_rate",
            "baseline_value": "1e-4",
            "experiment_value": "5e-5",
        },
        "B2": {
            "description": "higher LoRA dropout 0.1 → 0.15",
            "variable_name": "lora_dropout",
            "baseline_value": "0.1",
            "experiment_value": "0.15",
        },
        "B3": {
            "description": "L2 weight decay 0.0 → 0.01",
            "variable_name": "weight_decay",
            "baseline_value": "0.0",
            "experiment_value": "0.01",
        },
    }
    meta = experiment_meta[exp_id]

    if verbose:
        print(f"\n=== {exp_id}: {meta['description']} ===")

    set_seeds(args.seed)

    decode_cfg = _load_decode_config(args.decode_config)
    beam_size = decode_cfg["beam_size"]
    temperature = float(decode_cfg["temperature"])

    normalizer = create_normalizer()
    baseline_metrics = _load_baseline_metrics(args.baseline_metrics)

    train_df = load_manifest(args.manifest_path, "train")
    val_df = load_manifest(args.manifest_path, "val")

    if verbose:
        print(f"  Train: {len(train_df)} samples | Val: {len(val_df)} samples")

    # Setup fresh model (NOT from v1 checkpoint — single-variable control)
    model, processor, actual_device = setup_model_and_processor(
        model_name=args.model,
        lora_rank=args.lora_rank,
        device=args.device,
        lora_dropout=args.dropout,
    )

    train_hf = create_hf_dataset(train_df)
    val_hf = create_hf_dataset(val_df)
    train_prepared = prepare_dataset(train_hf, processor, normalizer, verbose=verbose)
    val_prepared = prepare_dataset(val_hf, processor, normalizer, verbose=verbose)

    # Build training config — only the experiment variable differs from M3
    training_config = TrainingConfig(
        output_dir=str(exp_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=actual_device,
        disable_tqdm=not verbose,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )

    if verbose:
        print(f"  lr={args.lr}, dropout={args.dropout}, weight_decay={args.weight_decay}")

    start = time.time()
    model, _ = train_model(
        model=model,
        processor=processor,
        train_dataset=train_prepared,
        eval_dataset=val_prepared,
        normalizer=normalizer,
        config=training_config,
        verbose=verbose,
    )
    training_time = time.time() - start

    checkpoint_dir = save_checkpoint(
        model=model,
        processor=processor,
        output_dir=exp_dir,
        training_config={
            "experiment_id": exp_id,
            "model": args.model,
            "lora_rank": args.lora_rank,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    )

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
        beam_size=beam_size,
        temperature=temperature,
    )

    exp_val_wer = eval_results["aggregate"]["wer"]
    generate_predictions_csv(predictions_df, exp_dir / "val_predictions.csv")

    val_metrics_path = exp_dir / "val_metrics.json"
    with open(val_metrics_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    decode_hash = hashlib.sha256(json.dumps(decode_cfg, sort_keys=True).encode()).hexdigest()[:16]
    frozen = {
        "run_id": exp_id,
        "description": meta["description"],
        "timestamp": datetime.now().isoformat(),
        "dataset": {"manifest_path": args.manifest_path},
        "base_model": args.model,
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_rank * 2,
            "dropout": args.dropout,
        },
        "training": {
            "learning_rate": args.lr,
            "max_epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "weight_decay": args.weight_decay,
        },
        "seeds": {
            "python": args.seed,
            "numpy": args.seed,
            "torch": args.seed,
            "deterministic_mode": False,
        },
        "decode_config": args.decode_config,
        "decode_config_hash": decode_hash,
        "textnorm": "textnorm_v1",
        "textnorm_module": "scripts/baseline_eval/normalization.py",
    }
    write_frozen_config(exp_dir, frozen)

    # insertion_delta left as 0: v1 insertion count not available at runtime.
    # WER delta is the primary decision signal; insertion guard is a secondary check.
    insertion_delta = 0

    entry = create_b_category_entry(
        experiment_id=exp_id,
        timestamp=datetime.now().isoformat(),
        description=meta["description"],
        variable_name=meta["variable_name"],
        baseline_value=meta["baseline_value"],
        experiment_value=meta["experiment_value"],
        model_v1_val_wer=_MODEL_V1_VAL_WER,
        experiment_val_wer=exp_val_wer,
        training_time_sec=training_time,
        checkpoint_path=str(checkpoint_dir),
        insertion_delta=insertion_delta,
        notes=f"seed={args.seed}",
    )
    log_path = out_dir / "controlled_experiment_log.csv"
    append_controlled_experiment_log(log_path, entry)

    delta = _MODEL_V1_VAL_WER - exp_val_wer
    print(
        f"\n  {exp_id} Result: val WER={exp_val_wer:.4f}, delta={delta:+.4f} → {entry['decision']}"
    )
    print(f"  {entry['decision_rationale']}")
    if verbose:
        print(f"  Updated: {log_path}")

    return 0


def _run_assembly(args: argparse.Namespace, out_dir: Path, verbose: bool) -> int:
    """
    Assembly: read experiment log, select best ADOPT, build Model v1.1 checkpoint.
    """
    if verbose:
        print("\n=== Assembly: Building Model v1.1 ===")

    log_path = out_dir / "controlled_experiment_log.csv"
    if not log_path.exists():
        raise FileNotFoundError(f"Controlled experiment log not found: {log_path}")

    adopted = get_adopt_experiments(log_path)
    best_b = get_best_b_experiment(log_path)

    c_adopted = [e for e in adopted if e["category"] == "C"]

    if best_b:
        b_decision = best_b["decision"]
        print(
            f"  Best B-experiment: {best_b['experiment_id']} "
            f"({b_decision}, delta={best_b['val_wer_delta']})"
        )
        included = [best_b["experiment_id"]]
        best_checkpoint = best_b["checkpoint_path"]

        if c_adopted:
            print(
                f"  Including C ({c_adopted[0]['experiment_id']}) + B ({best_b['experiment_id']})"
            )
            included = [c_adopted[0]["experiment_id"]] + included
    elif adopted:
        best = adopted[0]
        print(f"  No B-experiment to include. Best ADOPT: {best['experiment_id']}")
        included = [best["experiment_id"]]
        best_checkpoint = best["checkpoint_path"]
    else:
        print(
            "  No ADOPT or INVESTIGATE experiments found. Model v1.1 = Model v1 (no improvement)."
        )
        included = []
        best_checkpoint = args.model_v1_checkpoint

    v1_1_dir = out_dir / "model_v1.1_checkpoint"
    if best_checkpoint and Path(best_checkpoint).exists():
        shutil.copytree(best_checkpoint, str(v1_1_dir), dirs_exist_ok=True)
    else:
        shutil.copytree(args.model_v1_checkpoint, str(v1_1_dir), dirs_exist_ok=True)

    write_included_experiments(v1_1_dir, included)

    decode_cfg = _load_decode_config(args.decode_config)
    beam_size = decode_cfg["beam_size"]
    temperature = float(decode_cfg["temperature"])

    normalizer = create_normalizer()
    baseline_metrics = _load_baseline_metrics(args.baseline_metrics)
    val_df = load_manifest(args.manifest_path, "val")

    model, processor, actual_device = load_checkpoint(
        checkpoint_path=str(v1_1_dir),
        base_model_name=args.model,
        device=args.device,
    )
    val_hf = create_hf_dataset(val_df)
    val_prepared = prepare_dataset(val_hf, processor, normalizer, verbose=verbose)

    v1_1_predictions_df, v1_1_eval_results = run_full_evaluation(
        model=model,
        processor=processor,
        manifest_df=val_df,
        prepared_dataset=val_prepared,
        normalizer=normalizer,
        baseline_metrics=baseline_metrics,
        split="val",
        device=actual_device,
        verbose=verbose,
        beam_size=beam_size,
        temperature=temperature,
    )

    v1_1_val_wer = v1_1_eval_results["aggregate"]["wer"]

    v1_pred_path = (
        out_dir / "experiments" / (included[0] if included else "C1") / "val_predictions.csv"
    )
    if v1_pred_path.exists():
        v1_pred_df = pd.read_csv(v1_pred_path)
        generate_v1_1_predictions_csv(out_dir, v1_pred_df, v1_1_predictions_df)

    generate_v1_1_metrics_json(
        out_dir=out_dir,
        v1_1_metrics=v1_1_eval_results,
        included_experiments=included,
        model_v1_val_wer=_MODEL_V1_VAL_WER,
        model_v1_checkpoint=args.model_v1_checkpoint,
        lora_rank=args.lora_rank,
    )

    all_entries = read_controlled_experiment_log(log_path)
    generate_improvement_analysis_report(
        out_dir=out_dir,
        experiment_log_entries=all_entries,
        model_v1_val_wer=_MODEL_V1_VAL_WER,
        v1_1_val_wer=v1_1_val_wer,
        included_experiments=included,
    )

    delta = _MODEL_V1_VAL_WER - v1_1_val_wer
    print(f"\n  Model v1.1 val WER: {v1_1_val_wer:.4f} (delta={delta:+.4f} vs v1)")
    print(f"  Included experiments: {included}")
    print(f"  Checkpoint: {v1_1_dir}")

    return 0


def run_controlled_experiment_pipeline(args: argparse.Namespace) -> int:
    """
    Controlled experiment pipeline for systematic single-variable ablations.

    Routes to inference hygiene (C1/C2), training regularization (B1-B3),
    or model assembly based on --experiment_id.
    All outputs accumulate in --out_dir (no per-run timestamp subdir).
    """
    verbose = args.verbose and not args.quiet

    if not args.model_v1_checkpoint:
        raise ValueError("--model_v1_checkpoint required for controlled experiments")
    if not Path(args.model_v1_checkpoint).exists():
        raise FileNotFoundError(f"Model v1 checkpoint not found: {args.model_v1_checkpoint}")
    if not args.decode_config:
        raise ValueError("--decode_config required for controlled experiments")
    if not Path(args.decode_config).exists():
        raise FileNotFoundError(f"DECODE_V1.json not found: {args.decode_config}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_id = args.experiment_id
    try:
        if exp_id in ("C1", "C2"):
            return _run_c_category(args, out_dir, verbose)
        elif exp_id in ("B1", "B2", "B3"):
            return _run_b_category(args, out_dir, verbose)
        elif exp_id == "assemble":
            return _run_assembly(args, out_dir, verbose)
        else:
            raise ValueError(f"Unknown experiment_id: {exp_id}")
    except StopIteration:
        print(f"\n  Stop condition reached after {exp_id}.")
        return 3
