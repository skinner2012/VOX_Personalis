# S1-M4 Targeted Improvements & Controlled Model Upgrades — Specification

## Purpose

Systematically reduce error rates beyond Model v1 by applying **controlled,
hypothesis-driven improvements** — without breaking evaluation discipline.

This milestone answers one question:

> *Can we achieve measurable improvement through disciplined, single-variable
> experiments while maintaining rigorous evaluation integrity?*

______________________________________________________________________

## Dependencies

- S1-M3 Personalization & Fine-Tuning completed
- Model v1 checkpoint saved (`best_model/` from S1-M3)
- Dataset v1 manifest frozen (no modifications allowed)
- Model v1 val and test WER recorded in `experiment_log.csv`
- `configs/DECODE_V1.json` created from M3 final decoding configuration
- Evaluation framework with slice analysis operational

______________________________________________________________________

## Test Policy

**M4 does not run test set evaluation.**

The test set has been evaluated in prior milestones and is considered
contaminated for single-shot purposes. All M4 experiment selection is
**val-only**. No decision in M4 may reference test results.

______________________________________________________________________

## Baseline Contract

All M4 experiments MUST compare against the **locked Model v1**. Fill every
`{fill from M3 ...}` field before running any M4 experiment.

### Model v1 Reference (Locked)

| Property      | Value                           | Source            |
| ------------- | ------------------------------- | ----------------- |
| Checkpoint    | `out/fine_tuning/{M3}/best/`    | S1-M3 output      |
| Base Model    | `whisper-base.en`               | M3 training       |
| LoRA Rank     | `r=` **{fill from M3}**         | M3 experiment log |
| Best Epoch    | **{fill from M3}**              | M3 training log   |
| Val WER       | **{fill from M3}** (e.g. 0.664) | `experiment_log`  |
| Decode Config | `configs/DECODE_V1.json`        | Frozen            |

### Text Normalization (Locked)

Text normalization is applied at inference time via `textnorm_v1` as defined
in `scripts/fine_tuning/normalization.py`. The pipeline in M4 MUST call the
same function — do not inline or reimplement.

| Step | Rule                               |
| ---- | ---------------------------------- |
| 1    | Lowercase                          |
| 2    | Remove all punctuation             |
| 3    | Collapse multiple spaces to single |
| 4    | Strip leading/trailing whitespace  |

### DECODE_V1.json (Frozen Decoding Configuration)

`configs/DECODE_V1.json` is the single source of truth for all decode
parameters. This file MUST NOT be modified in M4.

```json
{
  "beam_size": 5,
  "temperature": 0,
  "language": "en",
  "task": "transcribe",
  "fp16": false,
  "no_repeat_ngram_size": null,
  "length_penalty": null,
  "compression_ratio_threshold": null,
  "logprob_threshold": null,
  "no_speech_threshold": null,
  "condition_on_prev_tokens": false,
  "config_version": "DECODE_V1",
  "locked_at": "S1-M3"
}
```

Parameters set to `null` use library defaults and are recorded here for
reproducibility. If any parameter is later set, a new version (`DECODE_V2`)
must be created — never modify `DECODE_V1.json`.

### Reproducibility Seed Policy

All training experiments MUST set the following seeds identically:

| Scope         | API                                  | Value      |
| ------------- | ------------------------------------ | ---------- |
| Python        | `random.seed(N)`                     | **{fill}** |
| NumPy         | `np.random.seed(N)`                  | Same N     |
| PyTorch       | `torch.manual_seed(N)`               | Same N     |
| DataLoader    | `worker_init_fn` with fixed seed     | Same N     |
| Deterministic | `torch.use_deterministic_algorithms` | Off (log)  |

`{fill}` = same seed used in M3 (record in frozen_config.json per run).
Deterministic mode is not required but its status must be recorded.

### Locked Configuration (MUST NOT Change)

| Component       | Specification                        |
| --------------- | ------------------------------------ |
| Dataset         | Dataset v1 manifest (frozen)         |
| Train/Val/Test  | Original splits from S1-M1           |
| Text Norm       | `textnorm_v1` via `normalization.py` |
| Decoding Policy | `DECODE_V1.json` (frozen)            |
| Scoring Library | `jiwer>=3.0` with default settings   |

______________________________________________________________________

## Scope

### Must Have

- **C1** (attention mask) + **C2** (generation config) — inference hygiene
- **B1** (learning rate) + **B2** (dropout) — training regularization
- Controlled Experiment Log with all required fields
- Model v1.1 checkpoint (best val config)
- Improvement Analysis Report

**No test evaluation in M4.**

### Stretch Goal

- **B3** (weight decay) — one additional regularization experiment

### Out of Scope

- **Whisper small.en (A1)** — deferred; will be revisited in M5 if
  M4 improvements are insufficient
- Dataset split modifications
- Data augmentation
- Broad hyperparameter sweeps
- Decoding policy changes
- Architecture changes
- Multi-variable experiments

### Maximum Training Run Count

**≤ 3 training runs** (B1, B2, optional B3). C1 and C2 are inference-only
and do not count.

______________________________________________________________________

## Controlled Experiment Framework

S1-M4 operates under strict **single-variable control**. Each experiment
changes exactly one variable from Model v1.

### Experiment Execution Rules

1. **One variable per experiment** — only the specified variable changes
1. **Fixed seed** — same seed value across all experiments (see above)
1. **Max epochs fixed** — all training runs use `max_epochs = {M3 value}`;
   early stopping may select an earlier checkpoint but cannot exceed this
1. **Val-only comparison** — all decisions based on validation split only
1. **Decide before proceeding** — record ADOPT/REJECT/INVESTIGATE before
   starting the next experiment

### Category C: Inference Hygiene (Start Here — No Retraining)

| Exp | Variable Changed        | Model v1      | New Value        |
| --- | ----------------------- | ------------- | ---------------- |
| C1  | Attention mask handling | Implicit      | Explicit         |
| C2  | Generation config       | Per-call args | `DECODE_V1.json` |

**Hypothesis:** Eliminating decoding instability reduces artificial errors
and improves reproducibility.

**Success criterion for C1/C2:** Reproducibility, not WER.

Run the same model 3× with identical inputs. Acceptable result: WER variance
< 0.2 absolute pts across 3 runs. WER improvement is a bonus, not required.

**Why first:** No training required. Establishes a clean, stable baseline
before any training experiments.

### Category B: Training Regularization

| Exp | Variable      | Model v1 Value | New Value | Max Epochs     |
| --- | ------------- | -------------- | --------- | -------------- |
| B1  | Learning rate | 1e-4           | 5e-5      | = M3 max       |
| B2  | Dropout       | 0.1            | 0.15      | = M3 max       |
| B3  | Weight decay  | 0.0            | 0.01      | = M3 max (opt) |

**Hypothesis:** Improved regularization reduces val → test generalization
gap.

**Max epochs contract:** Every B-category run MUST cap at
`max_epochs = {M3 value}`. This keeps training amount constant and
isolates the variable.
Early stopping only determines which checkpoint is selected.

______________________________________________________________________

## Controlled Experiment Log

### Log Schema (`controlled_experiment_log.csv`)

| Column                 | Type  | Description                              |
| ---------------------- | ----- | ---------------------------------------- |
| `experiment_id`        | str   | C1, C2, B1, B2, B3                       |
| `timestamp`            | str   | ISO 8601 start timestamp                 |
| `category`             | str   | B (regularization) or C (inference)      |
| `hypothesis`           | str   | Stated hypothesis                        |
| `variable_name`        | str   | Exact variable changed                   |
| `baseline_value`       | str   | Model v1 value                           |
| `experiment_value`     | str   | New value tested                         |
| `model_v1_val_wer`     | float | Model v1 val WER (e.g. 0.664)            |
| `experiment_val_wer`   | float | This experiment's val WER                |
| `val_wer_delta`        | float | Absolute improvement (positive = better) |
| `relative_improvement` | float | (delta / model_v1_val_wer) × 100%        |
| `hypothesis_supported` | bool  | True if results support hypothesis       |
| `decision`             | str   | ADOPT / INVESTIGATE / REJECT             |
| `decision_rationale`   | str   | Brief explanation                        |
| `training_time_sec`    | int   | Duration (0 for inference-only)          |
| `checkpoint_path`      | str   | Path to experiment checkpoint            |
| `notes`                | str   | Additional observations                  |

### Decision Thresholds

| Outcome                      | Decision    | Action                    |
| ---------------------------- | ----------- | ------------------------- |
| Val WER improved ≥ 2.0 pts   | ADOPT       | Include in v1.1 candidate |
| Val WER improved 0.5–1.9 pts | INVESTIGATE | Consider combining        |
| Val WER improved ≤ 0.4 pts   | REJECT      | Do not include            |
| Slice regression > 3 pts     | REJECT      | Even if aggregate better  |
| Insertions increased         | REJECT      | Regardless of WER         |

**Note for C1/C2:** Decision threshold is reproducibility (variance < 0.2
pts across 3 runs), not WER delta. Record `val_wer_delta = 0.0` if hygiene
only.

______________________________________________________________________

## Model v1.1 Assembly Rules

1. **Best single change first** — select the ADOPT experiment with highest
   val WER improvement
1. **One optional combination** — if multiple ADOPT results exist, allow
   exactly one (C) + (B) combination
1. **Re-validate combination** — run val on combined checkpoint before
   finalizing; record in experiment log
1. **No multi-B stacking** — do not combine B1 + B2 + B3 (confounding)
1. **Document selection** — record which experiments are included and why
   in `included_experiments.json`

______________________________________________________________________

## Stop Conditions

Stop when ANY condition is met:

| Condition               | Threshold                        | Action           |
| ----------------------- | -------------------------------- | ---------------- |
| **Target achieved**     | Val WER < 60% (absolute)         | Finalize v1.1    |
| **Diminishing returns** | 2 consecutive runs < 1.0 abs pts | Stop experiments |
| **Run cap reached**     | 3 training runs completed        | Stop experiments |

When stopping, record in experiment log: which condition triggered, current
val WER, experiments completed.

______________________________________________________________________

## Execution Plan

Start with inference hygiene (free), then training regularization (moderate
cost). This order ensures we fix correctness issues before training, and
validates improvement incrementally.

### Phase 1: Inference Hygiene (C1, C2)

| Step                           | Output                 |
| ------------------------------ | ---------------------- |
| Implement explicit attn mask   | Code change            |
| Run C1: val eval × 3           | Reproducibility report |
| Implement `DECODE_V1.json` use | Code change            |
| Run C2: val eval × 3           | Reproducibility report |
| Record C1, C2 decisions        | Experiment log entries |

### Phase 2: Training Regularization (B1, B2, optional B3)

| Step                          | Output                   |
| ----------------------------- | ------------------------ |
| Train B1 (lr=5e-5)            | Checkpoint + val metrics |
| Record B1 decision            | Experiment log entry     |
| Train B2 (dropout=0.15)       | Checkpoint + val metrics |
| Record B2 decision            | Experiment log entry     |
| [Optional] Train B3 (wd=0.01) | Checkpoint + val metrics |

### Phase 3: Assembly & Reporting

| Step                           | Output                   |
| ------------------------------ | ------------------------ |
| Assemble Model v1.1 checkpoint | `model_v1.1_checkpoint/` |
| Generate all output artifacts  | Reports + logs           |

______________________________________________________________________

## Evaluation Protocol

### Metrics

| Metric | Description               |
| ------ | ------------------------- |
| WER    | Word Error Rate (primary) |
| CER    | Character Error Rate      |

**Unit convention:** "absolute WER points" throughout M4.
Example: 66.4% → 62.0% = **4.4 absolute WER points** improvement.
Relative: (4.4 / 66.4) × 100% = 6.6% relative improvement.

### Evaluation Slices

#### Duration Bins (from S1-M1)

| Slice  | Duration Range |
| ------ | -------------- |
| Short  | (1, 3\] sec    |
| Medium | (3, 10\] sec   |
| Long   | (10, 30\] sec  |

#### Error-Type Slices (from S1-M2 baseline)

| Slice          | Definition                   |
| -------------- | ---------------------------- |
| Deletion-heavy | Baseline deletions > 50%     |
| Subst-heavy    | Baseline substitutions > 50% |
| Low-error      | Baseline WER < 0.1           |
| High-error     | Baseline WER > 0.5           |

#### Top-20 Worst Samples

Report the 20 highest-WER samples for Model v1 vs Model v1.1:

| Column          | Type  | Description                        |
| --------------- | ----- | ---------------------------------- |
| `file_name`     | str   | Audio file                         |
| `pair_sha256`   | str   | Sample identity hash               |
| `ref_norm`      | str   | Ground truth (normalized)          |
| `hyp_v1_norm`   | str   | Model v1 prediction (normalized)   |
| `hyp_v1_1_norm` | str   | Model v1.1 prediction (normalized) |
| `wer_v1`        | float | Model v1 WER                       |
| `wer_v1_1`      | float | Model v1.1 WER                     |
| `delta`         | float | Absolute improvement               |

______________________________________________________________________

## Outputs

All outputs written to `--out_dir/YYYYMMDD-HHMMSS/`:

### Per-Experiment Outputs

Each experiment writes to `experiments/{experiment_id}/`:

```text
experiments/B1/
├── val_predictions.csv
├── val_metrics.json
├── checkpoint/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_args.json
└── frozen_config.json
```

### frozen_config.json (Per-Run Reproducibility Snapshot)

Written to every experiment output directory. Contains the full
reproducible state of that run:

```json
{
  "run_id": "B1",
  "timestamp": "2026-XX-XXTXX:XX:XXZ",
  "dataset": {
    "manifest_path": "out/dataset_v1/YYYYMMDD/dataset_v1_manifest.csv",
    "manifest_sha256": "{hash}"
  },
  "model_v1_checkpoint": "out/fine_tuning/{M3_RUN_ID}/best_model/",
  "base_model": "whisper-base.en",
  "lora": {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.1
  },
  "training": {
    "learning_rate": 5e-5,
    "max_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "weight_decay": 0.0
  },
  "seeds": {
    "python": 42,
    "numpy": 42,
    "torch": 42,
    "dataloader_worker": 42,
    "deterministic_mode": false
  },
  "decode_config": "configs/DECODE_V1.json",
  "decode_config_hash": "{sha256 of DECODE_V1.json}",
  "textnorm": "textnorm_v1",
  "textnorm_module": "scripts/fine_tuning/normalization.py"
}
```

### Final Outputs

#### `model_v1.1_checkpoint/`

```text
model_v1.1_checkpoint/
├── adapter_config.json
├── adapter_model.safetensors
├── training_args.json
├── frozen_config.json
└── included_experiments.json    ← e.g. ["C1", "B1"]
```

#### `controlled_experiment_log.csv`

Complete experiment audit trail (see schema above).

#### `model_v1.1_val_predictions.csv`

Per-utterance val predictions for Model v1.1.

| Column          | Type  | Description                |
| --------------- | ----- | -------------------------- |
| `file_name`     | str   | Audio file name            |
| `pair_sha256`   | str   | Sample identity hash       |
| `duration_sec`  | float | Audio duration             |
| `duration_bin`  | str   | Duration bin label         |
| `ref_norm`      | str   | Ground truth (normalized)  |
| `hyp_v1_1_norm` | str   | Model v1.1 prediction      |
| `wer`           | float | Word error rate            |
| `cer`           | float | Character error rate       |
| `wer_v1`        | float | Model v1 WER (same sample) |
| `improvement`   | float | Absolute WER improvement   |

#### `model_v1.1_metrics.json`

```json
{
  "model_version": "v1.1",
  "base_model": "whisper-base.en",
  "experiments_included": ["C1", "B1"],
  "model_v1_reference": {
    "checkpoint_path": "out/fine_tuning/{M3_RUN_ID}/best_model/",
    "val_wer": 0.664,
    "lora_rank": 8
  },
  "val_results": {
    "wer": 0.620,
    "cer": 0.220,
    "absolute_improvement_pts": 4.4,
    "relative_improvement_pct": 6.6
  },
  "by_duration_bin": {},
  "by_error_type": {}
}
```

#### `improvement_analysis_report.md`

1. Executive Summary
1. Controlled Experiment Summary (all decisions)
1. WER/CER comparison (val only) — absolute and relative vs Model v1
1. Error type breakdown — deletions / insertions / substitutions
1. Slice-by-slice analysis
1. Top-20 worst samples analysis
1. Recommendations for future milestones

______________________________________________________________________

## CLI Interface

M4 reuses `scripts.fine_tuning` with additional M4-specific flags.

```bash
python -m scripts.fine_tuning \
  --manifest_path "./out/dataset_v1/YYYYMMDD/dataset_v1_manifest.csv" \
  --baseline_metrics "./out/baseline_eval/YYYYMMDD/baseline_metrics.json" \
  --model_v1_checkpoint "./out/fine_tuning/{M3_RUN_ID}/best_model" \
  --decode_config "./configs/DECODE_V1.json" \
  --out_dir "./out/model_improvement" \
  --model whisper-base.en \
  --method lora \
  --lora_rank {M3_BEST_RANK} \
  --experiment_id B1 \
  --lr 5e-5 \
  --device cpu \
  --verbose
```

### Required Arguments

| Argument                | Description                           |
| ----------------------- | ------------------------------------- |
| `--manifest_path`       | Path to Dataset v1 manifest CSV       |
| `--baseline_metrics`    | Path to S1-M2 baseline_metrics.json   |
| `--model_v1_checkpoint` | Path to Model v1 checkpoint directory |
| `--decode_config`       | Path to `DECODE_V1.json`              |

### Optional Arguments

| Argument               | Default              | Description           |
| ---------------------- | -------------------- | --------------------- |
| `--out_dir`            | `./out/model_improv` | Output directory      |
| `--experiment_id`      | None                 | Experiment log label  |
| `--lr`                 | Model v1 value       | Override LR           |
| `--dropout`            | Model v1 value       | Override dropout      |
| `--weight_decay`       | 0.0                  | Override weight decay |
| `--explicit_attn_mask` | False                | Explicit mask (C1)    |
| `--device`             | `cpu`                | Device (cpu only)     |
| `-v, --verbose`        | False                | Show progress         |
| `-q, --quiet`          | False                | Suppress output       |

### Exit Codes

| Code | Meaning                                          |
| ---- | ------------------------------------------------ |
| 0    | Success                                          |
| 1    | Fatal error (files not found, model load failed) |
| 2    | Validation error (no samples, split mismatch)    |
| 3    | Stop condition reached (not an error)            |
| 130  | Interrupted by user (Ctrl+C)                     |

______________________________________________________________________

## Error Handling

| Scenario                      | Behavior                         | Exit |
| ----------------------------- | -------------------------------- | ---- |
| Manifest not found            | Fatal error                      | 1    |
| Model v1 checkpoint not found | Fatal error                      | 1    |
| `DECODE_V1.json` not found    | Fatal error                      | 1    |
| `{fill from M3}` fields empty | Fatal error, list missing fields | 1    |
| Out of memory                 | Suggest reducing batch size      | 1    |
| Training diverges (NaN loss)  | Stop experiment, log failure     | 0    |
| Stop condition reached        | Normal exit with status message  | 3    |
| Keyboard interrupt            | Save checkpoint, clean exit      | 130  |

______________________________________________________________________

## Package Dependencies

### Inherited from S1-M3 (no new packages required)

| Package        | Version  | Purpose              |
| -------------- | -------- | -------------------- |
| `transformers` | >=4.36.0 | HuggingFace models   |
| `peft`         | >=0.7.0  | LoRA implementation  |
| `datasets`     | >=2.14.0 | HuggingFace datasets |
| `accelerate`   | >=0.25.0 | Training utilities   |
| `jiwer`        | >=3.0    | WER/CER (locked)     |
| `torch`        | >=2.0    | PyTorch backend      |
| `pandas`       | >=2.0    | Data manipulation    |

______________________________________________________________________

## Hardware Constraints

- Apple M4 Pro MacBook, 12-16 GB unified memory
- CPU training only (MPS unreliable for Whisper training)

| Configuration            | Estimated Memory | Batch Size |
| ------------------------ | ---------------- | ---------- |
| `whisper-base.en` + LoRA | ~2 GB            | 8          |

Training time per run: calibrate from M3 actuals before scheduling M4.

______________________________________________________________________

## Success Criteria

### Must Have (Required)

1. C1, C2, B1, B2 experiments completed with documented decisions
1. Controlled Experiment Log fully populated
1. Model v1.1 checkpoint saved
1. `frozen_config.json` present in every experiment output
1. Improvement Analysis Report generated

### Target (Quantitative)

| Metric            | Threshold                                       |
| ----------------- | ----------------------------------------------- |
| Val WER reduction | ≥ 5 absolute WER points OR ≥ 10% relative vs v1 |
| Slice regression  | No slice regresses > 3 absolute WER points      |
| Insertions        | No increase vs Model v1                         |

### Stretch

- B3 (weight decay) experiment completed
- Val WER < 60% absolute

**Note:** M4 is successful if the controlled experiment process is completed
with documented findings, even if quantitative targets are not met.

______________________________________________________________________

## Failure Modes if Skipped

- Model v1 remains best available without systematic improvement attempt
- No frozen decode contract (`DECODE_V1.json`) established
- No understanding of which changes work for this speaker
- Future milestones lack a controlled methodology baseline

______________________________________________________________________

## Implementation Notes

M4 does not introduce a new tool package.

- **Reuse** `scripts/fine_tuning/` — extend with M4-specific CLI flags
- **Add** `configs/DECODE_V1.json` — frozen decode configuration
- **Add** `scripts/fine_tuning/experiment_log.py` — experiment log writer

______________________________________________________________________

## References

- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- S1-M3 Personalization & Fine-Tuning Specification
