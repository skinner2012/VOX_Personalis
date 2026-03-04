# S1-M5 Model Capacity Scaling — small.en LoRA Fine-Tuning — Specification

## Purpose

Test whether scaling from `base.en` (74M) to `small.en` (244M) resolves the
persistent failure pattern identified in M4a.

This milestone answers one question:

> *Does a 3.3× larger model break through the capacity ceiling that
> `base.en` fine-tuning could not?*

M5 is a single training run. Regardless of the outcome, the project moves to
Min Viable Serving next — no further optimization cycles.

**Interpretation scope:** M5 is a capacity hypothesis check on validation only.
It does not claim final generalization until serving feedback confirms.

______________________________________________________________________

## Context

M4a error analysis diagnosed the primary bottleneck as model capacity:

| Evidence                          | Value                     |
| --------------------------------- | ------------------------- |
| Persistent failures (WER > 0.5    | 205 / 361 samples (56.8%) |
| in both baseline and v1.1)        |                           |
| Top-50 worst persistent failure % | 100% (50/50)              |
| Normalization artifact (H1)       | 1.67 WER pts (minor)      |
| Audio quality correlation (H3)    | r ≈ 0.1 (rejected)        |
| M4a decision                      | `upgrade_model`           |

After ruling out normalization (H1: small, applied in M4b) and audio quality
(H3: no correlation), model capacity (H2) is the remaining hypothesis.

**v1.1 reference WER:** 0.6404 (textnorm_v1) / 0.6237 (textnorm_v2)

**Model naming convention:** This spec uses shorthand names (`base.en`,
`small.en`). These map to HuggingFace IDs `openai/whisper-base.en` and
`openai/whisper-small.en` respectively (see `MODEL_MAPPING` in
`scripts/fine_tuning/models.py`). The CLI accepts both short and long forms.

______________________________________________________________________

## Dependencies

- S1-M4b complete (`textnorm_v2` ready in `scripts/baseline_eval/normalization.py`)
- Dataset v1 manifest (`out/dataset_v1/20260206-142756/dataset_v1_manifest.csv`)
- DECODE_V1.json (`configs/DECODE_V1.json`)
- Baseline metrics (`out/baseline_eval/20260210-220646/baseline_metrics.json`)
- Model v1 checkpoint (optional — for report comparison only; if absent, use
  v1.1 reference WER = 0.6237)
- Training infrastructure (`scripts/fine_tuning/`)

______________________________________________________________________

## Scope

### In Scope

- Single `small.en` LoRA fine-tuning run on Dataset v1
- Val WER/CER evaluation with `textnorm_v2`
- Comparison against Model v1.1 (both scored with `textnorm_v2`)
- Model v2 checkpoint and reproducibility snapshot
- Brief comparison report

### Out of Scope

- Hyperparameter grid search or ablation
- Decoding parameter changes (DECODE_V1 remains frozen)
- Architecture changes beyond model size
- Data augmentation or new data collection
- Test set evaluation
- Further optimization if results are poor

______________________________________________________________________

## Baseline Contract

All M5 results compare against Model v1.1 (locked).

### Model v1.1 Reference (Locked)

| Property      | Value                                         |
| ------------- | --------------------------------------------- |
| Checkpoint    | `out/model_improvement/model_v1.1_checkpoint` |
| Base Model    | `base.en` (74M)                               |
| LoRA Rank     | 16                                            |
| LoRA Dropout  | 0.15                                          |
| Val WER       | 0.6404 (textnorm_v1) / 0.6237 (textnorm_v2)   |
| Decode Config | `configs/DECODE_V1.json`                      |

### Text Normalization

M5 uses `textnorm_v2` exclusively. The v1.1 comparison baseline under
`textnorm_v2` is **0.6237** (measured in M4b via per-sample re-scoring).

### Locked Configuration (MUST NOT Change)

| Component      | Specification                                    |
| -------------- | ------------------------------------------------ |
| Dataset        | Dataset v1 manifest (frozen)                     |
| Train/Val/Test | Original splits from S1-M1                       |
| Text Norm      | `textnorm_v2` via `create_normalizer(version=2)` |
| Decoding       | `DECODE_V1.json` (frozen)                        |
| Scoring        | `jiwer >= 3.0` with default settings             |
| Seeds          | 42 (Python, NumPy, PyTorch)                      |

______________________________________________________________________

## Training Configuration (Model v2)

| Parameter               | Value          | Rationale                     |
| ----------------------- | -------------- | ----------------------------- |
| Base model              | `small.en`     | 244M params (3.3× base.en)    |
| LoRA rank               | 16             | Same as v1.1                  |
| LoRA alpha              | 32             | 2× rank (same as v1.1)        |
| LoRA dropout            | 0.15           | v1.1 winner, carry forward    |
| LoRA target modules     | q_proj, v_proj | Same as v1.1                  |
| Learning rate           | 1e-4           | M3/M4 default (5e-5 diverged) |
| Epochs                  | 3              | Same as v1.1                  |
| Batch size              | 4              | Constrained by memory         |
| Gradient accumulation   | 4              | Effective batch = 16          |
| Warmup steps            | 100            | Same as v1.1                  |
| Weight decay            | 0.0            | Same as v1.1                  |
| Early stopping patience | 2              | Same as v1.1                  |
| fp16                    | false          | CPU training                  |
| Device                  | cpu            | Apple M4 Pro (MPS unreliable) |

**Only intended variable from v1.1:** base model (`base.en` → `small.en`).
If OOM fallback triggers, batch/accum change is logged as a variant (see
Hardware Considerations).

______________________________________________________________________

## Hardware Considerations

| Model             | Est. Memory | Batch Size |
| ----------------- | ----------- | ---------- |
| `base.en` + LoRA  | ~2 GB       | 4–8        |
| `small.en` + LoRA | ~4 GB       | 4          |

Training time estimate: `base.en` took ~31 min/run (M4 actuals). `small.en`
estimated ~90–120 min/run (3–4× scaling).

**OOM fallback protocol:** If OOM occurs at batch_size=4, one restart is
permitted with batch_size=2 and gradient_accumulation=8 (preserving effective
batch = 16). The restart MUST:

- Use experiment_id `small_en_oom_fallback`
- Record `"oom_fallback": true` and actual batch/accum in `frozen_config.json`
- Note "OOM fallback variant" in `comparison_report.md`

No other restarts are allowed — any further failure is a fatal error.

______________________________________________________________________

## Code Changes

Minimal — reuse existing `scripts/fine_tuning/` pipeline.

### 1. Normalizer version

Update `create_normalizer()` → `create_normalizer(version=2)` in:

- `scripts/fine_tuning/cli.py` (lines 291, 531)
- `scripts/fine_tuning/experiments.py` (lines 89, 211, 417)

### 2. Frozen config

Record `"textnorm": "textnorm_v2"` in `frozen_config.json` output.

### 3. CLI invocation

Pass `--model small.en` (already supported by arg parser).

______________________________________________________________________

## CLI Interface

```bash
python -m scripts.fine_tuning \
  --manifest_path "./out/dataset_v1/20260206-142756/dataset_v1_manifest.csv" \
  --baseline_metrics \
    "./out/baseline_eval/20260210-220646/baseline_metrics.json" \
  --decode_config "./configs/DECODE_V1.json" \
  --out_dir "./out/capacity_scaling" \
  --model small.en \
  --lora_rank 16 \
  --dropout 0.15 \
  --experiment_id small_en \
  --seed 42 \
  --device cpu \
  --verbose
```

`--model_v1_checkpoint` is optional. If provided, the pipeline uses it for
baseline comparison in the report. If omitted, the report uses the v1.1
reference WER (0.6237) directly.

### Exit Codes

| Code | Meaning                                          |
| ---- | ------------------------------------------------ |
| 0    | Success                                          |
| 1    | Fatal error (files not found, model load failed) |
| 2    | Validation error (no samples, split mismatch)    |
| 130  | Interrupted by user (Ctrl+C)                     |

______________________________________________________________________

## Outputs

All outputs written to `out/capacity_scaling/YYYYMMDD-HHMMSS/`:

```text
out/capacity_scaling/YYYYMMDD-HHMMSS/
├── val_predictions.csv          # Per-utterance predictions (textnorm_v2)
├── val_metrics.json             # Aggregate WER/CER
├── checkpoint/                  # Model v2 LoRA adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── training_args.json
├── frozen_config.json           # Full reproducibility snapshot
└── comparison_report.md          # v1.1 vs v2 comparison
```

### frozen_config.json

```json
{
  "run_id": "small_en",
  "description": "Model capacity scaling: base.en → small.en",
  "timestamp": "2026-XX-XXTXX:XX:XXZ",
  "dataset": {
    "manifest_path": "out/dataset_v1/20260206-142756/dataset_v1_manifest.csv"
  },
  "base_model": "small.en",
  "lora": {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.15
  },
  "training": {
    "learning_rate": 1e-4,
    "max_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "warmup_steps": 100,
    "weight_decay": 0.0,
    "oom_fallback": false
  },
  "seeds": {
    "python": 42,
    "numpy": 42,
    "torch": 42,
    "deterministic_mode": false
  },
  "decode_config": "configs/DECODE_V1.json",
  "textnorm": "textnorm_v2",
  "textnorm_module": "scripts/baseline_eval/normalization.py"
}
```

### val_predictions.csv Schema

| Column         | Type  | Description                       |
| -------------- | ----- | --------------------------------- |
| `file_name`    | str   | Audio file name                   |
| `pair_sha256`  | str   | Sample identity hash              |
| `duration_sec` | float | Audio duration                    |
| `duration_bin` | str   | Duration bin label                |
| `reference`    | str   | Ground truth (textnorm_v2)        |
| `hypothesis`   | str   | Model v2 prediction (textnorm_v2) |
| `wer`          | float | Word error rate                   |
| `cer`          | float | Character error rate              |

### Comparison Framework

#### Aggregate

| Model  | Base     | Params | textnorm | Val WER    | Delta vs v1.1 |
| ------ | -------- | ------ | -------- | ---------- | ------------- |
| v1.1   | base.en  | 74M    | v2       | 0.6237     | —             |
| **v2** | small.en | 244M   | v2       | (measured) | (measured)    |

Both scored with `textnorm_v2` for fair comparison.

#### Diagnostic Slices

These slices directly answer the M4a diagnosis. No new pipeline — just subset
the val predictions using existing columns.

| Slice              | Definition                 | v1.1 WER   | v2 WER     |
| ------------------ | -------------------------- | ---------- | ---------- |
| Persistent failure | WER > 0.5, baseline & v1.1 | (from M4a) | (measured) |
| Short utterance    | `duration_sec` ≤ 3s        | (from M4a) | (measured) |

The persistent-failure slice is the primary hypothesis check (H2: capacity).
The short-utterance slice tracks the hallucination pattern flagged by advisor
review.

______________________________________________________________________

## Error Handling

| Scenario                     | Behavior                      | Exit |
| ---------------------------- | ----------------------------- | ---- |
| Manifest not found           | Fatal error                   | 1    |
| `DECODE_V1.json` not found   | Fatal error                   | 1    |
| Out of memory                | Reduce batch_size to 2, retry | 1    |
| Training diverges (NaN loss) | Stop, save partial artifacts  | 1    |
| Keyboard interrupt           | Save checkpoint, clean exit   | 130  |

______________________________________________________________________

## Completion Criteria

M5 is complete when:

1. Single training run of `small.en` + LoRA finished (all epochs or early stop)
1. Val predictions CSV generated with `textnorm_v2` scoring
1. Val WER/CER computed and recorded in `val_metrics.json`
1. Comparison against v1.1 (textnorm_v2 baseline = 0.6237) documented
1. `frozen_config.json` written with full reproducibility snapshot
1. Model v2 checkpoint saved
1. `comparison_report.md` generated

**No WER target.** The outcome is classified and documented; the project
proceeds to Min Viable Serving regardless.

### Outcome Classification (labels, not targets)

| Label        | Criterion (val, textnorm_v2) |
| ------------ | ---------------------------- |
| Breakthrough | ΔWER ≥ 5 pts vs v1.1         |
| Marginal     | ΔWER 1–4.9 pts vs v1.1       |
| No effect    | ΔWER < 1 pt vs v1.1          |
| Regression   | v2 WER higher than v1.1      |

______________________________________________________________________

## Package Dependencies

### Inherited from S1-M3 (no new packages required)

| Package        | Version   | Purpose              |
| -------------- | --------- | -------------------- |
| `transformers` | >= 4.36.0 | HuggingFace models   |
| `peft`         | >= 0.7.0  | LoRA implementation  |
| `datasets`     | >= 2.14.0 | HuggingFace datasets |
| `accelerate`   | >= 0.25.0 | Training utilities   |
| `jiwer`        | >= 3.0    | WER/CER (locked)     |
| `torch`        | >= 2.0    | PyTorch backend      |
| `pandas`       | >= 2.0    | Data manipulation    |

______________________________________________________________________

## References

- S1-M4a Error Analysis — H2 (capacity hypothesis) and `decision.json`
- S1-M4b Normalization Fix — `textnorm_v2`
- S1-M4 Targeted Improvements — training infrastructure, `frozen_config` schema
- S1-M3 Personalization & Fine-Tuning — LoRA config, training setup
- `out/error_analysis/20260302-202827/decision.json`
