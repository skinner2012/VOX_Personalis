# S1-M3 Personalization & Fine-Tuning — Specification

## Purpose

Establish the **first personalized ASR model** for a Deaf speaker and build
systematic understanding of fine-tuning techniques using Whisper.

This milestone answers one question:

> *Can we measurably improve ASR accuracy for this speaker through fine-tuning,
> and what techniques work best?*

______________________________________________________________________

## Dependencies

- S1-M2 Baseline Evaluation completed
- Dataset v1 manifest with train/val/test splits
- Baseline WER/CER metrics for comparison

______________________________________________________________________

## Baseline Contract

All fine-tuned models MUST be compared against a **locked baseline** to ensure
fair and reproducible comparisons.

### Baseline Reference

| Property     | Value                                | Source           |
| ------------ | ------------------------------------ | ---------------- |
| Run ID       | `out/baseline_eval/YYYYMMDD-HHMMSS/` | S1-M2 output dir |
| Model        | Whisper `base.en` or `small.en`      | As used in S1-M2 |
| Metrics File | `baseline_metrics.json`              | From S1-M2 run   |

### Locked Configuration

The following MUST match S1-M2 exactly:

| Component    | Specification                                        |
| ------------ | ---------------------------------------------------- |
| Text Norm    | `jiwer`: lowercase, rm punct, collapse spaces, strip |
| Scoring Lib  | `jiwer>=3.0` with default settings                   |
| Norm Version | `textnorm_v1` (as defined in S1-M2)                  |

### Baseline Metrics to Compare

| Split | Metric | Value        | Source                  |
| ----- | ------ | ------------ | ----------------------- |
| Val   | WER    | (from S1-M2) | `baseline_metrics.json` |
| Val   | CER    | (from S1-M2) | `baseline_metrics.json` |
| Test  | WER    | (from S1-M2) | `baseline_metrics.json` |
| Test  | CER    | (from S1-M2) | `baseline_metrics.json` |

**Important:** The `baseline_metrics.json` path MUST be recorded in every
experiment's output for traceability.

______________________________________________________________________

## Test Set Usage Policy

The test set is **single-shot**. This policy prevents evaluation contamination
and ensures reported improvements are genuine.

### Rules

1. All hyperparameter selection and ablation comparisons use val split ONLY
1. Test split is used exactly ONCE per model — after selecting the best
   configuration based on val performance
1. No peeking: Do not run test evaluation "just to check" during development

### Audit Requirements

Every test set evaluation MUST record:

| Field                  | Description                                  |
| ---------------------- | -------------------------------------------- |
| `test_run_timestamp`   | ISO 8601 timestamp of test evaluation        |
| `git_commit_sha`       | Git commit hash at time of evaluation        |
| `config_hash`          | SHA256 of training config (hyperparameters)  |
| `val_wer_at_selection` | Val WER that justified this config selection |
| `justification`        | Brief note: "Best val WER among ablations"   |

### Test Audit Log

All test evaluations are logged to `test_audit_log.csv`:

| Column               | Type  | Description                |
| -------------------- | ----- | -------------------------- |
| `run_id`             | str   | Unique run identifier      |
| `model`              | str   | Model name                 |
| `test_run_timestamp` | str   | ISO 8601 timestamp         |
| `git_commit_sha`     | str   | 40-char git SHA            |
| `config_hash`        | str   | SHA256 of config           |
| `val_wer`            | float | Val WER at selection       |
| `test_wer`           | float | Final test WER             |
| `justification`      | str   | Why this config was chosen |

### Violation Consequences

If test set is used more than once per model configuration:

- Results MUST be flagged as "potentially contaminated"
- All test runs MUST be documented in audit log
- Final report MUST disclose multiple test evaluations

______________________________________________________________________

## Scope

### Must Have

- **Whisper `base.en`** fine-tuning with LoRA (r=8 and r=16 ablation)
- Evaluation on val split for all experiments
- Single final test evaluation (best config only)
- Comprehensive findings report with error-type analysis

### Optional / Stretch Goals

- **Whisper `small.en`** ablation (r=8, r=16)

### Out of Scope

- Full model fine-tuning (all parameters)
- Alternative model architectures (e.g., Wav2Vec2, other ASR models)
- Human-in-the-loop corrections
- Streaming or real-time inference
- Cloud-based training
- Production deployment

______________________________________________________________________

## Models

### Whisper

| Property     | Value                                                |
| ------------ | ---------------------------------------------------- |
| Architecture | Encoder-decoder                                      |
| Library      | HuggingFace Transformers                             |
| Must Have    | `base.en` (74M)                                      |
| Stretch      | `small.en` (244M)                                    |
| Precedent    | Google Euphonia fine-tuned Whisper for accessibility |

______________________________________________________________________

## Fine-Tuning Method: LoRA

**LoRA** = Low-Rank Adaptation

### Fine-Tuning Methods Comparison

| Method       | Params | Mem  | Speed | Perf      | HW      | ASR      |
| ------------ | ------ | ---- | ----- | --------- | ------- | -------- |
| **Full FT**  | 100%   | 16GB | Slow  | Best      | GPU     | Standard |
| **LoRA**     | 1-5%   | 2GB  | Fast  | Near full | CPU/GPU | Proven   |
| **Adapters** | 2-10%  | 4GB  | Med   | Good      | CPU/GPU | Limited  |
| **Prefix**   | \<1%   | Low  | Fast+ | Variable  | CPU     | Minimal  |
| **QLoRA**    | 1-5%   | 1GB  | Fast  | Near LoRA | CPU     | Emerging |

**Notes:**

- Mem: Memory for Whisper base.en/small.en on consumer hardware
- Perf: Performance relative to full fine-tuning baseline
- HW Req: Hardware requirements (GPU rec = GPU recommended, CPU/GPU = CPU
  or consumer GPU)
- ASR Record: Track record for speech recognition models

### Why LoRA

**Chosen for this milestone because:**

1. **Hardware fit**: Runs on M4 Pro (12-16GB) without external GPU
1. **Proven for Whisper**: Google Euphonia team used LoRA for accessibility ASR
1. **Fast iteration**: 4-6 hours per experiment vs 12+ hours for full fine-tuning
1. **PEFT library**: Mature HuggingFace implementation, well-documented
1. **Performance**: Typically achieves 90-95% of full fine-tuning performance
1. **Reversible**: LoRA adapters can be easily removed or swapped

**Alternative methods considered:**

- **Full fine-tuning**: Rejected (too slow, GPU required, 100% params)
- **Adapter layers**: Possible future comparison (less proven for Whisper)
- **QLoRA**: Deferred (emerging, less validated for ASR, quantization complexity)

### LoRA Configuration

| Parameter      | Default                  | Ablation Values |
| -------------- | ------------------------ | --------------- |
| `r` (rank)     | 8                        | 8, 16           |
| `alpha`        | 16                       | 2× rank         |
| `dropout`      | 0.1                      | Fixed           |
| Target modules | Query, Value projections | Model-specific  |

______________________________________________________________________

## Execution Plan

**Execution Flow Rationale:**

The plan follows a risk-mitigated, single-shot test approach:

1. **1A (Fast First Win)**: Validate that fine-tuning helps at all —
   derisk the entire milestone early with one quick experiment
1. **1B (Systematic Ablations)**: Compare hyperparameters (r=8 vs r=16)
   on val split to find best config
1. **1C (Final Test Evaluation)**: Single-shot test evaluation of best
   config (respects test set policy)
1. **Phase 2 (Analysis & Reporting)**: Synthesize all findings after
   experiments complete

This order ensures: (a) early validation, (b) systematic comparison
without test contamination, (c) final generalization check, (d)
comprehensive analysis.

______________________________________________________________________

### Phase 1: Whisper Fine-Tuning

#### 1A: Fast First Win

Goal: Achieve any measurable WER improvement quickly.

| Step                          | Output               |
| ----------------------------- | -------------------- |
| Setup HuggingFace + PEFT      | Environment ready    |
| Data prep: Audio → Dataset    | Train/val DataLoader |
| Training: `base.en`, LoRA r=8 | Checkpoint           |
| Evaluation on **val only**    | Val delta report     |

**Success criterion:** Any measurable WER improvement on val split.

#### 1B: Systematic Ablations

Goal: Understand hyperparameter sensitivity on val split.

**Must Have:**

| Experiment | Model     | LoRA Rank |
| ---------- | --------- | --------- |
| 1          | `base.en` | r=8       |
| 2          | `base.en` | r=16      |

**Stretch (if time permits):**

| Experiment | Model      | LoRA Rank |
| ---------- | ---------- | --------- |
| 3          | `small.en` | r=8       |
| 4          | `small.en` | r=16      |

**Success criterion:** Consistent improvement on val; documented hyperparameter
sensitivity.

#### 1C: Final Test Evaluation (after ablations complete)

- Select best config based on val WER
- Run **single** test evaluation
- Record in test audit log

### Phase 2: Analysis & Reporting

Goal: Synthesize findings into actionable recommendations.

| Deliverable            | Content                                         |
| ---------------------- | ----------------------------------------------- |
| Ablation analysis      | LoRA rank impact, training dynamics             |
| Error pattern analysis | Which error types benefit most from fine-tuning |
| Slice-based evaluation | Performance by duration, error type             |
| Recommendations        | Best config, next steps for M4                  |

______________________________________________________________________

## Training Configuration

These settings are passed to HuggingFace `TrainingArguments` when using
the `Trainer` API with PEFT for LoRA fine-tuning.

### Common Settings

| Parameter             | Value |
| --------------------- | ----- |
| Epochs                | 3     |
| Batch size            | 4-8   |
| Learning rate         | 1e-4  |
| Warmup steps          | 100   |
| Gradient accumulation | 4     |
| Mixed precision       | No    |

**Epochs (3):** Standard for fine-tuning; balances convergence vs
overfitting risk. More epochs (5-10) typical for training from scratch, but
pre-trained models converge faster. Early stopping monitors for actual
convergence.

**Batch size (4-8):** Hardware constraint for M4 Pro with 12-16GB memory.
Whisper base.en (~2GB model) + gradients + optimizer states fit with
batch=4-8. Larger batches cause OOM (out of memory).

**Learning rate (1e-4):** LoRA standard is 10× higher than full fine-tuning
(1e-5) because only ~1-5% of params trained. Lower than training from
scratch (1e-3) because base model already pretrained. PEFT library default.

**Warmup steps (100):** Gradual LR ramp prevents large gradient spikes in
early training. ~7% of total steps (~1450 samples ÷ batch 8 × 3 epochs ≈
544 steps). Standard practice for small datasets.

**Gradient accumulation (4):** Simulates larger batch with effective
batch = 4-8 × 4 = 16-32. Larger effective batches stabilize training
without increasing memory. Trade-off: 4× slower per epoch.

**Mixed precision (No):** CPU limitation - fp16 (half precision) requires
GPU/MPS. CPU training uses fp32. Would enable if using MPS, but MPS has
operator fallbacks for Whisper (unstable).

### Device Strategy

| Platform | Device | Notes                               |
| -------- | ------ | ----------------------------------- |
| PyTorch  | CPU    | MPS unreliable (operator fallbacks) |

### Early Stopping

- Monitor validation WER every epoch
- Patience: 2 epochs without improvement
- Save best checkpoint by validation WER

______________________________________________________________________

## Data Pipeline

### Input Format

Use Dataset v1 manifest (`dataset_v1_manifest.csv`) with canonical columns
**exactly as defined in S1-M1**:

| Column                | Type  | Description                 |
| --------------------- | ----- | --------------------------- |
| `file_name`           | str   | Original audio file name    |
| `audio_path_resolved` | str   | Relative path to audio file |
| `transcript_raw`      | str   | Original transcript text    |
| `split`               | str   | train/val/test assignment   |
| `duration_sec`        | float | Audio duration              |
| `duration_bin`        | str   | Duration bin label          |
| `pair_sha256`         | str   | Sample identity hash        |

**Important:** Do NOT rename columns. Use the manifest as-is from S1-M1.

### Text Normalization (Runtime Only)

Text normalization is applied **at training/evaluation time**, not stored in
the manifest:

```python
# Generate normalized transcript at runtime
transcript_norm = textnorm_v1(row["transcript_raw"])
```

The manifest contains `transcript_raw` only. Normalization is a **view**, not
a data modification.

### Preprocessing Pipeline

1. Load audio at 16kHz (model requirement)
1. Normalize audio amplitude
1. Apply `textnorm_v1` to transcript (same as S1-M2)
1. Tokenize for model-specific format

### Data Splits

| Split | Samples     | Usage                                    |
| ----- | ----------- | ---------------------------------------- |
| Train | ~1450 (80%) | Fine-tuning                              |
| Val   | ~360 (10%)  | Hyperparameter selection, early stopping |
| Test  | ~360 (10%)  | **Single-shot final evaluation only**    |

______________________________________________________________________

## Evaluation Protocol

### Metrics

Same as S1-M2 baseline:

| Metric | Description                      |
| ------ | -------------------------------- |
| WER    | Word Error Rate (primary)        |
| CER    | Character Error Rate (secondary) |

### Comparison to Baseline

Report both absolute and relative improvement:

```text
Absolute: WER_baseline - WER_finetuned
Relative: (WER_baseline - WER_finetuned) / WER_baseline × 100%
```

### Evaluation Slices

#### Required: Duration Bins (from S1-M1)

| Slice  | Duration Range |
| ------ | -------------- |
| Short  | (1, 3\] sec    |
| Medium | (3, 10\] sec   |
| Long   | (10, 30\] sec  |

#### Required: Error-Type Slices (from S1-M2 baseline)

Classify samples based on baseline error patterns:

| Slice          | Definition                   | Source                     |
| -------------- | ---------------------------- | -------------------------- |
| Deletion-heavy | Baseline deletions > 50%     | `baseline_predictions.csv` |
| Subst-heavy    | Baseline substitutions > 50% | `baseline_predictions.csv` |
| Low-error      | Baseline WER < 0.1           | `baseline_predictions.csv` |
| High-error     | Baseline WER > 0.5           | `baseline_predictions.csv` |

These slices reveal **which error types benefit most from fine-tuning**.

______________________________________________________________________

## CLI Interface

```bash
python -m scripts.fine_tuning \
  --manifest_path "./out/dataset_v1/YYYYMMDD/dataset_v1_manifest.csv" \
  --baseline_metrics "./out/baseline_eval/YYYYMMDD/baseline_metrics.json" \
  --out_dir "./out/fine_tuning" \
  --model whisper-base.en \
  --method lora \
  --lora_rank 8 \
  --epochs 3 \
  --device cpu \
  --eval_split val \
  --verbose
```

### Required Arguments

| Argument             | Description                         |
| -------------------- | ----------------------------------- |
| `--manifest_path`    | Path to Dataset v1 manifest CSV     |
| `--baseline_metrics` | Path to S1-M2 baseline_metrics.json |

### Optional Arguments

| Argument        | Default             | Description           |
| --------------- | ------------------- | --------------------- |
| `--out_dir`     | `./out/fine_tuning` | Output dir            |
| `--model`       | `whisper-base.en`   | Model to tune         |
| `--method`      | `lora`              | Method (lora only)    |
| `--lora_rank`   | `8`                 | LoRA rank             |
| `--epochs`      | `3`                 | Epochs                |
| `--batch_size`  | `4`                 | Batch size            |
| `--lr`          | `1e-4`              | Learning rate         |
| `--device`      | `cpu`               | Device (cpu)          |
| `--eval_split`  | `val`               | Eval split (val/test) |
| `-v, --verbose` | False               | Show progress         |
| `-q, --quiet`   | False               | Suppress output       |

### Test Evaluation Mode

To run final test evaluation (single-shot):

```bash
python -m scripts.fine_tuning \
  --manifest_path "./out/dataset_v1/YYYYMMDD/dataset_v1_manifest.csv" \
  --baseline_metrics "./out/baseline_eval/YYYYMMDD/baseline_metrics.json" \
  --checkpoint_path "./out/fine_tuning/YYYYMMDD/whisper_base_r8_checkpoint/" \
  --eval_split test \
  --eval_only \
  --justification "Best val WER among r=8,16 ablations"
```

This mode:

- Loads existing checkpoint (no training)
- Runs evaluation on test split
- Appends to `test_audit_log.csv`
- Requires `--justification` argument

### Exit Codes

| Code | Meaning                                             |
| ---- | --------------------------------------------------- |
| 0    | Success                                             |
| 1    | Fatal error (manifest not found, model load failed) |
| 2    | Validation error (no samples in split)              |
| 130  | Interrupted by user (Ctrl+C)                        |

______________________________________________________________________

## Output Artifacts

All outputs written to `--out_dir/YYYYMMDD-HHMMSS/`:

### Per-Experiment Outputs

#### `{model}_{approach}_predictions.csv`

Per-utterance predictions and scores.

| Column                | Type  | Description                             |
| --------------------- | ----- | --------------------------------------- |
| `file_name`           | str   | Audio file name                         |
| `pair_sha256`         | str   | Sample identifier                       |
| `split`               | str   | val (or test for final run)             |
| `duration_sec`        | float | Audio duration                          |
| `duration_bin`        | str   | Duration bin label                      |
| `baseline_error_type` | str   | deletion-heavy/substitution-heavy/mixed |
| `reference`           | str   | Ground truth (normalized)               |
| `hypothesis`          | str   | Model prediction (normalized)           |
| `wer`                 | float | Word error rate                         |
| `cer`                 | float | Character error rate                    |

#### `{model}_{approach}_metrics.json`

Aggregate metrics and configuration.

```json
{
  "model": "whisper-base.en",
  "approach": "base_r8",
  "method": "lora",
  "lora_rank": 8,
  "epochs_trained": 3,
  "training_time_sec": 14400,
  "device": "cpu",
  "baseline_reference": {
    "metrics_file": "./out/baseline_eval/20260210/baseline_metrics.json",
    "baseline_wer": 0.234,
    "normalization_version": "textnorm_v1",
    "jiwer_version": "3.0.4"
  },
  "val_results": {
    "wer": 0.201,
    "cer": 0.075,
    "sample_count": 361,
    "absolute_improvement": 0.033,
    "relative_improvement_pct": 14.1
  },
  "by_duration_bin": {
    "(1.0, 3]": { "wer": 0.22, "sample_count": 105 },
    "(3.0, 10]": { "wer": 0.18, "sample_count": 239 },
    "(10.0, 30]": { "wer": 0.16, "sample_count": 17 }
  },
  "by_error_type": {
    "deletion_heavy": { "wer": 0.19, "n": 85, "baseline_wer": 0.31 },
    "substitution_heavy": { "wer": 0.21, "n": 120, "baseline_wer": 0.28 },
    "low_error": { "wer": 0.05, "n": 95, "baseline_wer": 0.07 },
    "high_error": { "wer": 0.35, "n": 61, "baseline_wer": 0.62 }
  }
}
```

#### `{model}_{approach}_checkpoint/`

Saved LoRA weights and configuration.

```text
{model}_{approach}_checkpoint/
├── adapter_config.json
├── adapter_model.safetensors
└── training_args.json
```

### Final Outputs

#### `experiment_log.csv`

Summary of all experiments.

| Column                     | Type  | Description           |
| -------------------------- | ----- | --------------------- |
| `experiment_id`            | str   | Unique ID             |
| `model`                    | str   | Model name            |
| `approach`                 | str   | Experiment label      |
| `lora_rank`                | int   | LoRA rank             |
| `epochs`                   | int   | Epochs                |
| `device`                   | str   | Device                |
| `training_time_sec`        | int   | Training time (sec)   |
| `baseline_wer`             | float | Baseline WER          |
| `val_wer`                  | float | Val WER               |
| `relative_improvement_pct` | float | Rel WER reduction (%) |
| `checkpoint_path`          | str   | Checkpoint path       |

#### `test_audit_log.csv`

Audit trail for all test evaluations (see Test Set Usage Policy).

#### `fine_tuning_report.md`

Comprehensive human-readable report with:

1. Executive summary
1. Ablation findings (val-based comparison of r=8 vs r=16)
1. Final test results (single-shot evaluation of best config)
1. Error-type slice analysis (which errors improved most)
1. Hyperparameter sensitivity analysis
1. Recommendations for M4 (human-in-the-loop)
1. Appendix: detailed metrics tables

#### `best_model/`

Copy of the best-performing checkpoint for easy access.

______________________________________________________________________

## Error Handling

| Scenario                     | Behavior                       | Exit Code |
| ---------------------------- | ------------------------------ | --------- |
| Manifest not found           | Fatal error                    | 1         |
| Baseline metrics not found   | Fatal error                    | 1         |
| Model download fails         | Fatal error, check network     | 1         |
| Out of memory                | Fatal error, reduce batch size | 1         |
| Audio file missing           | Log warning, skip sample       | 0         |
| Training diverges (NaN loss) | Stop early, save last valid    | 0         |
| Keyboard interrupt           | Save checkpoint, clean exit    | 130       |

### Checkpoint Recovery

If training is interrupted:

- Last valid checkpoint is saved
- Can resume with `--resume_from` flag
- Partial results are preserved in experiment log

______________________________________________________________________

## Package Dependencies

### New Packages (add to pyproject.toml)

| Package        | Version | Purpose              |
| -------------- | ------- | -------------------- |
| `transformers` | ≥4.36.0 | HuggingFace models   |
| `peft`         | ≥0.7.0  | LoRA implementation  |
| `datasets`     | ≥2.14.0 | HuggingFace datasets |
| `accelerate`   | ≥0.25.0 | Training utilities   |

### Existing Packages (from S1-M2)

- `openai-whisper` — Baseline model reference
- `torch` — PyTorch backend
- `jiwer>=3.0` — WER/CER computation (version locked)
- `pandas` — Data manipulation
- `tqdm` — Progress bars

______________________________________________________________________

## Hardware Constraints

### Target Platform

- Apple M4 Pro MacBook
- 12-16 GB unified memory
- No external GPU

### Memory Budget

| Model                     | Estimated VRAM | Batch Size |
| ------------------------- | -------------- | ---------- |
| `whisper-base.en` + LoRA  | ~2 GB          | 8          |
| `whisper-small.en` + LoRA | ~4 GB          | 4          |

### Training Time Estimates

| Model              | Device | Per Epoch | 3 Epochs |
| ------------------ | ------ | --------- | -------- |
| `whisper-base.en`  | CPU    | ~1.5 hrs  | ~4.5 hrs |
| `whisper-small.en` | CPU    | ~2 hrs    | ~6 hrs   |

______________________________________________________________________

## Success Criteria

S1-M3 is complete when:

### Must Have (Required)

1. Whisper `base.en` fine-tuned with LoRA (r=8 and r=16 ablations on val)
1. **Measurable and consistent** WER improvement on val split
1. Single-shot test evaluation for best config
1. Test audit log with all required fields
1. Comprehensive findings report with error-type analysis
1. Best model checkpoint saved

### Target (Aspirational, Not Hard KPI)

- 10-15% relative WER reduction on val
- Confirmed improvement direction on test (not required to hit specific %)

### Stretch (Optional)

- Whisper `small.en` ablation complete

**Note:** The 15% target is **aspirational**, not a gate. M3 is successful if
we achieve consistent improvement and understand why, even if below 15%.

______________________________________________________________________

## Failure Modes if Skipped

If this milestone is skipped:

- No personalized model exists
- Cannot validate whether fine-tuning helps this speaker
- No understanding of which techniques work
- Future milestones (human-in-the-loop) have no starting point

______________________________________________________________________

## Implementation Module Structure

```text
scripts/fine_tuning/
├── __init__.py          # __version__ = "0.1.0"
├── __main__.py          # Entry: python -m scripts.fine_tuning
├── cli.py               # CLI parsing + pipeline orchestration
├── data.py              # Data loading and preprocessing
├── models.py            # Model loading and LoRA configuration
├── training.py          # Training loop and checkpointing
├── evaluation.py        # WER/CER computation and comparison
├── slices.py            # Duration and error-type slice computation
├── audit.py             # Test audit log management
└── reporting.py         # Generate all output files
```

______________________________________________________________________

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [Google Euphonia](https://sites.research.google/euphonia/) — Accessibility ASR
