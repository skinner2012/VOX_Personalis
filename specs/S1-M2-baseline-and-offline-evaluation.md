# S1-M2 Baseline Model & Offline Evaluation Framework — Specification

## Purpose

Establish a **non-personalized baseline** and a **stable offline evaluation framework**
to define the performance floor for Dataset v1.

This milestone answers one question only:

> *How does a generic ASR model behave on this speaker’s data,
> before any personalization is applied?*

______________________________________________________________________

## Dependencies

- Dataset v1 completed and locked (S1-M1)
- Deterministic train / val / test split available via Dataset v1 manifest

______________________________________________________________________

## Scope

### In Scope

- Selection of a single non-personalized ASR baseline
- Offline inference on Dataset v1 test split
- Computation of core error metrics (WER and/or CER)
- Error breakdown and qualitative pattern analysis
- Structured baseline evaluation report

### Out of Scope

- Any form of personalization or fine-tuning
- Hyperparameter optimization
- Streaming or real-time evaluation
- UI, product, or latency concerns

______________________________________________________________________

## Baseline Model

### Why Whisper

Whisper is chosen as the baseline model because:

- Google's Euphonia project fine-tuned Whisper for accessibility speech,
  making it a meaningful reference point for this speaker's data
- It represents the current state-of-the-art for general-purpose ASR
- Speaker-independent by design (trained on 680k hours of diverse audio)
- Provides a meaningful performance floor before any personalization

### Model Requirements

The baseline model MUST satisfy:

- Speaker-independent (no exposure to this speaker's data)
- No fine-tuning on Dataset v1
- Used in inference-only mode
- Deterministic inference configuration (fixed decoding parameters)

The baseline is treated as a **black box recognizer**.
Model choice is not optimized; its role is to define a reference floor.

### Model Configuration

| Setting       | Value            | Rationale                     |
| ------------- | ---------------- | ----------------------------- |
| Library       | `openai-whisper` | Local inference, no API costs |
| Default Model | `small.en`       | 244M params, accuracy/speed   |
| Alternative   | `base.en`        | 74M params, faster iteration  |

### Inference Parameters

| Parameter     | Value               | Rationale                    |
| ------------- | ------------------- | ---------------------------- |
| `temperature` | 0                   | Deterministic greedy decode  |
| `language`    | `"en"`              | Force English, skip detect   |
| `task`        | `"transcribe"`      | Transcription, not translate |
| `fp16`        | False CPU, True MPS | Device-appropriate precision |
| `beam_size`   | 5                   | Default Whisper beam search  |

### Device Support

| Device            | Support      | Notes                 |
| ----------------- | ------------ | --------------------- |
| Apple Silicon MPS | Primary      | 5-10x faster than CPU |
| CPU               | Fallback     | Works but slower      |
| CUDA              | Not required | Project runs on macOS |

______________________________________________________________________

## Evaluation Setup

### Evaluation Splits

- **Primary:** Dataset v1 test split (365 samples, ~29 min audio)
- **Secondary:** Dataset v1 val split (361 samples, ~28 min audio)
- Test split MUST NOT be modified after this milestone

Running on both splits provides:

- Test: Immutable benchmark for future comparisons
- Val: Sanity check, helps detect anomalies

### Inference Mode

- Offline, full-utterance transcription
- No streaming or partial decoding
- One hypothesis per utterance
- Audio loaded at original sample rate (Whisper resamples internally to 16kHz)

______________________________________________________________________

## Text Normalization

WER and CER metrics are highly sensitive to text normalization.
Both reference (ground truth) and hypothesis (model output) MUST be normalized
identically before comparison.

### Normalization Pipeline

Apply the following transforms in order:

1. Convert to lowercase
1. Remove all punctuation: `.,!?;:"'-—:`
1. Collapse multiple spaces to single space
1. Strip leading/trailing whitespace

### Implementation

Use `jiwer` library transforms:

```python
from jiwer import Compose, ToLowerCase, RemovePunctuation, \
                  RemoveMultipleSpaces, Strip

normalizer = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])

# Apply to both reference and hypothesis
reference_normalized = normalizer(reference_raw)
hypothesis_normalized = normalizer(hypothesis_raw)
```

### Normalization Examples

| Original                  | Normalized               |
| ------------------------- | ------------------------ |
| `"What's the weather?"`   | `"whats the weather"`    |
| `"Turn on."`              | `"turn on"`              |
| `"A rainbow in the sky:"` | `"a rainbow in the sky"` |
| `"Isn't there any way?"`  | `"isnt there any way"`   |

**Note:** Apostrophes are removed by `RemovePunctuation()`. This is intentional
for consistent WER comparison, even though it changes contractions.

______________________________________________________________________

## Core Metrics

### Primary Metrics

| Metric | Description                                                     |
| ------ | --------------------------------------------------------------- |
| WER    | Word Error Rate — primary metric                                |
| CER    | Character Error Rate — optional, useful for non-standard speech |

At least one MUST be reported; WER is preferred as the primary metric.

______________________________________________________________________

## Error Decomposition

Error counts MUST be decomposed into:

- Deletions
- Insertions
- Substitutions

Both absolute counts and relative proportions SHOULD be reported.

______________________________________________________________________

## Evaluation Slices

The evaluation MUST include slices to reveal performance patterns
beyond aggregate metrics.

### Required Slices: Duration Bins

Use the same duration bins as S1-M1 Dataset Versioning for consistency.
The `duration_bin` column from `dataset_v1_manifest.csv` provides this directly.

| Slice Name | Duration Range | Expected Test Count | Expected Val Count |
| ---------- | -------------- | ------------------- | ------------------ |
| Very Short | (0, 1\] sec    | ~0 (rare)           | ~0                 |
| Short      | (1, 3\] sec    | ~106                | ~105               |
| Medium     | (3, 10\] sec   | ~240                | ~239               |
| Long       | (10, 30\] sec  | ~19                 | ~17                |
| Very Long  | (30, inf\] sec | ~0 (none)           | ~0                 |

Report WER/CER for each non-empty bin.

### Deferred Slices (Future Milestones)

The following slices are explicitly **out of scope** for S1-M2:

- **Acoustic/noise estimation:** Requires SNR computation, adds complexity
- **Phoneme-level analysis:** Requires phoneme recognizer

These may be added in future milestones if baseline analysis reveals the need.

### Slice Documentation Requirements

For each slice, the report MUST include:

- Slice definition (duration range or criteria)
- Sample count in slice
- WER and CER values
- Comparison to aggregate (higher/lower/similar)

______________________________________________________________________

## Error Pattern Analysis

Beyond numeric metrics, the evaluation MUST include:

- Most frequent substitutions (token → token)
- Common deletion patterns
- Notable insertion artifacts
- Speaker-specific or phonetic error clusters

The goal is **interpretability**, not exhaustiveness.

______________________________________________________________________

## Baseline Evaluation Report

### Report Format

- Markdown
- Human-readable
- Deterministic given the same inputs

### Required Sections

#### 1. Overview

- Dataset version (v1)
- Baseline model identifier
- Evaluation split
- Run configuration (high-level)

#### 2. Aggregate Metrics

- WER (and CER if applicable)
- Total utterance count
- Total word / character count

#### 3. Error Breakdown

- Deletions / Insertions / Substitutions
- Relative proportions
- High-level observations

#### 4. Evaluation Slices

For each slice:

- Slice definition
- Metric values
- Notable differences vs aggregate

#### 5. Error Pattern Analysis

- Top substitution pairs
- Common deletion cases
- Representative examples (text only)

#### 6. Key Takeaways

- What the baseline does poorly
- What it does reasonably well
- Which error types are likely addressable via personalization

#### 7. Limitations

- Single-speaker bias
- Limited linguistic diversity
- Baseline model constraints

______________________________________________________________________

## Reproducibility Requirements

- Evaluation MUST be repeatable with:
  - Dataset v1
  - Same baseline model
  - Same evaluation code
- All parameters affecting decoding or scoring MUST be fixed and documented

______________________________________________________________________

## Failure Mode if Skipped

If this milestone is skipped:

- No performance floor is defined
- Improvement claims cannot be validated
- Personalization risks optimizing noise instead of signal

______________________________________________________________________

## Completion Criteria

S1-M2 is complete when:

- Baseline inference on test split is complete
- Core metrics are computed
- Evaluation slices are analyzed
- Baseline evaluation report is generated and reviewed

______________________________________________________________________

## CLI Interface

```bash
python -m scripts.baseline_eval \
  --manifest_path "./out/dataset_v1/YYYYMMDD/dataset_v1_manifest.csv" \
  --out_dir "./out/baseline_eval" \
  --model_size small.en \
  --device mps \
  --splits test,val \
  --verbose
```

### Required Arguments

| Argument          | Description                     |
| ----------------- | ------------------------------- |
| `--manifest_path` | Path to Dataset v1 manifest CSV |

### Optional Arguments

| Argument    | Default               | Description              |
| ----------- | --------------------- | ------------------------ |
| `--out_dir` | `./out/baseline_eval` | Output directory         |
| `--model`   | `small.en`            | Whisper model size       |
| `--device`  | `mps`                 | Device: `cpu`, `mps`     |
| `--splits`  | `test,val`            | Comma-separated splits   |
| `-v`        | False                 | Verbose: progress + logs |
| `-q`        | False                 | Quiet: errors only       |

### Exit Codes

| Code | Meaning                                             |
| ---- | --------------------------------------------------- |
| 0    | Success                                             |
| 1    | Fatal error (manifest not found, model load failed) |
| 2    | Validation failed (no samples in requested splits)  |
| 130  | Interrupted by user (Ctrl+C)                        |

### Output

The script prints:

- Output directory path
- Model loading status
- Progress (if `--verbose`)
- Aggregate metrics summary
- Path to generated report

______________________________________________________________________

## Output Artifacts

All outputs written to `--out_dir/YYYYMMDD-HHMMSS/`:

### 1. `baseline_predictions.csv`

Per-utterance predictions and scores.

| Column               | Type  | Description                          |
| -------------------- | ----- | ------------------------------------ |
| `file_name`          | str   | Audio file name                      |
| `pair_sha256`        | str   | Sample identity hash (from manifest) |
| `split`              | str   | `train`, `val`, or `test`            |
| `duration_sec`       | float | Audio duration in seconds            |
| `duration_bin`       | str   | Duration bin label                   |
| `reference_raw`      | str   | Ground truth transcript (original)   |
| `hypothesis_raw`     | str   | Model prediction (original)          |
| `reference`          | str   | Ground truth (normalized)            |
| `hypothesis`         | str   | Model prediction (normalized)        |
| `wer`                | float | Word error rate (0.0 to 1.0+)        |
| `cer`                | float | Character error rate                 |
| `word_insertions`    | int   | Inserted word count                  |
| `word_deletions`     | int   | Deleted word count                   |
| `word_substitutions` | int   | Substituted word count               |

### 2. `baseline_metrics.json`

Machine-readable aggregate metrics.

```json
{
  "dataset_version": "v1",
  "baseline_model": {
    "name": "whisper",
    "size": "small.en",
    "library": "openai-whisper",
    "library_version": "20231117"
  },
  "evaluation_config": {
    "temperature": 0,
    "language": "en",
    "device": "mps",
    "normalization": "jiwer_standard"
  },
  "splits_evaluated": ["test", "val"],
  "aggregate": {
    "test": {
      "sample_count": 365,
      "total_duration_sec": 1735.09,
      "total_words": 1847,
      "total_chars": 9234,
      "wer": 0.234,
      "cer": 0.089,
      "insertions": 45,
      "deletions": 123,
      "substitutions": 89
    },
    "val": {
      "sample_count": 361,
      "total_duration_sec": 1649.65,
      "total_words": 1802,
      "total_chars": 9012,
      "wer": 0.241,
      "cer": 0.092,
      "insertions": 42,
      "deletions": 118,
      "substitutions": 85
    }
  },
  "by_duration_bin": {
    "test": {
      "(1.0, 3]": {
        "sample_count": 106,
        "wer": 0.28,
        "cer": 0.11
      },
      "(3.0, 10]": {
        "sample_count": 240,
        "wer": 0.21,
        "cer": 0.08
      },
      "(10.0, 30]": {
        "sample_count": 19,
        "wer": 0.19,
        "cer": 0.07
      }
    },
    "val": {}
  },
  "skipped_samples": {
    "count": 0,
    "reasons": {},
    "files": []
  },
  "created_timestamp": "2026-02-06T15:30:00Z",
  "tool_versions": {
    "python": "3.13.7",
    "whisper": "20231117",
    "jiwer": "3.0.4",
    "torch": "2.1.0"
  }
}
```

### 3. `baseline_errors.csv`

Error pattern analysis for interpretability.

| Column             | Type | Description                                   |
| ------------------ | ---- | --------------------------------------------- |
| `error_type`       | str  | `substitution`, `deletion`, or `insertion`    |
| `reference_token`  | str  | Expected word (empty for insertions)          |
| `hypothesis_token` | str  | Predicted word (empty for deletions)          |
| `count`            | int  | Frequency across all samples                  |
| `example_files`    | str  | Comma-separated file_names (up to 5 examples) |

Sorted by `count` descending. Include top 50 patterns.

### 4. `baseline_report.md`

Human-readable report following the structure in "Baseline Evaluation Report" section.

______________________________________________________________________

## Error Handling

| Scenario               | Behavior                      | Exit |
| ---------------------- | ----------------------------- | ---- |
| Manifest not found     | Fatal error, print path       | 1    |
| No rows for split      | Fatal error, check `--splits` | 2    |
| Model download fails   | Fatal error, check network    | 1    |
| Model load fails       | Fatal error, check memory     | 1    |
| Audio file not found   | Log warning, skip, continue   | 0    |
| Audio decode fails     | Log warning, skip, continue   | 0    |
| Empty model prediction | Log warning, use empty string | 0    |
| Keyboard interrupt     | Clean exit, no partial save   | 130  |

### Skipped Sample Policy

If samples are skipped due to audio errors:

- Continue processing remaining samples
- Report skipped count in console output
- Include skipped sample details in `baseline_metrics.json`
- Log warning if >5% of samples skipped

______________________________________________________________________

## Dependencies / Environment

### Required Packages

| Package          | Version   | Purpose                 |
| ---------------- | --------- | ----------------------- |
| `openai-whisper` | ≥20231117 | ASR model and inference |
| `torch`          | ≥2.0      | Whisper backend         |
| `jiwer`          | ≥3.0      | WER/CER computation     |
| `pandas`         | ≥2.0      | Data manipulation       |
| `tqdm`           | ≥4.0      | Progress bars           |

### Installation

```bash
pip install openai-whisper jiwer pandas tqdm
```

### Whisper Model Download

Models are downloaded automatically on first use:

- Storage location: `~/.cache/whisper/`
- Size: `small.en` ≈ 461 MB, `base.en` ≈ 139 MB

### Platform Requirements

- Python 3.11+
- macOS (Apple Silicon recommended for MPS acceleration)
- 4+ GB RAM (8+ GB recommended for `small.en`)

______________________________________________________________________

## Performance Constraints

### Expected Runtime

| Model      | Device | Speed Factor   | 365 test samples (~29 min) |
| ---------- | ------ | -------------- | -------------------------- |
| `base.en`  | MPS    | ~10x realtime  | ~3 minutes                 |
| `small.en` | MPS    | ~5x realtime   | ~6 minutes                 |
| `small.en` | CPU    | ~0.8x realtime | ~36 minutes                |
| `base.en`  | CPU    | ~2x realtime   | ~15 minutes                |

With both test and val splits (~58 min audio): double the above times.

### Memory Requirements

| Model       | Peak RAM Usage |
| ----------- | -------------- |
| `base.en`   | ~1 GB          |
| `small.en`  | ~2 GB          |
| `medium.en` | ~5 GB          |

### Recommendations

- Use MPS (Metal) on Apple Silicon for 5-10x speedup
- Batch size 1 is optimal (Whisper processes full utterances)
- For quick debugging, use `base.en` with `--splits test`
- For final baseline, use `small.en` with `--splits test,val`
