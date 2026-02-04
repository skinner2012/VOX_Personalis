# S1-M1 Dataset Versioning (Dataset v1) — Specification

## Purpose

Turn raw audio + transcript files into a **reproducible, comparable dataset asset**
by defining Dataset v1 with explicit identity, traceability, and split policy.

This milestone establishes a stable data foundation for all subsequent
experiments and evaluations.

______________________________________________________________________

## Dependencies

- S1-M0 Data Inventory completed
- Raw data files and manifest verified as readable and auditable

______________________________________________________________________

## Scope

Dataset v1 is created by **filtering and annotating existing data only**.

### Explicitly In Scope

- Deterministic filtering based on documented rules
- Metadata augmentation (duration, hashes, source)
- Deterministic dataset split assignment
- Dataset identity and version locking

### Explicitly Out of Scope

- Adding new recordings
- Modifying audio content (no trimming, resampling, denoise)
- Modifying transcript text
- Model training or evaluation

______________________________________________________________________

## Dataset v1 Identity

Dataset v1 is defined **solely** by its manifest file.

### Canonical Artifact

- `dataset_v1_manifest.csv`

All downstream usage MUST reference this manifest.
Any materialized directory structure is considered a derived convenience only.

______________________________________________________________________

## Dataset v1 Manifest Schema

Each row represents one audio–transcript pair **after cleaning**.

### Required Columns

#### Identity

- `dataset_version`
  - Constant value: `v1`
- `file_name`
  - Original audio file name
- `source`
  - Origin of the sample (constant: `euphonia` for Dataset v1)
- `manifest_row_index`
  - Original row index from input manifest (for traceability)

#### Audio

- `audio_path_resolved`
  - **Relative path** to audio file from manifest directory
  - Format: `<relative_path_from_manifest_dir>/<file_name>`
  - Example: `../data/audio/euphonia_abc123.wav` (if manifest is in `./out/dataset_v1/`)
  - Rationale: Enables portability across machines while maintaining traceability
- `duration_sec`
  - Audio duration (float, from inventory)
- `duration_bin`
  - Duration bin label (e.g., "(3, 10\]", for stratification tracking)

#### Transcript

- `transcript_raw`
  - Original transcript text
- `transcript_len_chars`
  - Character count (from inventory)
- `transcript_len_words`
  - Word count (from inventory)

#### Metadata (Optional but Recommended)

- `timestamp_ms`
  - Recording timestamp in milliseconds (if available from source)
  - May be NULL if not available
  - Used for temporal clustering detection (see Split Policy Step 4)
- `recording_device`
  - Recording device type (constant: `macbook_pro` for Dataset v1)

#### Integrity / Hashing

- `audio_sha256`
  - SHA256 hash of audio file bytes (lowercase hex string, 64 chars)
  - Computed: `hashlib.sha256(audio_file_bytes).hexdigest()`
- `transcript_sha256`
  - SHA256 hash of transcript text UTF-8 bytes (lowercase hex string, 64 chars)
  - Computed: `hashlib.sha256(transcript.encode('utf-8')).hexdigest()`
- `pair_sha256`
  - SHA256 of concatenated hex strings: `audio_sha256 + transcript_sha256`
  - Computed: `hashlib.sha256((audio_sha256 + transcript_sha256).encode('utf-8')).hexdigest()`
  - Example: if `audio_sha256 = "abc..."` and `transcript_sha256 = "def..."`, then
    `pair_sha256 = sha256("abc...def...")`
  - **Critical:** All hashes must be lowercase hex strings before concatenation

#### Split Assignment

- `split`
  - One of: `train`, `val`, `test`

#### Quality Flags

- `duplicate_audio_flag`
  - Boolean: True if `audio_sha256` appears multiple times with different transcripts
  - Default: False

______________________________________________________________________

## Cleaning Rules (Dataset v1)

### Included Rules

Dataset v1 excludes samples that meet **any** of the following conditions:

1. **Audio file unreadable** (from inventory: `audio_read_ok = False`)
1. **Duration is zero or undefined** (`duration_sec IS NULL` or `<= 0`)
1. **Transcript is empty or null** (`transcript_is_blank = True`)
1. **Duplicate audio-transcript pairs** (same `audio_sha256` + `transcript_sha256`)
   - Keep first occurrence by `manifest_row_index`
   - Mark subsequent occurrences with `excluded_reason = "duplicate_audio_transcript"`

Each excluded sample MUST be logged with:

- `file_name`
- `manifest_row_index`
- `excluded_reason` (one of the above)

Exclusions MUST be deterministic and reproducible.

### Explicitly Not Cleaned

The following are intentionally left untouched in Dataset v1:

- Silence at beginning, middle, or end of audio
- Background noise or low SNR
- Pronunciation quality or clarity
- Transcript wording or normalization (capitalization, punctuation)
- Audio with `duplicate_audio_flag = True` (different transcripts, kept for review)

This ensures Dataset v1 remains a faithful representation of the original data
and does not introduce subjective quality filters.

______________________________________________________________________

## Split Policy

### Goals

The split policy aims to:

- Enable fair, repeatable evaluation across duration ranges
- Prevent data leakage (duplicates, temporal clustering)
- Ensure test set represents train distribution for reliable WER measurement
- Remain simple and defensible for a single-speaker dataset

### Why Split Policy Matters for Fine-Tuning

**Poor splits directly cause fine-tuning failure:**

1. **Duration Bias Example:**

   - If test set contains only 10s+ clips but training had mostly 3-5s clips
   - Model learns attention patterns for short sequences
   - Test WER will be artificially high and misleading
   - You won't know if poor WER is due to model capacity or data mismatch

1. **Temporal Leakage Example:**

   - Recording session at 10:00 AM: "What's the weather?"
   - Same session at 10:01 AM: "What's the temperature?"
   - If first → train, second → test, model may "memorize" session acoustics
   - Test WER will be artificially low (overfitting to recording conditions)

1. **Duplicate Leakage Example:**

   - Exact same audio + transcript appears in train and test
   - Model achieves 0% WER on that sample via memorization
   - Inflates test metrics without real generalization

**Result:** You waste GPU hours training on biased splits, get misleading metrics,
and can't trust evaluation to guide model improvements.

### Constraints

- Single speaker (no speaker-level stratification possible)
- Small to medium dataset size
- Single source: Euphonia recordings via web upload on Macbook Pro
- Temporal metadata available: `timestamp_ms`

### Policy Definition

#### Determinism

- Split assignment MUST be deterministic.
- A fixed random seed (`--seed`, default `42`) MUST be used.
- Given the same Dataset v1 inputs, split assignments MUST not change.

#### Split Ratios (Default)

- Train: 80%
- Validation: 10%
- Test: 10%

Ratios may be adjusted only by updating the specification.

#### Assignment Mechanism: Stratified Splitting

Dataset v1 uses **duration-stratified splitting** to ensure representative
evaluation and prevent bias.

##### Step 1: Remove Duplicates

Before splitting, detect and handle duplicate audio:

1. Group samples by `audio_sha256`
1. If duplicate audio exists with **same transcript** (same `transcript_sha256`):
   - Keep first occurrence by `manifest_row_index`
   - Mark others with `excluded_reason = "duplicate_audio_transcript"`
   - Do NOT include in dataset v1 manifest
1. If duplicate audio exists with **different transcripts**:
   - Keep all occurrences (may indicate labeling error)
   - Flag with `duplicate_audio_flag = True` for manual review
   - Log warning in dataset summary

##### Step 2: Define Duration Bins

Use the same bins as S1-M0 inventory for consistency:

- Bin 1: (0s, 1s\]
- Bin 2: (1s, 3s\]
- Bin 3: (3s, 10s\]
- Bin 4: (10s, 30s\]
- Bin 5: (30s, inf\]

##### Step 3: Stratified Assignment by Duration

For each duration bin independently:

1. Extract all samples in this bin
1. Sort by `pair_sha256` (for determinism)
1. Assign to train/val/test using 80/10/10 ratio based on sorted index

This ensures each split has proportional representation of all duration ranges.

**Example:**

- If dataset has 40% samples in (3s, 10s\] bin
- Then train will have ~40% in (3s, 10s\], val ~40%, test ~40%

##### Step 4: Temporal Clustering Check (Optional but Recommended)

To detect potential temporal leakage:

**If `timestamp_ms` is available for all or most samples:**

1. Sort samples by `timestamp_ms`
1. Flag samples where `timestamp_diff < 60000` ms (1 minute) as "session cluster"
1. Report in `dataset_v1_summary.json`:
   - `temporal_clusters_crossing_splits`: count of session clusters that span
     train/test boundary
   - If count > 0, log WARNING (manual review may be needed)

**If `timestamp_ms` is missing or sparse (\<50% of samples have timestamps):**

1. Skip temporal clustering check
1. Report in `dataset_v1_summary.json`:
   - `temporal_clusters_crossing_splits`: `null`
   - `temporal_check_status`: `"skipped_insufficient_timestamps"`
1. Log WARNING: "Temporal leakage check skipped due to insufficient timestamp data"

**Note:** For Dataset v1, temporal clustering is **detected but not enforced** in
split assignment. If significant leakage is detected, it may be addressed in v2.

##### Step 5: Validate Split Quality

After assignment, compute and report:

1. **Duration Distribution per Split:**

   - Histogram of duration bins for train/val/test
   - Expected: all splits have similar distributions

1. **Minimum Sample Counts:**

   - Default minimum thresholds:
     - Train: ≥ 100 samples total, ≥ 10 minutes duration
     - Val: ≥ 20 samples total, ≥ 2 minutes duration
     - Test: ≥ 20 samples total, ≥ 2 minutes duration
   - If constraints not met:
     - Emit ERROR with validation details
     - Allow override with `--allow_small_splits` flag (with WARNING logged)
     - Rationale: Enables milestone completion on small datasets (e.g., pilot data)
       while making the risk explicit

1. **Transcript Length Distribution:**

   - Report character count histogram per split
   - Ensure no severe imbalance (e.g., test has only very short transcripts)

#### Test Set Immutability

##### Version Locking

- Once Dataset v1 is finalized, generate `test_set_v1_frozen.csv` containing:
  - `file_name`, `pair_sha256`, `audio_sha256`, `transcript_sha256`
- Commit this file to version control alongside `dataset_v1_manifest.csv`

##### Future Dataset Versions (v2, v3, ...)

- MUST load `test_set_v1_frozen.csv`
- MUST preserve all v1 test samples in test split (by matching `pair_sha256`)
- MAY add new samples to test (document explicitly in changelog)
- MUST NOT move v1 test samples to train/val
- MUST NOT remove v1 test samples from test

This ensures evaluation continuity across dataset versions.

______________________________________________________________________

## Outputs

All outputs are written to `--out_dir` (default: `./out/dataset_v1/YYYYMMDD-HHMMSS`).

### Required Artifacts

#### 1. `dataset_v1_manifest.csv`

The canonical dataset definition. Contains all columns from manifest schema above.

#### 2. `dataset_v1_summary.json`

Machine-readable summary containing:

**Cleaning Summary:**

- `input_manifest_rows`: Total rows in input manifest
- `excluded_count`: Total excluded samples
- `excluded_breakdown`: Dict of `{reason: count}`
  - Example: `{"duplicate_audio_transcript": 5, "audio_unreadable": 2}`
- `included_count`: Final dataset size

**Split Summary:**

- `split_counts`: Dict of `{split: count}`
  - Example: `{"train": 800, "val": 100, "test": 100}`
- `split_durations_sec`: Dict of `{split: total_duration}`
- `split_durations_hours`: Dict of `{split: total_duration / 3600}`

**Duration Distribution per Split:**

- `split_duration_distributions`: Dict of `{split: {bin: count}}`
  - Example: `{"train": {"(0, 1]": 80, "(1, 3]": 160, ...}}`

**Quality Checks:**

- `duplicate_audio_different_transcript_count`: Samples flagged with
  `duplicate_audio_flag = True`
- `temporal_clusters_crossing_splits`: Count of recording sessions that span
  train/test boundary (if timestamp available)

**Validation Status:**

- `min_sample_validation_passed`: Boolean
- `min_duration_validation_passed`: Boolean
- `split_quality_warnings`: List of warning messages (empty if all checks pass)

**Metadata:**

- `dataset_version`: `"v1"`
- `created_timestamp`: ISO 8601 timestamp
- `spec_version`: Git SHA of this spec file (if available)
- `seed`: Random seed used for splitting
- `tool_versions`: Dict of `{library: version}` (e.g.,
  `{"python": "3.11.5", "pandas": "2.0.3"}`)

#### 3. `dataset_v1_excluded.csv`

Log of excluded samples with columns:

- `file_name`
- `manifest_row_index`
- `excluded_reason`
- `audio_sha256` (if available)
- `transcript_sha256` (if available)

#### 4. `test_set_v1_frozen.csv`

Test set lock file for version continuity. Contains:

- `file_name`
- `pair_sha256`
- `audio_sha256`
- `transcript_sha256`

This file MUST be committed to version control.

#### 5. `dataset_v1_report.md`

Human-readable 1-2 page summary (see Report Structure below).

### Optional (Derived Artifacts)

#### Materialized Directory Structure

- `dataset_v1_materialized/`
  - `train/` (symlinks or copies of audio files)
  - `val/`
  - `test/`

**Important:** This is a **convenience view only** and MUST NOT be treated as the
dataset definition. The manifest CSV is the single source of truth.

______________________________________________________________________

## Report Structure (dataset_v1_report.md)

Human-readable 1-2 page summary following this structure:

### 1. Overview

- Dataset version: `v1`
- Source: Euphonia recordings via web upload on Macbook Pro
- Created: `YYYY-MM-DD HH:MM:SS`
- Input manifest path
- Output directory path

### 2. Cleaning Summary

- Input samples: `N`
- Excluded samples: `M` (`X%`)
  - Breakdown by exclusion reason (table)
- Final dataset size: `N - M`
- Total duration: `X.X hours`

### 3. Split Summary

Table with columns:

- Split (train/val/test)
- Count
- Duration (hours)
- Percentage of total

### 4. Duration Distribution

Per-split histograms showing sample counts in each duration bin.
Visualize balance across splits.

### 5. Quality Checks

- Duplicate audio with different transcripts: `N` (flag for review)
- Temporal session leakage: `N` clusters crossing splits (if available)
- Minimum sample validation: PASS/FAIL
- Minimum duration validation: PASS/FAIL

### 6. Split Quality Assessment

- Duration distribution comparison across splits (visual/heuristic)
  - Compare bin counts: train vs val, train vs test
  - Flag if any bin differs by >20% (relative to train proportion)
- Optional: Statistical tests if scipy available (KS-test, chi-squared)
  - Not required; heuristic checks are sufficient for Dataset v1
- Any warnings about imbalanced splits
- Recommendation: READY FOR TRAINING / NEEDS REVIEW

### 7. Test Set Lock

- Test set frozen: `test_set_v1_frozen.csv`
- Test samples count: `N`
- Instructions for future dataset versions

### 8. Next Steps

- Recommended: Proceed to S1-M2 (Audio Preprocessing) if quality checks pass
- If issues detected: Manual review of flagged samples

## CLI Interface

```bash
python -m vox_dataset_versioning \
  --inventory_dir "./out/inventory/YYYYMMDD-HHMMSS" \
  --out_dir "./out/dataset_v1" \
  --seed 42 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --verbose
```

### Required Arguments

- `--inventory_dir`: Path to S1-M0 inventory output directory
  - Must contain `inventory_files.csv` and `inventory_summary.json`

### Optional Arguments

- `--out_dir`: Output directory (default: `./out/dataset_v1/YYYYMMDD-HHMMSS`)
- `--seed`: Random seed for deterministic splitting (default: `42`)
- `--train_ratio`: Train split ratio (default: `0.8`)
- `--val_ratio`: Validation split ratio (default: `0.1`)
- `--test_ratio`: Test split ratio (default: `0.1`)
  - Note: Ratios must sum to 1.0
- `--duration_bins`: Duration bin edges in seconds (default: `"1,3,10,30"`)
- `--skip_temporal_check`: Skip temporal clustering analysis (default: False)
- `--allow_small_splits`: Allow splits below minimum sample/duration thresholds
  with WARNING (default: False)
- `--verbose` / `-v`: Enable progress bar and detailed logging

### Output

The script prints:

- Output directory path
- Cleaning summary (excluded count by reason)
- Split summary table (counts and durations)
- Validation status (PASS/FAIL with details)
- Exit code:
  - `0`: Success, validation passed
  - `1`: Fatal error (e.g., inventory files unreadable)
  - `2`: Validation failed (minimum sample/duration constraints not met)

## Reproducibility Guarantees

Dataset v1 is reproducible if:

- Raw data inputs are unchanged
- S1-M0 inventory outputs are unchanged
- This specification is unchanged
- The same seed and split ratios are used
- The same hashing and split logic is applied

Any deviation requires a new dataset version.

## Error Handling

- Inventory files unreadable → fatal error (exit code 1)
- Insufficient samples for minimum constraints → fatal error (exit code 2)
  - Unless `--allow_small_splits` flag provided, then WARNING and continue
- Duplicate audio with different transcripts → warning (continue, flag in manifest)
- Temporal leakage detected → warning (continue, log in summary)
- Temporal check impossible (missing timestamps) → warning (skip check, log status)
- Audio file unreadable during hashing → skip sample, log in excluded.csv
- Hash computation format mismatch → fatal error (implementation bug, must fix)

______________________________________________________________________

## Failure Mode if Skipped

If Dataset Versioning is skipped:

- Experiments become non-comparable
- Results cannot be reliably reproduced
- Data changes are not traceable
- Debugging becomes speculative (“it worked last time”)

______________________________________________________________________

## Dependencies / Environment

- Python 3.11+
- Required libraries:
  - `pandas` (for manifest processing)
  - `numpy` (for stratified sampling)
  - Standard library: `hashlib`, `pathlib`, `json`, `csv`
- Optional libraries:
  - `scipy` (for statistical tests in split quality report)
  - If not available, use heuristic checks instead
- Input: S1-M0 inventory outputs (`inventory_files.csv`, `inventory_summary.json`)
- Must run locally on macOS (Apple Silicon)

## Performance Constraints

- Should handle up to 50k samples
- Hashing should be efficient (use file streams, not full load into memory)
- Split assignment should be O(N log N) at most (sorting-based)

## Completion Criteria

S1-M1 is considered complete when:

1. **Artifacts Generated:**

   - `dataset_v1_manifest.csv` generated with all required columns
   - `dataset_v1_summary.json` contains all specified metrics
   - `dataset_v1_excluded.csv` logs all excluded samples
   - `test_set_v1_frozen.csv` locks test set for future versions
   - `dataset_v1_report.md` provides human-readable summary

1. **Validation Passed:**

   - All minimum sample constraints met (≥100 train, ≥20 val/test)
     OR `--allow_small_splits` flag used with explicit WARNING in summary
   - All minimum duration constraints met (≥10min train, ≥2min val/test)
     OR `--allow_small_splits` flag used with explicit WARNING in summary
   - Duration distribution balanced across splits (visual inspection or
     heuristic checks)

1. **Version Control:**

   - `dataset_v1_manifest.csv` committed to git
   - `test_set_v1_frozen.csv` committed to git
   - `dataset_v1_summary.json` committed to git
   - Cleaning and split logic documented in this spec

1. **Identity Locked:**

   - Dataset version explicitly set to `v1` in all artifacts
   - No further modifications allowed without creating v2

______________________________________________________________________

## Creating Future Versions (v2, v3, ...)

This specification **defines Dataset v1 specifically**.

When you need to create Dataset v2 or later (new data, improved rules, or bug
fixes):

- See: [DATASET-VERSIONING-STRATEGY.md](../DATASET-VERSIONING-STRATEGY.md)
  (explains why v1 is hard-coded, how versions work, step-by-step v2 process)
- Create snapshot: `dataset-versions/v2/dataset_v2_spec.md`
- Update: `dataset-versions/CHANGELOG.md`
- Run with: `python -m vox_dataset_versioning --dataset_version v2`

Test set from v1 will be automatically preserved in v2 and all future versions.
