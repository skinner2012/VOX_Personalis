# S1-M0 Data Inventory Report — Specification

## Goal
Produce a deterministic, reproducible "Data Inventory Report" for a labeled ASR dataset
(audio files + transcript CSV). The report must answer:

1) What data do we have (counts, durations, formats)?
2) Are there obvious metadata inconsistencies (sample rate, channels, missing files)?
3) Are transcripts plausibly usable (blank / mismatch signals)?
4) Rough estimate of long-silence / noise issues (coarse metrics only).
5) Initial conclusion: do we need major cleanup, and what are dominant failure modes?

## Non-Goals
- No audio modification (no trimming, no resampling, no denoise).
- No model inference / WER evaluation.
- No heavy ML pipelines.

## Inputs
### Required
- `--data_dir`: directory containing audio files (wav/flac/mp3/…)
- `--manifest_csv`: CSV with at least:
  - `file_name` (string): audio file name (with extension)
  - `transcript` (string): ground-truth text

### Optional
- `--audio_glob`: default `**/*` under data_dir
- `--file_col`: default `file_name`
- `--text_col`: default `transcript`
- `--encoding`: default `utf-8`
- `--stratify`: Enable stratified sampling by duration bins (default: True)
- `--stratify_bins`: Custom duration bins in seconds, comma-separated (default: "1,3,10,30")

### Display / Logging
- `--verbose` / `-v`: Enable progress bar and detailed logging (default: False)
- `--quiet` / `-q`: Suppress all output except errors (default: False)

## Outputs
The script produces:
1) `inventory_summary.json` (machine-readable summary)
2) `inventory_files.csv` (per-file metadata table)
3) `inventory_report.md` (human-readable 1–2 pages, markdown)
4) `inventory_samples.csv` (sampled transcript checks list with manual columns)

All outputs are written to `--out_dir` (default `./out/inventory/YYYYMMDD-HHMMSS`).

## Determinism / Reproducibility
- Any sampling must be controlled by `--seed` (default `42`).
- Sorting must be stable (lexicographic by `file_name`).
- All metrics should be computed from the raw input and recorded with tool versions.

## Dependencies / Environment
- Python 3.11+
- Allowed libraries:
  - standard lib
  - `pandas`
  - `numpy`
  - `soundfile` (preferred) or `librosa` (if needed)
  - `webrtcvad` (required for VAD-based silence detection)
  - `tqdm` (for progress bar when --verbose is enabled)
- Must run locally on macOS (Apple Silicon) with CPU-only dependencies.

## Data Integrity Checks (must be included)
### Manifest validity
- Rows count
- Duplicate `file_name` count
- Empty / null transcript count
- Empty / null file_name count

### File existence
- Missing audio files count + list (top 50 in report, full in CSV)
- Extra audio files count (in data_dir but not in manifest) + list (top 50)

### Readability
- Audio read failures count + list (top 50)

## Per-file Metadata (inventory_files.csv columns)
For each manifest row, output:

### Identity
- `file_name`
- `manifest_row_index`

### Transcript
- `transcript_raw`
- `transcript_len_chars`
- `transcript_len_words` (split on whitespace)
- `transcript_is_blank` (bool)
- `transcript_has_non_ascii_ratio` (float 0–1; for debugging)

### Audio
- `audio_path_resolved`
- `audio_exists` (bool)
- `audio_read_ok` (bool)
- `duration_sec` (float; null if unreadable)
- `sample_rate_hz` (int; null if unreadable)
- `channels` (int; null if unreadable)
- `format` (string; e.g., WAV/FLAC/MP3 if available)
- `bit_depth` (int if available; else null)

### Coarse Silence / Noise (optional but recommended)
Compute coarse metrics using Voice Activity Detection (VAD). Provide null if unreadable.
- `silence_ratio_est` (0–1): % frames classified as non-speech by VAD
- `longest_silence_sec_est` (float): longest consecutive silence span
- `rms_db_est` (float): approximate RMS in dBFS-like scale

**VAD Implementation**:
- Use `webrtcvad` (mode 3 = most aggressive non-speech detection)
- Frame size: 30ms (required by webrtcvad)
- Resample to 16kHz only for VAD computation (do not modify source audio)
- Rationale: VAD-based detection is more robust than energy thresholding for:
  - Distinguishing speech from background noise
  - Handling varying recording volumes across files
  - Detecting pauses in natural speech vs. dead air

**Fallback**: If audio is unreadable or resampling fails, set all silence metrics to null.

## Aggregate Summary Metrics (inventory_summary.json)
### Dataset size
- `num_manifest_rows`
- `num_unique_files`
- `total_duration_sec` (sum readable)
- `duration_histogram` (bucketed):
  - Bins: [0-1s, 1-3s, 3-10s, 10-30s, 30-60s, >60s]
- `transcript_len_histogram` (character count, bucketed):
  - Bins: [0-10, 10-50, 50-100, 100-200, >200]

### Audio distributions
- `sample_rate_distribution` (value -> count)
- `channels_distribution` (value -> count)
- `format_distribution` (value -> count)
- `read_failure_count`
- `missing_file_count`

### Transcript sanity
- `blank_transcript_count`
- `very_short_transcript_count` (config: <= 2 words)
- `duplicate_transcript_count` (exact duplicates; optional)

### Silence / Noise coarse
- `silence_ratio_distribution` (bucketed):
  - Bins: [0-0.1, 0.1-0.2, 0.2-0.4, 0.4-0.6, >0.6]
- `longest_silence_distribution` (seconds, bucketed):
  - Bins: [0-0.5s, 0.5-1s, 1-2s, 2-5s, >5s]
- `rms_db_distribution` (dBFS, bucketed):
  - Bins: [<-60, -60 to -40, -40 to -20, -20 to -10, >-10]

## Transcript Quality Sampling (inventory_samples.csv)
Goal: enable manual review without listening to entire dataset.

**Sampling Strategy**:
- Sample `N = --sample_n` (default 100) from readable audio files
- Use **stratified sampling by duration** to ensure diverse coverage:
  - 0-1s: 10% of samples (catch truncation issues)
  - 1-3s: 20% of samples (short phrases)
  - 3-10s: 40% of samples (normal sentences, typically majority of dataset)
  - 10-30s: 20% of samples (long sentences/paragraphs)
  - >30s: 10% of samples (outliers, potential segmentation issues)
- If a duration bin has fewer files than allocated samples, distribute extras proportionally to other bins
- Sampling must be deterministic by `--seed` (use stratified sampling with fixed seed)

**Rationale for duration-based stratification**:
Duration is the most informative feature for ASR quality control because:
- Short clips (<3s) often indicate truncation or segmentation errors
- Long clips (>30s) may have attention/memory issues for models
- Duration correlates with labeling difficulty and error rates
- Balances representation across the natural variance in the dataset

**CSV columns**:
  - `file_name`
  - `duration_sec`
  - `transcript_raw`
  - `audio_path_resolved` (full path for reviewers)
  - `manual_obvious_error` (empty; to be filled by human)
  - `manual_blank_or_garbled` (empty)
  - `manual_mismatch_signal` (empty)
  - `notes` (empty)

## Report (inventory_report.md) Structure (1–2 pages)
### 1. Overview
- Dataset name (from --dataset_name)
- Data paths used
- Run timestamp
- Tool versions (python, pandas, soundfile)

### 2. Inventory Summary
- # files, total duration (hours)
- Read failures / missing files
- File format distribution
- Sample rate / channel distribution

### 3. Transcript Sanity
- blank transcript ratio
- very short transcript ratio
- duplicates ratio (optional)
- Notes on language / character anomalies (non-ascii ratio heuristic)

### 4. Coarse Silence / Noise
- silence_ratio_est distribution
- longest_silence_sec_est distribution
- highlight "red flags" thresholds:
  - e.g., silence_ratio_est > 0.4
  - longest_silence_sec_est > 2.0

### 5. Initial Conclusion
Provide explicit bullets:
- Major cleanup required? (Yes/No/Conditional)
- Dominant failure modes (top 3)
- Recommended next milestone (e.g., Cleanup Policy, Label Repair, VAD trimming)

## CLI Interface
Example:
python -m vox_inventory \
  --dataset_name "vox-personalis-v1" \
  --data_dir "./data/audio" \
  --manifest_csv "./data/labels.csv" \
  --out_dir "./out" \
  --seed 42 \
  --sample_n 100 \
  --verbose

The module should print:
- output directory path
- summary table (counts and total duration)
- exit code non-zero if fatal errors (e.g., manifest unreadable)

## Error Handling Rules
- Manifest unreadable => fatal error
- Missing audio files => not fatal, but reported
- Audio read error => not fatal, but reported
- Any exception should not stop processing remaining files

## Performance Constraints
- Should handle up to 50k files
- Avoid loading full waveform into memory if not needed
- Use streaming read when possible (or read minimal to compute duration if library supports)
