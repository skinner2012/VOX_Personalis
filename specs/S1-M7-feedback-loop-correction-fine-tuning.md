# S1-M7: Feedback Loop & Correction Fine-Tuning

## Purpose

Close the loop from error observation to model improvement.

Google's Project Euphonia demonstrates that ASR systems for atypical speakers
improve dramatically when users correct transcription errors and those
corrections feed back into training. Our data originates from Euphonia
recordings (S1-M1); our fine-tuning pipeline (M3-M5) already trains on
`(audio, transcript)` pairs. M7 experiments with capturing corrections during live use and feeding them
back through that pipeline to see if targeted corrections can push past the
44% WER wall.

This milestone answers one question:

> *Can we collect user corrections during serving, batch fine-tune with those
> corrections using the existing LoRA pipeline, and measure WER improvement
> on the same val set?*

______________________________________________________________________

## Context

Model v2 (small.en + LoRA r=16, `textnorm_v2`) achieved 44.02% val WER —
an 18.35-point breakthrough from v1.1 (62.37%). But 44% is a wall. M6
serves the model live, and during daily use the user observes transcription
errors — but they are ephemeral. There is no way to capture what went wrong
or tell the model the correct answer.

The fine-tuning pipeline (M3-M5) already loads `(audio_path, transcript)`
pairs from a manifest CSV and trains LoRA adapters. Feedback data must produce
WAV + corrected text compatible with that manifest format. No new training
infrastructure is needed — only a bridge from the serving UI to the training
pipeline.

**Why not continue scaling model capacity?** M5 already explored this lever:
base.en (74M) → small.en (244M) gained 18 points. The next step would be
medium.en (769M) — 3× the parameters, significantly slower inference, and
higher overfitting risk on ~700 training samples. More importantly, scaling
alone does not reveal which errors the model makes or build a mechanism to
fix them. M7 tests a different lever: whether targeted corrections from live
use can improve the model where additional capacity may not.

______________________________________________________________________

## Approach Analysis

Four approaches to breaking the 44% WER wall:

**1. Scale model capacity** — Use medium.en (769M) or large (1550M).
Already explored in M5: base→small gave 18 pts. But does not reveal
specific error patterns or build a reusable fix mechanism. Also: 3-6×
compute, slower inference, overfitting risk on ~700 samples.

**2. Language model rescoring** — Add N-gram or neural LM to rescore
beam search output. Fixes word-boundary substitution errors, but does
not fix acoustic errors (mishearing sounds), adds inference complexity,
and requires an external LM.

**3. More training data** — Record additional speech samples. More data
always helps, but expensive to collect, does not target specific error
patterns, and has a slow feedback cycle.

**4. Fine-tune on corrections ★** — User corrects errors → (audio,
corrected_text) pairs added to training set → LoRA fine-tune. Directly
targets model's actual errors, reuses M3-M5 pipeline, minimal new
infrastructure, zero additional inference cost. 25-40 corrections
proves the mechanism; ~100 for stronger signal. Risk of catastrophic
forgetting mitigated by mixing with original data.

**Selected approach:** Supervised fine-tuning on user corrections, using the
same LoRA pipeline from M3-M5. Original training data mixed with corrections
to prevent catastrophic forgetting (see Design Decisions).

______________________________________________________________________

## Dependencies

- S1-M6 complete (serving pipeline operational)
- M3-M5 fine-tuning pipeline (`scripts/fine_tuning/`)
- Original training manifest (`out/dataset_v1/.../dataset_v1_manifest.csv`)
- Model v2 checkpoint (`out/capacity_scaling/20260304-103236/checkpoint`)
- See [Package Dependencies](#package-dependencies) — reuses existing deps

______________________________________________________________________

## Scope

### In Scope

- Audio retention buffer (hold raw PCM in memory per segment after inference)
- `POST /feedback` endpoint on the serving API
- Storage: `out/feedback/NNN/audio.wav + correction.json`
- Demo UI: inline text editing with submit button
- Memory cleanup on WebSocket disconnect
- CLI `--feedback_out` argument for serving
- Status tracking: consumed vs pending corrections (`consumed.marker`)
- Batch fine-tuning integration module (`scripts/feedback_finetune/`)
- Evaluation: v2 vs v2+corrections on same val set
- Comparison report with WER delta and breakdowns

### Out of Scope

- Auto-retraining or training triggers (manual batch only)
- Dataset quality scoring, statistics, or validation gates
- Audio playback in UI
- Professional UI styling
- Multiple fine-tuning rounds (M7 proves one round works)
- Undo/delete corrections, multi-session correction history
- Manifest CSV generation as a separate tool (integrated into fine-tune script)

______________________________________________________________________

## Design Decisions

Each decision below explains the "why" — following the format of prior specs.

### Why save audio only for corrected segments

Disk writes are proportional to error rate, not total usage. If the user speaks
100 utterances and corrects 10, only 10 audio files are saved (~30-50 KB each).
Saving all audio would require a different storage strategy and produce data
the pipeline cannot use (no correction = no signal).

### Why directory-per-correction

Each correction is an atomic unit: `audio.wav` + `correction.json` in one
directory. Partial writes corrupt one entry, not the entire dataset. The
structure is inspectable (`ls out/feedback/`), easy to back up, and directly
compatible with the fine-tuning manifest format (each directory maps to one
manifest row).

### Why inline edit (not modal dialog)

Inline editing matches reading flow: the user sees an error in the transcript,
clicks on it, fixes it, and submits. A modal dialog interrupts context and
adds clicks. The transcript text is already visible — making it editable is
the minimal UX change.

### Why same M5 pipeline (not new training code)

The M3-M5 fine-tuning pipeline is proven: it took WER from 62% to 44%. Adding
corrections to the training data is the minimal change — same model, same LoRA
config, same hyperparameters. Using different hyperparameters or a different
training approach would confound the comparison: we would not know whether
improvement came from corrections or from config changes.

### Why mix original data + corrections (not corrections alone)

Training only on corrections would cause catastrophic forgetting — the model
overfits to correction patterns and loses general capability. Mixing original
training data (~700 samples) with corrections maintains the model's existing
knowledge while incorporating targeted fixes. The corrections are a
supplement, not a replacement.

### Why batch fine-tune (not continuous retraining)

Collect corrections at natural pace over multiple sessions, then batch
fine-tune once. This decouples collection from training, avoids continuous
retraining complexity (trigger logic, validation gates, checkpoint management,
rollback strategy), and produces a clean before/after comparison.

______________________________________________________________________

## Audio Retention Strategy

**Integration point:** `scripts/serving/api.py:_handle_utterance()` line ~211

After successful transcription, before `audio_bytes` goes out of scope:

```python
_audio_buffer.store(segment_id, audio_bytes)
```

**AudioRetentionBuffer class** (new file: `scripts/serving/feedback.py`):

| Property     | Value                                                   |
| ------------ | ------------------------------------------------------- |
| Storage      | In-memory dict: `{segment_id: bytes}`                   |
| Max segments | 100 (~6-16 MB for 2-5s utterances at 16kHz 16-bit mono) |
| Eviction     | FIFO when over limit                                    |
| Cleanup      | `clear()` on WebSocket disconnect                       |
| Retrieval    | `pop(segment_id)` on successful feedback submission     |

The buffer is per-connection, scoped by `session_id` (UUID4 assigned on
WebSocket connect). Internally keyed as `(session_id, segment_id)` to avoid
collision across reconnects. Since this is a single-user demo, `POST /feedback`
uses the current active session — no session_id in the request body.

Audio is raw PCM: 16kHz, 16-bit signed integer, mono (per M6 WebSocket
protocol). `duration_sec` is computed as `len(pcm) / (16000 * 2)`.
Conversion to WAV happens at write time in `FeedbackStore`.

______________________________________________________________________

## Feedback Endpoint: `POST /feedback`

### Request

```json
{
  "segment_id": 3,
  "original_text": "Turn on the lights.",
  "corrected_text": "Turn on the light."
}
```

### Response (200 OK)

```json
{
  "status": "saved",
  "feedback_id": "0007",
  "path": "out/feedback/0007"
}
```

### Error Responses

| Code | Condition                                                             |
| ---- | --------------------------------------------------------------------- |
| 400  | Missing required field                                                |
| 400  | `original_text` and `corrected_text` are identical                    |
| 404  | Audio for `segment_id` expired (evicted from buffer or session ended) |
| 500  | Disk write failure                                                    |

### Server Logic

1. Validate request fields
1. Pop audio bytes from retention buffer (`_audio_buffer.pop(segment_id)`)
1. Create `out/feedback/NNN/` directory (sequential numbering)
1. Write `audio.wav` (16kHz, 16-bit signed integer, mono — via stdlib `wave`)
1. Write `correction.json` (metadata)

______________________________________________________________________

## Storage Format

```text
out/feedback/
  0001/
    audio.wav           # 16kHz, 16-bit signed integer, mono
    correction.json     # metadata
  0002/
    audio.wav
    correction.json
    consumed.marker     # added by fine-tuning script after use
```

### correction.json Schema

```json
{
  "feedback_id": "0007",
  "segment_id": 3,
  "session_id": "a1b2c3d4-...",
  "original_text": "Turn on the lights.",
  "corrected_text": "Turn on the light.",
  "duration_sec": 2.43,
  "timestamp": "2026-03-13T14:23:01Z",
  "model_version": "v2"
}
```

`model_version` is derived from the checkpoint directory name passed
via `--checkpoint` at server start (e.g., `"v2"` from the current default).

### Status Tracking

| State    | Indicator                 | Meaning                             |
| -------- | ------------------------- | ----------------------------------- |
| Pending  | No `consumed.marker` file | Available for next fine-tune batch  |
| Consumed | `consumed.marker` present | Already used in a fine-tuning batch |

**When to write `consumed.marker`:** Only after the merged manifest is saved
AND the checkpoint is successfully written. Evaluation failure does not block
the marker — the corrections were consumed by training regardless of whether
eval succeeded.

The marker contains:

```json
{
  "batch_id": "batch_001",
  "timestamp": "2026-03-15T10:00:00Z",
  "output_checkpoint": "out/feedback_finetune/batch_001/checkpoint",
  "evaluation_status": "success"
}
```

If evaluation failed, `"evaluation_status": "failed"` is recorded so the gap
is visible without re-running the batch.

______________________________________________________________________

## Demo UI Changes

Modify `scripts/serving/static/demo.html`:

### Interaction Flow

1. Transcript segment appears after inference (existing behavior)
1. User clicks on segment text → `contentEditable = true`, submit button appears
1. User edits the text to correct errors
1. User clicks Submit → `POST /feedback` with segment_id, original, corrected
1. On success: green border + checkmark icon, text locked (no further editing)
1. On error: brief red flash, text remains editable

### Data Attributes

Each `.segment` element stores:

- `data-segment-id` — matches the segment_id from inference
- `data-original-text` — captured when segment first renders (before any edit)

______________________________________________________________________

## Fine-Tuning Integration

### Module Structure

```text
scripts/feedback_finetune/
  __init__.py
  __main__.py           # Entry point: python -m scripts.feedback_finetune
  cli.py                # CLI argument parsing + pipeline orchestration
  manifest.py           # Scan corrections, generate manifest rows, merge
```

### Pipeline

1. **Scan** `out/feedback/*/` for directories without `consumed.marker`
1. **Generate** manifest rows from each pending correction:
   - Read `correction.json` → `corrected_text`, `duration_sec`
   - Resolve absolute path to `audio.wav`
   - Compute `duration_bin`, `pair_sha256`
   - Set `split = "train"`, `source = "feedback"`
1. **Load** original training manifest via `scripts.fine_tuning.data.load_manifest()`
1. **Merge** original train rows + correction rows into combined manifest
1. **Write** merged manifest to `out/feedback_finetune/batch_NNN/merged_manifest.csv`
1. **Train** from base small.en + fresh LoRA (same as M5 — not continuing
   from v2 checkpoint, so the only variable is the expanded dataset):
   - `setup_model_and_processor()` from `scripts.fine_tuning.models`
   - `create_hf_dataset()` + `prepare_dataset()` from `scripts.fine_tuning.data`
   - `train_model()` from `scripts.fine_tuning.training`
1. **Evaluate** on same val set:
   - `run_full_evaluation()` from `scripts.fine_tuning.evaluation`
1. **Mark** used corrections with `consumed.marker`
1. **Report** comparison results

### Reused Functions (not modified)

From `scripts/fine_tuning/data.py`:

- `load_manifest(path, split)` — load and filter manifest by split
- `create_hf_dataset(df)` — convert DataFrame to HF Dataset
- `prepare_dataset(dataset, processor, normalizer)` — feature
  extraction + tokenization

From `scripts/fine_tuning/models.py`:

- `setup_model_and_processor(model, rank, device, dropout)` — load
  Whisper + apply LoRA

From `scripts/fine_tuning/training.py`:

- `train_model(model, processor, train, eval, normalizer, config)`
  — training loop with early stopping
- `save_checkpoint(model, processor, output_dir)` — save LoRA adapter

From `scripts/fine_tuning/evaluation.py`:

- `run_full_evaluation(model, processor, manifest_df, dataset, ...)`
  — evaluate + generate metrics

CLI usage: see [CLI Interface](#cli-interface) below.

### Hyperparameters (same as M5)

| Parameter      | Value             |
| -------------- | ----------------- |
| Model          | `small.en` (244M) |
| LoRA rank      | 16                |
| LoRA alpha     | 32 (2× rank)      |
| Dropout        | 0.15              |
| Learning rate  | 1e-4              |
| Epochs         | 3                 |
| Batch size     | 4                 |
| Early stopping | patience=2        |

If OOM or clear training instability occurs, batch size may be reduced
(e.g., 4 → 2). Log the change as a variant — no other hyperparameters
should be adjusted, to keep the comparison clean.

______________________________________________________________________

## Evaluation Protocol

### Goal

Compare v2 WER vs v2+corrections WER on the same val set used in M5.
Apples-to-apples: same val samples, same normalizer (`textnorm_v2`), same
evaluation code.

**Interpretation boundary:** M7 measures whether the feedback loop produces
measurable validation gains. It does not claim unbiased final generalization —
the val set is fixed and small. The purpose is to prove the mechanism works,
not to publish a benchmark.

### Comparison Report

```text
| Model              | Val WER  | Delta      | # Train Samples |
|--------------------|----------|------------|-----------------|
| v2 (M5 baseline)   | 44.02%   | —          | ~700            |
| v2+corrections     | ??.??%   | -X.XX pts  | ~700 + N corr.  |
```

### Breakdowns

Reusing existing evaluation infrastructure from `scripts/fine_tuning/evaluation.py`:

- **By duration bin**: ≤3s, (3s, 10s\], >10s
- **Per-sample**: which val samples improved, degraded, or stayed the same
- **Aggregate error types**: insertions, deletions, substitutions

### Output

Report saved to: `out/feedback_finetune/batch_NNN/comparison_report.md`

______________________________________________________________________

## Systematic Correction Protocol

**Targets:** Required: 25-40 corrections (proves the loop end-to-end).
Stretch: ~100 corrections (stronger signal, if time allows).

1. Start MVS server (`python -m scripts.serving`)
1. Speak naturally — mix short commands, medium sentences, long phrases
1. After each utterance, review the transcript in the demo UI
1. If incorrect: click the text, correct it, click Submit
1. Target diverse utterances (do not only fix easy one-word errors)
1. Track progress: `ls out/feedback/ | wc -l`
1. At 25-40 corrections: run first batch fine-tune and evaluate
1. Optionally continue collecting to ~100 and run a second batch

______________________________________________________________________

## CLI Interface

### Serving (modified)

```bash
python -m scripts.serving \
  --checkpoint out/capacity_scaling/20260304-103236/checkpoint \
  --feedback_out out/feedback     # NEW: feedback output directory
```

### Feedback Fine-Tuning (new)

```bash
python -m scripts.feedback_finetune \
  --feedback_dir out/feedback \
  --original_manifest out/dataset_v1/20260206-142756/dataset_v1_manifest.csv \
  --baseline_metrics out/baseline_eval/.../baseline_metrics.json \
  --output_dir out/feedback_finetune \
  --model small.en \
  --lora_rank 16 \
  --dropout 0.15 \
  --device cpu
```

______________________________________________________________________

## Output Artifacts

Feedback collection layout: see [Storage Format](#storage-format).
New source files: see [Module Structure](#module-structure).

```text
out/feedback_finetune/
  batch_001/
    merged_manifest.csv         # Original + correction rows
    checkpoint/                 # LoRA adapter (v2+corrections)
    predictions.csv             # Per-sample predictions on val set
    metrics.json                # Aggregate metrics + comparison
    comparison_report.md        # Human-readable report
    logs/                       # Training logs
```

______________________________________________________________________

## Error Handling

| Scenario                       | Behavior                             | Exit |
| ------------------------------ | ------------------------------------ | ---- |
| Audio buffer full (100 segs)   | FIFO eviction, log warning           | 0    |
| Feedback with expired segment  | Return 404, brief red flash          | 0    |
| Feedback with identical text   | Return 400, no disk write            | 0    |
| Disk write failure (WAV/JSON)  | Return 500, no partial directory     | 0    |
| Feedback dir not writable      | Log error, disable feedback endpoint | 0    |
| No pending corrections         | Print message, exit cleanly          | 0    |
| Original manifest not found    | Print path, fail fast                | 1    |
| Fine-tuning OOM                | Print error, suggest smaller batch   | 1    |
| Evaluation failure after train | Save checkpoint anyway, log error    | 0    |

______________________________________________________________________

## Package Dependencies

Prefer no new packages. All dependencies listed are already installed from
prior milestones. New packages allowed if they replace significant glue code.

| Package        | Version  | Used By           | Purpose                      |
| -------------- | -------- | ----------------- | ---------------------------- |
| `wave`         | stdlib   | feedback.py       | Write WAV files from raw PCM |
| `uuid`         | stdlib   | feedback.py       | Generate session IDs         |
| `json`         | stdlib   | feedback.py       | Write correction.json        |
| `transformers` | existing | feedback_finetune | Model + training             |
| `peft`         | existing | feedback_finetune | LoRA adapters                |
| `jiwer`        | existing | feedback_finetune | WER computation              |
| `librosa`      | existing | feedback_finetune | Audio loading for training   |

______________________________________________________________________

## Completion Criteria

### Collection

1. `POST /feedback` saves `audio.wav` + `correction.json` to numbered directory
1. Demo UI allows inline editing of any transcript segment
1. Submit button sends correction via `POST /feedback`
1. Visual feedback (green border, checkmark) after successful submission
1. Audio retained in memory after inference, associated with segment_id
1. Audio cleaned up on WebSocket disconnect
1. WAV files loadable by `librosa.load(path, sr=16000)`
1. `correction.json` contains all required fields per schema
1. Directory numbers are sequential, no overwrites on server restart
1. `--feedback_out` CLI argument controls output directory

### Fine-Tuning

1. Integration script scans pending corrections (no `consumed.marker`)
1. Merged manifest contains original train rows + correction rows
1. Fine-tuning completes using M5 hyperparameters
1. `consumed.marker` written to each used correction directory
1. New checkpoint saved to `out/feedback_finetune/batch_NNN/checkpoint`

### Evaluation

1. v2+corrections model evaluated on same val set as M5
1. Comparison report shows WER delta and breakdown by duration bin
1. Report saved to `out/feedback_finetune/batch_NNN/comparison_report.md`

______________________________________________________________________

## Failure Modes if Skipped

- Transcription errors remain ephemeral — no mechanism to improve the model
  from observed failures
- The 44% WER wall becomes permanent without targeted correction data
- Fine-tuning pipeline (M3-M5) has no path to receive new training signal
  from live usage
- Portfolio lacks the most compelling piece: a closed-loop data flywheel
  demonstrating that the system learns from its mistakes

______________________________________________________________________

## Implementation Notes

### Audio Format Verification

Corrections produce WAV files from raw PCM received over WebSocket. The
`FeedbackStore` must verify the WAV is valid before marking the correction
as saved. Verification: `librosa.load(path, sr=16000)` should return audio
without error and duration should approximately match `duration_sec` in
`correction.json`.

### Sequential Directory Numbering

On server start, scan `out/feedback/` for the highest existing directory
number and continue from there. This prevents overwrites across server
restarts. Use zero-padded 4-digit names (`0001`, `0002`, ..., `9999`).

### Manifest Row Generation

Each correction becomes one row in the merged manifest. Required columns
(matching `scripts/fine_tuning/data.py:load_manifest()`):

| Column                | Source                                                |
| --------------------- | ----------------------------------------------------- |
| `file_name`           | `audio.wav` filename                                  |
| `audio_path_resolved` | Absolute path to `out/feedback/NNN/audio.wav`         |
| `transcript_raw`      | `corrected_text` from `correction.json`               |
| `split`               | `"train"` (all corrections go to training set)        |
| `duration_sec`        | From `correction.json`                                |
| `duration_bin`        | Computed from `duration_sec` using same binning as M1 |
| `pair_sha256`         | SHA256 of audio+transcript content                    |

______________________________________________________________________

## References

- S1-M3: Personalization & Fine-Tuning (LoRA pipeline, manifest format)
- S1-M5: Model Capacity Scaling (v2 training, hyperparameters, evaluation)
- S1-M6: Minimal Viable Serving (serving architecture, demo UI, WebSocket)
- Google Project Euphonia: ASR personalization for atypical speakers
