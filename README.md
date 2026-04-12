# VOX Personalis

VOX Personalis is an engineering-focused project exploring
**personalized speech recognition** for a single Deaf speaker.

The emphasis is **not** on training a new ASR model from scratch, but on
**data inspection, risk analysis, evaluation methodology, and
reproducible system design** for personalization workflows.

This project treats speech recognition as a **system engineering problem**,
not a demo.

## Table of Contents

- [Motivation](#motivation)
- [Milestones](#milestones)
  - [S1-M0 — Data Inventory & Risk Scan](#s1-m0--data-inventory--risk-scan)
  - [S1-M1 — Dataset Versioning (v1)](#s1-m1--dataset-versioning-v1)
  - [S1-M2 — Baseline Model & Offline Evaluation](#s1-m2--baseline-model--offline-evaluation)
  - [S1-M3 — Personalization & Fine-Tuning](#s1-m3--personalization--fine-tuning)
  - [S1-M4 — Targeted Improvements](#s1-m4--targeted-improvements--controlled-model-upgrades)
  - [S1-M4a — Error Analysis](#s1-m4a--error-analysis--targeted-improvement-hypotheses)
  - [S1-M4b — Normalization Fix](#s1-m4b--normalization-fix)
  - [S1-M5 — Model Capacity Scaling](#s1-m5--model-capacity-scaling)
  - [S1-M6 — Minimal Viable Serving](#s1-m6--minimal-viable-serving-mvs)
  - [S1-M7 — Feedback Loop](#s1-m7--feedback-loop--correction-fine-tuning)
  - [S1-M8 — Minimal Ops Layer](#s1-m8--minimal-ops-layer)
- [Repository Structure](#repository-structure)
- [What This Project Is Not](#what-this-project-is-not)
- [Status](#status)
- [Development](#development)
- [License](#license)

______________________________________________________________________

## Motivation

This project exists to solve two problems at the same time.

**Problem 1: General-purpose ASR does not work well for my voice.**
As a Deaf speaker, my pronunciation patterns fall outside what most speech
recognition systems are trained on. Services like Google's Project Euphonia
and Relate exist, but they are platform-locked — I cannot use them in my own
tools, on my own terms, or in contexts those products were not designed for.
I want a personalized model I actually own: one I can run locally, embed in
any workflow, and improve whenever I collect new data.

**Problem 2: Building this is how I learn.** Instead of studying ML theory
in isolation, I wanted to learn by solving a real problem — model fine-tuning,
evaluation methodology, serving infrastructure, feedback loops, and the
operational discipline that holds it all together. Every milestone in this
repo is a lesson in how ML systems actually work, from data inventory through
production serving.

VOX Personalis is the result: an engineering-first exploration of
single-speaker ASR personalization, answering one practical question:

> *Given a small, labeled, single-speaker dataset, can personalization be done
> in a controlled, explainable, and engineering-sound way?*

The repository follows a **spec-first, milestone-driven** approach:

- Specifications define behavior and scope **before** implementation.
- Each milestone answers a concrete engineering question.
- All outputs are reproducible and intentionally scoped.
- Success is measured by **clarity and correctness**, not just metric improvement.

______________________________________________________________________

## Milestones

### S1-M0 — Data Inventory & Risk Scan

Goal:

- Understand what data exists and whether it is suitable for personalization.

Key questions:

- What audio and transcript data do we actually have?
- Are there obvious metadata inconsistencies or failures?
- Are there systematic silence or noise issues?
- Does the dataset warrant further investment?

Specification:

- [`specs/S1-M0-data-inventory.md`](specs/S1-M0-data-inventory.md)

Key concepts:

- Per-file metadata inventory (CSV)
- Aggregate dataset summary (JSON)
- Human-readable inventory report (Markdown)

> No audio is modified and no model is trained at this stage.

### S1-M1 — Dataset Versioning (v1)

Goal:

- Create a reproducible, immutable Dataset v1 with deterministic split policy
  and frozen test set for fair model evaluation.

Key questions:

- How do we create deterministic, reproducible train/val/test splits?
- How do we detect and handle duplicate or leaked samples?
- How do we freeze the test set for future version continuity?

Specification:

- [`specs/S1-M1-dataset-versioning.md`](specs/S1-M1-dataset-versioning.md)

Key concepts:

- Duration-stratified splitting to prevent evaluation bias
- Duplicate and temporal leakage detection
- Test set frozen for future version continuity

Design details:

- [`DATASET-VERSIONING-STRATEGY.md`](DATASET-VERSIONING-STRATEGY.md) (explains
  dataset version lineage, why v1 is immutable, how to create v2+)

### S1-M2 — Baseline Model & Offline Evaluation

Goal:

- Establish a non-personalized baseline and stable offline evaluation framework
  to define the performance floor for Dataset v1.

Key questions:

- How does a generic ASR model behave on this speaker's data?
- What error patterns emerge before any personalization?
- Which errors are likely addressable via personalization?

Specification:

- [`specs/S1-M2-baseline-and-offline-evaluation.md`](specs/S1-M2-baseline-and-offline-evaluation.md)

Key concepts:

- Whisper `small.en` as baseline (Euphonia reference point)
- WER/CER metrics with jiwer normalization
- Duration-stratified evaluation slices
- Error pattern analysis for interpretability

### S1-M3 — Personalization & Fine-Tuning

Goal:

- Fine-tune Whisper using LoRA to achieve measurable WER improvement
  on this Deaf speaker's voice.

Key questions:

- Does LoRA fine-tuning improve transcription accuracy for this speaker?
- What is the optimal LoRA rank (r=8 vs r=16)?
- What decoding configuration yields best results?
- Does the model generalize from validation to test set?

Specification:

- [`specs/S1-M3-personalization-fine-tuning.md`](specs/S1-M3-personalization-fine-tuning.md)

Key concepts:

- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- HuggingFace Trainer + PEFT library
- Systematic ablation: LoRA rank, decoding parameters
- Single-shot test evaluation with audit trail enforcement

Results:

| Metric   | Baseline | Fine-tuned | Improvement |
| -------- | -------- | ---------- | ----------- |
| Test WER | 238.69%  | 66.41%     | **72.2%**   |
| Test CER | -        | 50.75%     | -           |

Key findings:

- **LoRA r=16** outperformed r=8 by ~9 pts on validation
- **Decoding matters**: beam=5, temp=0.0 yields best results
- **Excellent generalization**: Val-to-test gap only 1.63 pts
- **Model is usable**: WER reduced from ~2.4x reference to ~0.66x reference

### S1-M4 — Targeted Improvements & Controlled Model Upgrades

Goal:

- Systematically reduce error rates beyond Model v1 via controlled,
  hypothesis-driven experiments — one variable at a time.

Key questions:

- Can we achieve measurable improvement through disciplined,
  single-variable experiments while maintaining evaluation integrity?
- Which hyperparameter changes (learning rate, dropout) improve
  generalization for this speaker?

Specification:

- [`specs/S1-M4-targeted-improvements.md`](specs/S1-M4-targeted-improvements.md)

Key concepts:

- Strict single-variable control (one change per experiment)
- Frozen decode configuration (`DECODE_V1.json`) — locked after M3
- Controlled Experiment Log with ADOPT / REJECT decision thresholds
- Model v1.1 assembled from the best validated changes

Results:

| Metric  | Model v1 | Model v1.1 | Improvement  |
| ------- | -------- | ---------- | ------------ |
| Val WER | 64.78%   | 64.04%     | **0.74 pts** |

Key findings:

- **Best experiment**: training_2 (dropout=0.15), val WER 64.04% — ADOPT
- **v1.1 assembled from**: inference_1 (attention mask hygiene) + training_2
- **Lower learning rate diverged**: training_1 (lr=5e-5) rejected
- **No test evaluation**: test set reserved per single-shot policy from M3

### S1-M4a — Error Analysis & Targeted Improvement Hypotheses

Goal:

- Diagnose root causes of the persistent ~64% WER ceiling using validation
  data only.

Key questions:

- Why is WER stuck at ~64%, and what is the highest-leverage intervention?
- Is the bottleneck model capacity, normalization artifacts, or data quality?

Specification:

- [`specs/S1-M4a-error-analysis.md`](specs/S1-M4a-error-analysis.md)

Key concepts:

- Error decomposition: insertions, deletions, substitutions by duration bin
- Comparative analysis: baseline → v1 → v1.1 (what each training stage fixed)
- Normalization artifact detection via contraction map
- Audio quality correlation using M0 inventory metrics
- Decision gate with quantitative criteria

Key findings:

- **Capacity bottleneck confirmed**: 205/361 val samples (56.8%) fail
  persistently in both baseline and v1.1
- **Normalization artifact measured**: contraction mismatches inflate WER by
  1.67 pts (addressable without retraining)
- **Audio quality rejected**: WER vs silence/RMS correlation r ≈ 0.1
- **Decision**: fix normalization first (S1-M4b), then upgrade to `small.en`
  (S1-M5)

### S1-M4b — Normalization Fix

Goal:

- Implement the contraction-expansion fix identified in M4a and verify the
  1.67 WER pt recovery.

Key questions:

- Does expanding contractions in the normalizer recover the measured
  1.67 WER pts?

Specification:

- [`specs/S1-M4b-normalization-fix.md`](specs/S1-M4b-normalization-fix.md)

Key concepts:

- `textnorm_v2`: adds contraction expansion to the `textnorm_v1` pipeline
- `CONTRACTION_MAP` — 36-entry deterministic rule table, no model changes
- Applied symmetrically to both reference and hypothesis before scoring
- Default normalizer for all subsequent milestones

Results:

| Model      | textnorm_v1 WER | textnorm_v2 WER | Improvement  |
| ---------- | --------------- | --------------- | ------------ |
| Model v1.1 | 64.04%          | 62.37%          | **1.67 pts** |

Key findings:

- **H1 confirmed**: contraction artifact accounts for exactly 1.67 WER pts
  as estimated in M4a
- **No regressions**: zero previously-correct samples were degraded
- **textnorm_v2 is the default for M5**: all subsequent WER uses
  `create_normalizer(version=2)`

### S1-M5 — Model Capacity Scaling

Goal:

- Test whether scaling from `base.en` (74M) to `small.en` (244M) breaks
  through the persistent failure ceiling identified in M4a.

Key questions:

- Does a 3.3× larger model resolve the capacity bottleneck?

Specification:

- [`specs/S1-M5-model-capacity-scaling.md`](specs/S1-M5-model-capacity-scaling.md)

Key concepts:

- Single `small.en` LoRA fine-tuning run (same config as v1.1)
- Comparison against v1.1 baseline scored with `textnorm_v2`
- Outcome classification: Breakthrough / Marginal / No effect / Regression
- Capacity hypothesis check — project moves to Min Viable Serving regardless

Results:

| Metric  | Model v1.1 (base.en) | Model v2 (small.en) | Improvement   |
| ------- | -------------------- | ------------------- | ------------- |
| Val WER | 62.37%               | 47.95%              | **14.42 pts** |
| Val CER | —                    | 30.29%              | —             |

> **Note:** Val WER is the deterministic result (temperature=0.0, greedy
> decoding). An earlier sampling evaluation reported 44.02%; the deterministic
> re-run gives 47.95%, which is the canonical figure.

Key findings:

- **Breakthrough outcome**: Δ = −14.42 pts (23.1% relative improvement vs
  v1.1)
- **Capacity hypothesis confirmed**: 3.3× model scale resolved the persistent
  failure pattern
- **Next step**: Min Viable Serving with Model v2 checkpoint

### S1-M6 — Minimal Viable Serving (MVS)

Goal:

- Move Model v2 from offline evaluation into a real service with streaming
  transcription, measurable SLAs, and a browser demo.

Key questions:

- Can we expose Model v2 as a real service with WebSocket streaming, VAD-triggered
  transcription, and production-minded architecture?

Specification:

- [`specs/S1-M6-minimal-viable-serving.md`](specs/S1-M6-minimal-viable-serving.md)

Key concepts:

- WebSocket streaming with VAD-triggered batch-per-utterance transcription
- In-memory SLA metrics (`/metrics`) with rolling JSONL persistence
- Health/readiness endpoint (`/health`) with startup warm-up gating
- Self-contained browser demo UI (`/demo`) with AudioWorklet capture
- Dockerfile for cloud-readiness (non-root user, health check, volume mounts)
- Provider-agnostic 12-factor design (same code local and cloud)

Results:

> Demo results (screenshot / screen recording) to be added after formal demo
> session. All endpoints verified functional during smoke testing.

### S1-M7 — Feedback Loop & Correction Fine-Tuning

Goal:

- Close the loop from transcription error observation to model improvement:
  collect corrections during live serving and feed them back through the
  fine-tuning pipeline.

Key questions:

- Can we capture user corrections via the serving UI, batch fine-tune with
  those corrections, and measure a WER improvement on the same val set?

Specification:

- [`specs/S1-M7-feedback-loop-correction-fine-tuning.md`](specs/S1-M7-feedback-loop-correction-fine-tuning.md)

Key concepts:

- `POST /feedback` endpoint stores audio + corrected text per segment
- In-memory audio retention buffer (100-segment FIFO, per WebSocket session)
- `consumed.marker` tracks which corrections have been used in each batch
- Merged manifest: original training data + corrections (prevents catastrophic forgetting)
- `--checkpoint_path` flag: continued training from v2's LoRA weights (critical for signal)
- `PeftModel.from_pretrained(..., is_trainable=True)` required for gradient flow on resumed LoRA

Results (batch 2, `batch_20260317_110057`):

| Model                           | Val WER    | Delta          | Training Samples |
| ------------------------------- | ---------- | -------------- | ---------------- |
| v2 (M5 baseline, deterministic) | 47.95%     | —              | 2,897            |
| v2 + 108 corrections            | **34.05%** | **−13.90 pts** | 3,005            |

Key findings:

- **Batch 1 produced no improvement** (fresh LoRA, 67 corrections = 2.3% of data — signal
  diluted by relearning from scratch)
- **Batch 2 breakthrough**: starting from v2's checkpoint gave corrections leverage on
  existing knowledge — 13.90 absolute pts / 29.0% relative improvement
- **Implementation detail**: `is_trainable=True` in `PeftModel.from_pretrained()` is
  required; `model.train()` alone does not re-enable LoRA gradients

### S1-M8 — Minimal Ops Layer

Goal:

- Prevent system and cost from silently drifting out of control.

Key questions:

- How much does each inference cost, and how do we keep spend bounded?
- How do we trace individual requests through the system?

Key concepts:

- Per-inference cost estimation (rough is acceptable)
- Daily / weekly usage aggregation
- Cost guardrails: reject, degrade, or queue when limits are exceeded
- Basic observability: request ID traceability

> **Status:** Not yet started.

______________________________________________________________________

## Repository Structure

```text
VOX_Personalis/
├── specs/                    # Authoritative specifications (contract-first)
│   └── README.md             # Spec index with links to all milestones
├── configs/
│   └── DECODE_V1.json        # Frozen decoding config (locked after M3)
├── scripts/
│   ├── data_inventory/       # S1-M0: Data inventory CLI
│   ├── dataset_versioning/   # S1-M1: Dataset versioning CLI
│   ├── baseline_eval/        # S1-M2: Baseline evaluation + normalization
│   ├── fine_tuning/          # S1-M3/M4/M5: LoRA fine-tuning CLI
│   ├── error_analysis/       # S1-M4a: Error analysis CLI
│   ├── serving/              # S1-M6: Minimal Viable Serving
│   ├── feedback_finetune/    # S1-M7: Feedback-loop fine-tuning CLI
│   └── code_quality_check.sh # Ruff, mypy, mdformat, shellcheck runner
├── results/                  # Per-milestone result archives
│   ├── M0_data_inventory/    # Inventory report + summary JSON
│   ├── M1_dataset_v1/        # Dataset v1 report + summary JSON
│   ├── M2_baseline_eval/     # Baseline metrics (base.en, small.en)
│   ├── M3_fine_tuning/       # Experiment log, test audit, run metrics
│   ├── M4_model_improvement/ # Controlled experiment log, v1.1 metrics
│   ├── M4a_error_analysis/   # Error distribution, hypotheses, worst samples
│   ├── M4b_normalization_fix/# Normalization experiment results
│   ├── M5_capacity_scaling/  # Val metrics, predictions, comparison report
│   └── M7_feedback_finetune/ # Merged manifest, predictions, training config
├── data/                     # Local audio + transcripts (not committed)
├── out/                      # Generated artifacts (gitignored)
├── pyproject.toml            # Project metadata, dependencies, tool config
├── Dockerfile                # Cloud-readiness artifact (S1-M6)
├── AGENTS.md                 # Claude Code agent instructions
├── DATASET-VERSIONING-STRATEGY.md
├── LICENSE                   # MIT License
└── README.md
```

______________________________________________________________________

## What This Project Is Not

- ❌ Not a speech recognition demo
- ❌ Not an end-user product
- ❌ Not ML research or novel model architecture work
- ❌ Not a benchmark leaderboard chase

This project is about **engineering judgment**, not hype.

______________________________________________________________________

## Status

- Latest milestone: **S1-M7 — Feedback Loop & Correction Fine-Tuning** (complete)
- Latest result: val WER **34.05%** (v2 + 108 corrections, continued from v2 checkpoint)
- Serving: Model v2 live via WebSocket (P50 ~1.5s CPU latency, S1-M6)
- Model v2 (S1-M5): val WER 47.95% deterministic (small.en + LoRA r=16)
- Platform: macOS (Apple Silicon, CPU inference)
- Data: single-speaker, labeled audio + transcripts (not included in repo)

______________________________________________________________________

## Development

### Setup

#### **Create and activate a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

#### **Install system dependencies:**

```bash
# macOS (required for S1-M2 baseline evaluation)
brew install ffmpeg
```

> **Note:** `ffmpeg` is required by Whisper for audio processing. On Linux, use
> `sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg` (Fedora).

#### **Install the project with development dependencies:**

```bash
pip install -e ".[dev]"
```

> **Important:** Always activate the virtual environment before running
> any scripts or tools.

### Code Quality Tools

This project uses:

- **Python:** Ruff (formatting & linting), mypy (type checking)
- **Markdown:** mdformat (formatting), pymarkdownlnt (linting)
- **Shell Scripts:** shfmt (formatting), ShellCheck (linting)

#### Quick Usage (with venv activated)

```bash
# Check code quality (no changes)
./scripts/code_quality_check.sh                     # Everything
./scripts/code_quality_check.sh scripts/            # Specific directory
./scripts/code_quality_check.sh file.py             # Specific Python file
./scripts/code_quality_check.sh README.md           # Specific Markdown file
./scripts/code_quality_check.sh script.sh           # Specific shell script

# Check and auto-fix issues
./scripts/code_quality_check.sh --fix               # Everything
./scripts/code_quality_check.sh --fix scripts/      # Specific directory
./scripts/code_quality_check.sh --fix file.py       # Specific file
```

#### Direct Tool Usage

```bash
# Python
ruff format .                    # Format Python code
ruff check --fix .               # Lint and auto-fix
mypy scripts/                    # Type check

# Markdown
mdformat .                       # Format Markdown files
mdformat --check .               # Check Markdown formatting
pymarkdown scan .                # Lint Markdown files

# Shell Scripts
shfmt -i 2 -bn -ci -w .          # Format shell scripts
shfmt -i 2 -bn -ci -d .          # Check shell script formatting
shellcheck scripts/*.sh          # Lint shell scripts
```

______________________________________________________________________

## License

MIT License.
