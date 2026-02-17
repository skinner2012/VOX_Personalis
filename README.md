# VOX Personalis

VOX Personalis is an engineering-focused project exploring
**personalized speech recognition** for a single Deaf speaker.

The emphasis is **not** on training a new ASR model from scratch, but on
**data inspection, risk analysis, evaluation methodology, and
reproducible system design** for personalization workflows.

This project treats speech recognition as a **system engineering problem**,
not a demo.

______________________________________________________________________

## Motivation

General-purpose ASR systems often struggle with non-standard speech patterns,
including Deaf accents and atypical pronunciation.

As a Deaf speaker myself, I want to answer a very practical question:

> *Given a small, labeled, single-speaker dataset, can personalization be done
> in a controlled, explainable, and engineering-sound way?*

Before touching any model training, this project focuses on **understanding the data,
its risks, and its constraints**.

______________________________________________________________________

## Project Philosophy

This repository follows a **spec-first, milestone-driven** approach.

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

______________________________________________________________________

## Repository Structure

```text
VOX_Personalis/
├── specs/                    # Authoritative specifications (contract-first)
├── scripts/
│   ├── data_inventory/       # S1-M0: Data inventory CLI
│   ├── dataset_versioning/   # S1-M1: Dataset versioning CLI
│   ├── baseline_eval/        # S1-M2: Baseline evaluation CLI
│   └── fine_tuning/          # S1-M3: LoRA fine-tuning CLI
├── data/                     # (Local only; not committed)
├── out/                      # Generated artifacts (gitignored)
├── DATASET-VERSIONING-STRATEGY.md
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

- Latest milestone: **S1-M3 — Personalization & Fine-Tuning** (complete)
- Latest result: **72.2% relative WER improvement** (238.69% → 66.41%)
- Platform: macOS (Apple Silicon, CPU training)
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
