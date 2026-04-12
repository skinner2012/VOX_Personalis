# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VOX Personalis is a personalized speech recognition project for a single Deaf speaker.
It follows a **spec-first, milestone-driven** approach — specifications define behavior
before implementation, and each milestone answers a concrete engineering question.

Current state: S1-M7 complete (val WER 34.05%), S1-M8 (Minimal Ops Layer) next.

## Commands

### Code Quality (all-in-one)

```bash
./scripts/code_quality_check.sh              # Check everything
./scripts/code_quality_check.sh --fix        # Check + auto-fix
./scripts/code_quality_check.sh README.md    # Single file
./scripts/code_quality_check.sh scripts/     # Directory
```

### Direct Tools

```bash
ruff format .                    # Format Python
ruff check --fix .               # Lint Python
mypy scripts/                    # Type check (mypy_path = scripts/)
mdformat --check .               # Check Markdown formatting
pymarkdown scan .                # Lint Markdown (100-char line limit)
```

### Running CLI Modules

Every package under `scripts/` is invoked via `python -m`:

```bash
python -m scripts.data_inventory --dataset_name v1 --data_dir ./data --manifest_csv ./labels.csv
python -m scripts.baseline_eval --manifest_path ./out/dataset_v1/.../manifest.csv --model_size small.en
python -m scripts.fine_tuning --manifest_path ./out/dataset_v1/.../manifest.csv --baseline_metrics ./out/baseline_eval/.../metrics.json
python -m scripts.error_analysis --predictions ./out/fine_tuning/.../predictions.csv --manifest ./out/dataset_v1/.../manifest.csv
python -m scripts.serving --checkpoint ./out/fine_tuning/.../checkpoint --decode_config ./configs/DECODE_V1.json
python -m feedback_finetune --feedback_dir ./out/feedback --original_manifest ./out/dataset_v1/.../manifest.csv --checkpoint_path ./out/fine_tuning/.../checkpoint
```

### Setup

```bash
python3 -m venv venv && source venv/bin/activate
brew install ffmpeg              # Required by Whisper
pip install -e ".[dev]"          # Core + dev tools
pip install -e ".[dev,serving]"  # Include FastAPI/uvicorn
```

## Architecture

### Spec → Code → Results Pipeline

Each milestone follows: `specs/S1-MX-*.md` defines the contract →
`scripts/<module>/` implements it → `results/MX_*/` stores reproducible output artifacts.
Specs are authoritative — code must conform to the spec, not the other way around.

### CLI Module Pattern

All 7 script packages share the same structure:

- `__main__.py` — entry point, calls `cli.main()`
- `cli.py` — argparse setup + pipeline orchestration, returns 0/1
- Domain modules — one file per concern (e.g., `metrics.py`, `training.py`, `reporting.py`)

### Key Cross-Cutting Concepts

**Text normalization** (`scripts/baseline_eval/normalization.py`):

- `textnorm_v1`: lowercase → remove punctuation → collapse spaces
- `textnorm_v2` (default since M4b): adds contraction expansion via 36-entry `CONTRACTION_MAP`
  ("dont" → "do not", etc.) — recovered 1.67 WER pts
- Use `create_normalizer(version=2)` for all new work

**Frozen decoding** (`configs/DECODE_V1.json`): beam=5, temperature=0.0, locked at M3.
All evaluation from M3 onward uses this config. Do not modify.

**Dataset v1 is immutable**: deterministic train/val/test split. Test set is frozen.
Future dataset versions get new specs, not edits to `S1-M1-dataset-versioning.md`.

**LoRA fine-tuning** (`scripts/fine_tuning/models.py`): targets `q_proj` + `v_proj` only,
rank=16, alpha=32. When resuming from a checkpoint for continued training,
`PeftModel.from_pretrained(..., is_trainable=True)` is required — `model.train()` alone
does not re-enable LoRA gradients.

**Feedback loop** (`scripts/feedback_finetune/manifest.py`): merges original training data
with corrections to prevent catastrophic forgetting. `consumed.marker` tracks which
corrections have been used per batch.

## Commit Messages

Conventional Commits with optional Stage-Milestone scope:

```text
type([SX-MY/]component): subject
```

Examples: `feat(S1-M7/serving): add feedback endpoint`, `fix(api): handle null in user lookup`,
`chore(deps): add webrtcvad`. Imperative mood, under ~72 chars.
