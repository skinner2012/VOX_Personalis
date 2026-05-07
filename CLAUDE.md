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

## gstack

Use the `/browse` skill from gstack for all web browsing. Never use `mcp__claude-in-chrome__*` tools.

Available gstack skills: `/office-hours`, `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/design-consultation`, `/design-shotgun`, `/design-html`, `/review`, `/ship`, `/land-and-deploy`, `/canary`, `/benchmark`, `/browse`, `/connect-chrome`, `/qa`, `/qa-only`, `/design-review`, `/setup-browser-cookies`, `/setup-deploy`, `/setup-gbrain`, `/retro`, `/investigate`, `/document-release`, `/codex`, `/cso`, `/autoplan`, `/plan-devex-review`, `/devex-review`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/gstack-upgrade`, `/learn`

## GBrain Configuration (configured by /setup-gbrain)

- Engine: pglite
- Config file: ~/.gbrain/config.json (mode 0600)
- Setup date: 2026-05-06
- MCP registered: yes
- Memory sync: full
- Current repo policy: read-write

## GBrain Search Guidance (configured by /sync-gbrain)
<!-- gstack-gbrain-search-guidance:start -->

GBrain is set up and synced on this machine. The agent should prefer gbrain
over Grep when the question is semantic or when you don't know the exact
identifier yet. Two indexed corpora available via the `gbrain` CLI:
- This repo's code (registered as `gstack-code-<repo>` source).
- `~/.gstack/` curated memory (registered as `gstack-brain-<user>` source via
  the existing federation pipeline).

Prefer gbrain when:
- "Where is X handled?" / semantic intent, no exact string yet:
    `gbrain search "<terms>"` or `gbrain query "<question>"`
- "Where is symbol Y defined?" / symbol-based code questions:
    `gbrain code-def <symbol>` or `gbrain code-refs <symbol>`
- "What calls Y?" / "What does Y depend on?":
    `gbrain code-callers <symbol>` / `gbrain code-callees <symbol>`
- "What did we decide last time?" / past plans, retros, learnings:
    `gbrain search "<terms>" --source gstack-brain-<user>`

Grep is still right for known exact strings, regex, multiline patterns, and
file globs. The brain auto-syncs incrementally on every gstack skill start.
Run `/sync-gbrain` to force-refresh, `/sync-gbrain --full` for full reindex.

<!-- gstack-gbrain-search-guidance:end -->

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore

## Commit Messages

Conventional Commits with optional Stage-Milestone scope:

```text
type([SX-MY/]component): subject
```

Examples: `feat(S1-M7/serving): add feedback endpoint`, `fix(api): handle null in user lookup`,
`chore(deps): add webrtcvad`. Imperative mood, under ~72 chars.
