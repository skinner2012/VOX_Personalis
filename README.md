# VOX Personalis

VOX Personalis is an engineering-focused project exploring **personalized speech recognition**
for a single Deaf speaker.

The emphasis is **not** on training a new ASR model from scratch, but on
**data inspection, risk analysis, evaluation methodology, and reproducible system design**
for personalization workflows.

This project treats speech recognition as a **system engineering problem**, not a demo.

---

## Motivation

General-purpose ASR systems often struggle with non-standard speech patterns,
including Deaf accents and atypical pronunciation.

As a Deaf speaker myself, I want to answer a very practical question:

> *Given a small, labeled, single-speaker dataset, can personalization be done
> in a controlled, explainable, and engineering-sound way?*

Before touching any model training, this project focuses on **understanding the data,
its risks, and its constraints**.

---

## Project Philosophy

This repository follows a **spec-first, milestone-driven** approach.

- Specifications define behavior and scope **before** implementation.
- Each milestone answers a concrete engineering question.
- All outputs are reproducible and intentionally scoped.
- Success is measured by **clarity and correctness**, not just metric improvement.

---

## Milestones

### S1-M0 — Data Inventory & Risk Scan (current)

Goal:
- Understand what data exists and whether it is suitable for personalization.

Key questions:
- What audio and transcript data do we actually have?
- Are there obvious metadata inconsistencies or failures?
- Are there systematic silence or noise issues?
- Does the dataset warrant further investment?

Specification:
- [`specs/S1-M0-data-inventory.md`](specs/S1-M0-data-inventory.md)

Deliverables (local, reproducible):
- Per-file metadata inventory (CSV)
- Aggregate dataset summary (JSON)
- Human-readable inventory report (Markdown)

> No audio is modified and no model is trained at this stage.

---

## Repository Structure

```
VOX_Personalis/
├── specs/          # Authoritative specifications (contract-first)
├── src/            # Implementation code
├── scripts/        # CLI / helper scripts
├── data/           # (Local only; not committed)
├── out/            # Generated artifacts (gitignored)
└── README.md
```
---

## What This Project Is Not

- ❌ Not a speech recognition demo
- ❌ Not an end-user product
- ❌ Not ML research or novel model architecture work
- ❌ Not a benchmark leaderboard chase

This project is about **engineering judgment**, not hype.

---

## Status

- Current phase: **S1-M0 — Data Inventory**
- Platform: macOS (CPU-only, local execution)
- Data: single-speaker, labeled audio + transcripts (not included in repo)

---

## Development

### Setup

1. **Create and activate a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

2. **Install the project with development dependencies:**

```bash
pip install -e ".[dev]"
```

> **Important:** Always activate the virtual environment before running any scripts or tools.

### Code Quality Tools

This project uses **Ruff** for formatting and linting, and **mypy** for type checking.

#### Quick Usage (with venv activated)

```bash
# Check code quality (no changes)
./scripts/code_qaulity_check.sh                     # Everything
./scripts/code_qaulity_check.sh scripts/            # Specific directory
./scripts/code_qaulity_check.sh file.py             # Specific file

# Check and auto-fix issues
./scripts/code_qaulity_check.sh --fix               # Everything
./scripts/code_qaulity_check.sh --fix scripts/      # Specific directory
./scripts/code_qaulity_check.sh --fix file.py       # Specific file
```

#### Direct Tool Usage

```bash
# Format code
ruff format .

# Lint and auto-fix
ruff check --fix .

# Type check
mypy scripts/
```

---

## License

MIT License.