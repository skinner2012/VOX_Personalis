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

## License

MIT License.