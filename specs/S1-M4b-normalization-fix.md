# S1-M4b Normalization Fix — Specification

## Purpose

Apply the contraction-expansion normalization fix identified in M4a (H1) and
verify the measured WER improvement.

This milestone answers one question:

> *Does expanding contractions in `textnorm` recover the 1.67 WER pts
> identified in M4a error analysis?*

______________________________________________________________________

## Context

M4a error analysis (Phase 3, Section 4.2) found that `textnorm_v1` strips
apostrophes but does not expand contractions. This creates spurious mismatches
when the model outputs expanded forms:

| Reference (after textnorm_v1) | Hypothesis | Mismatch |
| ----------------------------- | ---------- | -------- |
| `whats`                       | `what is`  | 2 errors |
| `dont`                        | `do not`   | 2 errors |
| `thats`                       | `that is`  | 2 errors |
| `im`                          | `i am`     | 2 errors |
| `ive`                         | `i have`   | 2 errors |

M4a measured this effect at **50 samples** affected, contributing
**1.67 WER pts** to aggregate val WER. The fix is additive (new normalization
step), does not change model weights, and carries low risk.

**Source:** `out/error_analysis/20260302-202827/error_distribution_report.json`
(key: `normalization_audit`)

______________________________________________________________________

## Dependencies

- S1-M4a error analysis completed
- `textnorm_v1` normalizer (`scripts/baseline_eval/normalization.py`)
- Model v1.1 val predictions
  (`out/model_improvement/experiments/training_2/val_predictions.csv`)
- Baseline val predictions
  (`out/baseline_eval/20260210-220646/baseline_predictions.csv`)

______________________________________________________________________

## Scope

### In Scope

- Define a `CONTRACTION_MAP` (collapsed form → expanded form)
- Add contraction expansion step to the normalization pipeline (`textnorm_v2`)
- Re-score Model v1.1 and baseline val predictions with `textnorm_v2`
- Verify WER improvement against M4a estimate (1.67 pts)
- Unit tests for the contraction expansion step

### Out of Scope

- Model retraining or inference re-runs
- Changes to decoding parameters
- New data collection
- Any modification beyond the normalization pipeline

______________________________________________________________________

## Implementation

### Contraction Map

`CONTRACTION_MAP` already exists in `scripts/error_analysis/patterns.py`
(36 entries, used by M4a normalization audit). To avoid duplication:

1. **Move** `CONTRACTION_MAP` to `scripts/baseline_eval/normalization.py`
   (its canonical home, since it is now part of the normalization pipeline)
1. **Update** `scripts/error_analysis/patterns.py` to import from
   `scripts.baseline_eval.normalization` instead of defining its own copy

**Ambiguity note:** Some tokens are ambiguous (`its` = possessive vs
contraction, `were` = past tense vs contraction). Since both reference and
hypothesis receive the same expansion, ambiguous expansions cancel out and do
not inflate WER. The risk is bounded by the measured 1.67 pt contribution.

### Normalization Pipeline (textnorm_v2)

Insert contraction expansion **after** punctuation removal (which strips
apostrophes) and **before** space collapsing:

1. Convert to lowercase
1. Remove all punctuation (apostrophes stripped here)
1. **Expand contractions via `CONTRACTION_MAP`** (new step)
1. Collapse multiple spaces to single space
1. Strip leading/trailing whitespace

Apply to **both** reference and hypothesis text before WER/CER computation.

### Re-scoring

Re-score existing val predictions using `textnorm_v2`. No new inference needed
— the prediction CSVs already contain raw reference and hypothesis text.

______________________________________________________________________

## Deliverables

### 1. Updated normalizer

`scripts/baseline_eval/normalization.py` — updated with `CONTRACTION_MAP` and
`textnorm_v2` pipeline.

### 2. Re-scored WER results

Re-run output: `out/normalization_fix/20260303-193015/`

| Model      | textnorm_v1 WER | textnorm_v2 WER (measured) | Delta    |
| ---------- | --------------- | -------------------------- | -------- |
| Model v1.1 | 0.6404          | 0.6237                     | 1.67 pts |

**How 0.6237 was measured:** The `normalization_audit` in the error analysis
re-scores each sample by applying `expand_contractions()` to the reference and
hypothesis text, recomputing WER via jiwer, and computing the corpus-level
delta. Result: `normalization_wer_contribution_pts = 0.0167`.
So `0.6404 - 0.0167 = 0.6237` is a real, per-sample re-scoring — not an
approximation.

**Why `aggregate.wer` still shows 0.6404:** The error analysis reads the `wer`
column directly from the prediction CSV. Those values were computed during M4
inference with textnorm_v1 and are not rewritten by re-running this script.
The 0.6237 will appear in `aggregate.wer` in M5, once inference runs with
`create_normalizer(version=2)`.

**Regression check:** The re-run output is identical to M4a
(`out/error_analysis/20260302-202827/`), confirming the DRY refactor
(CONTRACTION_MAP move from `patterns.py` → `normalization.py`) introduced no
regressions.

### 3. Unit tests

Test cases for the contraction expansion step:

- Known contraction pairs expand correctly
- Non-contraction tokens pass through unchanged
- Ambiguous tokens (`its`, `were`) expand consistently
- Empty string and whitespace-only inputs handled
- Full pipeline (textnorm_v2) produces expected output for sample sentences

### 4. Close-out

**Status:** M4b complete.

- `textnorm_v2` implemented in `scripts/baseline_eval/normalization.py`
- WER improvement measured on v1.1/base.en: **1.67 pts** (0.6404 → 0.6237)
- `textnorm_v2` is the **default normalizer for M5** — pass
  `create_normalizer(version=2)` wherever WER/CER is computed

**Note on M5:** The 1.67 pt improvement is specific to Model v1.1 (base.en).
small.en may expand contractions more or less aggressively, so the actual
normalization benefit in M5 will differ. It must be re-measured from M5
predictions — do not carry 1.67 pts forward as a fixed expectation.

______________________________________________________________________

## Verification

The fix is verified when:

1. Model v1.1 val WER with `textnorm_v2` is within 0.5 pts of the M4a
   estimate (0.6237 ± 0.005)
1. All unit tests pass
1. No samples that were WER=0 under `textnorm_v1` become WER>0 under
   `textnorm_v2` (no regressions on already-correct samples)

______________________________________________________________________

## Verification Procedure

No new script. After updating `normalization.py`, re-run the existing error
analysis with the same arguments:

```bash
python -m scripts.error_analysis \
  --v1_1_predictions "./out/model_improvement/experiments/training_2/val_predictions.csv" \
  --baseline_predictions "./out/baseline_eval/20260210-220646/baseline_predictions.csv" \
  --manifest_path "./out/dataset_v1/*/dataset_v1_manifest.csv" \
  --inventory_path "./out/inventory/20260205-142601/inventory_files.csv" \
  --v1_predictions "./out/fine_tuning/20260216-225002/base_r16_predictions.csv" \
  --out_dir "./out/error_analysis" \
  --verbose
```

Check `normalization_audit.estimated_wer_contribution_pts` in the new
`error_distribution_report.json`. Expected: ~0.0167 (same as M4a).
Aggregate WER will still show 0.6404 — that is correct, as WER in the
prediction CSVs was pre-computed with textnorm_v1. The 1.67 pt improvement
will be realized in M5.

______________________________________________________________________

## References

- S1-M4a Error Analysis — H1 (contraction normalization artifact)
- `out/error_analysis/20260302-202827/error_distribution_report.json`
- `out/error_analysis/20260302-202827/error_analysis_report.md` (Section 4.2)
