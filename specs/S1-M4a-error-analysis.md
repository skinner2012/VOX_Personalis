# S1-M4a Error Analysis & Targeted Improvement Hypotheses — Specification

## Purpose

Diagnose the root causes of the current ~64% WER ceiling using validation data
only.

This milestone is **diagnostic, not performance-driven.** The goal is not to
further tune hyperparameters, but to:

- Understand dominant error categories and their distribution
- Identify systematic recognition failures vs random noise
- Quantify how much of the WER is model error vs normalization artifact
- Formulate evidence-based improvement hypotheses
- Make a clear decision on the next intervention: model capacity upgrade,
  data/normalization cleanup, or serving transition

This milestone answers one question:

> *Why is WER stuck at 64%, and what is the highest-leverage intervention to
> reduce it?*

______________________________________________________________________

## Dependencies

### Required (analysis will not run without these)

- S1-M4 Targeted Improvements completed

- Model v1.1 val predictions (`out/model_improvement/experiments/training_2/val_predictions.csv`)

  **Why training_2?** M4 assembled Model v1.1 from `inference_1 + training_2`
  (see `model_v1.1_metrics.json` → `experiments_included`). training_2
  (dropout=0.15) produced the best single-variable val WER (0.6404). M4 did
  not generate a top-level `model_v1.1_val_predictions.csv`, so this
  experiment file is the canonical source.

- Baseline val predictions (`out/baseline_eval/20260210-220646/baseline_predictions.csv`)

- Dataset v1 manifest (`out/dataset_v1/*/dataset_v1_manifest.csv`)

- M0 inventory data (`out/inventory/20260205-142601/inventory_files.csv`)
  — for audio quality correlation (Phase 4)

### Optional (used if available, gracefully skipped if not)

- Model v1 val predictions (`out/fine_tuning/20260216-225002/base_r16_predictions.csv`)
  — enables three-way comparison (baseline → v1 → v1.1)
- Baseline error patterns (`out/baseline_eval/20260210-220646/baseline_errors.csv`)
  — if not provided, recomputed from `baseline_predictions.csv`
- Frozen decode config (`configs/DECODE_V1.json`) — for reference documentation
- `textnorm_v1` normalization pipeline (`scripts/baseline_eval/normalization.py`)

______________________________________________________________________

## Scope

### In Scope

- Validation set analysis only
- Error decomposition and pattern mining
- Comparative analysis: baseline vs v1 vs v1.1 (what each stage fixed vs what persists)
- Audio quality correlation using M0 inventory metrics
- Text normalization audit
- Error concentration analysis
- Up to 2 targeted follow-up experiments (inference-only or normalization-only)
- Decision gate with quantitative criteria

### Out of Scope

- Test set usage — single-shot rule established in S1-M3 applies; val-only
- New training runs (follow-up experiments are inference/normalization only)
- Model architecture changes
- Data augmentation implementation
- Dataset modifications
- Hyperparameter tuning
- Production deployment considerations

______________________________________________________________________

## Analysis Framework

### Phase 1: Error Decomposition (Deliverable 1)

**Why:** Before diagnosing root causes, we need to establish what the error
landscape looks like at v1.1. Are errors concentrated in a handful of bad
samples, or spread uniformly? Is the model hallucinating or primarily making
substitutions? This phase answers: *how bad is it, and where is it bad?*
Without this foundation, any hypothesis about root causes is speculation.

#### 1.1 Global Metrics

Compute from Model v1.1 val predictions:

- Aggregate WER and CER
- Total word count (reference), total error count
- Insertions / deletions / substitutions — absolute counts and proportions

#### 1.2 Error Concentration Analysis

Determine whether errors are uniformly distributed or concentrated.

- Sort val samples by per-sample WER descending
- Compute **cumulative error contribution**: what % of total word errors come
  from the top N% of samples?
- Report key concentration metrics:
  - `top_10pct_error_share`: % of total errors from worst 36 samples
  - `top_20pct_error_share`: % of total errors from worst 72 samples
  - `zero_wer_sample_count`: samples with perfect recognition
  - `catastrophic_sample_count`: samples with WER > 1.0 (more errors than words)

**Interpretation guide:**

- If top 10% contribute >40% of errors → errors are concentrated → targeted
  intervention may help
- If top 10% contribute \<25% of errors → errors are uniform → model capacity
  or systematic issue

#### 1.3 Error Density by Duration Bin

Using duration bins from S1-M1 (consistent across all milestones):

| Bin    | Duration Range |
| ------ | -------------- |
| Short  | (1, 3\] sec    |
| Medium | (3, 10\] sec   |
| Long   | (10, 30\] sec  |

For each bin, report:

- Sample count, total words, total errors
- WER, CER
- Insertion / deletion / substitution proportions
- Mean and median per-sample WER

#### 1.4 Hallucination Analysis

**Definition:** A sample is "hallucination-heavy" if:

- `hypothesis_word_count > 1.5 × reference_word_count` (insertion-dominated), OR
- `word_insertions > reference_word_count` (more inserted words than reference)

Report:

- Count and percentage of hallucination-heavy samples
- Common inserted phrases or repeated tokens
- Correlation with duration bin

#### 1.5 Top-50 Worst Samples

Report the 50 highest-WER samples:

| Column               | Type  | Description                         |
| -------------------- | ----- | ----------------------------------- |
| `file_name`          | str   | Audio file name                     |
| `pair_sha256`        | str   | Sample identity hash                |
| `duration_sec`       | float | Audio duration                      |
| `duration_bin`       | str   | Duration bin label                  |
| `reference`          | str   | Ground truth (normalized)           |
| `hypothesis`         | str   | Model v1.1 prediction (normalized)  |
| `wer`                | float | Word error rate                     |
| `word_insertions`    | int   | Insertion count                     |
| `word_deletions`     | int   | Deletion count                      |
| `word_substitutions` | int   | Substitution count                  |
| `dominant_error`     | str   | insertion / deletion / substitution |
| `hallucination`      | bool  | Meets hallucination criteria        |
| `baseline_wer`       | float | Baseline (pre-fine-tuning) WER      |
| `wer_improvement`    | float | baseline_wer - v1.1_wer             |

______________________________________________________________________

### Phase 2: Comparative Analysis — What Each Stage Fixed (Deliverable 2a)

**Why:** We have three model snapshots (baseline, v1, v1.1) but no per-sample
view of what each training stage actually fixed. Without this, we cannot
distinguish "what M3 solved" from "what M4 solved" from "what neither solved."
Samples that remain high-WER across all three stages signal a model capacity
problem; errors that appeared only after fine-tuning signal regression. This
phase turns aggregate WER numbers into an actionable error history.

Join val predictions from all available models by `pair_sha256`.
Three-way comparison (baseline → v1 → v1.1) if v1 predictions are available;
otherwise baseline → v1.1 only.

#### 2.1 Improvement Distribution

For each model transition (baseline→v1, v1→v1.1):

- Samples where WER improved: count, mean improvement
- Samples where WER regressed: count, mean regression
- Samples unchanged: count

#### 2.2 Error Type Migration

Track how error types shifted across stages:

- Baseline was insertion-dominated (2083 insertions on val). How many insertions
  remain in v1? In v1.1?
- Did fine-tuning convert insertions to substitutions, or eliminate them?
- Did M4 improvements change the error profile, or just reduce magnitude?
- Are the remaining errors the same tokens the baseline confused, or new ones?

#### 2.3 Persistent Errors

Identify samples with WER > 0.5 in **both** baseline and v1.1:

- These represent fundamental recognition failures unaffected by fine-tuning
- Classify by suspected cause: acoustic difficulty, unusual vocabulary, or
  normalization artifact

______________________________________________________________________

### Phase 3: Systematic Pattern Analysis (Deliverable 2b)

**Why:** Random errors suggest a model capacity ceiling and call for a larger
model. Structured, repeatable patterns suggest fixable root causes that do not
require more training. The normalization artifact hypothesis is the
highest-priority test here: if contraction mismatches (e.g., "what's" vs
"what is") account for several WER points, the fix is a rule change to the
text pipeline — not retraining. Identifying patterns before committing to
`small.en` can save significant compute.

#### 3.1 Substitution Pattern Mining

From Model v1.1 val predictions, extract all substitution pairs
(reference_token → hypothesis_token).

Report top 30 substitution pairs by frequency, with columns:

| Column             | Type | Description                                  |
| ------------------ | ---- | -------------------------------------------- |
| `reference_token`  | str  | Expected word                                |
| `hypothesis_token` | str  | Predicted word                               |
| `count`            | int  | Frequency across val set                     |
| `also_in_baseline` | bool | Same pair appeared in baseline top-50 errors |
| `example_files`    | str  | Up to 5 example file names                   |

#### 3.2 Normalization Artifact Detection

**Critical analysis.** Preliminary data shows cases like:

- Reference: `"what's the news"` → normalized: `"whats the news"`
- Hypothesis: `"what is the news"` → normalized: `"what is the news"`
- Result: 66% WER on semantically correct recognition

Detect normalization-inflated errors using a **deterministic, rule-based
contraction map**. No semantic or NLP-based matching — only explicit pairs.

**Contraction Map (hardcoded, ~30–50 pairs):**

```text
what's    ↔ what is
don't     ↔ do not
it's      ↔ it is
i'm       ↔ i am
can't     ↔ cannot / can not
won't     ↔ will not
they're   ↔ they are
we're     ↔ we are
you're    ↔ you are
isn't     ↔ is not
aren't    ↔ are not
wasn't    ↔ was not
weren't   ↔ were not
doesn't   ↔ does not
didn't    ↔ did not
hasn't    ↔ has not
haven't   ↔ have not
hadn't    ↔ had not
couldn't  ↔ could not
wouldn't  ↔ would not
shouldn't ↔ should not
let's     ↔ let us
that's    ↔ that is
there's   ↔ there is
here's    ↔ here is
he's      ↔ he is
she's     ↔ she is
who's     ↔ who is
how's     ↔ how is
where's   ↔ where is
when's    ↔ when is
... (extend as needed based on data)
```

**Method:**

1. For each val sample, apply the contraction map to normalize both
   `reference` and `hypothesis` to a canonical form (expand all contractions)
1. Re-score WER with contraction-normalized texts
1. Compare against original WER to measure normalization contribution

**Report:**

- `normalization_artifact_count`: samples where contraction-normalized WER
  differs from original WER
- `normalization_wer_contribution_pts`: aggregate WER difference
  (original - contraction-normalized), in absolute WER points
- `contraction_expansion_pairs`: list of detected pairs with frequency counts

#### 3.3 Short Utterance Instability

For samples with ≤ 3 reference words:

- WER is disproportionately affected by single errors (1 error = 33%+ WER)
- Report: count, mean WER, median WER, proportion of total errors
- Compare to longer utterances to assess whether short utterance scoring
  inflates aggregate WER

#### 3.4 Long Utterance Analysis

For samples with ≥ 10 reference words:

- Do long utterances have systematically higher WER than medium ones?
- Report: count, mean WER, median WER, comparison to medium-length samples
- Observational only — no token-position alignment required

#### 3.5 Domain-Specific Token Failures

Identify tokens/phrases with consistently high error rates:

- Group errors by reference token
- Report tokens where error rate > 50% (failed more often than recognized)
- Classify as: function words, command verbs, proper nouns, numbers, other

______________________________________________________________________

### Phase 4: Audio Quality Correlation (Deliverable 2c)

**Why:** A strong correlation between audio quality (high silence ratio, low
RMS) and WER would redirect the intervention from model capacity to data
quality. If the worst-WER samples are also the noisiest recordings, the right
next step is not a larger model but targeted re-recording or audio
preprocessing. We have M0 inventory metrics for every file, so this check is
low-cost and can rule out (or confirm) audio quality as a primary driver before
committing to a training run.

Join val predictions with M0 inventory metrics by `file_name`. If the join
produces \<80% match rate, report a warning and proceed with matched rows only.

#### 4.1 Overlap Statistic

- Of the top-50 worst WER samples, how many have poor audio quality?
  - Poor audio defined as: `silence_ratio_est > 0.4` OR `rms_db_est < -40`
- Report: `poor_audio_in_top50_worst` count and percentage

#### 4.2 Correlation Analysis

- Pearson/Spearman correlation coefficients (WER vs `silence_ratio_est`,
  WER vs `rms_db_est`)
- Use `scipy.stats` if available; fall back to `numpy` correlation with a note

______________________________________________________________________

### Phase 5: Targeted Hypotheses (Deliverable 3)

Based on Phases 1–4, formulate **maximum 3 hypotheses**.

Each hypothesis MUST include:

| Field                   | Description                                        |
| ----------------------- | -------------------------------------------------- |
| `hypothesis_id`         | H1, H2, H3                                         |
| `observed_pattern`      | Specific error pattern from analysis               |
| `evidence`              | Quantitative evidence (counts, percentages)        |
| `suspected_root_cause`  | Why this pattern exists                            |
| `proposed_intervention` | Minimal, specific change to test                   |
| `experiment_type`       | inference-only / normalization-only                |
| `expected_impact`       | Estimated WER point range                          |
| `risk_assessment`       | What could go wrong, regression risk               |
| `testable_in_m4a`       | Yes/No — can this be validated without retraining? |

**Example hypotheses (illustrative only — actual hypotheses must come from data):**

Hypothesis 1: Contraction expansion inflates WER

- Pattern: "what's" → "what is" scored as substitution error
- Evidence: N samples affected, estimated M WER points
- Root cause: `textnorm_v1` removes apostrophes but doesn't normalize
  contractions
- Intervention: Add contraction normalization to text pipeline
- Expected impact: 1–5 pts
- Risk: Low (additive normalization, doesn't change model)
- Testable in M4a: Yes (re-score existing predictions with updated normalizer)

Hypothesis 2: Model capacity ceiling

- Pattern: Function word confusion persists from baseline (the↔do, the↔you)
- Evidence: N of top-30 substitution pairs also in baseline errors
- Root cause: `base.en` (74M params) lacks capacity for this speaker's
  acoustic patterns
- Intervention: Fine-tune `small.en` (244M) with same LoRA config
- Expected impact: 10–20 pts
- Risk: Medium (longer training, higher memory, may overfit with same data)
- Testable in M4a: No (requires S1-M5 training)

______________________________________________________________________

### Phase 6: Targeted Follow-Up Experiments (Deliverable 4)

**Maximum 2 experiments.** These are **inference-only or normalization-only** —
no new training runs.

#### Experiment Protocol

1. Select hypotheses marked `testable_in_m4a = Yes`
1. Implement the minimal intervention
1. Re-score Model v1.1 val predictions (or re-run inference with changed
   normalization)
1. Compare against Model v1.1 val WER
1. Record results in experiment log

#### Experiment Log Schema (`m4a_experiment_log.csv`)

| Column                 | Type  | Description                                |
| ---------------------- | ----- | ------------------------------------------ |
| `experiment_id`        | str   | m4a_exp_1, m4a_exp_2                       |
| `hypothesis_id`        | str   | H1, H2, etc.                               |
| `timestamp`            | str   | ISO 8601                                   |
| `intervention`         | str   | What was changed                           |
| `experiment_type`      | str   | inference-only / normalization-only        |
| `model_v1_1_val_wer`   | float | Model v1.1 val WER (reference)             |
| `experiment_val_wer`   | float | Val WER after intervention                 |
| `val_wer_delta`        | float | Absolute improvement (positive = better)   |
| `hypothesis_supported` | bool  | True if results support hypothesis         |
| `decision`             | str   | VALIDATED / PARTIALLY_VALIDATED / REJECTED |
| `notes`                | str   | Observations                               |

______________________________________________________________________

## Decision Gate

After completing analysis and follow-up experiments, make ONE decision from
the following options. The decision MUST be recorded in the final report with
supporting evidence.

### Decision Criteria (Default Heuristics)

The following thresholds are **default heuristics**, not hard rules.
If the data suggests a different threshold is more appropriate, the report
MUST document the adjustment and its justification.

| Finding                                                         | Default Decision                |
| --------------------------------------------------------------- | ------------------------------- |
| >50% of top-50 error samples share persistent baseline patterns | **Upgrade model** → small.en M5 |
| Normalization artifacts account for >3 WER points               | **Fix normalization** first     |
| >30% of high-WER samples have poor audio quality (M0 metrics)   | **Data quality** intervention   |
| Errors are uniformly distributed, no dominant pattern           | **Upgrade model** → small.en M5 |
| Normalization fix alone drops WER below 55%                     | **Consider serving** + feedback |

### Decision Options

1. **Upgrade model capacity** → Plan S1-M5: Fine-tune `small.en` with LoRA

   - Triggered by: capacity-related findings (persistent function word confusion,
     uniform error distribution, baseline patterns surviving fine-tuning)
   - Includes: decoding parameter search (unfreezing DECODE_V1)

1. **Apply data/normalization cleanup** → Implement normalization fix, then
   re-evaluate whether model upgrade is still needed

   - Triggered by: normalization artifacts contributing >3 WER points
   - Quick win before committing to longer training runs

1. **Proceed to serving with feedback loop** → Accept current WER, build
   correction pipeline

   - Triggered by: errors are fundamentally acoustic and no clear model
     intervention will help without new data
   - Includes: user correction interface for continuous improvement

1. **Collect targeted data** → Record 50–100 specific utterances

   - Triggered by: >30% of errors cluster around specific phonemes or word
     types underrepresented in training data
   - Includes: phoneme gap analysis and recording protocol

**Note:** Options are not mutually exclusive. The decision may be "fix
normalization first (quick), then upgrade model (M5)."

______________________________________________________________________

## Targeted Data Plan (Conditional)

This section is activated ONLY if analysis reveals systematic acoustic gaps.

If >30% of high-WER val samples share common phoneme patterns or word types
not well-represented in training data:

1. **List specific gaps**: phoneme clusters, word types, or utterance patterns
1. **Propose recording list**: 50–100 targeted utterances
1. **Estimate impact**: expected WER improvement range
1. **Assess effort cost**: recording time, re-training time
1. **Version implications**: would require Dataset v2 (per DATASET-VERSIONING-STRATEGY.md)

______________________________________________________________________

## Output Artifacts

All outputs written to `--out_dir` (default: `./out/error_analysis/YYYYMMDD-HHMMSS/`).

### 1. `error_distribution_report.json`

Machine-readable error decomposition.

```json
{
  "model_version": "v1.1",
  "val_sample_count": 361,
  "aggregate": {
    "wer": null,
    "cer": null,
    "total_reference_words": null,
    "total_errors": null,
    "insertions": null,
    "deletions": null,
    "substitutions": null
  },
  "error_concentration": {
    "top_10pct_error_share": null,
    "top_20pct_error_share": null,
    "zero_wer_sample_count": null,
    "catastrophic_sample_count": null
  },
  "by_duration_bin": {},
  "hallucination": {
    "count": null,
    "percentage": null,
    "common_inserted_tokens": []
  },
  "normalization_audit": {
    "artifact_count": null,
    "estimated_wer_contribution_pts": null,
    "contraction_pairs": []
  },
  "audio_quality_overlap": {
    "join_match_rate": null,
    "poor_audio_in_top50_worst": null,
    "poor_audio_in_top50_worst_pct": null,
    "wer_vs_silence_ratio_corr": null,
    "wer_vs_rms_db_corr": null
  }
}
```

### 2. `worst_samples.csv`

Top 50 worst samples (schema in Phase 1.5 above).

### 3. `substitution_patterns.csv`

Top 30 substitution pairs (schema in Phase 3.1 above).

### 4. `comparative_analysis.csv`

Per-sample comparison across available models:

| Column                    | Type  | Description                               |
| ------------------------- | ----- | ----------------------------------------- |
| `file_name`               | str   | Audio file name                           |
| `pair_sha256`             | str   | Sample identity hash                      |
| `duration_sec`            | float | Audio duration                            |
| `reference`               | str   | Ground truth (normalized)                 |
| `baseline_wer`            | float | Baseline (zero-shot) WER                  |
| `v1_wer`                  | float | Model v1 WER (null if v1 preds unavail)   |
| `v1_1_wer`                | float | Model v1.1 WER                            |
| `improvement_total`       | float | baseline_wer - v1_1_wer                   |
| `improvement_m3`          | float | baseline_wer - v1_wer (null if unavail)   |
| `improvement_m4`          | float | v1_wer - v1_1_wer (null if unavail)       |
| `baseline_dominant_error` | str   | insertion/deletion/substitution           |
| `v1_dominant_error`       | str   | insertion/deletion/substitution (or null) |
| `v1_1_dominant_error`     | str   | insertion/deletion/substitution           |
| `persistent_failure`      | bool  | WER > 0.5 in both baseline and v1.1       |

### 5. `hypotheses.json`

Structured hypotheses (schema in Phase 5 above).

### 6. `m4a_experiment_log.csv`

Follow-up experiment results (schema in Phase 6 above).

### 7. `error_analysis_report.md`

Human-readable report with all findings. Structure:

1. Executive Summary
1. Error Distribution (global, by duration, concentration)
1. Comparative Analysis (baseline → v1 → v1.1 — what each stage fixed)
1. Systematic Patterns (substitutions, normalization artifacts, short/long
   utterance effects)
1. Audio Quality Correlation (overlap statistic + WER correlation coefficients)
1. Hypotheses (each with full evidence)
1. Follow-Up Experiment Results
1. Decision Gate Outcome
1. Recommendations

### 8. `decision.json`

Final decision record:

```json
{
  "decision": "upgrade_model | fix_normalization | serve_with_feedback | collect_data",
  "primary_evidence": "summary of key finding",
  "supporting_metrics": {},
  "next_milestone": "S1-M5 description",
  "normalization_fix_first": true,
  "timestamp": "ISO 8601"
}
```

______________________________________________________________________

## CLI Interface

```bash
python -m scripts.error_analysis \
  --v1_1_predictions "./out/model_improvement/experiments/training_2/val_predictions.csv" \  # Model v1.1 (see Dependencies)
  --baseline_predictions "./out/baseline_eval/20260210-220646/baseline_predictions.csv" \
  --manifest_path "./out/dataset_v1/*/dataset_v1_manifest.csv" \
  --inventory_path "./out/inventory/20260205-142601/inventory_files.csv" \
  --v1_predictions "./out/fine_tuning/20260216-225002/base_r16_predictions.csv" \
  --out_dir "./out/error_analysis" \
  --verbose
```

### Required Arguments

| Argument                 | Description                            |
| ------------------------ | -------------------------------------- |
| `--v1_1_predictions`     | Path to Model v1.1 val predictions CSV |
| `--baseline_predictions` | Path to baseline val predictions CSV   |
| `--manifest_path`        | Path to Dataset v1 manifest CSV        |
| `--inventory_path`       | Path to M0 inventory CSV               |

### Optional Arguments

| Argument           | Default                | Description                          |
| ------------------ | ---------------------- | ------------------------------------ |
| `--v1_predictions` | None                   | Path to Model v1 val predictions CSV |
| `--out_dir`        | `./out/error_analysis` | Output directory                     |
| `--top_n_worst`    | 50                     | Number of worst samples              |
| `--top_n_subs`     | 30                     | Number of substitution pairs         |
| `-v, --verbose`    | False                  | Detailed logging                     |

### Exit Codes

| Code | Meaning                                       |
| ---- | --------------------------------------------- |
| 0    | Success                                       |
| 1    | Fatal error (input files not found)           |
| 2    | Validation error (prediction format mismatch) |

______________________________________________________________________

## Success Criteria

S1-M4a is complete when:

### Required

1. Error distribution report generated with all specified metrics
1. Comparative analysis (baseline vs v1.1, and vs v1 if available) completed
1. At least 2 systematic error patterns identified with quantitative evidence
1. Normalization audit completed with WER contribution estimate
1. At least 2 testable hypotheses proposed; up to 2 follow-up experiments
   attempted and outcomes documented (VALIDATED / PARTIALLY_VALIDATED / REJECTED)
1. Decision gate outcome recorded in `decision.json`
1. `error_analysis_report.md` generated with all sections

### Target

- At least 1 follow-up experiment shows ≥1 WER point improvement
- Decision gate leads to a concrete next milestone definition

**Note:** The success bar is "attempted and documented," not "validated."
A well-documented REJECTED hypothesis is a valid M4a outcome — it eliminates
a hypothesis and narrows the decision space.

______________________________________________________________________

## Error Handling

| Scenario                        | Behavior                   | Exit |
| ------------------------------- | -------------------------- | ---- |
| Prediction file not found       | Fatal error                | 1    |
| Manifest not found              | Fatal error                | 1    |
| Prediction schema mismatch      | Fatal error, show expected | 2    |
| Inventory file not found        | Fatal error                | 1    |
| Join key mismatch (pair_sha256) | Warning, report unmatched  | 0    |
| No patterns found               | Report "uniform errors"    | 0    |

______________________________________________________________________

## Dependencies / Environment

### Packages (all inherited — no new dependencies)

| Package  | Version | Purpose                 |
| -------- | ------- | ----------------------- |
| `pandas` | >=2.0   | Data manipulation       |
| `jiwer`  | >=3.0   | WER/CER computation     |
| `numpy`  | any     | Statistical computation |

### Optional

| Package | Version | Purpose                |
| ------- | ------- | ---------------------- |
| `scipy` | any     | Correlation statistics |

If `scipy` is not available, fall back to `numpy` for correlation computation
or skip correlation analysis with a note in the report.

### Platform

- Python 3.11+
- macOS (Apple Silicon)
- No GPU required (analysis only)

______________________________________________________________________

## Implementation Notes

### Module Structure

```text
scripts/error_analysis/
├── __init__.py
├── __main__.py          # Entry: python -m scripts.error_analysis
├── cli.py               # CLI parsing + pipeline orchestration
├── decomposition.py     # Error decomposition and concentration
├── comparative.py       # v1.1 vs baseline comparison
├── patterns.py          # Substitution mining, normalization audit
├── audio_correlation.py # Audio quality vs WER correlation
├── hypotheses.py        # Hypothesis generation framework
└── reporting.py         # Generate all output files
```

### Reuse

- Text normalization: import `textnorm_v1` from `scripts.baseline_eval.normalization`
- WER computation: use `jiwer` with same settings as M2/M3/M4
- Duration bins: use same bin definitions as S1-M1

______________________________________________________________________

## Failure Modes if Skipped

- Model upgrade decision is uninformed (guessing rather than evidence-based)
- Normalization artifacts may persist undetected, inflating WER by unknown amount
- Training compute wasted if root cause is normalization, not model capacity
- No understanding of error distribution → no ability to set realistic targets

______________________________________________________________________

## References

- S1-M2 Baseline Evaluation (baseline error patterns)
- S1-M3 Personalization & Fine-Tuning (Model v1 training)
- S1-M4 Targeted Improvements (Model v1.1, controlled experiments)
- M0 Data Inventory (audio quality metrics)
