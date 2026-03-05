# S1-M4a Error Analysis Report

## 1. Executive Summary

**Model v1.1 val WER:** 0.6404 (64.0%)  
**Model v1.1 val CER:** 0.4661 (46.6%)  
**Val samples:** 361  
**Total reference words:** 1766  
**Total errors:** 1131 (I:242 D:65 S:824)

**Decision:** `upgrade_model`
> 50 of top-50 worst samples remain high-WER (>0.5) in both baseline and v1.1 (100.0% > 50% threshold). Fine-tuning has not resolved these persistent failures — model capacity is the bottleneck.

## 2. Error Distribution

### 2.1 Global Metrics

| Metric | Value |
|--------|-------|
| WER | 0.6404 |
| CER | 0.4661 |
| Insertions | 242 (21.4%) |
| Deletions | 65 (5.8%) |
| Substitutions | 824 (72.9%) |

### 2.2 Error Concentration

- Top 10% worst samples (36 samples): **27.8%** of total errors
- Top 20% worst samples (72 samples): **45.7%** of total errors
- Zero-WER samples: 71
- Catastrophic (WER > 1.0): 64

_Interpretation: Errors are **moderately concentrated**._

### 2.3 By Duration Bin

| Bin | Samples | WER | CER | Ins% | Del% | Sub% | Mean WER | Median WER |
|-----|---------|-----|-----|------|------|------|----------|------------|
| (1.0, 3] | 105 | 0.5815 | 0.3835 | 19.8% | 7.6% | 72.6% | 0.7311 | 0.6667 |
| (10.0, 30] | 17 | 0.6571 | 0.3998 | 23.2% | 5.8% | 71.0% | 0.7422 | 0.5333 |
| (3.0, 10] | 239 | 0.6501 | 0.4939 | 21.4% | 5.4% | 73.2% | 0.6920 | 0.6667 |

### 2.4 Hallucination Analysis

Hallucination-heavy samples: **32** (8.9%)
Top inserted tokens: `you` (10), `i` (6), `to` (4), `are` (4), `how` (3)
By duration bin: {'(3.0, 10]': 19, '(1.0, 3]': 10, '(10.0, 30]': 3}

## 3. Comparative Analysis (baseline → v1 → v1.1)

**Insertion reduction:** baseline=2083 → v1=219 → v1.1=242

### Improvement Distribution (baseline → v1.1)
- Improved: 315 samples (mean +1.6559 WER)
- Regressed: 24 samples (mean -0.4477 WER)
- Unchanged: 22 samples

**Persistent failures** (WER > 0.5 in both baseline and v1.1): 205 samples

## 4. Systematic Patterns

### 4.1 Substitution Patterns

Top 10 substitution pairs (of 30 total):

| Reference | Hypothesis | Count | In Baseline? |
|-----------|-----------|-------|--------------|
| `whats` | `is` | 6 | No |
| `i` | `im` | 4 | No |
| `its` | `is` | 3 | No |
| `why` | `what` | 3 | No |
| `dont` | `is` | 3 | No |
| `thats` | `is` | 3 | No |
| `in` | `at` | 3 | No |
| `them` | `name` | 2 | No |
| `thermostat` | `window` | 2 | No |
| `on` | `at` | 2 | No |

### 4.2 Normalization Artifacts

- Samples affected: **50**
- WER contribution: **1.67 WER pts**

Top detected contraction pairs:
| Reference Token | Hypothesis Token | Count |
|----------------|-----------------|-------|
| `dont` | `dont` | 7 |
| `whats` | `what` | 6 |
| `i` | `im` | 5 |
| `im` | `im` | 4 |
| `thats` | `that` | 4 |
| `dont` | `is` | 3 |
| `im` | `i` | 2 |
| `ive` | `i` | 2 |
| `a` | `youre` | 1 |
| `amber` | `im` | 1 |

### 4.3 Short Utterance Instability

Short utterances (≤3 ref words): **123** samples  
Mean WER: 0.8984 vs longer utterances: 0.6062  
Error share of total: 20.6%

### 4.4 Long Utterance Analysis

Long utterances (≥10 ref words): **16** samples  
Mean WER: 0.5074, Median WER: 0.5000  
Medium utterances mean WER: 0.6133  
Systematically higher WER than medium: **False**

### 4.5 Domain-Specific Token Failures

Tokens with error rate > 50%: **103**

| Token | Error Rate | Occurrences | Category |
|-------|-----------|-------------|----------|
| `spotify` | 100.0% | 2 | other |
| `onto` | 100.0% | 2 | other |
| `only` | 100.0% | 3 | other |
| `23rd` | 100.0% | 2 | other |
| `july` | 100.0% | 2 | other |
| `ends` | 100.0% | 2 | other |
| `monaco` | 100.0% | 2 | other |
| `reservation` | 100.0% | 2 | other |
| `eat` | 100.0% | 2 | other |
| `highway` | 100.0% | 2 | other |
| `weather` | 100.0% | 3 | other |
| `shut` | 100.0% | 2 | other |
| `wind` | 100.0% | 2 | other |
| `southern` | 100.0% | 2 | other |
| `warm` | 100.0% | 2 | other |
| `lower` | 100.0% | 3 | other |
| `whats` | 100.0% | 6 | other |
| `please` | 100.0% | 3 | other |
| `meal` | 100.0% | 3 | other |
| `elaborate` | 100.0% | 2 | other |

## 5. Audio Quality Correlation

Join match rate: 100.0% (361 samples matched)
Poor-audio samples in top-50 worst WER: **12** (24.0%)
WER vs silence_ratio: Pearson=0.0956, Spearman=0.0583
WER vs rms_db: Pearson=-0.0428, Spearman=-0.0703

## 6. Hypotheses

### H1: Contraction mismatch inflates WER: model expands contraction...

**Evidence:** 50 val samples affected; contraction normalization reduces aggregate WER by 0.0167 points (1.67 WER pts).
**Root cause:** textnorm_v1 removes apostrophes but does not expand contractions. The fine-tuned model outputs expanded forms while normalized reference retains the collapsed form, creating spurious mismatches.
**Intervention:** Add contraction expansion step to normalization pipeline: apply CONTRACTION_MAP to both reference and hypothesis before WER scoring.
**Experiment type:** normalization-only  
**Expected impact:** 1.7 WER pts (measured)  
**Risk:** Low risk. Additive rule; does not change model weights. Possible false positives for ambiguous tokens (e.g. 'its', 'were'), but overall impact is bounded by measured contribution.  
**Testable in M4a:** True

### H2: 205 samples remain high-WER (>0.5) in both baseline and v1.1...

**Evidence:** Baseline had 2083 insertions; v1.1 has 242 (reduction: 1841). 2 of top-30 substitution pairs persist from baseline, suggesting the model has not learned certain acoustic patterns despite fine-tuning.
**Root cause:** base.en (74M parameters) may lack capacity for this speaker's acoustic patterns. Fine-tuning with LoRA adapts the existing representations but cannot add new capacity to distinguish phonetically similar tokens.
**Intervention:** Fine-tune small.en (244M parameters) with same LoRA config (r=16, alpha=32) and Dataset v1. Unfreeze DECODE_V1 decoding parameters for joint search.
**Experiment type:** training-required  
**Expected impact:** 10–20 WER pts (estimated from model size scaling)  
**Risk:** Medium risk. Longer training (~3–4× compute), higher memory usage. Risk of overfitting with same dataset size. Requires S1-M5 training run; not testable in M4a.  
**Testable in M4a:** False

### H3: 24.0% of top-50 worst-WER samples have poor audio quality (s...

**Evidence:** 12 of top-50 worst samples classified as poor audio. Correlation with silence ratio: 0.096. Suggests audio quality is a contributing factor to high WER.
**Root cause:** High silence ratio (pauses, breath noise) and low RMS level cause the model to hallucinate or produce low-confidence transcriptions.
**Intervention:** Targeted re-recording of high-silence, low-RMS utterances, or apply audio preprocessing (silence trimming, normalization) before inference.
**Experiment type:** inference-only  
**Expected impact:** 2–8 WER pts (estimated based on affected sample share)  
**Risk:** Low-medium risk. Re-recording is labor-intensive but targeted. Audio preprocessing may degrade other samples. Requires Dataset v2 if re-recording is chosen.  
**Testable in M4a:** False

## 7. Follow-Up Experiment Results

| ID | Hypothesis | Type | v1.1 WER | Exp WER | Delta | Decision |
|----|-----------|------|----------|---------|-------|----------|
| m4a_exp_1 | H1 | normalization-only | 0.6404 | 0.6237 | +0.0167 | VALIDATED |

**m4a_exp_1** — Apply contraction expansion (CONTRACTION_MAP) to both reference and hypothesis before WER scoring. Re-score all val predictions.  
Notes: 50 samples affected. WER reduction: 1.67 pts. 20 distinct contraction pairs detected.

## 8. Decision Gate Outcome

**Decision:** `upgrade_model`
**Primary evidence:** 50 of top-50 worst samples remain high-WER (>0.5) in both baseline and v1.1 (100.0% > 50% threshold). Fine-tuning has not resolved these persistent failures — model capacity is the bottleneck.
**Normalization fix first:** True
**Next milestone:** S1-M5: Fine-tune small.en (244M) with LoRA r=16; unfreeze DECODE_V1.

## 9. Recommendations

1. **Quick win:** Apply normalization fix first to recover estimated WER points.
2. **S1-M5:** Fine-tune small.en (244M) with LoRA r=16, alpha=32.
3. **Unfreeze DECODE_V1** decoding parameters during S1-M5 training.
