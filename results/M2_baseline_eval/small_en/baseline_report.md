# Baseline Evaluation Report

## 1. Overview

- **Dataset version:** v1
- **Baseline model:** Whisper small.en
- **Device:** cpu
- **Evaluation splits:** test, val
- **Created:** 2026-02-11 06:20:22 UTC

## 2. Aggregate Metrics

| Split | Samples | Duration | Words | WER    | CER    |
| ----- | ------- | -------- | ----- | ------ | ------ |
| test  | 365     | 28.9 min | 1,804 | 238.7% | 196.3% |
| val   | 361     | 27.5 min | 1,766 | 205.8% | 143.9% |

## 3. Error Breakdown

### TEST Split

| Error Type    | Count     | Proportion |
| ------------- | --------- | ---------- |
| Substitutions | 1,425     | 33.1%      |
| Deletions     | 147       | 3.4%       |
| Insertions    | 2,734     | 63.5%      |
| **Total**     | **4,306** | 100%       |

### VAL Split

| Error Type    | Count     | Proportion |
| ------------- | --------- | ---------- |
| Substitutions | 1,440     | 39.6%      |
| Deletions     | 112       | 3.1%       |
| Insertions    | 2,083     | 57.3%      |
| **Total**     | **3,635** | 100%       |

## 4. Evaluation Slices

Performance by duration bin:

### TEST Split

| Duration Bin | Samples | WER    | CER    | vs Aggregate |
| ------------ | ------- | ------ | ------ | ------------ |
| (1.0, 3\]    | 106     | 137.0% | 107.8% | -43% lower   |
| (10.0, 30\]  | 19      | 199.2% | 166.2% | -17% lower   |
| (3.0, 10\]   | 240     | 267.2% | 218.4% | +12% higher  |

### VAL Split

| Duration Bin | Samples | WER    | CER    | vs Aggregate |
| ------------ | ------- | ------ | ------ | ------------ |
| (1.0, 3\]    | 105     | 140.4% | 104.2% | -32% lower   |
| (10.0, 30\]  | 17      | 182.4% | 108.8% | -11% lower   |
| (3.0, 10\]   | 239     | 223.4% | 157.9% | +9% higher   |

## 5. Error Pattern Analysis

### Top 10 Substitutions

| Reference | Hypothesis | Count |
| --------- | ---------- | ----- |
| the       | do         | 10    |
| the       | you        | 7     |
| that      | it         | 7     |
| the       | it         | 6     |
| is        | to         | 6     |
| that      | to         | 6     |
| the       | to         | 6     |
| you       | it         | 5     |
| turn      | you        | 5     |
| it        | to         | 5     |

### Top 10 Deletions

| Deleted Word | Count |
| ------------ | ----- |
| me           | 12    |
| you          | 9     |
| to           | 9     |
| it           | 8     |
| do           | 7     |
| is           | 7     |
| on           | 6     |
| my           | 5     |
| are          | 4     |
| i            | 4     |

### Top 10 Insertions

| Inserted Word | Count |
| ------------- | ----- |
| do            | 374   |
| to            | 228   |
| dick          | 202   |
| you           | 158   |
| the           | 145   |
| who           | 140   |
| ting          | 136   |
| doo           | 130   |
| what          | 127   |
| no            | 115   |

## 6. Key Takeaways

### What the baseline does poorly

- High overall WER (238.7%) indicates significant recognition challenges
- Model tends to hallucinate words (insertions > deletions)

### What the baseline does reasonably well

### Errors likely addressable via personalization

- Speaker-specific vocabulary and proper nouns
- Acoustic patterns unique to this speaker
- Consistent substitution patterns (see error analysis)

## 7. Limitations

- **Single-speaker bias:** Results reflect one speaker's speech patterns
- **Limited linguistic diversity:** Dataset contains specific utterance types
- **Baseline model constraints:** Whisper optimized for general speech, not accessibility
- **Text normalization:** Punctuation removal may affect some comparisons
