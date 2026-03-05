# Baseline Evaluation Report

## 1. Overview

- **Dataset version:** v1
- **Baseline model:** Whisper base.en
- **Device:** mps
- **Evaluation splits:** test
- **Created:** 2026-02-10 22:49:21 UTC

## 2. Aggregate Metrics

| Split | Samples | Duration | Words | WER    | CER    |
| ----- | ------- | -------- | ----- | ------ | ------ |
| test  | 365     | 28.9 min | 1,804 | 254.3% | 224.5% |

## 3. Error Breakdown

### TEST Split

| Error Type    | Count     | Proportion |
| ------------- | --------- | ---------- |
| Substitutions | 383       | 8.3%       |
| Deletions     | 1,386     | 30.2%      |
| Insertions    | 2,818     | 61.4%      |
| **Total**     | **4,587** | 100%       |

## 4. Evaluation Slices

Performance by duration bin:

### TEST Split

| Duration Bin | Samples | WER    | CER    | vs Aggregate |
| ------------ | ------- | ------ | ------ | ------------ |
| (1.0, 3\]    | 106     | 287.2% | 226.9% | +13% higher  |
| (10.0, 30\]  | 19      | 181.5% | 157.4% | -29% lower   |
| (3.0, 10\]   | 240     | 261.6% | 237.8% | similar      |

## 5. Error Pattern Analysis

### Top 10 Substitutions

| Reference | Hypothesis | Count |
| --------- | ---------- | ----- |
| are       | i          | 5     |
| im        | i          | 4     |
| you       | i          | 3     |
| the       | i          | 3     |
| you       | we         | 3     |
| i         | uh         | 3     |
| let       | well       | 3     |
| me        | well       | 3     |
| what      | i          | 3     |
| can       | alright    | 2     |

### Top 10 Deletions

| Deleted Word | Count |
| ------------ | ----- |
| the          | 57    |
| to           | 40    |
| a            | 29    |
| you          | 28    |
| is           | 26    |
| it           | 25    |
| i            | 23    |
| me           | 21    |
| in           | 20    |
| on           | 20    |

### Top 10 Insertions

| Inserted Word | Count |
| ------------- | ----- |
| we            | 464   |
| do            | 295   |
| um            | 293   |
| i             | 209   |
| let           | 208   |
| really        | 207   |
| good          | 205   |
| okay          | 204   |
| get           | 204   |
| was           | 200   |

## 6. Key Takeaways

### What the baseline does poorly

- High overall WER (254.3%) indicates significant recognition challenges
- Model tends to miss words (deletions > substitutions)
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
