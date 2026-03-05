# Dataset v1 Report

## 1. Overview

- **Dataset version:** v1
- **Source:** Euphonia recordings via web upload on Macbook Pro
- **Created:** 2026-02-06T14:27:59.536364
- **Seed:** 42

## 2. Cleaning Summary

- **Input samples:** 3,623

- **Excluded samples:** 0 (0.0%)

- **Final dataset size:** 3,623

- **Total duration:** 4.68 hours

## 3. Split Summary

| Split | Count | Duration (hours) | Percentage |
| ----- | ----- | ---------------- | ---------- |
| train | 2,897 | 3.74             | 80.0%      |
| val   | 361   | 0.46             | 10.0%      |
| test  | 365   | 0.48             | 10.1%      |

## 4. Duration Distribution

| Duration Bin | Train | Val | Test |
| ------------ | ----- | --- | ---- |
| (0, 1\]      | 0     | 0   | 0    |
| (1.0, 3\]    | 843   | 105 | 106  |
| (10.0, 30\]  | 140   | 17  | 19   |
| (3.0, 10\]   | 1914  | 239 | 240  |
| (30.0, inf\] | 0     | 0   | 0    |

## 5. Quality Checks

- **Duplicate audio with different transcripts:** 0
- **Temporal session leakage:** 38 clusters crossing splits
- **Minimum sample validation:** PASS
- **Minimum duration validation:** PASS

## 6. Split Quality Assessment

Duration distributions are balanced across splits.

**Recommendation:** READY FOR TRAINING

## 7. Test Set Lock

- **Test set frozen:** `test_set_v1_frozen.csv`
- **Test samples count:** 365

**Instructions for future dataset versions:**

1. Load `test_set_v1_frozen.csv`
1. Preserve all v1 test samples in test split (match by `pair_sha256`)
1. MAY add new samples to test
1. MUST NOT move v1 test samples to train/val

## 8. Next Steps

- Proceed to S1-M2 (Audio Preprocessing)
