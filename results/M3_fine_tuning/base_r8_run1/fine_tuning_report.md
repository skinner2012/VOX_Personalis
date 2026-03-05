# Fine-Tuning Report

## 1. Overview

- **Model:** base.en
- **Approach:** base_r8
- **Method:** LoRA (rank=8)
- **Device:** cpu
- **Training time:** 40.1 minutes
- **Created:** 2026-02-17 06:43:30 UTC

## 2. Baseline Reference

- **Baseline file:** `./out/baseline_eval/20260210-220646/baseline_metrics.json`
- **Baseline WER:** 205.83%
- **Normalization:** textnorm_v1

## 3. VAL Results

| Metric | Value |
|--------|-------|
| Samples | 361 |
| WER | 79.50% |
| CER | 54.04% |
| Baseline WER | 205.83% |
| Absolute Improvement | 126.33% |
| Relative Improvement | 61.4% |

### Error Breakdown

| Error Type | Count |
|------------|-------|
| Insertions | 254 |
| Deletions | 117 |
| Substitutions | 1033 |

## 4. Performance by Duration

| Duration Bin | Samples | WER | CER |
|--------------|---------|-----|-----|
| (1.0, 3] | 105 | 72.22% | 55.42% |
| (10.0, 30] | 17 | 88.10% | 57.55% |
| (3.0, 10] | 239 | 79.63% | 53.18% |

## 6. Training History

| Epoch | Val WER |
|-------|---------|
| 1.0 | 91.56% |
| 2.0 | 79.50% |
| 3.0 | 87.83% |

## 7. Key Takeaways

- ✅ **Improvement achieved:** 61.4% relative WER reduction
- Model trained on CPU in 40.1 minutes
- LoRA rank 8 with ~1-3% trainable parameters
