# Fine-Tuning Report

## 1. Overview

- **Model:** base.en
- **Approach:** base_r16
- **Method:** LoRA (rank=16)
- **Device:** cpu
- **Training time:** 36.4 minutes
- **Created:** 2026-02-17 19:34:03 UTC

## 2. Baseline Reference

- **Baseline file:** `./out/baseline_eval/20260210-220646/baseline_metrics.json`
- **Baseline WER:** 205.83%
- **Normalization:** textnorm_v1

## 3. VAL Results

| Metric | Value |
|--------|-------|
| Samples | 361 |
| WER | 69.54% |
| CER | 46.58% |
| Baseline WER | 205.83% |
| Absolute Improvement | 136.29% |
| Relative Improvement | 66.2% |

### Error Breakdown

| Error Type | Count |
|------------|-------|
| Insertions | 217 |
| Deletions | 85 |
| Substitutions | 926 |

## 4. Performance by Duration

| Duration Bin | Samples | WER | CER |
|--------------|---------|-----|-----|
| (1.0, 3] | 105 | 62.22% | 46.24% |
| (10.0, 30] | 17 | 76.67% | 49.21% |
| (3.0, 10] | 239 | 69.91% | 46.55% |

## 5. Performance by Error Type

| Error Type | Samples | Fine-tuned WER | Baseline WER | Improvement |
|------------|---------|----------------|--------------|-------------|
| deletion_heavy | 1 | 0.00% | 50.00% | +50.00% |
| high_error | 351 | 70.28% | 217.26% | +146.98% |
| insertion_heavy | 2 | 33.33% | 35.00% | +1.67% |
| low_error | 2 | 80.00% | 0.00% | -80.00% |
| mixed | 2 | 36.36% | 39.29% | +2.92% |
| substitution_heavy | 3 | 37.50% | 35.00% | -2.50% |

## 6. Training History

| Epoch | Val WER |
|-------|---------|
| 1.0 | 83.75% |
| 2.0 | 70.95% |
| 3.0 | 69.54% |

## 7. Key Takeaways

- ✅ **Improvement achieved:** 66.2% relative WER reduction
- Model trained on CPU in 36.4 minutes
- LoRA rank 16 with ~1-3% trainable parameters
