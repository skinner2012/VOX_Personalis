# Fine-Tuning Report

## 1. Overview

- **Model:** base.en
- **Approach:** base_r8
- **Method:** LoRA (rank=8)
- **Device:** cpu
- **Training time:** 32.0 minutes
- **Created:** 2026-02-17 18:54:23 UTC

## 2. Baseline Reference

- **Baseline file:** `./out/baseline_eval/20260210-220646/baseline_metrics.json`
- **Baseline WER:** 205.83%
- **Normalization:** textnorm_v1

## 3. VAL Results

| Metric | Value |
|--------|-------|
| Samples | 361 |
| WER | 76.84% |
| CER | 52.20% |
| Baseline WER | 205.83% |
| Absolute Improvement | 128.99% |
| Relative Improvement | 62.7% |

### Error Breakdown

| Error Type | Count |
|------------|-------|
| Insertions | 243 |
| Deletions | 91 |
| Substitutions | 1023 |

## 4. Performance by Duration

| Duration Bin | Samples | WER | CER |
|--------------|---------|-----|-----|
| (1.0, 3] | 105 | 69.63% | 54.04% |
| (10.0, 30] | 17 | 84.76% | 55.02% |
| (3.0, 10] | 239 | 77.06% | 51.19% |

## 5. Performance by Error Type

| Error Type | Samples | Fine-tuned WER | Baseline WER | Improvement |
|------------|---------|----------------|--------------|-------------|
| deletion_heavy | 1 | 0.00% | 50.00% | +50.00% |
| high_error | 351 | 77.77% | 217.26% | +139.49% |
| insertion_heavy | 2 | 33.33% | 35.00% | +1.67% |
| low_error | 2 | 80.00% | 0.00% | -80.00% |
| mixed | 2 | 36.36% | 39.29% | +2.92% |
| substitution_heavy | 3 | 37.50% | 35.00% | -2.50% |

## 6. Training History

| Epoch | Val WER |
|-------|---------|
| 1.0 | 91.85% |
| 2.0 | 92.36% |
| 3.0 | 76.84% |

## 7. Key Takeaways

- ✅ **Improvement achieved:** 62.7% relative WER reduction
- Model trained on CPU in 32.0 minutes
- LoRA rank 8 with ~1-3% trainable parameters
