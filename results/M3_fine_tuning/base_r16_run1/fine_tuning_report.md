# Fine-Tuning Report

## 1. Overview

- **Model:** base.en
- **Approach:** base_r16
- **Method:** LoRA (rank=16)
- **Device:** cpu
- **Training time:** 31.5 minutes
- **Created:** 2026-02-17 07:22:46 UTC

## 2. Baseline Reference

- **Baseline file:** `./out/baseline_eval/20260210-220646/baseline_metrics.json`
- **Baseline WER:** 205.83%
- **Normalization:** textnorm_v1

## 3. VAL Results

| Metric | Value |
|--------|-------|
| Samples | 361 |
| WER | 68.23% |
| CER | 45.34% |
| Baseline WER | 205.83% |
| Absolute Improvement | 137.60% |
| Relative Improvement | 66.8% |

### Error Breakdown

| Error Type | Count |
|------------|-------|
| Insertions | 219 |
| Deletions | 80 |
| Substitutions | 906 |

## 4. Performance by Duration

| Duration Bin | Samples | WER | CER |
|--------------|---------|-----|-----|
| (1.0, 3] | 105 | 61.48% | 46.07% |
| (10.0, 30] | 17 | 78.57% | 49.93% |
| (3.0, 10] | 239 | 67.96% | 44.69% |

## 6. Training History

| Epoch | Val WER |
|-------|---------|
| 1.0 | 82.79% |
| 2.0 | 70.89% |
| 3.0 | 68.23% |

## 7. Key Takeaways

- ✅ **Improvement achieved:** 66.8% relative WER reduction
- Model trained on CPU in 31.5 minutes
- LoRA rank 16 with ~1-3% trainable parameters
