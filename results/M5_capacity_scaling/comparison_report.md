# Capacity Scaling — Comparison Report

## 1. Overview

- **Model:** small.en (244M params)
- **Method:** LoRA (rank=16)
- **Device:** cpu
- **Training time:** 91.5 minutes
- **Created:** 2026-03-04 23:44:33 UTC
- **Normalization:** textnorm_v2

## 2. Aggregate Comparison (v1.1 vs v2)

| Model  | Base         | Params   | textnorm | Val WER    | Delta vs v1.1  |
| ------ | ------------ | -------- | -------- | ---------- | -------------- |
| v1.1   | base.en      | 74M      | v2       | 0.6237     | —              |
| **v2** | **small.en** | **244M** | **v2**   | **0.4795** | **+14.42 pts** |

Val samples: 361 | Absolute improvement: 0.1442 | Relative: 23.1%

> **Note on WER:** The original post-training evaluation used temperature=1.0 (sampling) and
> reported 0.4402. A deterministic re-evaluation with temperature=0.0 (greedy decoding)
> gives **0.4795**, which is the canonical figure used in all downstream comparisons.
> The training history (epoch 3 = 47.95%) already reflected the deterministic result.

## 3. Outcome Classification

**Breakthrough** (ΔWER = +14.42 pts vs v1.1)

| Label            | Criterion      |
| ---------------- | -------------- |
| **Breakthrough** | ΔWER >= 5 pts  |
| Marginal         | ΔWER 1-4.9 pts |
| No effect        | ΔWER < 1 pt    |
| Regression       | v2 WER > v1.1  |

## 4. Diagnostic Slices

| Slice                                             | Samples | v1.1 WER | v2 WER | Delta      |
| ------------------------------------------------- | ------- | -------- | ------ | ---------- |
| Persistent failure (WER > 0.5 in baseline & v1.1) | 205     | 1.0786   | 0.7081 | +37.05 pts |
| Short utterance (duration \<= 3s)                 | 105     | 0.7311   | 0.5002 | +23.09 pts |

The persistent-failure slice is the primary hypothesis check (H2: capacity).
The short-utterance slice tracks the hallucination pattern flagged by advisor review.

*Note: WER > 1.0 indicates more error words than reference words, typically caused by
heavy insertion/hallucination errors.*

## 5. Performance by Duration

| Duration Bin | Samples | WER    | CER    |
| ------------ | ------- | ------ | ------ |
| (1.0, 3\]    | 105     | 50.02% | 30.97% |
| (10.0, 30\]  | 17      | 66.23% | 37.76% |
| (3.0, 10\]   | 239     | 47.77% | 29.46% |

## 6. Error Breakdown

| Error Type    | Count |
| ------------- | ----- |
| Insertions    | 169   |
| Deletions     | 44    |
| Substitutions | 582   |

## 7. Training History

| Epoch | Val WER |
| ----- | ------- |
| 1.0   | 67.50%  |
| 2.0   | 52.21%  |
| 3.0   | 47.95%  |

## 8. Key Takeaways

- **Breakthrough:** +14.42 pts WER improvement over v1.1 (62.37% → 47.95%, deterministic)
- Model trained on CPU in 91.5 minutes
- LoRA rank 16 with 0.73% trainable parameters
- Persistent failures (n=205): +37.05 pts improvement
