# S1-M7 Feedback Fine-Tuning — Comparison Report

## Training Data

| Source | Samples |
| --- | --- |
| Original train | 2897 |
| Corrections | 108 |
| **Total** | **3005** |

## Val WER Comparison

| Model | Val WER |
| --- | --- |
| v2 (baseline) | 44.02% |
| v2 + corrections | 34.05% |

**Absolute improvement:** +9.97%
**Relative improvement:** +22.6%
**Result:** Improved ✓

## Notes

- Training mode: continued from checkpoint
- Merged original train + corrections to prevent catastrophic forgetting
- textnorm\_v2 applied to all transcripts during training and evaluation
