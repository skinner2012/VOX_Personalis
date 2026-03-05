# Improvement Analysis Report

Generated: 2026-02-25T06:36:55.654541+00:00

---

## 1. Executive Summary

- Model v1 val WER: **64.78%**
- Model v1.1 val WER: **64.04%**
- Absolute improvement: **0.0074 pts**
- Relative improvement: **1.1%**
- Experiments included in v1.1: ['inference_1', 'training_2']

---

## 2. Controlled Experiment Summary

| ID | Category | Variable | Baseline | New Value | Val WER | Delta | Decision |
| -- | -------- | -------- | -------- | --------- | ------- | ----- | -------- |
| inference_1 | inference | attention_mask | implicit | explicit (all ones) | 0.6478 | +0.0000 | **ADOPT** |
| inference_2 | inference | decode_config_source | per-call args | DECODE_V1.json | 0.6478 | +0.0000 | **ADOPT** |
| training_1 | training | learning_rate | 1e-4 | 5e-5 | 0.9083 | -0.2605 | **REJECT** |
| training_2 | training | lora_dropout | 0.1 | 0.15 | 0.6404 | +0.0074 | **INVESTIGATE** |
| training_3 | training | weight_decay | 0.0 | 0.01 | 0.6416 | +0.0062 | **INVESTIGATE** |

---

## 3. WER/CER Comparison (val only)

| Model | Val WER | Abs Improvement |
| ----- | ------- | --------------- |
| Model v1 | 64.78% | — |
| Model v1.1 | 64.04% | 0.0074 pts |

---

## 4. Recommendations for Future Milestones

- Review REJECT experiments for potential combination opportunities
- Consider Whisper small.en upgrade if further gains are needed
