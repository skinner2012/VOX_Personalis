# S2-M0: Stock Zipformer Baseline Evaluation

**Date:** 2026-05-09\
**Model:** `csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-21` (70M params, LibriSpeech + GigaSpeech XL)\
**Runtime:** sherpa-onnx 1.13.1, ONNX Runtime CPU provider\
**Normalization:** `create_normalizer(version=2)` — same as S1-M7 baseline

## Result

| Metric              | Value        |
| ------------------- | ------------ |
| Val WER             | **100.72%**  |
| Baseline to beat    | 34.05% (S1-M7 fine-tuned Whisper) |
| Beats gate          | NO           |
| Val clips evaluated | 361          |
| Latency p50         | 102ms        |
| Latency p95         | 265ms        |

## Decision

Per the M0 decision tree in `specs/S2-sherpa-zipformer-gemma-pipeline.md`:

> Stock 70M val WER > 50% → **M1 CRITICAL** — fine-tuning required.

WER of 100.72% confirms the stock model produces near-random output on Deaf-accent speech.
This is expected: the model was trained exclusively on clean read speech (LibriSpeech + GigaSpeech)
and has never encountered the user's atypical speech patterns.

**M1 cloud fine-tune is required and non-optional.**

## Artifacts

- `predictions.csv` — per-clip reference, hypothesis, WER, latency (361 val clips)
- `metrics.json` — aggregated metrics

## Reproducibility

```bash
source venv/bin/activate
python -m scripts.zipformer_eval \
  --manifest ./results/M7_feedback_finetune/predictions.csv \
  --split val \
  --output ./out/S2-M0_zipformer_baseline
```
