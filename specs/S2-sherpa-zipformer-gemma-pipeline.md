# VOX Personalis Stage 2 — Sherpa-ONNX + Streaming Zipformer + Gemma 4 Pipeline

**Status:** SUPERSEDED (2026-05-09)\
**Branch:** main\
**Superseded by:** [S2-whisperlivekit-gemma-pipeline.md](S2-whisperlivekit-gemma-pipeline.md)\
**Originally supersedes:** S1-M7 (Whisper fine-tuned baseline, 34.05% val WER), S2-moonshine-gemma-pipeline.md (Moonshine v2 path — abandoned because HF Transformers Moonshine does not expose true streaming inference)

______________________________________________________________________

## Postmortem (2026-05-09)

This spec was attempted end-to-end through M1. The streaming Zipformer fine-tune produced an
unusable model. The root cause is a mismatch between icefall's recipe defaults and our
dataset scale, compounded by a structural fact missed during planning: icefall has no
streaming-AND-PEFT recipe.

### What happened

| Milestone                                                              | Outcome                                                                                               | Result file                                                                                       |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| M0: Stock Zipformer baseline WER on 361 val clips                      | **100.72% val WER** (random output on Deaf accent)                                                    | [results/S2-M0_zipformer_baseline/metrics.json](../results/S2-M0_zipformer_baseline/metrics.json) |
| M1: Cloud fine-tune on Lambda Labs A10 (Candidate B, 66M, LibriSpeech) | Training appeared successful: val loss 1.71 → 0.05 in 20 epochs                                       | (cloud-only artifact, not committed)                                                              |
| M1: ONNX export + sherpa-onnx eval on val set                          | **97.90% val WER — alignment collapse** (model emits only first BPE token per utterance, then blanks) | (ephemeral, not committed)                                                                        |

### Root cause: dataset scale ≪ recipe assumption

icefall's `egs/librispeech/ASR/zipformer/finetune.py` defaults are tuned for ~100–1000h
datasets. We ran with 3.7h. Concretely:

- `--warm-step 2000` × `--max-duration 300` × 3.7h ≈ ~250 total optimizer steps. The LR
  schedule never finished warmup; optimizer ran at suboptimal LR throughout.
- `--enable-musan 0` (we lacked MUSAN on the cloud) removed regularization that prevents
  the joiner from collapsing to "blank everywhere".
- `--use-mux 0` (no LibriSpeech to mux against on cloud) removed the safety net the
  upstream icefall doc explicitly warns about.

The result: `pruned_rnnt_loss` minimized cleanly (low val loss) by assigning probability
mass to blanks across all positions. Greedy/beam decoding then emits only the first BPE
token (where the encoder's frame-1 representation is strongest) and blanks thereafter.
This is the canonical small-data RNN-T failure mode, well-documented in the literature
(NeMo issue #14140, icefall discussion #1580).

### The structural fact missed during planning

icefall ships three Zipformer recipes:

| Recipe                                       | Streaming export? | PEFT?                    |
| -------------------------------------------- | ----------------- | ------------------------ |
| `egs/librispeech/ASR/zipformer/` (used here) | Yes               | No (full fine-tune only) |
| `egs/librispeech/ASR/zipformer_lora/`        | **No**            | Yes                      |
| `egs/librispeech/ASR/zipformer_adapter/`     | **No**            | Yes                      |

There is no streaming + PEFT combination in icefall. The original spec quoted the upstream
maintainer's note that `zipformer_adapter` is non-streaming and chose full fine-tune as
the documented streaming path — but did not validate that 3.7h was sufficient for full
fine-tune of a 66M streaming RNN-T. Literature evidence (Tomanek 2023 Project Euphonia,
Takahashi 2025 Interspeech SAP winner) sets the practical threshold around 100h+.

### Lessons

1. **Validate dataset-vs-recipe scale before committing cloud time.** `warm_step=2000` on
   3.7h is a checkable red flag.
1. **Streaming + PEFT requires picking a model family that supports both** — Whisper +
   SimulStreaming, NeMo cache-aware Conformer + adapter, or NVIDIA nemotron-speech-streaming.
   icefall is not that family.
1. **Low val loss can hide alignment collapse** in transducer models. Always smoke-test
   inference output (not just metrics) before committing to a full export pipeline.

### Artifacts retained (still useful)

| Path                                                                      | Reason kept                                                                                                                      |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| [scripts/zipformer_eval/](../scripts/zipformer_eval/)                     | Generic Zipformer WER eval module — works on any sherpa-onnx Zipformer ONNX, reusable for future attempts                        |
| [scripts/zipformer_finetune/](../scripts/zipformer_finetune/)             | Lhotse manifest converter + A10 bootstrap script — validated end-to-end on the A10, reusable if revisited with proper data scale |
| [results/S2-M0_zipformer_baseline/](../results/S2-M0_zipformer_baseline/) | 100.72% stock baseline result on the S1 frozen val split                                                                         |
| `out/lhotse_manifests/` (gitignored)                                      | Pre-computed 16kHz fbank features + cuts for 3,005 train + 361 dev clips                                                         |
| `~/Downloads/voice_data_v2.tar.gz` (1.5 GB, local only)                   | Tarball ready for re-upload to A10 if streaming Zipformer is revisited                                                           |

### Pivot

The replacement pipeline reuses our existing [S1-M7 feedback-fine-tuned Whisper LoRA
checkpoint (34.05% val WER)](../results/M7_feedback_finetune/) served via WhisperLiveKit's
SimulStreaming/AlignAtt path for sub-second word-by-word display, paired with Gemma 4
Stage B correction. See the live spec linked at the top.

The remainder of this document is preserved as the original ACTIVE spec for historical
record.

______________________________________________________________________

______________________________________________________________________

## Table of Contents

- [Problem](#problem)
- [Why Streaming Zipformer (vs Moonshine, vs Whisper)](#why-streaming-zipformer-vs-moonshine-vs-whisper)
- [Core Product Flow](#core-product-flow)
- [Constraints](#constraints)
- [10x Vision](#10x-vision)
- [Hardware](#hardware)
- [What Is Out of Scope (Phase 1)](#what-is-out-of-scope-phase-1)
- [Accepted Phase 1 Limitations](#accepted-phase-1-limitations)
- [Baseline to Beat](#baseline-to-beat)
- [Objective](#objective)
- [Naming Conventions](#naming-conventions)
- [Milestone Sequence](#milestone-sequence)
- [Pre-work: Day 0 Checklist](#pre-work-day-0-checklist-blocks-everything)
- [System Architecture (Layer 1)](#system-architecture-layer-1)
- [Component Specs](#component-specs)
  - [TextConsumer Protocol](#textconsumer-protocol)
  - [WebConsumer (aiohttp HTTP+WS)](#webconsumer-aiohttp-httpws)
  - [Silero VAD Pre-filter](#silero-vad-pre-filter)
  - [Sherpa-ONNX Streaming Zipformer Integration](#sherpa-onnx-streaming-zipformer-integration)
  - [Gemma 4 GGUF Integration](#gemma-4-gguf-integration)
  - [Chrome Frontend](#chrome-frontend)
  - [Daemon CLI](#daemon-cli)
- [Module Structure](#module-structure)
- [Dataset](#dataset)
- [Cloud Fine-tuning Workflow](#cloud-fine-tuning-workflow)
- [Evaluation Specs](#evaluation-specs)
  - [M0: Stock Zipformer Baseline WER](#m0-stock-zipformer-baseline-wer)
  - [M1: Zipformer Streaming Fine-tuning (Cloud)](#m1-zipformer-streaming-fine-tuning-cloud)
  - [M2: Gemma 4 Correction Evaluation](#m2-gemma-4-correction-evaluation)
- [Error / Rescue Registry (Layer 1)](#error--rescue-registry-layer-1)
- [Success Criteria (5-Day Demo)](#success-criteria-5-day-demo)
- [Layer 2: AX Integration Stub (Post-demo)](#layer-2-ax-integration-stub-post-demo)
- [Pre-Implementation Blockers](#pre-implementation-blockers)

______________________________________________________________________

## Problem

Stage 1 ended with a fine-tuned Whisper small.en at 34.05% val WER. That is the
bar. Stage 2 answers: can a true streaming ASR — emitting partial transcripts
*while the user speaks*, not after they pause — beat the Whisper baseline, and
ship as a live demo on Apple Silicon within 5 days?

Whisper is being replaced. Moonshine v2 was the original candidate but its
HuggingFace Transformers code path does not expose true streaming (the model
card admits "the current Transformers code path does not yet implement fully
efficient streaming"). Streaming Zipformer via sherpa-onnx is the new ASR
engine: it emits partial transcripts mid-utterance (every 320ms encoder chunk,
~3 updates per second) during speech, runs natively on Apple Silicon, has a
documented fine-tuning path, and benchmarks significantly better than
Moonshine on LibriSpeech.

**This is not a captioning app.** The end use case — the Deaf speaker using
their own voice in a job interview — is the motivation. Stage 2 work is: prove
the new streaming stack is better, then put it in front of a real user.

______________________________________________________________________

## Why Streaming Zipformer (vs Moonshine, vs Whisper)

| Property                   | Whisper (S1)                    | Moonshine v2 (HF)                           | **Streaming Zipformer (S2)**                                               |
| -------------------------- | ------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------- |
| Streaming UX               | No (offline)                    | No (HF API is one-shot)                     | **Yes — partials every 320ms encoder chunk**                               |
| LibriSpeech test-clean WER | ~5% (small)                     | 6.65% (medium)                              | **2.43%** (70M LibriSpeech+GigaSpeech, greedy, 320ms chunk)                |
| Model size                 | 244M / ~1GB (small)             | 245M / 1.06GB (medium)                      | **70M / ~340MB**                                                           |
| Fine-tuning path           | LoRA via HF/PEFT (proven on S1) | LoRA via HF/PEFT (proven for non-streaming) | **Full fine-tune of streaming Zipformer via icefall (cloud GPU required)** |
| Apple Silicon inference    | PyTorch MPS                     | PyTorch MPS                                 | **sherpa-onnx (CPU or CoreML), pip wheel**                                 |
| First-token latency        | N/A (offline)                   | After VAD pause + encoder pass (~600ms)     | **~320ms chunk granularity, mid-utterance**                                |
| License (model + code)     | MIT (model), MIT (code)         | Apache-2.0 (small/medium)                   | **Apache-2.0 (model + sherpa-onnx + icefall)**                             |

The streaming property is decisive: the user wanted word-by-word display *while
speaking*, not after a VAD-detected pause. Only sherpa-onnx + streaming
Zipformer delivers that on the open-source path.

______________________________________________________________________

## Core Product Flow

```text
User speaks
  → sherpa-onnx OnlineRecognizer (streaming Zipformer, fine-tuned for Deaf accent)
      ↓ mic feeds 100ms audio chunks; encoder emits partial transcripts every 320ms
  → Stage A: processing text  — appears mid-utterance, updates ~3× per second
      ↓ on endpoint detection (rule-based: trailing-silence + utterance-length)
  → Stage B: committed text   — Gemma-corrected, replaces Stage A
  → display (Phase 1: Chrome tab / Phase C: Live Captions Type to Speak)
  → user reviews, submits
```

**Why partial Stage A?** The user gets immediate visual feedback as they speak.
No waiting. No "did the system hear me?" anxiety.

**Why Stage B at all?** Streaming output sacrifices some accuracy because the
encoder hasn't seen the full utterance. Gemma 4 fixes errors after the line
finalizes (endpoint detection): correcting accent-driven misrecognitions,
restoring punctuation, fixing disfluencies. Stage B replaces Stage A within
~800ms of the line ending.

**Why "text consumer agnostic"?** The ASR pipeline does not know or care how
text reaches the user. It calls three methods on a TextConsumer:
`on_stage_a()`, `on_stage_b()`, `on_clear()`. Today the consumer is a Chrome
WebSocket display. In Phase C it will be a Swift AX injector writing to Live
Captions. The pipeline does not change when the consumer changes.

______________________________________________________________________

## Constraints

- **Fully local inference.** No cloud ASR, no cloud LLM, no audio upload at
  runtime. Privacy is non-negotiable: the user's voice never leaves the machine
  during normal use.
- **Cloud is allowed for one-time fine-tuning.** The training set is uploaded
  to a Lambda Labs A10 instance, fine-tuning runs there, the resulting ONNX
  model is downloaded back. Audio is not retained on the cloud after training.
- **Single speaker.** The model is personalized to one voice. Not multi-user.
- **macOS only** (Phase 1 + Phase 2). Android is Phase 3 — out of scope here.
- **No automatic TTS triggering.** User controls when text is submitted/spoken.
- **No audio retention** on local machine without user opt-in.

______________________________________________________________________

## 10x Vision

WER drops below 20% after 6 months of feedback fine-tuning. The streaming
display feels indistinguishable from a fast typist seeing the speaker's mouth.
The Deaf speaker enters a job interview, speaks naturally, and their words
appear on screen — word-by-word as they speak — with the clarity of a fluent
typist, no corrections, no hesitation, no apology for the technology.

______________________________________________________________________

## Hardware

| Machine             | Specs                                           | Role                                                          |
| ------------------- | ----------------------------------------------- | ------------------------------------------------------------- |
| Mac mini            | M4 Pro, 64GB RAM                                | Local development, evaluation (M0, M2, M3), Gemma 4 inference |
| MBP M4 Pro          | M4 Pro, 48GB RAM                                | Live demo (M3), daily driver                                  |
| **Lambda Labs A10** | 1× A10 24GB, 226 GiB RAM, 1.3 TiB SSD, $1.29/hr | **One-time fine-tuning (M1) — rented for ~6–8h, ~$10–20**     |

**Inference framework (local):** sherpa-onnx (ONNX Runtime + optional CoreML)
via the `sherpa-onnx` PyPI wheel. arm64 wheels exist for Python 3.11–3.14. CPU
provider is the stable path; CoreML is benchmarked separately on M4 Pro
([known issue #2910](https://github.com/k2-fsa/sherpa-onnx/issues/2910): CoreML
sometimes slower than CPU on M2 — measure on M4 Pro before committing).

**Fine-tuning framework (cloud):** k2 + icefall + PyTorch with CUDA on Lambda
Labs A10. icefall recipe: `egs/librispeech/ASR/zipformer/` (the vanilla
recipe, which has both `finetune.py` and `export-onnx-streaming.py`).
**Full encoder fine-tune** with `--causal 1` for streaming. icefall does NOT
ship a streaming-AND-PEFT recipe — `zipformer_adapter/` is non-streaming only;
`zipformer_lora/` does not document streaming support. Full fine-tune is the
documented streaming path.

**Why cloud for training?** k2's `pruned_rnnt_loss` is C++/CUDA-only — there's
no Metal/MPS kernel. icefall *does* run on macOS CPU (verified), but throughput
is ~50–100× slower than a single A10. A full streaming Zipformer fine-tune on
M4 Pro CPU would be 50–200h per run, making hyperparameter iteration
impractical in a 5-day window.

**Gemma on M4 Pro (64GB):** Gemma 4 26B-A4B MoE Q4_K_M (~16.9 GB GGUF) is the
primary choice — fits comfortably on 64GB alongside ~340MB Zipformer ONNX.
Smaller alternatives if Q4_K_M latency exceeds the 800ms target: Q3_K_M
(~12.7 GB), IQ3_XXS (~11.4 GB), or fall back to Gemma 4 E2B Q4_K_M (~4 GB).
llama.cpp Gemma 4 support landed around build b9080 (May 2026) — use the
latest llama.cpp release. Day 0 benchmark confirms latency on M4 Pro.

______________________________________________________________________

## What Is Out of Scope (Phase 1)

| Item                                      | Deferred to                                              |
| ----------------------------------------- | -------------------------------------------------------- |
| Android client                            | Phase 3                                                  |
| FastAPI serving layer                     | Phase 2 (exists at scripts/serving/, untouched)          |
| AX injection into Live Captions           | Phase C (post-demo)                                      |
| IME (InputMethodKit) path                 | Phase C (if AX fails)                                    |
| Stage B suppression (user-edit detection) | Phase 2                                                  |
| Visual Stage A/B differentiation in AX    | AX limitation — plain text only                          |
| Automated test suite                      | Phase 2                                                  |
| Menu bar status app                       | Explicitly skipped                                       |
| Auto-submit after silence                 | Explicitly skipped                                       |
| Gemma rollback hotkey                     | Explicitly skipped                                       |
| Multi-language support                    | Phase 3 (en-only for now)                                |
| CoreML on-device fine-tuning              | Researchy — out of scope until icefall has Metal kernels |

______________________________________________________________________

## Accepted Phase 1 Limitations

- **Stage A is partial-transcript streaming, not perfect-transcript streaming.**
  Each chunk update overwrites the previous Stage A text. Words may flicker as
  the model revises hypotheses with more context. This is inherent to streaming
  ASR; the UX should make it visually clear that Stage A is "in progress."
- **Stage B is applied unconditionally.** If the user edits Stage A text
  manually before Stage B arrives, Stage B overwrites the edit. User-edit
  detection requires bidirectional IPC — deferred to Phase 2.
- **No visual distinction between Stage A and Stage B in the AX path (Phase
  C):** AXUIElement sets plain text only, no character-level styling. Chrome
  display (Phase B) uses color/italic to distinguish them.
- **`on_clear()` is a no-op in Phase 1.** The field retains Stage B text until
  the next utterance's Stage A fires.
- **First-token latency is ~320ms (not 50ms).** Streaming Zipformer's
  `--decode-chunk-len 32` configures the encoder to wait one chunk
  (320ms at 16kHz, hop_size=10ms) before emitting first hypothesis. This is
  perceptually live but not as snappy as theoretical sub-100ms streaming.

______________________________________________________________________

## Baseline to Beat

All prior evaluation used the same held-out val set and `create_normalizer(version=2)`
normalization. New evals must use identical normalization for apples-to-apples
comparison.

| Model                                                   | WER (val set, S1 frozen splits) | Notes                                                                                      |
| ------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------ |
| Whisper small.en (out-of-box)                           | ~55% (estimated)                | Pre-fine-tuning baseline                                                                   |
| Whisper small.en (fine-tuned, S1-M7)                    | **34.05%**                      | Current best — the bar to clear                                                            |
| Stock streaming Zipformer (LibriSpeech+GigaSpeech, 70M) | TBD — M0                        | icefall RESULTS: 2.43% / 6.0% on LibriSpeech test-clean / test-other (greedy, 320ms chunk) |
| Streaming Zipformer (fine-tuned, full)                  | Target: \<34.05%                | M1 gate                                                                                    |
| Streaming Zipformer + Gemma 4 correction                | Target: ≤ fine-tuned alone      | M2 gate                                                                                    |

**Model selection (decided 2026-05-08):** Inference uses
`csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-21` (70M params, ~340 MB
ONNX = 337 encoder + 2 decoder + 1 joiner, trained on LibriSpeech 960h +
GigaSpeech XL).

**PyTorch source checkpoint for fine-tuning — two candidates, Day 0 picks one:**

- **Candidate A (matches inference 70M):**
  `marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04`
  (281 MB `exp/pretrained.pt`). **Risk:** trained with the OLDER
  `pruned_transducer_stateless7_streaming_multi` recipe; new vanilla
  `zipformer/finetune.py` may not load the state dict cleanly. Mitigation:
  use `--init-modules` to selectively load encoder weights only; verify via
  smoke-test before the 6h training run.
- **Candidate B (66M LibriSpeech-only, recipe-matched):**
  `Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17`. Architecturally
  matches new `zipformer/` recipe so `finetune.py` loads cleanly. Tradeoff:
  smaller model + LibriSpeech only → higher OOB WER (~2.94 vs 2.43).

**Day 0 decision:** smoke-test loading Candidate A into vanilla `zipformer/`.
If state dict matches (or `--init-modules encoder` works): use A. Otherwise:
use B and accept the OOB WER tradeoff.

______________________________________________________________________

## Objective

Replace Whisper with streaming Zipformer + Gemma 4 as the ASR stack. Prove the
new stack beats fine-tuned Whisper (34.05% val WER), demonstrate true streaming
UX (partial transcripts during speech), and ship a working live-caption demo
for a hiring manager screen within 5 days.

Two-layer plan:

- **Layer 1 (now):** Cloud fine-tune → ONNX export → local Python pipeline → Chrome demo
- **Layer 2 (post-demo):** AX injection → Live Captions Type to Speak

______________________________________________________________________

## Naming Conventions

Two naming schemes are used in this spec.

**Phase 1 / Phase 2 / Phase 3** — scope phases of Stage 2:

- **Phase 1 (this spec):** macOS only — local Python daemon + Chrome display + AX injection
- **Phase 2 (future):** FastAPI serving layer + browser extension
- **Phase 3 (future):** Android client

**Phase A / Phase B / Phase C** — time sub-phases within Phase 1:

- **Phase A:** Prove the core — evaluate stock Zipformer, fine-tune on cloud, add Gemma 4 (M0–M2)
- **Phase B:** Ship the demo — Chrome live caption display (M3)
- **Phase C:** Production integration — AX injection into Live Captions (M4–M5, post-demo)

______________________________________________________________________

## Milestone Sequence

```text
PHASE A — PROVE THE CORE (Days 1–3)
┌─────────────────────────────────────────────────────────────────┐
│ M0 — Stock Zipformer Baseline (~2–3h, local M4 Pro)             │
│   Install sherpa-onnx, download stock 70M ONNX, run WER eval    │
│   on S1 val/test splits. Use create_normalizer(version=2).      │
│   Gate: stock Zipformer WER comparable to or better than        │
│   fine-tuned Whisper (34.05%)? If yes → Stage B Gemma correction│
│   may be enough; M1 fine-tune becomes optional. If no → M1.     │
│   Output: "Zipformer 70M stock val WER = X%, latency = Y ms".   │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ M1 — Streaming Zipformer Fine-tune on Cloud A10 (~1 day, $10–20)│
│   1. Convert S1 manifest CSV → lhotse cuts.jsonl format.         │
│   2. Rent Lambda Labs A10, install icefall + k2 + base ckpt.    │
│   3. Run zipformer/finetune.py: --causal 1, --base-lr 0.0045,    │
│      20 epochs. Full encoder fine-tune (no PEFT).                │
│   4. Export via zipformer/export-onnx-streaming.py.             │
│   5. rsync ONNX back to local M4 Pro.                            │
│   Gate: val WER < 34.05% (beats fine-tuned Whisper baseline).   │
│   Output: ./out/zipformer_finetune/{encoder,decoder,joiner}.onnx│
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ M2 — Gemma 4 Correction Evaluation (~4–8h)                       │
│   Zipformer (fine-tuned) → Gemma 4 correction → WER eval.       │
│   Gate: Stage B WER ≤ Stage A WER (correction helps, not hurts).│
│   Gate: false correction rate < 15% on val set samples.          │
│   Output: "Stage B WER = Y%" — go/no-go for Gemma in demo.       │
└────────────────────────┬────────────────────────────────────────┘
                         │

PHASE B — SHIP THE DEMO (Days 3–5)
┌────────────────────────▼────────────────────────────────────────┐
│ M3 — Chrome Live Caption Demo (~1–2 days)                       │
│   Python daemon: mic → sherpa-onnx OnlineRecognizer → Gemma →    │
│   aiohttp HTTP+WS server → Chrome.                               │
│   Stage A streams word-by-word (partials every 100ms), Stage B  │
│   replaces on endpoint detection.                                │
│   Gate: demo runs stably for 10 min of continuous speech.        │
│   This IS the hiring manager screen.                             │
└─────────────────────────────────────────────────────────────────┘

PHASE C — PRODUCTION INTEGRATION (Post-demo)
┌─────────────────────────────────────────────────────────────────┐
│ M4 — AX Proof of Concept (~4–8h)                                │
│   Swift CLI: inject hardcoded string → Live Captions Type to    │
│   Speak field. Go/no-go for AX path.                            │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ M5 — AX Daemon + Swift Injector (~1 week)                       │
│   Replace WebConsumer with AXConsumer (one config flag).        │
└─────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## Pre-work: Day 0 Checklist (~2h, blocks everything)

These must pass before writing any Phase A code. Local steps run on Mac mini
(M4 Pro, 64GB). Cloud steps run on a freshly-rented Lambda Labs A10.

### Local (Mac mini M4 Pro)

```bash
# 1. Verify sherpa-onnx installs and loads on Apple Silicon
pip install sherpa-onnx
python -c "
import sherpa_onnx
print(f'sherpa-onnx version: {sherpa_onnx.__version__}')
print('OnlineRecognizer available:', hasattr(sherpa_onnx, 'OnlineRecognizer'))
"

# 2. Download stock streaming Zipformer 70M (LibriSpeech+GigaSpeech)
mkdir -p ./models/sherpa-onnx-streaming-zipformer-en-2023-06-21
cd ./models/sherpa-onnx-streaming-zipformer-en-2023-06-21
# Use HF hub CLI or git LFS:
huggingface-cli download csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-21 \
    --include "*.onnx" "tokens.txt" --local-dir .

# 3. Smoke-test stock model on a single sample from S1 val set
python -c "
import sherpa_onnx, soundfile as sf, numpy as np
recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
    encoder='./models/sherpa-onnx-streaming-zipformer-en-2023-06-21/encoder-epoch-99-avg-1.onnx',
    decoder='./models/sherpa-onnx-streaming-zipformer-en-2023-06-21/decoder-epoch-99-avg-1.onnx',
    joiner='./models/sherpa-onnx-streaming-zipformer-en-2023-06-21/joiner-epoch-99-avg-1.onnx',
    tokens='./models/sherpa-onnx-streaming-zipformer-en-2023-06-21/tokens.txt',
    provider='cpu',
    sample_rate=16000, feature_dim=80,
)
audio, sr = sf.read('<path-to-a-val-set-clip>.wav')
if sr != 16000:  # S1 data is 44.1kHz; resample
    import librosa; audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
stream = recognizer.create_stream()
stream.accept_waveform(16000, audio.astype('float32'))
stream.input_finished()
while recognizer.is_ready(stream):
    recognizer.decode_stream(stream)
print('Transcript:', recognizer.get_result(stream))
"

# 4. Verify Gemma 4 builds via llama.cpp on M4 Pro
git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp
cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j
# Verify llama.cpp version supports Gemma 4 (initial support landed ~b9080 in
# May 2026). Pin to latest release tag if a build fails to load Gemma 4.
./build/bin/llama-cli --version

# Download Gemma 4 26B-A4B Q4_K_M GGUF (~16.9GB) from unsloth or bartowski:
huggingface-cli download unsloth/gemma-4-26B-A4B-it-GGUF \
    --include "*Q4_K_M*" --local-dir ../models/
./build/bin/llama-cli -m ../models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
    -p "Correct: 'i want to talk about kubernets'" -n 50
# Target: response in <800ms on M4 Pro. If exceeded, step down to Q3_K_M (~12.7GB)
# or IQ3_XXS (~11.4GB). Final fallback: Gemma 4 E2B Q4 (~4GB).

# 5. Verify aiohttp installs (HTTP+WebSocket server for daemon)
pip install aiohttp
python -c "import aiohttp; from aiohttp import web; print('aiohttp OK')"

# 6. Verify silero-vad installs and runs on a 100ms test chunk
pip install silero-vad
python -c "
import torch
from silero_vad import load_silero_vad
model = load_silero_vad()
# silero-vad v6 requires exactly 512 samples per call at 16kHz (32ms chunks).
# Our daemon feeds 1600-sample (100ms) chunks — split into 512-sample sub-chunks
# and take max() of probabilities.
chunk_512 = torch.zeros(512)
prob = model(chunk_512, 16000).item()
print(f'silero-vad v6 OK, silence prob={prob:.3f} (expect near 0)')
"
```

### Cloud (Lambda Labs A10) — only when ready to start M1

```bash
# Rent A10 24GB instance via lambdalabs.com, ssh in.
# 6. icefall + k2 install (use prebuilt CUDA wheels)
# Pick the latest k2 dev wheel matching the instance's CUDA + torch version.
# As of 2026-05-08, latest available builds include:
#   k2==1.24.4.dev20260423+cuda12.6.torch2.11.0
#   k2==1.24.4.dev20260306+cuda12.6.torch2.10.0
# Confirm available wheels at https://k2-fsa.github.io/k2/cuda.html before pinning.
pip install k2==1.24.4.dev20260423+cuda12.6.torch2.11.0 \
    -f https://k2-fsa.github.io/k2/cuda.html
git clone https://github.com/k2-fsa/icefall.git && cd icefall
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH

# 7. Verify k2 CUDA + RNN-T loss
python -c "
import torch, k2
print(f'CUDA: {torch.cuda.is_available()}')
print(f'k2 version: {k2.__version__}')
loss = k2.rnnt_loss_simple
print(f'rnnt_loss_simple available: {loss is not None}')
"

# 8. Download base PyTorch checkpoint
# Candidate A (70M, LibriSpeech+GigaSpeech, OLDER recipe — verify state dict loads):
huggingface-cli download \
    marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04 \
    --include "exp/pretrained.pt" "data/lang_bpe_500/*" \
    --local-dir ./base_ckpt_A
# Candidate B (66M, LibriSpeech-only, NEW recipe — guaranteed compatible):
huggingface-cli download \
    Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17 \
    --include "exp/pretrained.pt" "data/lang_bpe_500/*" \
    --local-dir ./base_ckpt_B

# 9. Day 0 smoke-test: does Candidate A load into vanilla zipformer/finetune.py?
# Try a 1-step training run to verify state dict compatibility.
cd icefall/egs/librispeech/ASR
python ./zipformer/finetune.py \
    --do-finetune 1 \
    --finetune-ckpt ~/base_ckpt_A/exp/pretrained.pt \
    --num-epochs 1 --start-epoch 1 \
    --causal 1 --chunk-size 32 --left-context-frames 128 \
    --max-duration 100 \
    --exp-dir /tmp/smoke_test_A 2>&1 | head -50
# Expected outcome:
#   Loads cleanly → use Candidate A (70M, better OOB).
#   State dict mismatch → use Candidate B (66M, recipe-matched), or pass
#                        --init-modules to selectively load only encoder weights.
```

### pyproject.toml updates

Add new deps via `pip install -e ".[dev]"` (not bare `pip install`) — keep the
project deps-declared invariant:

```text
# Add to pyproject.toml [project.dependencies]:
sherpa-onnx>=1.13.0      # streaming ASR runtime (local inference)
huggingface-hub>=0.20    # model downloads (likely already transitive)
aiohttp>=3.9             # HTTP+WebSocket server for daemon
sounddevice>=0.4         # mic capture for daemon
silero-vad>=6.0          # neural VAD pre-filter (gates audio before ASR; v6 requires 512-sample chunks)
lhotse>=1.20             # manifest format conversion (M1 prep, local-only)

# Cloud-only deps (do NOT add to local pyproject.toml — installed on A10):
# k2 (CUDA wheel), icefall (git clone + requirements.txt)
```

`transformers` is already imported in S1 code without being declared in
pyproject.toml — pre-existing gap, but worth fixing now.

### Decision trees

**Zipformer model decision tree:**

```text
M0 result on val set:
  Stock 70M val WER ≤ 34.05% → fine-tuning is OPTIONAL; ship demo with stock
                                weights + Gemma correction. Save M1 budget.
  Stock 70M val WER 34.05–50% → fine-tune on cloud (M1) for the gain.
  Stock 70M val WER > 50%    → fine-tuning critical; M1 must succeed.
```

**Gemma model decision tree:**

```text
Day 0 result:
  26B-A4B MoE Q4_K_M loads + p95 latency <800ms → use Q4_K_M (best correction)
  Q4_K_M >800ms                                  → step down to Q3_K_M (~12.7GB)
  Q3_K_M >800ms                                  → step down to IQ3_XXS (~11.4GB)
  All 26B variants >800ms                        → use E2B Q4_K_M (~4GB)
  E2B Q4_K_M >800ms                              → skip Gemma in demo (Stage A only)
```

______________________________________________________________________

## System Architecture (Layer 1)

```text
┌────────────────────────────────────────────────────────────────┐
│ PYTHON DAEMON  (scripts/vox_daemon/)                            │
│                                                                 │
│  Microphone capture (sounddevice, 16kHz, float32 mono)          │
│       │ 100ms audio chunks                                      │
│       ▼                                                         │
│  Silero VAD pre-filter (silero-vad, neural)                     │
│  threshold=0.5, hangover=12 chunks (1.2s post-speech tail)      │
│  Drops silence chunks during idle; forwards speech + tail       │
│       │ filtered chunks (speech + post-speech silence)          │
│       ▼                                                         │
│  sherpa-onnx OnlineRecognizer (streaming Zipformer)             │
│  provider="cpu" or "coreml" (benchmark Day 0)                   │
│  --decode-chunk-len 32 (320ms encoder window)                   │
│  --rule1-min-trailing-silence 1.2 (endpoint at 1.2s pause)      │
│       │                                                          │
│       ├── partial transcript per chunk ─► consumer.on_stage_a() │
│       │                                                          │
│       └── on endpoint detected ─► flush, queue Gemma correction │
│                                                                  │
│       ▼ (concurrent, thread-based)                              │
│  Gemma 4 GGUF (llama.cpp subprocess.Popen)                      │
│  Threaded reader pumps stdout → queue.Queue                     │
│  timeout=1500ms; on timeout → emit Stage A as Stage B           │
│       │                                                          │
│       └── corrected text ─────────► consumer.on_stage_b()       │
│                                                                  │
│  TextConsumer (Protocol)                                         │
│    Layer 1:  WebConsumer   ──► aiohttp HTTP+WS ──► Chrome       │
│    Layer 2:  AXConsumer    ──► Swift injector ──► Live Captions │
└────────────────────────────────────────────────────────────────┘
       │ aiohttp WebSocket (localhost:8765)
       ▼
┌────────────────────────────────────────────────────────────────┐
│ CHROME FRONTEND  (static HTML, opened automatically)           │
│                                                                 │
│  Stage A (gray italic, partials):                               │
│  "I want to talk about kubernet... kubernetes uh the netw..."   │
│       ↓ replaced by Stage B on endpoint                          │
│  Stage B (white, committed):                                    │
│  "I want to talk about Kubernetes networking."                  │
│                                                                 │
│  History: last 5 committed utterances (scrollable)              │
└────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## Component Specs

### TextConsumer Protocol

File: `scripts/vox_daemon/consumer.py`

```python
from typing import Protocol

class TextConsumer(Protocol):
    def on_stage_a(self, text: str, uid: int) -> None: ...
    def on_stage_b(self, text: str, uid: int) -> None: ...
    def on_clear(self) -> None: ...
```

The `uid` (utterance ID) is an incrementing integer assigned by the daemon at
each endpoint boundary. Stage A *partials* during an utterance and the
corresponding Stage B all share the same uid. Consumers use uid to discard
stale Stage B messages if a new utterance has already started.

**Note on on_clear():** No-op in Phase 1. Stage A of the next utterance
overwrites. Defined in the Protocol for Phase 2 migration (bidirectional IPC
enables Python to detect Return press → call on_clear() → clear the field).

______________________________________________________________________

### WebConsumer (aiohttp HTTP+WS)

File: `scripts/vox_daemon/consumer.py`

aiohttp serves both static HTML and the WebSocket on `localhost:8765` — single
server, single port. The Python `websockets` library is WS-only, so we use
aiohttp instead.

**Wire format (same JSON structure used for Layer 2 IPC):**

```json
{"type": "stage_a", "text": "hello world",   "uid": 42}
{"type": "stage_b", "text": "Hello, world.", "uid": 42}
{"type": "clear"}
```

**Server skeleton:**

```python
from aiohttp import web

async def index(request: web.Request) -> web.Response:
    return web.FileResponse("scripts/vox_daemon/static/index.html")

async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    request.app["clients"].add(ws)
    try:
        async for _ in ws:
            pass  # client doesn't send anything in Phase 1
    finally:
        request.app["clients"].discard(ws)
    return ws

def make_app() -> web.Application:
    app = web.Application()
    app["clients"] = set()
    app.router.add_get("/", index)
    app.router.add_get("/ws", websocket_handler)
    return app

async def broadcast(app: web.Application, msg: dict) -> None:
    payload = json.dumps(msg)
    for ws in list(app["clients"]):
        await ws.send_str(payload)
```

**Startup sequence:**

1. `aiohttp.web.AppRunner` on `localhost:8765` (HTTP + WS on same port)
1. `subprocess.run(['open', 'http://localhost:8765'])` — opens in default browser
1. Enter ASR loop

**Client-side race guard (mirrors Layer 2 D3 + D6 decisions):**

```javascript
let lastUID = -1;
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'stage_a') {
    lastUID = msg.uid;
    showStageA(msg.text);   // gray italic, replaces previous
  } else if (msg.type === 'stage_b') {
    if (msg.uid !== lastUID) return;   // stale Stage B, discard
    showStageB(msg.text);   // white, committed
  } else if (msg.type === 'clear') {
    clearDisplay();
  }
};
```

______________________________________________________________________

### Silero VAD Pre-filter

File: `scripts/vox_daemon/silero_vad.py`

**Role: pre-filter, not segmenter.** Streaming Zipformer is always-on — it
transcribes whatever audio it receives. A neural VAD in front of the recognizer
prevents two failure modes:

1. **Silence hallucinations.** Without a pre-filter, sherpa-onnx runs the
   encoder on background noise (HVAC, keyboard, room tone) and can emit
   spurious tokens. VAD gates these out before they reach the recognizer.
1. **Compute waste during idle.** Skipping silence chunks frees the encoder
   for actual speech.

VAD does NOT replace sherpa-onnx's endpoint detection. Sherpa-onnx still
decides "this utterance is complete" via its rule-based logic on its own
input stream. VAD just controls *what audio* sherpa-onnx sees.

**Why Silero (vs webrtcvad).** Silero is a neural VAD trained on diverse
speech. For Deaf/atypical speech with non-standard prosody, Silero is more
robust than rule-based webrtcvad. The S1 codebase already uses webrtcvad
in `scripts/serving/vad.py` (untouched, Phase 2 path); S2 uses Silero.

```python
from silero_vad import load_silero_vad
import torch

class SileroVADPreFilter:
    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 1600  # 100ms at 16kHz

    def __init__(self, *, threshold: float = 0.5, hangover_chunks: int = 12):
        """
        threshold: speech probability cutoff (0.0–1.0). Higher = stricter.
        hangover_chunks: how many post-speech chunks to keep forwarding before
                        dropping back to silence-gated mode. 12 chunks = 1.2s,
                        which matches sherpa-onnx's rule1_min_trailing_silence.
                        This lets sherpa-onnx see enough trailing silence to
                        fire its own endpoint detection.
        """
        self.model = load_silero_vad()
        self.threshold = threshold
        self.hangover_chunks = hangover_chunks
        self._silence_count = hangover_chunks   # start in idle state

    def is_speech(self, chunk_float32) -> bool:
        """True if chunk contains speech.

        silero-vad v6 requires exactly 512 samples per call at 16kHz.
        Split the 1600-sample (100ms) chunk into 512-sample sub-chunks
        and return True if any sub-chunk exceeds the threshold.
        """
        chunk_t = torch.from_numpy(chunk_float32)
        probs = []
        for i in range(0, len(chunk_t), 512):
            sub = chunk_t[i : i + 512]
            if len(sub) < 512:
                sub = torch.nn.functional.pad(sub, (0, 512 - len(sub)))
            probs.append(self.model(sub, self.SAMPLE_RATE).item())
        return max(probs) > self.threshold

    def should_forward(self, chunk_float32) -> bool:
        """
        Returns True if this chunk should be fed to the ASR recognizer.
        Behavior:
          - Speech detected → forward, reset silence counter.
          - Silence + within hangover window → forward (let ASR see trailing
            silence for its own endpoint detection).
          - Silence + past hangover → drop (idle state).
        """
        if self.is_speech(chunk_float32):
            self._silence_count = 0
            return True
        self._silence_count += 1
        return self._silence_count <= self.hangover_chunks
```

**Why hangover = 12 chunks (1.2s).** This matches
`rule1_min_trailing_silence=1.2` so sherpa-onnx receives enough trailing
silence in its input to fire endpoint detection naturally. Shorter hangover
risks cutting off the endpoint signal; longer hangover wastes compute.

**Configurable via CLI:** `--vad-threshold` and `--vad-hangover-ms` flags.

______________________________________________________________________

### Sherpa-ONNX Streaming Zipformer Integration

File: `scripts/vox_daemon/zipformer_asr.py`

The streaming model has 3 ONNX files: encoder, decoder, joiner. All three plus
`tokens.txt` (BPE vocabulary) are required to instantiate `OnlineRecognizer`.

**Architecture note:** Streaming Zipformer is a Transducer (RNN-T) model. The
encoder uses chunked attention with `--decode-chunk-len 32` (320ms at 16kHz,
hop=10ms). Partial transcripts are produced after each chunk. Endpointing is
**rule-based**, not model-implicit: configurable trailing-silence and
utterance-length thresholds determine when a line is "complete."

```python
import sherpa_onnx
import numpy as np

def make_recognizer(model_dir: str, provider: str = "cpu") -> sherpa_onnx.OnlineRecognizer:
    """Load streaming Zipformer recognizer. model_dir contains 3 .onnx files + tokens.txt."""
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=f"{model_dir}/encoder-epoch-99-avg-1.onnx",
        decoder=f"{model_dir}/decoder-epoch-99-avg-1.onnx",
        joiner=f"{model_dir}/joiner-epoch-99-avg-1.onnx",
        tokens=f"{model_dir}/tokens.txt",
        provider=provider,           # "cpu" | "coreml"
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        # Endpoint detection (rule-based, all in seconds):
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=1.2,   # 1.2s pause = endpoint
        rule2_min_trailing_silence=0.8,   # 0.8s pause IF utterance >= rule2_min_utterance_length
        rule3_min_utterance_length=20.0,  # force endpoint at 20s utterance
    )

# Daemon main loop (simplified):
def asr_loop(recognizer, vad, mic_chunks, on_partial, on_final):
    stream = recognizer.create_stream()
    uid = 0
    last_text = ""
    for chunk_float32 in mic_chunks:                       # 100ms each
        # Pre-filter: drop pure-silence chunks during idle
        if not vad.should_forward(chunk_float32):
            continue
        stream.accept_waveform(16000, chunk_float32)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        text = recognizer.get_result(stream)
        if text != last_text:
            on_partial(text, uid)                           # Stage A update
            last_text = text
        if recognizer.is_endpoint(stream):
            on_final(text, uid)                             # queue Gemma + Stage B
            recognizer.reset(stream)
            uid += 1
            last_text = ""
```

**Why provider="cpu" by default:** sherpa-onnx issue #2910 reports CoreML
slower than CPU on M2 for some models. Day 0 should benchmark both providers
on M4 Pro and pick the winner. CPU is the stable fallback.

**Audio resampling:** S1 data is 44.1kHz. Mic capture is 16kHz (sounddevice
default for ASR). For M0 eval against S1 val/test, resample 44.1kHz→16kHz via
`librosa.resample` once at manifest-load time. For runtime, mic feeds 16kHz
directly — no resampling.

**Error handling:**

- sherpa-onnx exception → log stderr + skip chunk, continue
- ONNX load failure at startup → fail fast with clear error
- Empty result during a partial chunk → suppress on_partial call (no Stage A update)

______________________________________________________________________

### Gemma 4 GGUF Integration

File: `scripts/vox_daemon/gemma.py`

Runs as a persistent subprocess (`subprocess.Popen`). Python communicates via
stdin/stdout (one correction request per line).

```text
┌──────────────────────────────────────────────────────────────┐
│ Concurrency model                                             │
│                                                               │
│ ASR thread                    Gemma worker thread            │
│   endpoint detected ─────────► queue.put(uid, stage_a_text)  │
│   emit Stage A immediately     worker.get() → Gemma prompt   │
│   continue capturing            Gemma stdin.write            │
│                                 reader_q.get(timeout=1.5)    │
│                                 → emit Stage B               │
└──────────────────────────────────────────────────────────────┘
```

**Gemma prompt template:**

```text
Correct the following speech transcript. Fix ASR errors, grammar, and
disfluencies. Output ONLY the corrected text. No explanation.

Input: {stage_a_text}
Output:
```

**Timeout handling.** `subprocess.Popen.stdout.readline()` is blocking and
does not accept a `timeout` kwarg. Use a threaded reader that drains stdout
into a `queue.Queue`, then `get(timeout=...)` on the main thread. The Gemma
worker is serialized (one prompt → one response → next prompt), so a
single-slot queue suffices.

```python
import queue, threading

# At daemon startup, after spawning gemma_proc:
gemma_q: queue.Queue[str] = queue.Queue()

def _gemma_reader() -> None:
    for line in iter(gemma_proc.stdout.readline, ""):
        gemma_q.put(line)

threading.Thread(target=_gemma_reader, daemon=True).start()

# Per correction request (Gemma worker thread):
gemma_proc.stdin.write(prompt + "\n")
gemma_proc.stdin.flush()
try:
    result = gemma_q.get(timeout=1.5)
except queue.Empty:
    log.warning("Gemma timeout on uid=%d — emitting Stage A as Stage B", uid)
    result = stage_a_text   # fallback: Stage A is the committed result
```

**Process failure:**

```python
# Monitored in background thread
if gemma_proc.poll() is not None:   # process died
    if restart_count < 3:
        gemma_proc = respawn_gemma()
        restart_count += 1
    else:
        log.error("Gemma restart limit reached — Stage A only for session")
        gemma_disabled = True
```

**Model selection (resolve on Day 0):**

- Primary: Gemma 4 26B-A4B MoE Q4_K_M (~16.9 GB) — best correction quality
- Step-down if Q4_K_M latency >800ms: Q3_K_M (~12.7 GB) or IQ3_XXS (~11.4 GB)
- Final fallback: Gemma 4 E2B Q4_K_M (~4 GB) — always within latency budget
- CLI flag: `--gemma-model path/to/model.gguf`

______________________________________________________________________

### Chrome Frontend

File: `scripts/vox_daemon/static/index.html`

Single-file HTML. No build step. No dependencies beyond browser WebSocket API.

```text
┌──────────────────────────────────────────────────────────────┐
│  VOX Personalis                              ● live           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Current utterance                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Stage A → "I want to talk about kubernet..."           │  │
│  │           (updates word-by-word as user speaks)         │  │
│  │ Stage B → "I want to talk about Kubernetes."           │  │
│  │           (replaces Stage A on endpoint)                │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  History                                                      │
│  ─────────────────────────────────────────────────────────   │
│  "Can you tell me about your experience with Kubernetes?"    │
│  "Yes, I have three years working with Kubernetes."          │
│  "We use it for container orchestration at my company."      │
└──────────────────────────────────────────────────────────────┘
```

- **Stage A styling:** `color: #888; font-style: italic` — signals "processing"
- **Stage B styling:** `color: #fff; font-weight: 600` — signals "committed"
- **Transition:** Stage A → Stage B uses a 150ms CSS color transition

______________________________________________________________________

### Daemon CLI

File: `scripts/vox_daemon/cli.py`

```bash
python -m scripts.vox_daemon \
  --zipformer-model ./out/zipformer_finetune \
  --gemma-model ./models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --provider cpu \
  --decode-chunk-len 32 \
  --rule1-min-trailing-silence 1.2 \
  --vad-threshold 0.5 \
  --vad-hangover-ms 1200 \
  --port 8765 \
  --debug            # prints Stage A and Stage B updates to stdout
```

`--zipformer-model` is a directory containing `encoder-*.onnx`,
`decoder-*.onnx`, `joiner-*.onnx`, and `tokens.txt`. After M1 fine-tuning,
this points to the local download of the cloud-trained ONNX export. Default
during M0: `./models/sherpa-onnx-streaming-zipformer-en-2023-06-21/` (stock).

`--consumer` flag (`web` | `ax`) is added in Phase C (post-demo). Phase 1
defaults to `web` and is hardcoded.

**Startup sequence:**

1. Load Zipformer recognizer via sherpa-onnx (fails fast if model files missing)
1. Load Silero VAD pre-filter
1. Spawn Gemma subprocess, verify it responds to a test prompt
1. Start aiohttp HTTP+WS server on `--port`
1. `subprocess.run(['open', 'http://localhost:{port}'])` (macOS)
1. Start mic capture loop (sounddevice → VAD pre-filter → recognizer.accept_waveform)

______________________________________________________________________

## Module Structure

```text
scripts/
  serving/
    vad.py               ← UNTOUCHED (webrtcvad, S1 code — not used in S2)
    api.py               ← UNTOUCHED (FastAPI, Phase 2)
  baseline_eval/
    normalization.py     ← REUSE: create_normalizer(version=2) (no changes)
  feedback_finetune/
    manifest.py          ← REUSE: imported for feedback corrections (no changes)
  vox_daemon/            ← NEW: Phase B daemon
    __main__.py
    cli.py
    consumer.py          # TextConsumer Protocol + WebConsumer (aiohttp)
    silero_vad.py        # Silero VAD pre-filter (gates audio before ASR)
    zipformer_asr.py     # sherpa-onnx OnlineRecognizer wrapper
    gemma.py             # Gemma 4 GGUF subprocess + threaded reader
    static/
      index.html         # Chrome frontend (single file)
  zipformer_eval/        ← NEW: M0 + M2 evaluation
    __main__.py
    cli.py
    eval.py              # WER eval — uses sherpa-onnx OfflineRecognizer for batch eval
                         # Reuses create_normalizer(version=2) from baseline_eval
  zipformer_finetune/    ← NEW: M1 fine-tuning helpers (run mostly on cloud)
    __main__.py
    cli.py
    manifest_to_lhotse.py  # Convert S1 CSV manifest → lhotse cuts.jsonl
    cloud_run_template.sh  # Lambda Labs A10 setup + finetune.py invocation
    export_onnx.py         # Run after fine-tune: TorchScript → ONNX (3 files)
```

**Existing code policy:** All S1 code under `scripts/` remains untouched. S2
creates new modules alongside. Shared utilities (`create_normalizer`,
`feedback_finetune.manifest`) are imported, not copied or modified.

______________________________________________________________________

## Dataset

**Source:** Project Euphonia takeout export at `~/Downloads/takeout-E407/`
(3,912 wav clips + `listing.csv` manifest). 44.1kHz mono 16-bit.

**S1 frozen splits** (`results/M1_dataset_v1/dataset_v1_summary.json`):

| Split | Clips | Duration |
| ----- | ----- | -------- |
| Train | 2,897 | 3.74h    |
| Val   | 361   | 0.46h    |
| Test  | 365   | 0.48h    |

**S1-M7 train set** (`results/M7_feedback_finetune/merged_manifest.csv`):
2,897 train + 108 feedback corrections = 3,005 rows / 4.02h. **This is the
training set for S2-M1.**

**S1 manifest CSV columns** (the canonical loader is
`scripts/fine_tuning/data.py:36-44`):

```text
file_name, audio_path_resolved, transcript_raw, split,
duration_sec, duration_bin, pair_sha256
```

Plus audit columns (timestamp_ms, recording_device, audio_sha256, etc.) that
are loaded but not required.

**Splits are frozen.** Use exact same train/val/test as S1 for apples-to-apples
WER comparison against the 34.05% baseline.

______________________________________________________________________

## Cloud Fine-tuning Workflow

End-to-end workflow for M1. Total wall-clock: ~6–8h. Total cost: ~$10–20.

### Step 1 — Local prep (Mac mini)

```bash
# Convert S1 CSV manifest → lhotse cuts.jsonl format
python -m scripts.zipformer_finetune.manifest_to_lhotse \
    --input  ./results/M7_feedback_finetune/merged_manifest.csv \
    --audio-root ~/Downloads/takeout-E407 \
    --output ./out/lhotse_manifests/

# Output: cuts_train.jsonl.gz, cuts_dev.jsonl.gz, cuts_test.jsonl.gz
# (lhotse handles 44.1→16kHz resampling on-the-fly during feature extraction)

# Tarball training data + manifests for cloud upload
tar czf voice_data.tar.gz \
    ~/Downloads/takeout-E407/ \
    ./out/lhotse_manifests/
# Estimated size: ~1.5GB (audio) + manifests
```

### Step 2 — Rent Lambda Labs A10

```bash
# Sign up at lambdalabs.com, launch a "1× A10 24GB" instance.
# Verified specs (as of 2026-05): $1.29/hr, 226 GiB RAM, 1.3 TiB SSD,
# 1× NVIDIA A10 24GB, Ubuntu 22 with PyTorch + CUDA preinstalled via Lambda Stack.
# Confirm exact CUDA/PyTorch versions via SSH:
ssh ubuntu@<A10-INSTANCE-IP>
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Step 3 — Cloud setup (one-time per instance)

```bash
# On A10. Install k2 matching the preinstalled torch + CUDA.
# Check available wheels at https://k2-fsa.github.io/k2/cuda.html and pick
# the closest match. Example assumes torch 2.11 + CUDA 12.6:
pip install k2==1.24.4.dev20260423+cuda12.6.torch2.11.0 \
    -f https://k2-fsa.github.io/k2/cuda.html
pip install lhotse kaldifeat
git clone https://github.com/k2-fsa/icefall.git
cd icefall && pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH

# Download base PyTorch checkpoint (Day 0 picked Candidate A or B):
# Candidate A: marcoyang/icefall-libri-giga-pruned-transducer-stateless7-streaming-2023-04-04
# Candidate B: Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17
huggingface-cli download <CHOSEN-CANDIDATE> --local-dir ./base_ckpt
ls ./base_ckpt/exp/pretrained.pt   # must exist
```

### Step 4 — Upload data + fine-tune

```bash
# From local Mac mini: upload tarball to A10
rsync -avzP voice_data.tar.gz ubuntu@<A10-IP>:~/

# On A10: extract + run fine-tune
tar xzf voice_data.tar.gz
cd icefall/egs/librispeech/ASR

# Symlink data into icefall's expected location
ln -s ~/voice_data/lhotse_manifests data/fbank
# (lhotse pre-computed features go here)

# Run the vanilla zipformer fine-tune recipe with --causal 1 for streaming.
# This is a FULL fine-tune (no PEFT). All encoder weights are trainable.
python ./zipformer/finetune.py \
    --world-size 1 \
    --num-epochs 20 \
    --start-epoch 1 \
    --use-fp16 1 \
    --base-lr 0.0045 \
    --do-finetune 1 \
    --finetune-ckpt ~/base_ckpt/exp/pretrained.pt \
    --causal 1 \
    --chunk-size 32 \
    --left-context-frames 128 \
    --max-duration 1000 \
    --bpe-model ~/base_ckpt/data/lang_bpe_500/bpe.model \
    --exp-dir ./zipformer/exp_finetune
# Wall-clock estimate: ~5-7h on a single A10 for 20 epochs / 4h training data
# (full fine-tune is heavier than adapter would be).
# Monitor with `tail -f zipformer/exp_finetune/log/log-train-*`.
# Early-stop if val WER stalls.

# If state dict load fails on Candidate A: retry with --init-modules:
#   --init-modules encoder
# This loads only matching keys and re-initializes the rest.
```

### Step 5 — Export ONNX

```bash
# On A10, after training:
python ./zipformer/export-onnx-streaming.py \
    --exp-dir ./zipformer/exp_finetune \
    --epoch 20 --avg 1 \
    --tokens ~/base_ckpt/data/lang_bpe_500/tokens.txt \
    --use-averaged-model 1
# Note: Streaming params (--causal, --chunk-size, --left-context-frames) are
# read from the checkpoint's saved config — not passed on the command line.
# Outputs: encoder-epoch-20-avg-1.onnx, decoder-*, joiner-*

# Tarball ONNX exports + tokens + frozen config
tar czf zipformer_finetune_v1.tar.gz \
    ./zipformer/exp_finetune/encoder-*.onnx \
    ./zipformer/exp_finetune/decoder-*.onnx \
    ./zipformer/exp_finetune/joiner-*.onnx \
    ~/base_ckpt/data/lang_bpe_500/tokens.txt \
    ./zipformer/exp_finetune/frozen_config.json
```

### Step 6 — Download to local + smoke-test

```bash
# From local: pull tarball
rsync -avzP ubuntu@<A10-IP>:~/icefall/egs/librispeech/ASR/zipformer_finetune_v1.tar.gz ./out/

mkdir -p ./out/zipformer_finetune
tar xzf ./out/zipformer_finetune_v1.tar.gz -C ./out/zipformer_finetune
# Smoke-test on a val clip (same script as Day 0 step 3, swap the model path)

# Terminate Lambda Labs instance (stop billing)
```

### Frozen config pattern

M1 must output a `frozen_config.json` alongside the ONNX files for
reproducibility (matching S1 pattern from `scripts/fine_tuning/reporting.py`).
Records: base checkpoint ID, all training hyperparameters (--causal,
--chunk-size, --base-lr, etc.), dataset manifest, git commit hash. Critical
for matching S1's M3 reproducibility discipline.

______________________________________________________________________

## Evaluation Specs

### M0: Stock Zipformer Baseline WER

Run stock streaming Zipformer 70M on S1 frozen val + test splits. No
fine-tuning. The output number determines whether M1 fine-tuning is necessary
or optional.

```bash
python -m scripts.zipformer_eval \
    --manifest ./results/M1_dataset_v1/dataset_v1_<id>/manifest.csv \
    --model-dir ./models/sherpa-onnx-streaming-zipformer-en-2023-06-21 \
    --provider cpu \
    --norm-version 2 \
    --output ./out/M0_zipformer_baseline/
```

**Implementation notes:**

- The streaming Zipformer ONNX files (encoder/decoder/joiner) have chunk-state
  cache inputs that `OfflineRecognizer` does NOT supply. Use
  `OnlineRecognizer.from_transducer(...)` for batch eval as well, feeding each
  full clip via `accept_waveform(...)` followed by `input_finished()` and
  draining `decode_stream(...)` until exhausted. This is the documented and
  supported pattern for batch evaluation of streaming ONNX models.
- `--norm-version 2` maps internally to `create_normalizer(version=2)` from
  `scripts/baseline_eval/normalization.py`. This is the same normalization
  applied to S1-M7's 34.05% number.
- For WER computation, use `jiwer` (already in pyproject.toml).

**Output (per split):**

- WER on val + test (compare to Whisper OOB and fine-tuned Whisper 34.05%)
- Per-clip predictions saved as CSV (for error analysis)
- p50 / p95 latency per chunk on M4 Pro (real-time factor measurement)

**Decision:**

```text
M0 result on val:
  Stock 70M val WER ≤ 34.05% → M1 OPTIONAL — ship demo with stock + Gemma.
  Stock 70M val WER 34.05–50% → M1 RECOMMENDED — fine-tune for the gain.
  Stock 70M val WER > 50%    → M1 CRITICAL — fine-tune or swap to Whisper-LoRA fallback.
```

### M1: Zipformer Streaming Fine-tuning (Cloud)

See [Cloud Fine-tuning Workflow](#cloud-fine-tuning-workflow) above for the
detailed steps. This subsection captures the gate and config.

**Hyperparameters (icefall vanilla zipformer/finetune.py + --causal):**

| Parameter               | Value         | Source                                            |
| ----------------------- | ------------- | ------------------------------------------------- |
| `--causal`              | 1             | Required for streaming export                     |
| `--chunk-size`          | 32            | 320ms encoder chunk (32 × 10ms hop)               |
| `--left-context-frames` | 128           | 1.28s left context per chunk                      |
| `--base-lr`             | 0.0045        | icefall recipe doc (1/10 of from-scratch)         |
| `--num-epochs`          | 20            | icefall recipe default                            |
| `--max-duration`        | 1000 (frames) | A10 24GB memory headroom                          |
| `--use-fp16`            | 1             | A10 supports FP16 well                            |
| `--do-finetune`         | 1             | Enables fine-tune mode                            |
| Trainable params        | All (~70M)    | Full fine-tune; no PEFT in icefall streaming path |

**Gate:** val WER < 34.05% (beats fine-tuned Whisper baseline).

**Overfitting risk.** Full fine-tune of a 70M model on 4h of data is data-hungry
relative to LoRA. Monitor val loss every epoch; if val degrades while train
loss drops, early-stop and reduce LR or epochs. The 4h training set is at the
small end of where full Zipformer fine-tune has been demonstrated.

**If M1 gate fails after 20 epochs:**

1. Check val loss curve — if still improving, extend to 30 epochs with
   `--start-epoch 21 --num-epochs 30`.
1. Try `--base-lr 0.0010` (lower) — may help if val loss is unstable.
1. Try `--init-modules encoder` to selectively load only encoder weights from
   the base ckpt, leaving decoder + joiner randomly initialized (or vice versa).
1. If still failing: fall back to S1 Whisper-LoRA pipeline (chunky-but-adapted),
   reframe Phase B as "show streaming UX with stock Zipformer + best-effort
   Gemma correction."

**Output:**

- `./out/zipformer_finetune/{encoder,decoder,joiner}.onnx`
- `./out/zipformer_finetune/tokens.txt`
- `./out/zipformer_finetune/frozen_config.json`
- val WER + test WER reported in `./out/M1_zipformer_finetune/metrics.json`

### M2: Gemma 4 Correction Evaluation

Run fine-tuned Zipformer on val set. For each transcription, run Gemma
correction. Measure:

1. WER of raw Zipformer output (Stage A baseline)
1. WER of Gemma-corrected output (Stage B)
1. False correction rate (cases where Gemma made it worse)

```bash
python -m scripts.zipformer_eval \
    --manifest <PATH_TO_VAL_MANIFEST> \
    --model-dir ./out/zipformer_finetune \
    --gemma-model ./models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
    --norm-version 2 \
    --output ./out/M2_zipformer_gemma_eval/
```

**Gate:** Stage B WER ≤ Stage A WER AND false correction rate < 15%.

If Stage B WER > Stage A WER: skip Gemma in demo.\
If false correction rate > 15%: tighten the Gemma prompt (more conservative).

______________________________________________________________________

## Error / Rescue Registry (Layer 1)

| Error                                  | Rescue                                                       |
| -------------------------------------- | ------------------------------------------------------------ |
| sherpa-onnx exception (any)            | Skip chunk, log stderr, continue                             |
| ONNX model file missing at startup     | Fail fast with clear error                                   |
| Empty Stage A partial                  | Suppress on_partial call (no Stage A update)                 |
| Gemma timeout (>1500ms)                | Emit Stage A text as Stage B                                 |
| Gemma OOM / crash                      | Auto-restart up to 3 times; Stage A only after limit         |
| Gemma malformed output (empty/garbled) | Emit Stage A as Stage B fallback                             |
| WebSocket client disconnected          | Continue broadcasting; reconnect on next page load           |
| Chrome not installed                   | Log warning; user opens <http://localhost:8765> manually     |
| Port 8765 in use                       | CLI error on startup: "Port 8765 in use — use --port N"      |
| CoreML provider slower than CPU        | Fall back to provider="cpu" (Day 0 benchmark guides default) |

______________________________________________________________________

## Success Criteria (5-Day Demo)

| Metric                       | Target                                                           | Milestone |
| ---------------------------- | ---------------------------------------------------------------- | --------- |
| Stock Zipformer val WER      | Measured and recorded                                            | M0        |
| Fine-tuned Zipformer val WER | < 34.05% (beats fine-tuned Whisper)                              | M1        |
| Stage B WER vs Stage A       | Stage B ≤ Stage A                                                | M2        |
| Stage A first-token latency  | < 600ms from speech onset (320ms chunk + lookahead + scheduling) | M3        |
| Stage A update cadence       | Partial transcript updates ~3× per second (every 320ms chunk)    | M3        |
| Stage B latency              | Stage B replaces Stage A within 800ms of endpoint                | M3        |
| Demo stability               | Runs for 10 min without crash                                    | M3        |

______________________________________________________________________

## Layer 2: AX Integration Stub (Post-demo)

All engineering decisions from the planning session are captured here.
Implementation begins after M3 demo is validated.

### Decisions Made (D1–D7)

| #   | Decision              | Choice                                                                        |
| --- | --------------------- | ----------------------------------------------------------------------------- |
| D1  | IPC wire format       | Newline-delimited JSON: `{"type":"stage_a","text":"...","uid":42}\n`          |
| D2  | Daemon lifecycle      | Python daemon spawns Swift injector as subprocess (owns lifecycle)            |
| D3  | Stage B race (Return) | Swift reads field before Stage B write — discard if field is empty            |
| D4  | on_clear() Phase 1    | No-op — Stage A of next utterance overwrites; defined in Protocol for Phase 2 |
| D5  | Tests                 | Deferred to Phase 2 (MVP 1 POC is the pass/fail gate)                         |
| D6  | Utterance ID          | `uid` in every IPC message; Swift discards Stage B if uid != last Stage A uid |
| D7  | Injector crash        | Auto-restart up to 3 times; after limit, fall to ASR-only mode                |

### IPC Wire Format

Same JSON structure as WebSocket consumer. Newline-delimited, UTF-8.
Max message size: ~500 bytes.\
Named pipe path: `/tmp/vox_personalis_${UID}.pipe`

### Swift AX Injector Contract

```swift
func applyStageA(text: String, uid: Int) {
    lastWrittenUID = uid
    writeFieldValue(text)  // unconditional — partials overwrite freely
}

func applyStageB(text: String, uid: Int) {
    guard uid == lastWrittenUID else { return }      // D6: stale guard
    guard !readFieldValue().isEmpty else { return }  // D3: Return race guard
    writeFieldValue(text)
}
```

### Swap to AX

TextConsumer Protocol means swapping WebConsumer → AXConsumer is a config flag
(added in Phase C):

```bash
python -m scripts.vox_daemon --consumer ax    # Phase C
python -m scripts.vox_daemon --consumer web   # Phase B (default)
```

### MVP 1 Pass Criterion

Swift CLI injects a hardcoded string into the Live Captions Type to Speak field
without manual focus intervention, on macOS 14.0+. If this fails: Chrome
fallback (WebConsumer) remains the production path; AX investigation continues
in parallel.

______________________________________________________________________

## Pre-Implementation Blockers

| #   | Blocker                                                                                                              | When     | Owner                                |
| --- | -------------------------------------------------------------------------------------------------------------------- | -------- | ------------------------------------ |
| 1   | sherpa-onnx loads + runs streaming Zipformer on M4 Pro arm64                                                         | Day 0    | Verify with pip install + smoke test |
| 2   | Stock Zipformer 70M ONNX downloads cleanly from HF                                                                   | Day 0    | huggingface-cli download             |
| 3   | aiohttp HTTP+WS server on port 8765                                                                                  | Day 0    | pip install + minimal test           |
| 3b  | silero-vad loads + runs on 100ms chunks at 16kHz                                                                     | Day 0    | pip install + smoke test             |
| 4   | Gemma 4 GGUF builds on macOS 14 via llama.cpp + Metal                                                                | Day 0    | cmake + Metal build                  |
| 5   | Gemma 4 p95 latency \<800ms on M4 Pro (26B-A4B or E2B)                                                               | Day 0    | Benchmark before M3                  |
| 6   | PyTorch source checkpoint A or B chosen via smoke-test (state dict loads cleanly into vanilla zipformer/finetune.py) | Day 0    | See Day 0 step 9                     |
| 7   | Lambda Labs A10 access + payment method on file                                                                      | M1 start | Sign up; ~$5 credit for first run    |
| 8   | k2 + icefall install on Lambda Labs A10 (CUDA 12.1, PyTorch 2.1)                                                     | M1 start | Follow icefall install doc           |
| 9   | lhotse manifest conversion script handles S1 CSV format                                                              | M1 prep  | Implement + test on local            |

**Dependency summary:**

```text
# Local pyproject.toml additions:
sherpa-onnx>=1.13.0      # streaming ASR runtime
huggingface-hub>=0.20    # model downloads
aiohttp>=3.9             # HTTP+WebSocket server
sounddevice>=0.4         # mic capture
silero-vad>=6.0          # neural VAD pre-filter (gates audio before ASR; v6 requires 512-sample chunks)
lhotse>=1.20             # manifest conversion (M1 prep only)

# Already in pyproject.toml:
torch>=2.0, peft>=0.4    # (peft no longer needed for S2 but stays for S1 compat)
jiwer>=3.0               # WER computation
soundfile, librosa       # audio I/O + resampling

# Cloud-only (Lambda Labs A10):
k2 (CUDA 12.1 wheel from k2-fsa.github.io)
icefall (git clone + requirements.txt)
```

______________________________________________________________________

## Open Questions / Risks

1. **Full fine-tune on 4h Deaf-accent data is data-hungry.** icefall full
   fine-tune recipes target larger datasets; 4h is at the small end. Without
   the parameter-efficient regime (icefall has no streaming-PEFT recipe today),
   overfitting risk is real. Mitigation: M0-first (stock baseline) gives a
   clean comparison point; if stock WER is already \<34.05% on Deaf accent,
   M1 can be skipped entirely.

1. **Base checkpoint state-dict compatibility.** Candidate A
   (`marcoyang/...-2023-04-04`) is the 70M LibriSpeech+GigaSpeech model that
   matches inference, but was trained under the older
   `pruned_transducer_stateless7_streaming_multi` recipe. The new
   `zipformer/finetune.py` may or may not load it cleanly. Day 0 smoke-test
   resolves; fall back to Candidate B (66M, recipe-matched) if A fails.

1. **CoreML vs CPU on M4 Pro — known performance variance.** sherpa-onnx
   issue #2910 reports CoreML slower than CPU on some M2 configs. M4 Pro is
   newer; benchmark required.

1. **Latency target of \<500ms first-token-on-screen is tight.** Budget:
   320ms encoder chunk + right-context lookahead + sherpa-onnx scheduling +
   WS hop. Measured first-output latency in production reports is typically
   400–600ms on CPU. Worth measuring early in M3; relax target to 600ms if
   the streaming model needs a longer right-padding window.

1. **Pause/silence tuning across VAD and endpoint detection.** Two parameters
   work together and must be tuned together for the user's natural pause
   patterns:

   - **Silero VAD `threshold`** (default `0.5`) gates whether audio reaches
     the recognizer. Too high → misses soft speech onsets and clips sustained
     quiet vocalizations. Too low → forwards background noise to the recognizer.
   - **sherpa-onnx `rule1_min_trailing_silence`** (default `1.2s`) decides
     when an utterance ends and Gemma correction fires. Too low → premature
     endpoints on natural mid-sentence pauses. Too high → laggy Stage B.

   The VAD hangover window (`1.2s`) deliberately matches
   `rule1_min_trailing_silence` so the recognizer sees enough trailing silence
   to fire its own endpoint naturally. If you change one, change the other.
   Phase 1 ships with defaults; M3 stability testing tunes both against the
   user's actual speech.
