# VOX Personalis Stage 2 — WhisperLiveKit + Whisper LoRA + Gemma 4 Pipeline

**Status:** ACTIVE\
**Branch:** main\
**Supersedes:** [S2-sherpa-zipformer-gemma-pipeline.md](S2-sherpa-zipformer-gemma-pipeline.md)
(streaming Zipformer fine-tune produced alignment collapse on 3.7h of personalization
data — see that spec's postmortem), [S2-moonshine-gemma-pipeline.md](S2-moonshine-gemma-pipeline.md)
(Moonshine v2 path — abandoned because HF Transformers Moonshine does not expose true
streaming inference), S1-M7 (Whisper fine-tuned baseline, 34.05% val WER — reused as the
ASR model in this pipeline rather than replaced).

______________________________________________________________________

## Table of Contents

- [Problem](#problem)
- [Why WhisperLiveKit (vs Streaming Zipformer, vs Whisper VAD-batch)](#why-whisperlivekit-vs-streaming-zipformer-vs-whisper-vad-batch)
- [Core Product Flow](#core-product-flow)
- [Constraints](#constraints)
- [10x Vision](#10x-vision)
- [Hardware](#hardware)
- [What Is Out of Scope (Phase 1)](#what-is-out-of-scope-phase-1)
- [Accepted Phase 1 Limitations](#accepted-phase-1-limitations)
- [Baseline to Reuse](#baseline-to-reuse)
- [Objective](#objective)
- [Naming Conventions](#naming-conventions)
- [Milestone Sequence](#milestone-sequence)
- [Pre-work: Day 0 Checklist](#pre-work-day-0-checklist)
- [System Architecture (Layer 1)](#system-architecture-layer-1)
- [Component Specs](#component-specs)
  - [LoRA Merge (one-time prep)](#lora-merge-one-time-prep)
  - [WhisperLiveKit Serving](#whisperlivekit-serving)
  - [Gemma 4 GGUF Integration](#gemma-4-gguf-integration)
  - [Stage A / Stage B Proxy](#stage-a--stage-b-proxy)
  - [Chrome Frontend](#chrome-frontend)
  - [Daemon CLI](#daemon-cli)
- [Module Structure](#module-structure)
- [Dataset](#dataset)
- [Evaluation Specs](#evaluation-specs)
- [Error / Rescue Registry (Layer 1)](#error--rescue-registry-layer-1)
- [Success Criteria (Demo Day)](#success-criteria-demo-day)
- [Layer 2: AX Integration Stub (Post-demo)](#layer-2-ax-integration-stub-post-demo)
- [Pre-Implementation Blockers](#pre-implementation-blockers)
- [Open Questions / Risks](#open-questions--risks)

______________________________________________________________________

## Problem

The user has a Deaf-accent ASR demo for a hiring manager screen on Monday morning
(2026-05-11). The prior streaming-Zipformer plan failed; the previously demoed S1-M7
pipeline (Whisper LoRA at 34.05% val WER) is offline-batch only, which feels laggy.

We need: a **live, word-by-word display** while the user speaks; a **Gemma 4 polish layer**
on each committed utterance; **fully local inference** at runtime; and **shipable in
under 36 hours** with no fresh training. The S1-M7 LoRA already exists — we reuse it.

This is not a captioning app. The end use case — the Deaf speaker using their own voice
in a live job interview — is the motivation. Stage 2 work is: take the S1 model that
already works offline, give it streaming UX, polish with Gemma, and put it in front of a
real user.

______________________________________________________________________

## Why WhisperLiveKit (vs Streaming Zipformer, vs Whisper VAD-batch)

Three approaches to a "live" Whisper experience were compared:

| Property                        | Whisper VAD-batch (S1 path)                      | Streaming Zipformer (failed)               | **WhisperLiveKit (this spec)**                            |
| ------------------------------- | ------------------------------------------------ | ------------------------------------------ | --------------------------------------------------------- |
| Streaming UX                    | After full VAD pause + encoder pass (3–10s)      | 320ms partials mid-utterance               | **600–1500ms commit cadence, partials in flight**         |
| Architecture                    | Whisper offline                                  | True chunked streaming                     | Whisper offline + AlignAtt sliding-window emission policy |
| Personalization for Deaf accent | LoRA on whisper-small.en (proven 34.05% val WER) | Full fine-tune (failed at 3.7h scale)      | **Same proven LoRA, merged into base**                    |
| Inference latency on M4 Pro     | ~real-time but blocks until pause                | ~real-time true-streaming                  | Real-time with sliding window, MLX-accelerated            |
| Apple Silicon path              | PyTorch / CTranslate2                            | sherpa-onnx                                | **MLX (`mlx-whisper`) → 4–6× speedup**                    |
| Open source maturity            | Mature                                           | icefall PEFT non-streaming-only (dead end) | **Mature; ships WebSocket frontend**                      |
| Time to ship                    | n/a (already exists)                             | 5+ days, high risk                         | **~36 hours**                                             |

WhisperLiveKit uses the AlignAtt streaming policy
([Papi et al., Interspeech 2023](https://arxiv.org/abs/2305.11408)): it monitors decoder
cross-attention to identify when each token is "stable enough" to commit, and holds
unstable tokens in a "buffer" tier. The user sees grey/italic in-flight text turning into
white/committed text on a sub-second cadence. Sub-second word-by-word feedback is
acceptable for the user's use case (live self-monitoring during speech is tolerant of
~1s latency; a Deaf speaker monitoring articulation does not need 320ms cadence).

Two orthogonal CLI choices in `wlk` (verify exact flag names with `wlk --help` in M2):
the **inference backend** selects which Whisper runtime to use (e.g.,
`mlx-whisper`, `faster-whisper`); the **streaming policy** selects AlignAtt
SimulStreaming. We want `mlx-whisper` + AlignAtt. The defaults may already be AlignAtt;
M2 confirms.

**The killer property:** WhisperLiveKit accepts a merged Whisper checkpoint as a regular
HF model. Our S1-M7 LoRA can be merged once and served with no special handling — we get
LoRA accuracy and MLX speed simultaneously. (The `--lora-path` flag is documented as
PyTorch-backend-only and would lose MLX speed; merging is the right move.)

______________________________________________________________________

## Core Product Flow

```text
User speaks
  → Mic capture (browser → WhisperLiveKit WebSocket → Whisper-small.en LoRA-merged on MLX)
      ↓ AlignAtt emits {committed_text, buffer_text} every ~500ms-1s
  → Stage A: word-by-word display
      • dark/white "committed" text (lags spoken word by 0.6-1.5s)
      • grey/italic "buffer" text (in-flight hypothesis, may revise)
      ↓ on VAD endpoint detection (WhisperLiveKit's built-in segmenter)
  → daemon proxy fires Gemma 4 correction on the committed line
      ↓ Gemma returns within ~400ms (warm)
  → Stage B: polished text replaces Stage A line (white, bold, committed)
  → display (Phase 1: Chrome tab / Phase C: Live Captions Type to Speak)
  → user reviews, submits
```

**Why partial Stage A?** The user gets immediate visual feedback as they speak. No
waiting for VAD pause. No "did the system hear me?" anxiety.

**Why Stage B at all?** Whisper's streaming output sometimes mis-recognizes mid-utterance
context, and lacks ASR-style punctuation. Gemma 4 fixes errors after the line finalizes:
correcting accent-driven misrecognitions, restoring punctuation, fixing disfluencies.
Stage B replaces Stage A within ~400ms of endpoint detection.

**Concrete trace** (modeled from AlignAtt + Simul-Whisper paper latencies + WhisperLiveKit
source — published frame-level traces do not exist; numbers are estimates verified by
M2/M3 measurements):

```text
t=0.00s   user begins speaking "I want to talk about Kubernetes networking"
t=0.40s   buffer "I"                       committed ""
t=0.80s   buffer "want"                    committed "I"
t=1.20s   buffer "to talk"                 committed "I want"
t=1.60s   buffer "talk about"              committed "I want to"
t=2.00s   buffer "about Kubernetes"        committed "I want to talk"
t=2.40s   user stops speaking; VAD endpoint fires after 1.2s trailing silence
t=2.80s   buffer flushes; committed = "I want to talk about Kubernetes networking"
t=3.20s   Gemma returns Stage B
          display: "I want to talk about Kubernetes networking."
                   (added punctuation, fixed any ASR errors)
```

End-to-end wait after speech ends: **~1.2-2.0s** (Whisper commit) + **~0.4s** (Gemma) =
**~1.6-2.4s** for the polished final text.

**Why "text consumer agnostic"?** The Stage A / Stage B → consumer interface is preserved
from the prior spec. Today the consumer is a Chrome WebSocket display. In Phase C it will
be a Swift AX injector writing to Live Captions. The pipeline does not change when the
consumer changes.

______________________________________________________________________

## Constraints

(unchanged from prior spec — listed for completeness)

- **Fully local inference.** No cloud ASR, no cloud LLM, no audio upload at runtime.
- **Single speaker.** The model is personalized to one voice. Not multi-user.
- **macOS only** (Phase 1 + Phase 2). Android is Phase 3.
- **No automatic TTS triggering.** User controls when text is submitted/spoken.
- **No audio retention** on local machine without user opt-in.
- **No new fine-tuning required** for Phase 1. The S1-M7 LoRA is the model.

______________________________________________________________________

## 10x Vision

(unchanged from prior spec)

WER drops below 20% after 6 months of feedback fine-tuning. The streaming display feels
indistinguishable from a fast typist seeing the speaker's mouth. The Deaf speaker enters
a job interview, speaks naturally, and their words appear on screen — word-by-word as
they speak — with the clarity of a fluent typist, no corrections, no hesitation, no
apology for the technology.

______________________________________________________________________

## Hardware

| Machine    | Specs            | Role                                           |
| ---------- | ---------------- | ---------------------------------------------- |
| Mac mini   | M4 Pro, 64GB RAM | Local development, evaluation, LoRA merge prep |
| MBP M4 Pro | M4 Pro, 48GB RAM | **Live demo (M5), daily driver**               |

**No cloud required for Phase 1.** The S1-M7 LoRA already exists. All Stage 2 work is
local CPU/Metal inference.

**Inference framework (local):** WhisperLiveKit with `mlx-whisper` backend (Apple Silicon
optimized, 4–6× faster than PyTorch per the upstream MLX-Whisper benchmarks). Falls back
to `faster-whisper` (CTranslate2) if MLX path has issues.

**Memory budget on MBP 48GB:**

```text
macOS + Chrome + daemon                    ~10 GB
Gemma 4 26B-A4B Q4_K_M                     ~17 GB
Whisper-small.en LoRA-merged (MLX fp16)    ~0.5 GB
Whisper streaming buffer + state           ~1 GB
Available headroom                         ~19 GB
```

Comfortable.

______________________________________________________________________

## What Is Out of Scope (Phase 1)

(largely unchanged — listed for completeness)

| Item                                      | Deferred to                                                                   |
| ----------------------------------------- | ----------------------------------------------------------------------------- |
| Android client                            | Phase 3                                                                       |
| FastAPI serving layer                     | Phase 2 (exists at scripts/serving/, untouched)                               |
| AX injection into Live Captions           | Phase C (post-demo)                                                           |
| IME (InputMethodKit) path                 | Phase C (if AX fails)                                                         |
| Stage B suppression (user-edit detection) | Phase 2                                                                       |
| Visual Stage A/B differentiation in AX    | AX limitation — plain text only                                               |
| Automated test suite                      | Phase 2                                                                       |
| Menu bar status app                       | Explicitly skipped                                                            |
| Auto-submit after silence                 | Explicitly skipped                                                            |
| Gemma rollback hotkey                     | Explicitly skipped                                                            |
| Multi-language support                    | Phase 3 (en-only)                                                             |
| Re-training Whisper-medium.en LoRA        | Optional Phase 1 stretch (M6, deferred unless small.en + Gemma underperforms) |

______________________________________________________________________

## Accepted Phase 1 Limitations

- **First-token latency is ~600-1000ms**, not the 320ms originally targeted with streaming
  Zipformer. AlignAtt + Whisper-small.en cannot match a true causal-streaming model. The
  user is monitoring their own articulation, not reading to make a real-time decision —
  sub-second is acceptable.
- **Buffer text flickers as the model revises.** The Chrome UI displays it in grey/italic
  to make this visually obvious; only committed text is "real".
- **Stage B is applied unconditionally.** If the user edits Stage A text manually before
  Stage B arrives, Stage B overwrites the edit. User-edit detection requires
  bidirectional IPC — deferred to Phase 2.
- **No visual distinction between Stage A and Stage B in the AX path** (Phase C):
  AXUIElement sets plain text only, no character-level styling. Chrome display (Phase B)
  uses color/italic to distinguish them.
- **Gemma timeout fallback:** if Gemma takes >1500ms or crashes, Stage A's committed text
  is used as Stage B verbatim (no correction).
- **No re-training in Phase 1.** S1-M7's 34.05% WER is the floor. Gemma post-correction
  is the only accuracy lever.

______________________________________________________________________

## Baseline to Reuse

| Model                                              | val WER (S1 frozen splits) | Notes                          |
| -------------------------------------------------- | -------------------------- | ------------------------------ |
| Whisper small.en (out-of-box)                      | ~55% (estimated)           | Pre-fine-tuning baseline       |
| Whisper small.en LoRA (S1-M3)                      | ~36%                       | Original LoRA fine-tune        |
| **Whisper small.en feedback-LoRA (S1-M7)**         | **34.05%**                 | **The ASR model in this spec** |
| WhisperLiveKit + S1-M7 LoRA-merged (Stage A)       | Target: ~34.05%            | M3 gate (parity check)         |
| WhisperLiveKit + S1-M7 LoRA-merged + Gemma Stage B | Target: ≤ Stage A WER      | M4 gate                        |

The S1-M7 number was measured with `create_normalizer(version=2)` from
`scripts/baseline_eval/normalization.py`. All Stage 2 evaluation must use the same
normalization for apples-to-apples comparison.

**Checkpoint location:** `out/feedback_finetune/batch_20260317_110057/checkpoint/` —
already transferred from MBP to Mac mini. Verified: `metrics.json` shows
`finetuned_wer: 0.34053156146179403` on the 361-clip val split. Adapter format: PEFT
LoRA (`adapter_config.json` + `adapter_model.safetensors`, 7.1 MB).

______________________________________________________________________

## Objective

Take the S1-M7 LoRA (already at 34.05% val WER, validated on Project Euphonia data),
merge it into Whisper-small.en, serve it via WhisperLiveKit with MLX acceleration, layer
Gemma 4 polish on every committed utterance, and ship a working live-caption demo for a
hiring manager screen by Monday morning (2026-05-11).

Two-layer plan (unchanged from prior spec):

- **Layer 1 (now):** Local LoRA merge → MLX convert → WhisperLiveKit serve → Chrome demo
- **Layer 2 (post-demo):** AX injection → Live Captions Type to Speak

______________________________________________________________________

## Naming Conventions

(unchanged from prior spec)

**Phase 1 / Phase 2 / Phase 3** — scope phases of Stage 2:

- **Phase 1 (this spec):** macOS only — local Python daemon + Chrome display + AX injection
- **Phase 2 (future):** FastAPI serving layer + browser extension
- **Phase 3 (future):** Android client

**Phase A / Phase B / Phase C** — time sub-phases within Phase 1:

- **Phase A:** Prove the core — checkpoint transfer, LoRA merge, WhisperLiveKit smoke test (M0–M2)
- **Phase B:** Ship the demo — Chrome live caption display + Gemma Stage B (M3–M5)
- **Phase C:** Production integration — AX injection into Live Captions (M7–M8, post-demo)

______________________________________________________________________

## Milestone Sequence

```text
PHASE A — PROVE THE CORE (Sun morning, ~3-4h)
┌─────────────────────────────────────────────────────────────────┐
│ M0 — Checkpoint sanity check (~15 min)                          │
│   Checkpoint at out/feedback_finetune/                          │
│     batch_20260317_110057/checkpoint/ (LoRA adapter, 7.1MB).    │
│   Base model: openai/whisper-small.en (HF Hub).                 │
│   Load via PeftModel.from_pretrained, transcribe 1 val clip.    │
│   Snippet — see "M0 sanity stub" below.                         │
│   Gate: transcript reasonably matches reference (sanity check;  │
│         not WER-blocking, just verifies LoRA loads + decodes).  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ M1 — LoRA merge + MLX conversion (~30 min)                      │
│   merge_and_unload() → out/whisper_small_en_s1m7_merged/        │
│   mlx-whisper convert → out/whisper_small_en_s1m7_merged_mlx/   │
│   Smoke test both formats on 1 val clip.                        │
│   Gate: both produce identical text.                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ M2 — WhisperLiveKit standalone (~1-2h)                          │
│   pip install whisperlivekit, run wlk with merged MLX model.    │
│   Open browser, mic capture, verify word-by-word UI.            │
│   Measure first-token latency + commit latency on M4 Pro.       │
│   Gate: streaming UI works; commit latency < 1500ms p95.        │
└────────────────────────┬────────────────────────────────────────┘
                         │

PHASE B — SHIP THE DEMO (Sun, ~6-8h)
┌────────────────────────▼────────────────────────────────────────┐
│ M3 — WER eval parity check (~1h)                                │
│   Write scripts/vox_daemon/eval_merged.py (baseline_eval        │
│   doesn't accept custom checkpoints). Iterate val split (361)   │
│   from out/dataset_v1/20260206-142756/dataset_v1_manifest.csv,  │
│   apply path remap, normalize with create_normalizer(version=2).│
│   Gate: WER within ±0.5% of S1-M7's 34.05% (proves merge        │
│   + format conversion didn't degrade anything).                 │
│   Output: results/S2-M3_whisper_merged_baseline/                │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ M4 — Gemma 4 Stage B integration + correction WER (~3-4h)       │
│   New scripts/vox_daemon/ — proxy + correction pipeline.        │
│   Subscribe to WhisperLiveKit's WebSocket, on committed line    │
│   trigger Gemma correction, re-broadcast Stage A + Stage B.     │
│   Eval Stage B WER on 361 val clips (offline mode).             │
│   Gate: Stage B WER ≤ Stage A WER + false-correction rate <15%. │
│   Output: results/S2-M4_gemma_correction_eval/                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ M5 — Chrome live caption demo (~2-3h)                           │
│   Custom static HTML in scripts/vox_daemon/static/index.html.   │
│   Connects to vox_daemon WebSocket (which proxies WhisperLive). │
│   Stage A: word-by-word, committed (white) + buffer (grey).    │
│   Stage B: replaces line on endpoint, white/bold.               │
│   History pane: last 5 committed utterances.                    │
│   Gate: 10 min continuous speech without crash.                 │
│   This IS the hiring manager screen.                            │
└─────────────────────────────────────────────────────────────────┘

(M6 — optional medium.en LoRA upgrade — DEFERRED unless M4 gate fails)

PHASE C — PRODUCTION INTEGRATION (Post-demo)
┌─────────────────────────────────────────────────────────────────┐
│ M7 — AX Proof of Concept (~4-8h)                                │
│   Swift CLI: inject hardcoded string → Live Captions field.     │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ M8 — AX Daemon + Swift Injector (~1 week)                       │
│   Replace WebConsumer with AXConsumer (one config flag).        │
└─────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## Pre-work: Day 0 Checklist

### Install state on Mac mini (verified 2026-05-10)

Already present:

- **Python 3.11.15 venv** at `./venv` — activate with `source venv/bin/activate`
- **`peft`, `transformers`, `torch`** — from S1, declared in
  [pyproject.toml](../pyproject.toml)
- **`silero-vad` 6.2.1** — reused as WhisperLiveKit's `--vac` Voice Activity Controller
  (v6 API)
- **llama.cpp + Gemma 4 GGUF** — `~/llama.cpp/build/bin/llama-cli` +
  `models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`
- **openai-whisper** Python package — installed from S1

Missing — install before M0:

- **ffmpeg** — required by `whisper`. Install: `brew install ffmpeg`
- **whisperlivekit** — install in M2 prep: `pip install "whisperlivekit[whisper]"`
- **mlx-whisper** — install in M1 prep: `pip install mlx-whisper`

### Day 0 commands to run before M0

```bash
brew install ffmpeg            # required by openai-whisper
source venv/bin/activate       # Python 3.11.15
pip install mlx-whisper        # Apple Silicon backend (also pulls in mlx, mlx-lm)
pip install "whisperlivekit[whisper]"
```

`mlx-whisper` will fail to install on non-macOS — that's fine, we're macOS-only by spec.

**No A10 work for Phase 1.** No new cloud spend.

### pyproject.toml updates

```text
# Add to [project.dependencies]:
whisperlivekit>=0.4         # WebSocket streaming server with AlignAtt SimulStreaming
mlx-whisper>=0.4            # Apple Silicon MLX backend (macOS only)
silero-vad>=6.2             # already present; reused by WhisperLiveKit --vac
```

______________________________________________________________________

## System Architecture (Layer 1)

```text
┌────────────────────────────────────────────────────────────────────────┐
│ BROWSER  (Chrome, opened to localhost:8000)                            │
│                                                                         │
│  Mic capture (Web Audio API, 16kHz, Opus or PCM via WebSocket)         │
│  Custom HTML at static/index.html (NOT WhisperLiveKit's bundled UI)    │
│       │  WS: audio frames + control messages                           │
│       ▼                                                                 │
└────────────────────────────────────────────────────────────────────────┘
              │
              │ WS: ws://localhost:8000/asr (proxy)
              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ VOX DAEMON  (scripts/vox_daemon/, on Mac mini → MBP for demo)         │
│                                                                         │
│  aiohttp server (HTTP static files + WebSocket)                        │
│  Acts as a transparent proxy + Gemma overlay:                          │
│                                                                         │
│   Client WS ──┬─→ forwards mic audio to WhisperLiveKit                 │
│   Client WS ←─┴── forwards Stage A events (committed + buffer)         │
│                  + injects Stage B events (Gemma-corrected)            │
│                                                                         │
│  On committed-line event from WhisperLiveKit:                          │
│    1. Forward Stage A as-is (instant)                                  │
│    2. Submit to Gemma worker queue (async)                             │
│    3. When Gemma returns, emit Stage B event                           │
│                                                                         │
│  Gemma 4 worker:                                                        │
│    subprocess.Popen(llama-cli ... --reasoning off)                     │
│    threaded stdout reader → queue.Queue                                │
│    timeout=1500ms; on timeout → emit Stage A as Stage B                │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
              │
              │ WS: ws://localhost:8001/asr (internal — WhisperLiveKit)
              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ WHISPERLIVEKIT  (subprocess, spawned by vox_daemon)                    │
│                                                                         │
│  wlk --backend mlx-whisper                                              │
│      --model out/whisper_small_en_s1m7_merged_mlx                      │
│      --host 127.0.0.1 --port 8001                                      │
│      --no-static                  (we serve our own HTML)              │
│                                                                         │
│  Internally:                                                            │
│    Audio buffer (rolling window) → MLX Whisper-small.en (LoRA-merged)  │
│    AlignAtt emission policy → {committed, buffer} per ~500ms-1s        │
│    VAD-based segmenter → endpoint events                                │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

**Why proxy instead of fork the frontend?** Two reasons:

1. We want full freedom over the Chrome UI (Stage A/B styling, history pane, debug
   overlays, future extensions). WhisperLiveKit's bundled HTML is fine but limits us.
1. Gemma correction has to happen *somewhere*; doing it in the daemon between WhisperLive
   and the browser is the cleanest layering and keeps the browser dumb.

______________________________________________________________________

## Component Specs

### M0 sanity stub

Single-purpose script the agent should write at `scripts/vox_daemon/m0_sanity.py`
(or run as a one-liner). Confirms the LoRA loads on top of the base and decodes one clip.

```python
"""M0: load S1-M7 LoRA + transcribe 1 val clip. Sanity check, no WER."""

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch, librosa

BASE = "openai/whisper-small.en"
LORA = "out/feedback_finetune/batch_20260317_110057/checkpoint"
# pick any val clip; example below is the first val row in dataset_v1_manifest.csv
WAV = "/Users/skinnercheng/Downloads/takeout-E407/euphonia_002f4f2d5ad6ecd94202d4ef92719c02.wav"
EXPECTED = "can you play some music"  # from results/M7_feedback_finetune/predictions.csv

base = WhisperForConditionalGeneration.from_pretrained(BASE)
proc = WhisperProcessor.from_pretrained(BASE)
model = PeftModel.from_pretrained(base, LORA).eval()
audio, _ = librosa.load(WAV, sr=16000)
inputs = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
with torch.no_grad():
    ids = model.generate(inputs, language="en", task="transcribe")
print("Hypothesis:", proc.batch_decode(ids, skip_special_tokens=True)[0])
print("Reference:", EXPECTED)
```

Gate: hypothesis is recognizable English close to reference (the val clip used here is
WER=0 in S1-M7, so a clean "can you play some music" is expected on MPS/CPU; small
floating-point variance is fine).

### LoRA Merge (one-time prep — M1)

File: `scripts/vox_daemon/merge_lora.py`

```python
"""One-time: merge S1-M7 LoRA into Whisper-small.en base, save HF + MLX formats."""

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_MODEL_ID = "openai/whisper-small.en"


def merge(lora_checkpoint: str, output_dir: str) -> None:
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

    peft_model = PeftModel.from_pretrained(base, lora_checkpoint)
    merged = peft_model.merge_and_unload()  # bake LoRA weights into base

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    processor.save_pretrained(out)
    # Preserve generation_config so downstream code doesn't need to re-set lang/task:
    merged.generation_config.save_pretrained(out)
    print(f"Saved merged HF model to {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lora", default="out/feedback_finetune/batch_20260317_110057/checkpoint")
    p.add_argument("--out", default="out/whisper_small_en_s1m7_merged")
    args = p.parse_args()
    merge(args.lora, args.out)


if __name__ == "__main__":
    main()
```

After merge, convert to MLX:

```bash
python -m mlx_whisper.convert \
  --torch-name-or-path ./out/whisper_small_en_s1m7_merged \
  --mlx-path ./out/whisper_small_en_s1m7_merged_mlx \
  --dtype float16
```

**Why `merge_and_unload` is safe:** the merge is a deterministic linear combination
(`W_merged = W_base + alpha/rank * B @ A`) — produces a model with identical numerical
behavior to the LoRA-active model under inference, with zero overhead. This is the
standard PEFT export path.

**mlx-whisper convert flag note:** the actual flag is `--torch-name-or-path` (it accepts
both HF Hub IDs and local HF Transformers checkpoint dirs); the non-existent `--hf-path`
flag is a common mistake. Verify with `python -m mlx_whisper.convert --help` after
install.

### WhisperLiveKit Serving

WhisperLiveKit is invoked as a subprocess by the daemon. We do NOT use its built-in
static-file server (we serve our own HTML).

```bash
wlk --backend mlx-whisper \
    --model ./out/whisper_small_en_s1m7_merged_mlx \
    --host 127.0.0.1 --port 8001 \
    --min-chunk-size 0.5 \
    --vac \                    # Silero-VAD-based voice activity controller
    --vac-chunk-size 0.04
```

Key flags:

- `--backend mlx-whisper`: Apple Silicon native; 4-6× faster than vanilla PyTorch
- `--min-chunk-size 0.5`: balance between latency and accuracy. 0.5s gives ~1s commit
  latency. Day 0 step in M2 will tune this against measured RTF.
- `--vac`: enable WhisperLiveKit's Voice Activity Controller. **This is Silero VAD
  internally** — WhisperLiveKit imports the `silero-vad` PyPI package (already in our
  pyproject) and feeds 32ms (`--vac-chunk-size 0.04` ≈ 512 samples at 16kHz, the v6 API
  contract) frames to the Silero ONNX model to gate endpoint detection. Required for
  endpoint events that trigger Gemma Stage B correction. No separate VAD pre-filter
  needed; we use one Silero VAD instance owned by WhisperLiveKit.
- **No `--lora-path`**: WhisperLiveKit's `--lora-path` flag is PyTorch-backend-only and
  would lose MLX speed. We pass a merged model via `--model` instead and simply don't
  pass `--lora-path` at all.
- `--no-static` is **assumed** but not verified — if it doesn't exist, suppress the
  bundled UI by binding port 8001 to localhost only and ignoring the served HTML.
- **M2 verification step:** run `wlk --help` and confirm exact backend names, flag
  spellings, and the WS endpoint path (this spec assumes `/asr`; upstream may use `/`
  or `/ws`). The flag list above is from upstream docs; expect minor drift.

**WebSocket protocol — from upstream README; verify against installed version in M2.**

Consumed by daemon, not browser:

```json
{
  "lines": [
    {"speaker": 1, "text": "I want to talk about Kubernetes networking",
     "start": 0.0, "end": 2.4}
  ],
  "buffer_transcription": "",
  "remaining_time_transcription": 0.0
}
```

The daemon detects "line just committed" by diffing successive `lines` arrays. Open
Question 2 covers the residual risk that the diff signal is unreliable (e.g., line text
mutates mid-utterance) — M2 logging captures the actual stream so M4 can finalize the
diff predicate.

### Gemma 4 GGUF Integration

(Mostly unchanged from prior spec — the integration we already validated in the prior
Day 0 step still works. New: the trigger is WhisperLiveKit's commit event, not a custom
endpoint detector.)

File: `scripts/vox_daemon/gemma.py`

```python
"""Gemma 4 GGUF subprocess wrapper. Submit text → get corrected text."""

import queue, subprocess, threading

GEMMA_MODEL = "./models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
LLAMA_CLI = "/Users/skinnercheng/llama.cpp/build/bin/llama-cli"

PROMPT_TEMPLATE = """Correct the following speech transcript. Fix ASR errors, grammar, and disfluencies. Output ONLY the corrected text. No explanation.

Input: {stage_a_text}
Output:"""


class GemmaWorker:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [LLAMA_CLI, "-m", GEMMA_MODEL,
             "--reasoning", "off",         # critical: disable thinking mode
             "-n", "60",
             "--single-turn", "--simple-io",
             "-t", "8"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        for line in iter(self.proc.stdout.readline, ""):
            self.q.put(line)

    def correct(self, stage_a_text: str, timeout: float = 1.5) -> str:
        prompt = PROMPT_TEMPLATE.format(stage_a_text=stage_a_text)
        self.proc.stdin.write(prompt + "\n")
        self.proc.stdin.flush()
        # llama-cli with --simple-io --single-turn emits multiple stdout lines and a
        # sentinel/EOT before going idle. Drain until the prompt sentinel "Output:" is
        # echoed back blank, or until quiet for `quiet_window` seconds. Tune this stub
        # against actual llama-cli output during M4.
        deadline = __import__("time").monotonic() + timeout
        chunks: list[str] = []
        while __import__("time").monotonic() < deadline:
            try:
                chunks.append(self.q.get(timeout=0.1))
            except queue.Empty:
                if chunks:
                    break
        if not chunks:
            return stage_a_text  # timeout fallback
        return " ".join(c.strip() for c in chunks if c.strip()).strip()
```

Reuses Day 0 validated config (`--reasoning off`, p95 ~430ms warm). The exact stdout
shape from `llama-cli --simple-io --single-turn` needs to be inspected against the
installed binary (`llama-cli --version`) — adjust drain loop accordingly in M4.

### Stage A / Stage B Proxy

File: `scripts/vox_daemon/proxy.py`

The proxy connects browser ↔ WhisperLiveKit ↔ Gemma:

- Forwards audio frames from browser → WhisperLiveKit
- Forwards Stage A events (committed text + buffer) from WhisperLiveKit → browser
  immediately
- On a *new* committed line (diff vs previous frame's `lines` array): submit it to a
  background Gemma task; when Gemma returns, push a Stage B event to the browser

**Wire format to browser** (extends WhisperLiveKit's protocol with `stage_b` events):

```json
// Stage A (every WhisperLiveKit update, ~1Hz):
{"type": "stage_a", "lines": [...], "buffer": "...", "uid": 42}

// Stage B (on new committed line, ~400ms after):
{"type": "stage_b", "uid": 42, "line_index": 0, "text": "Corrected text."}
```

`uid` increments per utterance to let the client discard stale Stage B events if a new
utterance has already started.

### Chrome Frontend

File: `scripts/vox_daemon/static/index.html`

Single-file HTML, no build step. Connects to `ws://localhost:8000/asr` (the daemon, not
WhisperLiveKit directly).

```text
┌──────────────────────────────────────────────────────────────┐
│  VOX Personalis                              ● live           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Current utterance                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Stage A → "I want to talk about kubernet..."           │  │
│  │           (committed white + buffer grey/italic)        │  │
│  │ Stage B → "I want to talk about Kubernetes."           │  │
│  │           (replaces Stage A on endpoint, bold)          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  History                                                      │
│  ─────────────────────────────────────────────────────────   │
│  "Can you tell me about your experience with Kubernetes?"    │
│  "Yes, I have three years working with Kubernetes."          │
│  "We use it for container orchestration at my company."      │
└──────────────────────────────────────────────────────────────┘
```

Styling:

- Buffer text: `color: #888; font-style: italic` (signals "in flight")
- Stage A committed: `color: #fff` (signals "Whisper emitted")
- Stage B committed: `color: #fff; font-weight: 600` (signals "Gemma polished")
- Stage A → Stage B uses 150ms CSS transition for smoothness

Mic capture uses Web Audio API + an `AudioWorklet` that resamples to 16kHz Float32 PCM
and pushes via WebSocket. WhisperLiveKit's standard input format.

### Daemon CLI

File: `scripts/vox_daemon/cli.py`

```bash
python -m scripts.vox_daemon \
  --whisper-model ./out/whisper_small_en_s1m7_merged_mlx \
  --gemma-model ./models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --port 8000 \
  --whisperlivekit-port 8001 \
  --debug
```

Startup sequence:

1. Spawn WhisperLiveKit subprocess on port 8001 with merged Whisper model
1. Spawn Gemma 4 subprocess via llama-cli with `--reasoning off`
1. Verify both respond to a test prompt
1. Start aiohttp HTTP+WS server on port 8000 (HTML at `/`, proxy at `/asr`)
1. `subprocess.run(['open', 'http://localhost:8000'])` (macOS) — opens default browser
1. Wait for connections; clean shutdown on SIGINT/SIGTERM

______________________________________________________________________

## Module Structure

```text
scripts/
  serving/                 ← UNTOUCHED (S1 FastAPI, Phase 2)
  baseline_eval/
    inference.py           ← REUSE pieces only (Whisper transcription helpers)
    metrics.py             ← REUSE: WER/CER computation
    normalization.py       ← REUSE: create_normalizer(version=2) — call explicitly
    cli.py                 ← UNTOUCHED (only accepts --model_size; not used in S2 evals)
  feedback_finetune/       ← UNTOUCHED (S1-M7 trainer; not used in S2)
  fine_tuning/             ← UNTOUCHED (S1 base trainer; not used in S2)
  zipformer_eval/          ← KEPT FROM ABANDONED ZIPFORMER PLAN (still useful for any future Zipformer eval)
  zipformer_finetune/      ← KEPT FROM ABANDONED ZIPFORMER PLAN
  vox_daemon/              ← NEW: M0 + M1 + M3 + M4 + M5
    __main__.py
    cli.py                 # Daemon CLI (M5)
    proxy.py               # Browser ↔ WhisperLiveKit ↔ Gemma proxy (M4)
    gemma.py               # Gemma 4 subprocess wrapper, validated Day 0
    merge_lora.py          # M1: merge S1-M7 LoRA → HF
    m0_sanity.py           # M0: load LoRA, transcribe 1 clip
    eval_merged.py         # M3: parity-check merged-model WER vs S1-M7
    eval_offline.py        # M4: Stage A + Stage B WER over val split
    static/
      index.html           # Custom Chrome frontend (M5)
```

Note: We do NOT need a separate WhisperLiveKit wrapper — `wlk` is invoked as a
subprocess. The daemon owns its lifecycle.

______________________________________________________________________

## Dataset

**No new training data required.** The S1-M7 LoRA was already trained on the merged
manifest (3,005 train + feedback corrections, ~4h total). Stage 2 reuses:

- **Manifest for eval** — `out/dataset_v1/20260206-142756/dataset_v1_manifest.csv`
  (3623 rows total: 2897 train / 365 test / **361 val**). Column
  `audio_path_resolved` has stale `/Users/skinner/...` paths — remap to
  `/Users/skinnercheng/...` before reading audio.
- **S1-M7 reference predictions** — `results/M7_feedback_finetune/predictions.csv`
  (361 val rows; columns: `file_name`, `reference`, `hypothesis`, `wer`). NOT a
  manifest (no audio path) — useful as ground-truth references in joins, not as
  iteration source.
- `~/Downloads/takeout-E407/` audio (Project Euphonia drop) for any Stage 2 smoke tests
- `~/Downloads/feedback/` audio (108 clips) for any Stage 2 feedback-loop testing

______________________________________________________________________

## Evaluation Specs

### M3: Merged-model parity check

Goal: confirm `merge_and_unload` + MLX conversion didn't change inference numerics.

**Implementation note:** the existing `scripts/baseline_eval/cli.py` only accepts a Whisper
size string (`--model_size {tiny,base,small,medium,large}.en`) — it cannot load a
custom HF checkpoint, and it calls `create_normalizer()` defaulting to v1 (the S1-M7
34.05% number was measured with v2). Two implementation options:

- **Option A (recommended): write a dedicated eval script** at
  `scripts/vox_daemon/eval_merged.py`. It loads the merged HF checkpoint via
  `WhisperForConditionalGeneration.from_pretrained(merged_dir)`, iterates the val split
  from `out/dataset_v1/20260206-142756/dataset_v1_manifest.csv` (filter `split=='val'`,
  361 rows), normalizes with `create_normalizer(version=2)`, computes WER via
  `scripts/baseline_eval/metrics.py`, writes `metrics.json` + `predictions.csv` to
  `results/S2-M3_whisper_merged_baseline/`. Mirrors the prior `M7_feedback_finetune/`
  output schema.
- **Option B:** extend `scripts/baseline_eval/cli.py` with `--checkpoint_path` and
  `--norm_version` flags. More invasive, but reusable for future LoRA evals.

**Manifest path-fixup gotcha:** `dataset_v1_manifest.csv` was written on a different
machine and stores `/Users/skinner/...` paths; on this Mac mini, audio lives at
`/Users/skinnercheng/...`. The eval script must remap before opening files:

```python
df["audio_path_resolved"] = df["audio_path_resolved"].str.replace(
    "/Users/skinner/", "/Users/skinnercheng/", regex=False
)
```

**Gate:** WER within ±0.5% of S1-M7's 34.05% on the same 361-clip val split.

### M4: Gemma Stage B correction eval

Goal: measure WER and false-correction rate of the Stage A → Stage B pipeline offline.

```bash
python -m scripts.vox_daemon.eval_offline \
  --manifest ./out/dataset_v1/20260206-142756/dataset_v1_manifest.csv \
  --split val \
  --whisper-model ./out/whisper_small_en_s1m7_merged_mlx \
  --gemma-model ./models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --output ./results/S2-M4_gemma_correction_eval/
```

`eval_offline.py` is a script written under `scripts/vox_daemon/` for this milestone.
Apply the same `/Users/skinner/` → `/Users/skinnercheng/` path remap as M3. It runs
offline (no WebSocket, no streaming) but uses the same Gemma worker the live daemon
uses — Stage B output is faithful to what the demo will produce.

**Gates:**

1. Stage B WER ≤ Stage A WER (Gemma helps overall)
1. False-correction rate < 15% (cases where Gemma made it worse)

If gate 1 fails: skip Gemma in the live demo (Stage A only).\
If gate 2 fails: tighten the Gemma prompt to be more conservative.

______________________________________________________________________

## Error / Rescue Registry (Layer 1)

| Error                                        | Rescue                                                                                        |
| -------------------------------------------- | --------------------------------------------------------------------------------------------- |
| WhisperLiveKit subprocess crash              | Auto-restart up to 3 times; emit alert to browser; if exhausted, surface error in UI          |
| WhisperLiveKit MLX backend errors at startup | Fall back to `--backend faster-whisper`; log warning; degrades to ~2× slower commit but works |
| Gemma timeout (>1500ms)                      | Emit Stage A text as Stage B (no correction; uid preserved)                                   |
| Gemma OOM / crash                            | Auto-restart up to 3 times; Stage A only after limit                                          |
| Gemma malformed output (empty/garbled)       | Emit Stage A as Stage B fallback                                                              |
| WebSocket client disconnected                | Daemon continues; reconnect on next page load                                                 |
| Chrome not installed                         | Log warning; user opens <http://localhost:8000> manually                                      |
| Port 8000 / 8001 in use                      | CLI error on startup: "Port N in use — use --port"                                            |
| Whisper merged model load failure            | Fail fast at daemon startup with clear error; instruct user to re-run merge_lora.py           |

______________________________________________________________________

## Success Criteria (Demo Day)

| Metric                                | Target                                 | Milestone |
| ------------------------------------- | -------------------------------------- | --------- |
| Merged model val WER                  | Within ±0.5% of 34.05%                 | M3        |
| Stage B WER vs Stage A                | Stage B ≤ Stage A                      | M4        |
| Stage A first-token latency on M4 Pro | < 1500ms p95                           | M2        |
| Stage A commit cadence                | < 1500ms p95                           | M2        |
| Stage B latency after endpoint        | < 600ms p95 (Gemma warm)               | M5        |
| Demo stability                        | 10 min continuous speech without crash | M5        |

______________________________________________________________________

## Layer 2: AX Integration Stub (Post-demo)

(unchanged from prior spec — all D1–D7 decisions still apply)

The Stage A / Stage B contract on the Chrome WebSocket is identical to what an AX
injector would consume. Phase C work swaps the consumer (browser → Swift AX agent) with
no daemon changes.

______________________________________________________________________

## Pre-Implementation Blockers

1. **[Done — M0]** Identify which `out/feedback_finetune/batch_<id>/` corresponds to S1-M7
   (val WER 34.05%). → `batch_20260317_110057/` (verified `metrics.json`).
1. **[Done — M0]** Transfer checkpoint dir from MBP → Mac mini via rsync. → checkpoint
   present locally.
1. **[Open — M2]** WhisperLiveKit installs cleanly on M4 Pro with MLX backend (pip
   install + smoke test).
1. **[Open — M1]** `mlx-whisper.convert` accepts a HF Whisper checkpoint (one-shot
   convert + load test).
1. **[Open — M2]** WhisperLiveKit's CLI flags + WebSocket protocol match the spec
   assumptions (run `wlk --help`, log a connection's stream, confirm `lines`/`buffer`
   field names + WS endpoint path before M4 proxy work begins).

______________________________________________________________________

## Open Questions / Risks

1. **mlx-whisper merged-LoRA support is empirically untested.** The `merge_and_unload`
   path produces a regular Whisper checkpoint, so in theory it should convert cleanly.
   Verify in M1.

1. **WhisperLiveKit's commit event format.** The frontend code shows `lines[]` arrays,
   but the exact "new line just committed" diff signal isn't documented — we'll have
   to read the protocol stream in M2 to confirm reliable detection.

1. **VAD/endpoint tuning.** WhisperLiveKit's `--vac` defaults may not match the user's
   speech patterns (Deaf speech often has different prosody). M5 stability testing tunes
   `--vac-chunk-size` and related flags.

1. **Gemma + Whisper Metal contention on MBP.** Both use Metal. If concurrent Whisper
   inference + Gemma generation contends, we may see Whisper stalling during Gemma's
   ~400ms generation. Mitigation: serialize, or run Gemma on CPU only (`-ngl 0` in
   llama.cpp). M5 measures.

1. **Final-day move from Mac mini → MBP.** All artifacts need to be on the MBP by
   Sunday evening for a full integration test on the demo machine. Concrete checklist:

   ```bash
   # From MBP, pulling from Mac mini (replace with Mac mini's hostname/IP):
   rsync -avzP <mac-mini>:~/Projects/VOX_Personalis/out/whisper_small_en_s1m7_merged_mlx \
        ~/Projects/VOX_Personalis/out/
   rsync -avzP <mac-mini>:~/Projects/VOX_Personalis/scripts/vox_daemon/ \
        ~/Projects/VOX_Personalis/scripts/vox_daemon/
   # Gemma GGUF (~17GB) — copy once if not already on MBP.
   # Run `git pull` on MBP for committed code.
   ```

1. **WhisperLiveKit version churn.** The project moves fast; record the installed
   version (`pip show whisperlivekit` after install) and pin in `pyproject.toml`.
   Blocker #5 covers the actual flag/protocol verification.
