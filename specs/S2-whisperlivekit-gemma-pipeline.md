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

WhisperLiveKit's `--backend simulstreaming` uses the AlignAtt policy
([Papi et al., Interspeech 2023](https://arxiv.org/abs/2305.11408)): it monitors decoder
cross-attention to identify when each token is "stable enough" to commit, and holds
unstable tokens in a "buffer" tier. The user sees grey/italic in-flight text turning into
white/committed text on a sub-second cadence. Sub-second word-by-word feedback is
acceptable for the user's use case (live self-monitoring during speech is tolerant of
~1s latency; a Deaf speaker monitoring articulation does not need 320ms cadence).

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

**Checkpoint location:** `out/feedback_finetune/batch_<timestamp>/checkpoint/` on the
MBP. Must be transferred to the development Mac mini for M1 prep work. The user will
identify the correct batch (the one whose `metrics.json` shows 34.05% val WER).

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
- **Phase C:** Production integration — AX injection into Live Captions (M6–M7, post-demo)

______________________________________________________________________

## Milestone Sequence

```text
PHASE A — PROVE THE CORE (Sat afternoon, ~3-4h)
┌─────────────────────────────────────────────────────────────────┐
│ M0 — Checkpoint transfer + sanity check (~30 min)               │
│   rsync S1-M7 LoRA from MBP → Mac mini.                         │
│   Load via PeftModel.from_pretrained, transcribe 1 val clip.    │
│   Gate: transcript matches reference (sanity, not WER).         │
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
│ M3 — WER eval parity check (~30 min)                            │
│   Reuse scripts/baseline_eval/ on the merged model.             │
│   361 val clips, create_normalizer(version=2).                  │
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

Day 0 work for the prior spec (sherpa-onnx install, llama.cpp build with Gemma 4 support,
silero-vad install) was completed on 2026-05-09 and is **already valid for this spec**.
The only new dependencies are:

```bash
# In the existing venv at /Users/skinnercheng/Projects/VOX_Personalis/venv:
pip install "whisperlivekit[whisper]"
pip install mlx-whisper        # Apple Silicon backend
```

Plus pre-existing project deps (`peft`, `transformers`, `torch`) — already declared in
[pyproject.toml](../pyproject.toml).

**No A10 work for Phase 1.** No new cloud spend.

### pyproject.toml updates

```text
# Add to [project.dependencies]:
whisperlivekit>=0.4         # WebSocket streaming server with AlignAtt SimulStreaming
mlx-whisper>=0.4            # Apple Silicon MLX backend (macOS only)
```

`mlx-whisper` will fail to install on non-macOS — that's fine, we're macOS-only by spec.
If we ever need cross-platform local dev, mark it with a platform marker in pyproject.

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

### LoRA Merge (one-time prep)

File: `scripts/vox_daemon/merge_lora.py`

```python
"""One-time: merge S1-M7 LoRA into Whisper-small.en base, save HF + MLX formats."""

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def merge(
    base_model_id: str,
    lora_checkpoint: str,
    output_dir: str,
) -> None:
    base = WhisperForConditionalGeneration.from_pretrained(base_model_id)
    processor = WhisperProcessor.from_pretrained(base_model_id)

    peft_model = PeftModel.from_pretrained(base, lora_checkpoint)
    merged = peft_model.merge_and_unload()  # bake LoRA weights into base

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    processor.save_pretrained(out)
    print(f"Saved merged HF model to {out}")
    print("Next: python -m mlx_whisper.convert "
          f"--hf-path {out} --mlx-path {out}_mlx")
```

**Why `merge_and_unload` is safe:** the merge is a deterministic linear combination
(`W_merged = W_base + alpha/rank * B @ A`) — produces a model with identical numerical
behavior to the LoRA-active model under inference, with zero overhead. This is the
standard PEFT export path.

### WhisperLiveKit Serving

WhisperLiveKit is invoked as a subprocess by the daemon. We do NOT use its built-in
static-file server (we serve our own HTML).

```bash
wlk --backend mlx-whisper \
    --model ./out/whisper_small_en_s1m7_merged_mlx \
    --host 127.0.0.1 --port 8001 \
    --min-chunk-size 0.5 \
    --vac \                    # voice activity controller for endpoint detection
    --vac-chunk-size 0.04
```

Key flags:

- `--backend mlx-whisper`: Apple Silicon native; 4-6× faster than vanilla PyTorch
- `--min-chunk-size 0.5`: balance between latency and accuracy. 0.5s gives ~1s commit
  latency. Day 0 step in M2 will tune this against measured RTF.
- `--vac`: enable Voice Activity Controller, required for endpoint events that trigger
  Gemma correction
- `--no-lora-path`: we use a merged model, not a separate LoRA — full speed on MLX

**WebSocket protocol** (consumed by daemon, not browser):

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

The daemon detects "line just committed" by diffing successive `lines` arrays.

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
        try:
            return self.q.get(timeout=timeout).strip()
        except queue.Empty:
            return stage_a_text  # fallback: no correction
```

Reuses Day 0 validated config (`--reasoning off`, p95 ~430ms warm).

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
    inference.py           ← REUSE: load merged Whisper checkpoint for offline WER eval
    metrics.py             ← REUSE: WER/CER computation
    normalization.py       ← REUSE: create_normalizer(version=2)
    cli.py                 ← REUSE (point at merged model dir)
  feedback_finetune/       ← UNTOUCHED (S1-M7 trainer; not used in S2)
  fine_tuning/             ← UNTOUCHED (S1 base trainer; not used in S2)
  zipformer_eval/          ← KEPT FROM ABANDONED ZIPFORMER PLAN (still useful for any future Zipformer eval)
  zipformer_finetune/      ← KEPT FROM ABANDONED ZIPFORMER PLAN
  vox_daemon/              ← NEW: M4 + M5 daemon
    __main__.py
    cli.py
    proxy.py               # Browser ↔ WhisperLiveKit ↔ Gemma proxy
    gemma.py               # Gemma 4 subprocess wrapper (validated Day 0)
    merge_lora.py          # One-time: merge S1-M7 LoRA → HF → MLX
    static/
      index.html           # Custom Chrome frontend
```

Note: We do NOT need a separate WhisperLiveKit wrapper — `wlk` is invoked as a
subprocess. The daemon owns its lifecycle.

______________________________________________________________________

## Dataset

**No new training data required.** The S1-M7 LoRA was already trained on the merged
manifest (3,005 train + feedback corrections, ~4h total). Stage 2 reuses:

- S1 frozen splits for evaluation (val=361, test=365 — `results/M7_feedback_finetune/predictions.csv`)
- `~/Downloads/takeout-E407/` audio for any Stage 2 smoke tests
- `~/Downloads/feedback/` audio (108 clips) for any Stage 2 feedback-loop testing

______________________________________________________________________

## Evaluation Specs

### M3: Merged-model parity check

Goal: confirm `merge_and_unload` + MLX conversion didn't change inference numerics.

```bash
source venv/bin/activate
python -m scripts.baseline_eval \
  --manifest_path ./results/M7_feedback_finetune/predictions.csv \
  --model_path ./out/whisper_small_en_s1m7_merged \
  --norm_version 2 \
  --output ./results/S2-M3_whisper_merged_baseline/
```

**Gate:** WER within ±0.5% of S1-M7's 34.05%. Tighter is better.

### M4: Gemma Stage B correction eval

Goal: measure WER and false-correction rate of the Stage A → Stage B pipeline offline.

```bash
python -m scripts.vox_daemon.eval_offline \
  --manifest ./results/M7_feedback_finetune/predictions.csv \
  --whisper-model ./out/whisper_small_en_s1m7_merged_mlx \
  --gemma-model ./models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --output ./results/S2-M4_gemma_correction_eval/
```

This is a script we'll write under `scripts/vox_daemon/`. It runs offline (no
WebSocket, no streaming) but uses the same Gemma worker the live daemon uses — so the
Stage B output is faithful to what the demo will produce.

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

| #   | Blocker                                                                                             | When     | Owner                                    |
| --- | --------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------- |
| 1   | Identify which `out/feedback_finetune/batch_<id>/` on the MBP corresponds to S1-M7 (val WER 34.05%) | M0 start | User checks `metrics.json` in each batch |
| 2   | Transfer that checkpoint dir from MBP → Mac mini via rsync                                          | M0 start | rsync -avzP from MBP IP                  |
| 3   | WhisperLiveKit installs cleanly on M4 Pro with MLX backend                                          | M2 start | pip install + smoke test                 |
| 4   | `mlx-whisper.convert` accepts a HF Whisper checkpoint                                               | M1 start | One-shot convert + load test             |
| 5   | WhisperLiveKit's WebSocket protocol matches the proxy assumptions                                   | M4 start | Inspect actual protocol via M2 logs      |

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

1. **Final-day move from Mac mini → MBP.** All artifacts (merged model, MLX-converted
   model, daemon code, Gemma GGUF) need to be on the MBP by Sunday evening. Plan Sunday
   night for full integration test on demo machine.

1. **WhisperLiveKit version churn.** The project moves fast; pinning a specific version
   is recommended. Pre-Implementation Blocker #5 includes verifying our assumptions
   against the installed version.
