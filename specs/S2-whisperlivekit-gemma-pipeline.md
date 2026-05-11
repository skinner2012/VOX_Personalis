<!-- pyml disable-num-lines 9999 line-length -->

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
- [Demo MBP Setup](#demo-mbp-setup)
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

| Property                        | Whisper VAD-batch (S1 path)                      | Streaming Zipformer (failed)               | **WhisperLiveKit (this spec)**                             |
| ------------------------------- | ------------------------------------------------ | ------------------------------------------ | ---------------------------------------------------------- |
| Streaming UX                    | After full VAD pause + encoder pass (3–10s)      | 320ms partials mid-utterance               | **600–1500ms commit cadence, partials in flight**          |
| Architecture                    | Whisper offline                                  | True chunked streaming                     | Whisper offline + AlignAtt sliding-window emission policy  |
| Personalization for Deaf accent | LoRA on whisper-small.en (proven 34.05% val WER) | Full fine-tune (failed at 3.7h scale)      | **Same proven LoRA, merged into base**                     |
| Inference latency on M4 Pro     | ~real-time but blocks until pause                | ~real-time true-streaming                  | Real-time with sliding window, faster-whisper (CT2) on CPU |
| Apple Silicon path              | PyTorch / CTranslate2                            | sherpa-onnx                                | **faster-whisper (CT2) — MLX abandoned (M2 postmortem)**   |
| Open source maturity            | Mature                                           | icefall PEFT non-streaming-only (dead end) | **Mature; ships WebSocket frontend**                       |
| Time to ship                    | n/a (already exists)                             | 5+ days, high risk                         | **~36 hours**                                              |

WhisperLiveKit uses the AlignAtt streaming policy
([Papi et al., Interspeech 2023](https://arxiv.org/abs/2305.11408)): it monitors decoder
cross-attention to identify when each token is "stable enough" to commit, and holds
unstable tokens in a "buffer" tier. The user sees grey/italic in-flight text turning into
white/committed text on a sub-second cadence. Sub-second word-by-word feedback is
acceptable for the user's use case (live self-monitoring during speech is tolerant of
~1s latency; a Deaf speaker monitoring articulation does not need 320ms cadence).

Two orthogonal CLI choices in `wlk`: the **inference backend** selects which Whisper
runtime to use (e.g., `mlx-whisper`, `faster-whisper`); the **streaming policy**
selects between SimulStreaming (AlignAtt, default) and LocalAgreement-2. After M2 we
ship with `faster-whisper` + LocalAgreement — `mlx-whisper` was eliminated by a
threading bug under wlk; either streaming policy works on faster-whisper, and
LocalAgreement was the one verified in M2's smoke test.

**The killer property:** WhisperLiveKit accepts a merged Whisper checkpoint as a regular
HF model. Our S1-M7 LoRA can be merged once and served with no special handling — we get
LoRA accuracy on a thread-safe CT2 runtime. (The `--lora-path` flag is documented as
PyTorch-backend-only and incurs runtime adapter overhead; merging is the right move
regardless of backend.)

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
│ M1 — LoRA merge + conversions (~45 min)                         │
│   merge_lora.py: merge_and_unload() →                           │
│     out/whisper_small_en_s1m7_merged/                           │
│   ct2-transformers-converter (serving path) →                   │
│     out/whisper_small_en_s1m7_merged_ct2/                       │
│   hf_to_mlx.py (batch-eval path) →                              │
│     out/whisper_small_en_s1m7_merged_mlx/                       │
│   (custom MLX script — pip mlx-whisper has no convert subcmd;   │
│    see "LoRA Merge" for key mapping + CT2 preprocessor hack)    │
│   Smoke test all three formats on 1 val clip.                   │
│   Gate: CT2 and MLX hypotheses match HF hypothesis (modulo fp16)│
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ M2 — WhisperLiveKit standalone (~1-2h)                          │
│   Run wlk --backend faster-whisper with merged CT2 model.       │
│   (mlx-whisper attempted first, abandoned due to thread-local   │
│    Stream(gpu, 1) bug under wlk's asyncio.to_thread workers —   │
│    see "M2 Postmortem" in WhisperLiveKit Serving section.)      │
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
  (this also pulls `faster-whisper` and `ctranslate2`, which are our serving backend)
- **mlx-whisper** — install in M1 prep: `pip install mlx-whisper` (used by M3/M4
  batch eval only; not by the wlk server — see M2 postmortem)
- **python-multipart** — `pip install python-multipart`. Required by wlk's REST
  endpoints; the wheel does not pull it transitively in 0.2.20.post1.

### Day 0 commands to run before M0

```bash
brew install ffmpeg                          # required by openai-whisper
source venv/bin/activate                     # Python 3.11.15
pip install mlx-whisper                      # M3/M4 batch eval only
pip install "whisperlivekit[whisper]"        # also pulls faster-whisper + ctranslate2
pip install python-multipart                 # required by wlk REST endpoints
```

`mlx-whisper` will fail to install on non-macOS — that's fine, we're macOS-only by spec.

**No A10 work for Phase 1.** No new cloud spend.

**Python 3.11 patch (already applied):** `whisperlivekit==0.2.20.post1` contains an
f-string with a backslash in `cli.py:371` that is a syntax error under Python 3.11
(allowed only in 3.12+). The line has been patched in-place in the venv to assign the
escape sequence to a variable before the f-string. No functional change.

### pyproject.toml updates

```text
# Add to [project.dependencies]:
whisperlivekit==0.2.20.post1  # latest PyPI release; 0.4 series does not exist
mlx-whisper==0.4.3            # Apple Silicon MLX backend (M3/M4 batch eval only)
silero-vad>=6.2               # already present; reused by WhisperLiveKit --vac
python-multipart>=0.0.20      # wlk REST endpoints; not pulled in transitively
# faster-whisper + ctranslate2 are pulled in transitively by whisperlivekit[whisper]
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
│  wlk serve --backend faster-whisper                                     │
│      --model_dir out/whisper_small_en_s1m7_merged_ct2                  │
│      --host 127.0.0.1 --port 8001                                      │
│      --backend-policy localagreement                                    │
│      --warmup-file ""                                                   │
│                                                                         │
│  Internally:                                                            │
│    Audio buffer (rolling window) → CT2 Whisper-small.en (LoRA-merged)  │
│    LocalAgreement-2 emission policy → {committed, buffer} per ~500ms-1s│
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

After merge, build **two** target formats:

1. **CTranslate2** — for wlk streaming (M2 production path; faster-whisper backend):

   ```bash
   # ct2-transformers-converter wants preprocessor_config.json; newer transformers
   # saves it as processor_config.json. Make a copy so the converter is happy.
   cp out/whisper_small_en_s1m7_merged/processor_config.json \
      out/whisper_small_en_s1m7_merged/preprocessor_config.json

   ct2-transformers-converter \
     --model out/whisper_small_en_s1m7_merged \
     --output_dir out/whisper_small_en_s1m7_merged_ct2 \
     --quantization float16 \
     --copy_files tokenizer.json preprocessor_config.json \
     --force
   ```

   Output: `out/whisper_small_en_s1m7_merged_ct2/` (~461MB; `model.bin` +
   `config.json` + `tokenizer.json` + `vocabulary.json` + `preprocessor_config.json`).

1. **MLX** — kept for our own batch eval scripts that drive MLX from the main thread
   (no wlk threading involvement; see M2 Postmortem). NOT used at serving time.

   ```bash
   python -m scripts.vox_daemon.hf_to_mlx \
     --hf-dir ./out/whisper_small_en_s1m7_merged \
     --mlx-dir ./out/whisper_small_en_s1m7_merged_mlx \
     --dtype float16
   ```

**Why `merge_and_unload` is safe:** the merge is a deterministic linear combination
(`W_merged = W_base + alpha/rank * B @ A`) — produces a model with identical numerical
behavior to the LoRA-active model under inference, with zero overhead. This is the
standard PEFT export path.

**Why a custom HF→MLX converter (and not `python -m mlx_whisper.convert`):** the
`mlx-whisper` PyPI package (0.4.3, latest) ships only the runtime; the `convert.py` tool
lives in the [`ml-explore/mlx-examples`](https://github.com/ml-explore/mlx-examples/blob/main/whisper/convert.py)
GitHub repo, not in the pip distribution. Even if vendored, that `convert.py` saves the
weights as `model.safetensors`, but `mlx_whisper.load_models` looks for
`weights.safetensors` (or `weights.npz`) — so a rename step would still be required.
Our `hf_to_mlx.py` is ~70 lines, reuses the installed `mlx_whisper.whisper` /
`torch_whisper` modules as a library, and writes the file with the correct name.

**Conversion key-mapping** (HF Transformers → MLX OpenAI-style), implemented in
`hf_to_mlx.py`:

```text
HF state_dict key                              → MLX key
─────────────────────────────────────────────────────────────────────
model.encoder.conv1.weight                     → encoder.conv1.weight
model.encoder.embed_positions.weight           → (dropped — sinusoids regenerated)
model.encoder.layers.N.self_attn.q_proj        → encoder.blocks.N.attn.query
model.encoder.layers.N.self_attn.k_proj        → encoder.blocks.N.attn.key
model.encoder.layers.N.self_attn.v_proj        → encoder.blocks.N.attn.value
model.encoder.layers.N.self_attn.out_proj      → encoder.blocks.N.attn.out
model.encoder.layers.N.self_attn_layer_norm    → encoder.blocks.N.attn_ln
model.encoder.layers.N.fc1                     → encoder.blocks.N.mlp1
model.encoder.layers.N.fc2                     → encoder.blocks.N.mlp2
model.encoder.layers.N.final_layer_norm        → encoder.blocks.N.mlp_ln
model.encoder.layer_norm                       → encoder.ln_post
model.decoder.embed_tokens.weight              → decoder.token_embedding.weight
model.decoder.embed_positions.weight           → decoder.positional_embedding
model.decoder.layers.N.encoder_attn.*          → decoder.blocks.N.cross_attn.*
model.decoder.layers.N.encoder_attn_layer_norm → decoder.blocks.N.cross_attn_ln
model.decoder.layer_norm                       → decoder.ln
proj_out.weight                                → (dropped — tied to token_embedding)
```

Conv1d weights additionally need an axis swap (`(out, in, kernel)` → `(out, kernel, in)`).
Output: `weights.safetensors` + `config.json` with the OpenAI-style dim keys
(`n_mels`, `n_audio_ctx`, `n_audio_state`, …, `n_text_layer`, `model_type: "whisper"`).

### WhisperLiveKit Serving

WhisperLiveKit is invoked as a subprocess by the daemon. We do NOT use its built-in
static-file server (we serve our own HTML). Bind to 127.0.0.1 to prevent external
access — `--no-static` does not exist in 0.2.20.

```bash
wlk serve \
    --backend faster-whisper \
    --model_dir ./out/whisper_small_en_s1m7_merged_ct2 \
    --host 127.0.0.1 --port 8001 \
    --min-chunk-size 0.5 \
    --vac-chunk-size 0.04 \
    --backend-policy localagreement \
    --warmup-file "" \
    --pcm-input
```

Key flags (verified against `wlk serve --help` on 0.2.20.post1):

- `--backend faster-whisper`: CTranslate2 runtime (C++, thread-safe). Reliable on M4
  Pro CPU at ~0.1× RTF for small.en — plenty fast for live streaming. We tried
  `mlx-whisper` first for the GPU speedup; it failed in wlk due to a thread-locality
  bug in MLX's runtime — see the **M2 Postmortem** below.
- `--model_dir`: path to a local Whisper checkpoint directory. **Not `--model`** — that
  flag accepts only a size string (e.g. `small.en`); `--model_dir` overrides it for
  local paths.
- `--backend-policy localagreement`: the LocalAgreement-2 emission policy from
  whisper_streaming. wlk 0.2.20's default is `simulstreaming` (AlignAtt). Both work
  with faster-whisper; LocalAgreement was the verified path in M2.
- `--vac-chunk-size 0.04`: VAC is **enabled by default** (Silero VAD internally, via
  onnxruntime). No `--vac` flag needed — pass `--no-vac` to disable. WhisperLiveKit
  feeds 32ms frames (`0.04s ≈ 512 samples at 16kHz`) to the Silero ONNX model for
  endpoint detection. Required for the endpoint events that trigger Gemma Stage B.
- `--warmup-file ""`: explicit empty to suppress wlk's default warmup-file fetch
  (which falls back to a missing path under our install).
- `--pcm-input`: required for the included `whisperlivekit.test_client` smoke test,
  which streams raw 16kHz mono PCM. The Chrome frontend will use Opus-over-WebRTC
  instead and won't need this flag — M5 will revisit.
- **No `--lora-path`**: LoRA is merged into the base weights before serving.
- **No `--no-static`**: flag does not exist. Localhost-only binding (`--host 127.0.0.1`)
  is sufficient to isolate WhisperLiveKit's bundled UI from external access.

#### M2 Postmortem — Why mlx-whisper is unusable inside wlk

Despite mlx-whisper being the documented "Apple Silicon native, 4-6× faster" backend,
**mlx-whisper running under wlk's `AudioProcessor` always raises**
`RuntimeError: There is no Stream(gpu, 1) in current thread.`

Root cause:

- wlk's `AudioProcessor.transcription_processor` calls
  `await asyncio.to_thread(self.transcription.process_iter)`
  ([audio_processor.py:370](../venv/lib/python3.11/site-packages/whisperlivekit/audio_processor.py#L370)),
  dispatching each transcription onto a Python `ThreadPoolExecutor` worker thread.
- MLX's runtime stores `Stream(gpu, 1)` as **thread-local state**, lazily created on
  the thread where MLX is first used. Worker threads from `to_thread` never have this
  stream — every MLX call raises.
- Both wlk backend policies hit the same wall, at different call sites:
  - LocalAgreement → `mlx_whisper/decoding.py:600` (`mx.async_eval`)
  - SimulStreaming → `simul_whisper/simul_whisper.py:265`
    (`torch.as_tensor(mlx_encoder_feature)` via dlpack bridge)

Things tried and discarded (all reverted; no live patches remain in venv):

1. **wlk + mlx-whisper + default (SimulStreaming) policy.**
   Fails at the dlpack bridge — same `Stream(gpu, 1)` error.
1. **Wrap `MLXWhisper.transcribe` in `with mx.stream(mx.gpu):`** (LocalAgreement path).
   `mx.stream(gpu)` selects default `Stream(gpu, 0)`; the internal `Stream(gpu, 1)`
   used by `async_eval` is still missing on the worker thread.
1. **Replace `mx.async_eval` with sync `mx.eval`** in `mlx_whisper/decoding.py`.
   Sync `mx.eval` also hard-requires `Stream(gpu, 1)` on the calling thread.
1. **Force `--backend-policy localagreement`** to dodge the SimulStreaming dlpack path.
   Same `Stream(gpu, 1)` error from a different call site.

A workable MLX fix would need either (a) a wlk patch that pins transcription to the
main event-loop thread (defeats wlk's concurrency model) or (b) an MLX patch that
makes streams global / thread-shared (upstream change). Neither is in scope for a
Monday demo. We accept the perf loss and ship faster-whisper.

`mlx-whisper` is still used in our own **batch eval** path (M3 / M4), where we drive
the model on the main thread and the threading bug doesn't trigger. The MLX
checkpoint at `out/whisper_small_en_s1m7_merged_mlx/` is retained for that purpose.

**WebSocket protocol — verified against `timed_objects.py` in 0.2.20.post1.**

Consumed by daemon, not browser. Full `FrontData.to_dict()` shape:

```json
{
  "status": "",
  "lines": [
    {"speaker": 1, "text": "I want to talk about Kubernetes networking",
     "start": "0.00", "end": "2.40"}
  ],
  "buffer_transcription": "networking",
  "buffer_diarization": "",
  "buffer_translation": "",
  "remaining_time_transcription": 0.0,
  "remaining_time_diarization": 0.0
}
```

Notes: `start`/`end` are formatted strings (e.g. `"2.40"`), not floats. `speaker` is 1
when diarization is off (mapped from internal -1). `buffer_transcription` is the
in-flight partial hypothesis (what the spec called `buffer_text`).

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
    merge_lora.py          # M1: merge S1-M7 LoRA → HF Transformers checkpoint
    hf_to_mlx.py           # M1: convert HF Transformers checkpoint → MLX format
    lora_sanity.py         # M0: load LoRA, transcribe 1 clip (sanity gate)
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

Goal: confirm `merge_and_unload` didn't change inference numerics. We evaluate the
**HF merged checkpoint** here (not MLX or CT2): driving the HF model on the main
thread is the most direct check that the LoRA bake-in is correct. M4's offline eval
covers the format-conversion-induced drift separately.

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

## Demo MBP Setup

Step-by-step to bring up the streaming pipeline on a clean MBP (the actual demo
machine on 2026-05-11). Source machine = the Mac mini where M0–M2 were built.

### 1. OS prerequisites on MBP

```bash
# Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# ffmpeg — required by whisper / openai-whisper / faster-whisper
brew install ffmpeg

# Python 3.11 — wlk 0.2.20 has a Python 3.12-only f-string on cli.py:371; we ship
# the in-place patch via this repo's `setup_demo.sh`, but only Python 3.11 has been
# validated end-to-end. Apple ships 3.13 on recent macOS; install 3.11 explicitly:
brew install python@3.11
```

### 2. Clone repo + create venv on MBP

```bash
git clone <repo-url> ~/Projects/VOX_Personalis
cd ~/Projects/VOX_Personalis
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pip install mlx-whisper                  # for batch eval (M3/M4)
pip install "whisperlivekit[whisper]"    # pulls faster-whisper + ctranslate2
pip install python-multipart             # wlk REST endpoints
```

### 3. Apply the wlk Python 3.11 f-string patch

`whisperlivekit==0.2.20.post1`'s `cli.py:371` contains an f-string with a backslash
that 3.11 rejects. Patch in place:

```bash
python - <<'PY'
import re, pathlib, whisperlivekit
p = pathlib.Path(whisperlivekit.__file__).parent / "cli.py"
src = p.read_text()
bad = 'print(f"  ffmpeg:       {\'found\' if _check_ffmpeg() else \'\\033[31mNOT FOUND\\033[0m (required)\'}")'
good = (
    '_ffmpeg_status = "found" if _check_ffmpeg() else "\\033[31mNOT FOUND\\033[0m (required)"\n'
    '    print(f"  ffmpeg:       {_ffmpeg_status}")'
)
if bad in src:
    p.write_text(src.replace(bad, good))
    print("patched")
else:
    print("already patched or upstream changed; inspect cli.py:371")
PY
```

### 4. Get the merged model onto the MBP

Two options — pick whichever is faster on the day. We do NOT commit the binaries
to git (too large; the LoRA adapter at
`out/feedback_finetune/batch_20260317_110057/checkpoint/` is already committed
and is the source of truth for accuracy).

**Option A — rsync from Mac mini** (fastest, ~1 min on LAN):

```bash
# CT2 checkpoint (serving — required for M2 / M5)
rsync -avzP <mac-mini>:~/Projects/VOX_Personalis/out/whisper_small_en_s1m7_merged_ct2 \
     ~/Projects/VOX_Personalis/out/

# Merged HF checkpoint (used by M3 eval_merged.py)
rsync -avzP <mac-mini>:~/Projects/VOX_Personalis/out/whisper_small_en_s1m7_merged \
     ~/Projects/VOX_Personalis/out/

# MLX checkpoint (used by M4 eval_offline.py)
rsync -avzP <mac-mini>:~/Projects/VOX_Personalis/out/whisper_small_en_s1m7_merged_mlx \
     ~/Projects/VOX_Personalis/out/
```

**Option B — rebuild on the MBP from the committed LoRA adapter**
(~3 min on M4 Pro; uses only files already in the repo):

```bash
source venv/bin/activate

# 1. Merge LoRA into base whisper-small.en → HF Transformers format
python -m scripts.vox_daemon.merge_lora
# Reads: out/feedback_finetune/batch_20260317_110057/checkpoint/  (LoRA, 7.1MB, in git)
# Writes: out/whisper_small_en_s1m7_merged/                       (HF, ~950MB)

# 2. Convert HF → CT2 for the wlk serving path
cp out/whisper_small_en_s1m7_merged/processor_config.json \
   out/whisper_small_en_s1m7_merged/preprocessor_config.json
ct2-transformers-converter \
  --model out/whisper_small_en_s1m7_merged \
  --output_dir out/whisper_small_en_s1m7_merged_ct2 \
  --quantization float16 \
  --copy_files tokenizer.json preprocessor_config.json \
  --force
# Writes: out/whisper_small_en_s1m7_merged_ct2/                   (CT2, ~461MB)

# 3. Convert HF → MLX for batch eval (M3/M4)
python -m scripts.vox_daemon.hf_to_mlx \
  --hf-dir ./out/whisper_small_en_s1m7_merged \
  --mlx-dir ./out/whisper_small_en_s1m7_merged_mlx \
  --dtype float16
# Writes: out/whisper_small_en_s1m7_merged_mlx/                   (MLX, ~481MB)
```

Both options produce bit-identical outputs (the merge is a deterministic linear
combo; CT2 quantization is deterministic given the same input).

### 5. Transfer Gemma GGUF (~17GB)

If the demo MBP doesn't already have `~/llama.cpp/build/bin/llama-cli` + the GGUF,
copy them over. The GGUF is the slowest single transfer — run it overnight:

```bash
# llama.cpp binary (assumes the source MBP has a working build)
rsync -avzP <mac-mini>:~/llama.cpp ~/llama.cpp

# Gemma 4 GGUF
mkdir -p ~/Projects/VOX_Personalis/models/gemma
rsync -avzP <mac-mini>:~/Projects/VOX_Personalis/models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
     ~/Projects/VOX_Personalis/models/gemma/
```

### 6. End-to-end smoke test on MBP

```bash
source venv/bin/activate

# Terminal 1: start wlk
wlk serve \
  --backend faster-whisper \
  --model_dir ./out/whisper_small_en_s1m7_merged_ct2 \
  --backend-policy localagreement \
  --host 127.0.0.1 --port 8001 \
  --min-chunk-size 0.5 \
  --warmup-file "" \
  --pcm-input \
  -l INFO

# Terminal 2: drive it with a known clip
python -m whisperlivekit.test_client \
  --url ws://localhost:8001/asr \
  --speed 1 \
  ~/Downloads/takeout-E407/euphonia_002f4f2d5ad6ecd94202d4ef92719c02.wav
```

Expected client output:

```text
[0:00:00.00 -> 0:00:03.38] can you play some music yes
[buffer] can you play some music yes
--- 90 responses | 49 updates | 3.4s audio ---
```

If the output text appears, **the MBP demo path is ready**. Subsequent milestones
(M3 WER parity, M4 Gemma Stage B, M5 daemon + Chrome) build on top of this.

### 7. Run the M5 demo

The vox_daemon orchestrator spawns wlk + Gemma + the proxy + the Chrome
frontend with one command. Stop any leftover wlk / llama-cli processes from
step 6 first; the daemon needs ports 8000 and 8001 free.

```bash
source venv/bin/activate
pkill -f "whisperlivekit.basic_server" 2>/dev/null
pkill -f llama-cli 2>/dev/null

python -m scripts.vox_daemon --open-browser
# (or: --port 8000 --wlk-port 8001 --gemma-correction-timeout 1.5)
```

The daemon prints these signposts in order; expect ~5 s end-to-end on M4 Pro:

```text
[daemon] spawning wlk: …
[wlk:err] INFO:     Application startup complete.
[daemon] wlk is ready
[daemon] starting Gemma worker (this can take ~3-50s)…
[daemon] Gemma worker ready in 3.2s
[daemon] serving at http://localhost:8000
```

Chrome opens to `http://localhost:8000`. First-time setup:

- **Allow microphone** in the browser permission prompt.
- macOS prerequisite: System Settings → Privacy & Security → Microphone →
  enable Google Chrome. If the OS-level toggle is off, the page-level
  permission silently fails with `NotAllowedError`/`NotFoundError`.

Then click **Start mic** and speak. You should see:

- Buffer (grey italic) growing as you speak.
- Committed line text appearing within ~1 s of words stabilising.
- Pulsing `•••` polishing indicator on the line while Gemma is mid-correction.
- Stage B (white bold) replacing the line ~1–2 s after a natural speech pause.
- Earlier utterances moving into the History pane (most-recent first, capped
  at 5).

Both Stage A and Stage B lines are `contenteditable` — click in to type. Per
M5 design, Stage B unconditionally overwrites whatever's there when it
arrives, so user edits inside the polishing window may be clobbered; that's
an accepted trade-off (see "Accepted Phase 1 Limitations").

Stop the daemon with Ctrl+C in the terminal; cli.py tears down Gemma and wlk
cleanly. Logs and state are ephemeral by design — no audio is written to disk.

______________________________________________________________________

## Pre-Implementation Blockers

1. **[Done — M0]** Identify which `out/feedback_finetune/batch_<id>/` corresponds to S1-M7
   (val WER 34.05%). → `batch_20260317_110057/` (verified `metrics.json`).
1. **[Done — M0]** Transfer checkpoint dir from MBP → Mac mini via rsync. → checkpoint
   present locally.
1. **[Done — Day 0]** WhisperLiveKit installs cleanly on M4 Pro with MLX backend.
   → `whisperlivekit==0.2.20.post1` + `mlx-whisper==0.4.3` installed. Python 3.11
   f-string patch applied in-place to `cli.py:371`. `wlk --help` works.
1. **[Resolved — Day 0]** `python -m mlx_whisper.convert` does NOT exist in the pip
   package (0.4.3); the conversion tool is GitHub-only (`ml-explore/mlx-examples`).
   Replaced with a custom `scripts/vox_daemon/hf_to_mlx.py` script. Final HF→MLX
   numerical parity is verified by M3's WER gate.
1. **[Done — Day 0]** WhisperLiveKit's CLI flags + WebSocket protocol verified.
   → WS endpoint: `/asr`. Flags: `--model_dir` (not `--model`),
   `--backend-policy simulstreaming` for AlignAtt, VAC on by default.
   Protocol: `lines[].{speaker, text, start, end}` + `buffer_transcription`
   (string). See "WhisperLiveKit Serving" section.
1. **[Resolved — M2]** mlx-whisper backend fails inside wlk with
   `RuntimeError: There is no Stream(gpu, 1) in current thread` because wlk runs
   transcription in `asyncio.to_thread` workers and MLX streams are thread-local.
   Resolution: pivot serving to `--backend faster-whisper` with a CTranslate2 build
   of the merged checkpoint. MLX checkpoint is retained for batch eval only. Full
   postmortem in the "WhisperLiveKit Serving" section.
1. **[Resolved — M2]** `ct2-transformers-converter` requires `preprocessor_config.json`,
   but newer `transformers` saves the processor as `processor_config.json`. We copy
   the file before invoking the converter; see "LoRA Merge".
1. **[Done — M2]** `python-multipart` is required by wlk's REST endpoints. It was
   missing in the default install path; installed manually and added to the demo
   setup checklist below.

______________________________________________________________________

## Open Questions / Risks

1. **mlx-whisper merged-LoRA support — resolved (negative).** The `merge_and_unload`
   path produces a regular HF Whisper checkpoint, and our `hf_to_mlx.py` script
   converts it to MLX format successfully. But mlx-whisper **cannot be used as wlk's
   serving backend** due to MLX's thread-local Stream(gpu, 1) requirement — see the
   "M2 Postmortem" in the WhisperLiveKit Serving section. Serving runs on
   `faster-whisper` + a CTranslate2 build of the merged checkpoint. The MLX
   checkpoint is retained for M3/M4 batch eval where the main-thread driver pattern
   sidesteps the bug.

1. **WhisperLiveKit's commit event format — resolved.** Protocol verified from source:
   `lines[]` grows as utterances commit; daemon diffs successive `lines` arrays to detect
   new committed lines. `buffer_transcription` is the in-flight partial. `start`/`end`
   are formatted strings, not floats.

1. **VAD/endpoint tuning.** WhisperLiveKit's VAC is on by default (Silero ONNX).
   Default `--vac-chunk-size` may not match the user's speech patterns (Deaf speech often
   has different prosody). M5 stability testing tunes this flag.

1. **Gemma + Whisper Metal contention on MBP.** Both use Metal. If concurrent Whisper
   inference + Gemma generation contends, we may see Whisper stalling during Gemma's
   ~400ms generation. Mitigation: serialize, or run Gemma on CPU only (`-ngl 0` in
   llama.cpp). M5 measures.

1. **Final-day move from Mac mini → MBP.** Detailed checklist in the
   **"Demo MBP Setup"** section below — covers OS prereqs, Python env, package
   install, model-artifact transfer (rsync vs HF Hub fallback), and the smoke-test
   command to confirm end-to-end before the demo.

1. **WhisperLiveKit version — resolved.** Installed and pinned: `==0.2.20.post1` (latest
   PyPI release; the `>=0.4` requirement in the original spec was incorrect — the package
   follows a 0.2.x version scheme). Pinned in `pyproject.toml`.
