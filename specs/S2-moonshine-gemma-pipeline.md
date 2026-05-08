# VOX Personalis Stage 2 — Moonshine v2 + Gemma 4 ASR Pipeline

**Status:** ACTIVE\
**Branch:** main\
**Supersedes:** S1-M7 (Whisper fine-tuned baseline, 34.05% val WER)

______________________________________________________________________

## Table of Contents

- [Problem](#problem)
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
- [Pre-work: Day 0 Checklist](#pre-work-day-0-checklist-2h-blocks-everything)
- [System Architecture (Layer 1)](#system-architecture-layer-1)
- [Component Specs](#component-specs)
  - [TextConsumer Protocol](#textconsumer-protocol)
  - [WebConsumer](#webconsumer)
  - [SileroVAD Config (Phase 1 values)](#silerovad-config-phase-1-values)
  - [Moonshine v2 Streaming Integration](#moonshine-v2-streaming-integration)
  - [Gemma 4 GGUF Integration](#gemma-4-gguf-integration)
  - [Chrome Frontend](#chrome-frontend)
  - [Daemon CLI](#daemon-cli)
- [Module Structure](#module-structure)
- [Evaluation Specs](#evaluation-specs)
  - [M0: Moonshine v2 Streaming Baseline WER](#m0-moonshine-v2-streaming-baseline-wer)
  - [M1: Moonshine v2 Streaming Fine-tuning](#m1-moonshine-v2-streaming-fine-tuning)
  - [M2: Gemma 4 Correction Evaluation](#m2-gemma-4-correction-evaluation)
- [Error / Rescue Registry (Layer 1)](#error--rescue-registry-layer-1)
- [Success Criteria (5-Day Demo)](#success-criteria-5-day-demo)
- [Layer 2: AX Integration Stub (Post-demo)](#layer-2-ax-integration-stub-post-demo)
  - [Decisions Made (D1–D7)](#decisions-made-d1d7)
  - [IPC Wire Format](#ipc-wire-format)
  - [Swift AX Injector Contract](#swift-ax-injector-contract)
  - [Swap to AX](#swap-to-ax)
  - [MVP 1 Pass Criterion](#mvp-1-pass-criterion)
- [Pre-Implementation Blockers](#pre-implementation-blockers)

______________________________________________________________________

## Problem

Stage 1 ended with a fine-tuned Whisper small.en model at 34.05% val WER.
That is the bar. The question Stage 2 answers: can Moonshine v2 + Gemma 4 do
better — and can it do so in real time, on-device, on Apple Silicon?

Whisper is being replaced. Moonshine v2 is the new ASR engine. Gemma 4 is the
correction layer on top of it. Stage 2 proves whether this new stack beats the
Whisper baseline, then ships a live demo of the pipeline in action.

**This is not a captioning app.** The end use case — the Deaf speaker using their
own voice in a job interview — is the motivation. The Stage 2 work is: prove the
new ASR stack is better, then put it in front of a real user.

______________________________________________________________________

## Core Product Flow

```text
User speaks
  → local ASR (Moonshine v2, fine-tuned for Deaf accent)
  → Stage A: processing text  — raw transcript, appears immediately
  → Stage B: committed text   — Gemma-corrected, replaces Stage A within 800ms
  → display (Phase 1: Chrome tab / Phase 2: Live Captions Type to Speak)
  → user reviews, submits
```

**Why two stages?** Every real-time ASR product faces a tension: show text fast
(low latency) or show text accurately (high quality). Stage A resolves the tension
by giving the user immediate visual feedback while Gemma corrects in the background.
The user sees their speech reflected instantly. The correction arrives before they
have time to read and react. No waiting, no stale display.

**Why "text consumer agnostic"?** The ASR pipeline (Moonshine + Gemma) does not
know or care how text reaches the user. It calls three methods on a TextConsumer:
`on_stage_a()`, `on_stage_b()`, `on_clear()`. Today the consumer is a Chrome
WebSocket display. In Phase C it will be a Swift AX injector writing to Live
Captions. The pipeline does not change when the consumer changes.

______________________________________________________________________

## Constraints

- **Fully local.** No cloud ASR, no cloud LLM, no audio upload. All inference runs
  on-device. Privacy is non-negotiable: the user's voice never leaves the machine.
- **Single speaker.** The model is personalized to one voice. Not multi-user.
- **macOS only** (Phase 1 + Phase 2). Android is Phase 3 — out of scope here.
- **No automatic TTS triggering.** User controls when text is submitted/spoken.
- **No audio retention** without user opt-in.

______________________________________________________________________

## 10x Vision

WER drops below 20% after 6 months of feedback fine-tuning. The system works in
any macOS text field (not just Live Captions). The user's voice becomes their most
precise communication tool, not their lossiest one. The Deaf speaker enters a
job interview, speaks naturally, and their words appear on screen with the clarity
of a fluent typist — no corrections, no hesitation, no apology for the technology.

______________________________________________________________________

## Hardware

| Machine    | Specs            | Role                                               |
| ---------- | ---------------- | -------------------------------------------------- |
| Mac mini   | M4 Pro, 64GB RAM | Fine-tuning (M1), evaluation (M0, M2), development |
| MBP M4 Pro | M4 Pro, 48GB RAM | Live demo (M3), daily driver                       |

Both machines run macOS 14.0+ (Sonoma), Apple Silicon. PyTorch MPS (Metal
Performance Shaders) backend and llama.cpp Metal acceleration apply to both.
Fine-tuning jobs run on Mac mini (desktop, can run overnight). The hiring
manager demo runs on MBP (portable).

**Inference framework:** PyTorch + HuggingFace Transformers with MPS backend
(`device = "mps"`). This is the same approach used for Whisper fine-tuning in
S1.

**Why not MLX?** MLX (Apple's Machine Learning eXtensions) was the original plan
for faster Apple Silicon inference and fine-tuning. However, as of 2026-05-08:

- `mlx-moonshine` does not exist as a pip package.
- `mlx-audio` (by Blaizzy) supports Moonshine v1 only, NOT v2 streaming.
- No `mlx-community/moonshine-streaming-*` weights exist on HuggingFace.
- Moonshine v2 streaming (`MoonshineStreamingForConditionalGeneration`) is only
  available via HuggingFace Transformers (PyTorch).
- `mlx-tune` cannot fine-tune Moonshine v2 streaming (no MLX model exists).

**Do not attempt MLX for Moonshine v2 streaming.** Use PyTorch + Transformers + MPS.

**Fine-tuning framework:** PyTorch + PEFT/LoRA on MPS. Same LoRA approach as S1
(target attention layers, rank=16, alpha=32) but different target layer names
(Moonshine v2 streaming architecture, not Whisper).

**Gemma on Mac mini (64GB):** 26B MoE Q4 is the primary choice — fits with 40GB+
to spare for the OS and Moonshine. E2B is the fallback if 26B MoE GGUF is not
yet available for llama.cpp.

**Gemma on MBP (48GB):** Same choice — 48GB is sufficient for 26B MoE Q4
(~10-12GB footprint) alongside Moonshine (~1GB for medium). Day 0 benchmark
confirms.

______________________________________________________________________

## What Is Out of Scope (Phase 1)

| Item                                      | Deferred to                                     |
| ----------------------------------------- | ----------------------------------------------- |
| Android client                            | Phase 3                                         |
| FastAPI serving layer                     | Phase 2 (exists at scripts/serving/, untouched) |
| AX injection into Live Captions           | Phase C (post-demo)                             |
| IME (InputMethodKit) path                 | Phase C (if AX fails)                           |
| Stage B suppression (user-edit detection) | Phase 2                                         |
| Visual Stage A/B differentiation in AX    | AX limitation — plain text only                 |
| Automated test suite                      | Phase 2                                         |
| Menu bar status app                       | Explicitly skipped                              |
| Auto-submit after silence                 | Explicitly skipped                              |
| Gemma rollback hotkey                     | Explicitly skipped                              |

______________________________________________________________________

## Accepted Phase 1 Limitations

- Stage B is applied unconditionally. If the user edits Stage A text manually
  before Stage B arrives, Stage B overwrites the edit. User-edit detection
  (Stage B suppression) requires bidirectional IPC — deferred to Phase 2.
- No visual distinction between Stage A and Stage B in the AX path (Phase C):
  AXUIElement sets plain text only, no character-level styling. Chrome display
  (Phase B) uses color/italic to distinguish them.
- `on_clear()` is a no-op in Phase 1. The field retains Stage B text until
  the next utterance's Stage A fires.

______________________________________________________________________

## Baseline to Beat

All prior evaluation used the same held-out val set and `textnorm_v2` normalization.
New evals must use identical normalization for apples-to-apples comparison.

| Model                                           | WER (val set)              | Notes                                      |
| ----------------------------------------------- | -------------------------- | ------------------------------------------ |
| Whisper small.en (out-of-box)                   | ~55% (estimated)           | Pre-fine-tuning baseline                   |
| Whisper small.en (fine-tuned, S1-M7)            | **34.05%**                 | Current best — the bar to clear            |
| moonshine-streaming-small (out-of-box)          | TBD — M0                   | HF reported: 7.84% on general benchmarks   |
| moonshine-streaming-medium (out-of-box)         | TBD — M0                   | HF reported: 6.65% on general benchmarks   |
| moonshine-streaming-{small,medium} (fine-tuned) | Target: \<34.05%           | M1 gate — model chosen based on M0 results |
| moonshine-streaming-{chosen} + Gemma 4          | Target: ≤ fine-tuned alone | M2 gate                                    |

**Model selection:** M0 benchmarks both `UsefulSensors/moonshine-streaming-small`
(123M params, 561MB) and `UsefulSensors/moonshine-streaming-medium` (245M params,
1.06GB) on the val set. The model with better WER-to-latency tradeoff proceeds to
M1 fine-tuning. If the gap is marginal (\<1% WER after fine-tuning), prefer small
for faster iteration and lower resource usage.

______________________________________________________________________

## Objective

Replace Whisper with Moonshine v2 + Gemma 4 as the ASR stack. Prove the new
stack beats the fine-tuned Whisper baseline (34.05% val WER), then ship a
working live-caption demo for a hiring manager screen within 5 days.

Two-layer plan:

- **Layer 1 (now):** Evaluate + Python pipeline → Chrome demo
- **Layer 2 (post-demo):** AX injection → Live Captions Type to Speak

______________________________________________________________________

## Naming Conventions

Two naming schemes are used in this spec. They are related but distinct.

**Phase 1 / Phase 2 / Phase 3** — scope phases of Stage 2:

- **Phase 1 (this spec):** macOS only — Python daemon + Chrome display + AX injection
- **Phase 2 (future):** FastAPI serving layer + browser extension
- **Phase 3 (future):** Android client

**Phase A / Phase B / Phase C** — time sub-phases within Phase 1:

- **Phase A:** Prove the core — evaluate and fine-tune Moonshine v2, add Gemma 4 (M0–M2)
- **Phase B:** Ship the demo — Chrome live caption display (M3)
- **Phase C:** Production integration — AX injection into Live Captions (M4–M5, post-demo)

All milestones in this spec are Phase 1. Phase A/B/C are the delivery sub-phases
within it.

______________________________________________________________________

## Milestone Sequence

```text
PHASE A — PROVE THE CORE (Days 1–3)
┌─────────────────────────────────────────────────────────────────┐
│ M0 — Moonshine v2 Baseline (~2–3h)                              │
│   Install transformers+torch, run WER eval on existing val set. │
│   Benchmark BOTH moonshine-streaming-small and -medium.         │
│   Gate: does Moonshine v2 OOB WER beat Whisper OOB WER?        │
│   (Fine-tuned Whisper = 34.05% — that's the bar to clear.)     │
│   Output: "small = X% WER, medium = Y% WER" + latency data.    │
│   Decision: pick model for M1 based on WER-to-latency tradeoff.│
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ M1 — Moonshine v2 Fine-tune (~1–2 days)                         │
│   LoRA via PyTorch + PEFT on existing Deaf accent dataset.      │
│   Uses MPS backend on M4 Pro (same approach as S1 Whisper).     │
│   Gate: val WER < 34.05% (beats fine-tuned Whisper baseline).   │
│   Output: fine-tuned Moonshine v2 streaming checkpoint.         │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│ M2 — Gemma 4 Correction Evaluation (~4–8h)                      │
│   Moonshine v2 (fine-tuned) → Gemma 4 correction → WER eval.   │
│   Gate: Stage B WER < Stage A WER (correction helps, not hurts).│
│   Gate: false correction rate < 15% on val set samples.         │
│   Output: "Moonshine+Gemma = Y% WER" — go/no-go for Stage B.   │
└────────────────────────┬────────────────────────────────────────┘
                         │

PHASE B — SHIP THE DEMO (Days 3–5)
┌────────────────────────▼────────────────────────────────────────┐
│ M3 — Chrome Live Caption Demo (~1–2 days)                       │
│   Python daemon: mic → VAD → Moonshine → Gemma → WebSocket.     │
│   Chrome: Stage A (gray, live) → Stage B (white, committed).    │
│   Gate: demo runs stably for 10 min of continuous speech.       │
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
│   All D1–D7 decisions apply here — see Layer 2 stub below.      │
└─────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## Pre-work: Day 0 Checklist (~2h, blocks everything)

These must pass before writing any Phase A code. Run on Mac mini (M4 Pro, 64GB).

```bash
# 1. Verify Moonshine v2 streaming loads via HuggingFace Transformers
pip install transformers torch torchaudio  # install latest; MoonshineStreaming requires recent transformers
python -c "
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
import torch

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'Device: {device}')

# Test BOTH models
for model_id in [
    'UsefulSensors/moonshine-streaming-small',   # 123M params, 561MB
    'UsefulSensors/moonshine-streaming-medium',   # 245M params, 1.06GB
]:
    print(f'Loading {model_id}...')
    processor = AutoProcessor.from_pretrained(model_id)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).to(device)
    print(f'  OK — {sum(p.numel() for p in model.parameters())/1e6:.0f}M params')
"
# Both must load without error. If MPS fails: fallback to CPU (slower but works).
# If model download fails: check HF token / network.

# 2. Verify SileroVAD loads
pip install silero-vad
python -c "
from silero_vad import load_silero_vad
model = load_silero_vad()
print('SileroVAD OK')
"

# 3. Verify Gemma 4 loads via llama.cpp
# Download Gemma 4 E2B GGUF Q4 (~4GB) — confirm URL from Hugging Face
# Build llama.cpp with Metal support on macOS
cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j
./build/bin/llama-cli -m gemma-4-e2b-Q4_K_M.gguf -p "Correct: 'i want to talk about kubernets'" -n 50
# Target: response in <800ms on M4 Pro

# 4. Verify Gemma 4 larger model (optional, if 26B MoE GGUF exists)
# Check: does llama.cpp support Gemma 4 MoE architecture?
# Check: actual VRAM footprint of 26B MoE Q4 on M4 Pro
# If verified: upgrade Gemma spec to 26B MoE; else: stay with E2B

# 5. Verify PEFT/LoRA loads for Moonshine streaming
pip install peft
python -c "
from peft import get_peft_model, LoraConfig
print('PEFT OK')
"
```

**Moonshine model decision tree:**

```text
Day 0 result:
  Both small + medium load on MPS → benchmark both in M0, pick winner
  Only one loads on MPS → use whichever loads
  Neither loads on MPS → fallback to CPU inference (slower, still works)
```

**Gemma model decision tree:**

```text
Day 0 result:
  26B MoE Q4 loads + p95 latency <800ms → use 26B MoE (better correction)
  26B MoE unavailable or >800ms → use E2B Q4 (already committed)
  E2B Q4 >800ms → try E2B Q3 or reduce VAD max_utterance_sec to 2.0s
```

______________________________________________________________________

## System Architecture (Layer 1)

```text
┌────────────────────────────────────────────────────────────────┐
│ PYTHON DAEMON  (scripts/vox_daemon/)                            │
│                                                                 │
│  Microphone capture (sounddevice, 16kHz, 16-bit mono)          │
│       │ audio chunks                                            │
│       ▼                                                         │
│  SileroVAD (silero-vad, neural network VAD)                    │
│  threshold=0.5, min_speech_ms=300, max_utterance_sec=3.0       │
│  (replaces webrtcvad — better for atypical/Deaf speech)        │
│       │ on segment boundary (complete utterance buffer)         │
│       ▼                                                         │
│  Moonshine v2 Streaming (HuggingFace Transformers, MPS)        │
│       │ Stage A: raw transcript  (~73–107ms, model-dependent)    │
│       ├──────────────────────────────────────► consumer.on_stage_a()
│       │                                                         │
│       ▼ (concurrent, thread-based)                              │
│  Gemma 4 GGUF (llama.cpp subprocess.Popen)                     │
│  timeout=1500ms; on timeout → emit Stage A as Stage B           │
│       │ Stage B: corrected text (~400–800ms)                    │
│       └──────────────────────────────────────► consumer.on_stage_b()
│                                                                  │
│  TextConsumer (Protocol)                                         │
│    Layer 1:  WebConsumer   ──► WebSocket server ──► Chrome       │
│    Layer 2:  AXConsumer    ──► Swift injector ──► Live Captions  │
└────────────────────────────────────────────────────────────────┘
       │ WebSocket (localhost:8765)
       ▼
┌────────────────────────────────────────────────────────────────┐
│ CHROME FRONTEND  (static HTML, opened automatically)           │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Stage A (gray italic, live updates):                    │  │
│  │  "I want to talk about kubernets uh the networking..."   │  │
│  └──────────────────────────────────────────────────────────┘  │
│       ↓ replaced by Stage B within 800ms                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Stage B (white, committed):                             │  │
│  │  "I want to talk about Kubernetes networking."           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  History: last 5 committed utterances (scrollable)             │
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

The `uid` (utterance ID) is an incrementing integer assigned by the daemon per
VAD segment. Both stage_a and stage_b for the same utterance carry the same uid.
Consumers use uid to avoid applying a stale Stage B after a new Stage A fires.

**Note on on_clear():** No-op in Phase 1. Stage A of the next utterance overwrites.
Defined in the Protocol for Phase 2 migration (bidirectional IPC enables Python to
detect Return press → call on_clear() → clear the field).

______________________________________________________________________

### WebConsumer

File: `scripts/vox_daemon/consumer.py`

WebSocket server over `localhost:8765`. Broadcasts JSON messages to all connected
clients. Opens Chrome automatically on start.

**Wire format (same JSON structure used for Layer 2 IPC):**

```json
{"type": "stage_a", "text": "hello world",   "uid": 42}
{"type": "stage_b", "text": "Hello, world.", "uid": 42}
{"type": "clear"}
```

**Startup sequence:**

1. Create WebSocket server on `localhost:8765`
1. Serve static HTML from `scripts/vox_daemon/static/index.html`
1. `subprocess.run(['open', 'http://localhost:8765'])` — opens in default browser
1. Enter listen loop

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

### SileroVAD Config (Phase 1 values)

File: `scripts/vox_daemon/silero_vad.py`

**Why SileroVAD instead of webrtcvad?** The existing `scripts/serving/vad.py`
uses webrtcvad (rule-based signal processing). For Deaf/accented speech with
atypical prosody and pauses, a neural VAD is more robust — it was trained on
diverse speech data and handles unusual vocal patterns better. The existing
webrtcvad code in `scripts/serving/vad.py` remains untouched.

```python
from silero_vad import load_silero_vad, get_speech_timestamps
import torch

class SileroVADSegmenter:
    SAMPLE_RATE = 16000

    def __init__(
        self,
        *,
        threshold: float = 0.5,          # speech probability threshold
        min_silence_ms: int = 1000,       # 1s silence ends utterance
        min_speech_ms: int = 300,         # skip clips shorter than 300ms
        max_utterance_sec: float = 3.0,   # keeps ASR in 150ms budget
    ):
        self.model = load_silero_vad()
        # ... segment buffering logic, same interface pattern as VADSegmenter
```

**Interface contract:** `SileroVADSegmenter` exposes the same method pattern as
the existing `VADSegmenter` — `feed(audio_chunk) -> bytes | None`, `flush()`,
`buffer_duration_ms()` — so the daemon pipeline can use either interchangeably.

**Why 3.0s max:** Moonshine v2 streaming on M4 Pro processes ~3s audio in
~73–107ms (model-dependent). Segments longer than 3s risk missing the Stage A
\<300ms latency target. Configurable: `--max-utterance-sec` CLI flag. Default:
3.0. Hard upper bound: 8.0.

______________________________________________________________________

### Moonshine v2 Streaming Integration

File: `scripts/vox_daemon/moonshine.py`

**Architecture note:** Moonshine v2 streaming uses an ergodic encoder with
sliding-window attention — a fundamentally different architecture from Whisper.
The eval and fine-tuning code are NEW (not adapted from existing S1 Whisper
code). The HuggingFace Transformers API is the only supported inference path;
there is no MLX implementation of Moonshine v2 streaming.

```python
import torch
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

# Load once at daemon startup
device = "mps" if torch.backends.mps.is_available() else "cpu"

def load_moonshine(model_id: str = "UsefulSensors/moonshine-streaming-small"):
    """Load model + processor. Use checkpoint path for fine-tuned model."""
    processor = AutoProcessor.from_pretrained(model_id)
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32
    ).to(device)
    return model, processor

model, processor = load_moonshine(checkpoint_path or "UsefulSensors/moonshine-streaming-small")

def transcribe(audio_bytes: bytes) -> str:
    """Synchronous. Runs on ASR worker thread. Returns raw transcript."""
    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    try:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        log.error("Moonshine exception: %s — skipping segment", e)
        return ""   # empty string signals skip: no Stage A update
```

**Error handling:**

- Exception → log stderr + return `""` → daemon skips segment, continues
- Empty string return → daemon skips Stage A + Stage B for this segment
- OOM → same path (exception caught)

**Model IDs (HuggingFace):**

| Model ID                                   | Params | File size | General WER |
| ------------------------------------------ | ------ | --------- | ----------- |
| `UsefulSensors/moonshine-streaming-small`  | 123M   | 561 MB    | 7.84%       |
| `UsefulSensors/moonshine-streaming-medium` | 245M   | 1.06 GB   | 6.65%       |

Default: `moonshine-streaming-small`. M0 benchmarks both; winner proceeds to M1.

**Checkpoint:** defaults to out-of-box HuggingFace model. After M1 fine-tuning,
pass `--moonshine-model path/to/finetuned` to the daemon CLI (this points to a
local directory containing the PEFT-merged checkpoint in Transformers format).

______________________________________________________________________

### Gemma 4 GGUF Integration

File: `scripts/vox_daemon/gemma.py`

Runs as a persistent subprocess (`subprocess.Popen`). Python communicates via
stdin/stdout (one correction request per line).

```text
┌──────────────────────────────────────────────────────────────┐
│ Concurrency model                                            │
│                                                              │
│ ASR thread                    Gemma thread (concurrent)     │
│   segment ready ──────────────► queue.put(uid, stage_a_text)│
│   emit Stage A immediately     worker.get() → Gemma prompt  │
│   continue capturing            Gemma subprocess.stdin.write│
│                                 Gemma subprocess.stdout.read │
│                                 (1500ms timeout)            │
│                                 queue.put(uid, corrected)   │
│                                 main thread: emit Stage B   │
└──────────────────────────────────────────────────────────────┘
```

**Gemma prompt template:**

```text
Correct the following speech transcript. Fix ASR errors, grammar, and
disfluencies. Output ONLY the corrected text. No explanation.

Input: {stage_a_text}
Output:
```

**Timeout handling:**

```python
try:
    result = gemma_proc.stdout.readline(timeout=1.5)
except TimeoutError:
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

- Primary: Gemma 4 26B MoE Q4 (if llama.cpp supports and p95 \<800ms)
- Fallback: Gemma 4 E2B Q4 (~4GB, always available)
- CLI flag: `--gemma-model path/to/model.gguf`

______________________________________________________________________

### Chrome Frontend

File: `scripts/vox_daemon/static/index.html`

Single-file HTML. No build step. No dependencies beyond browser WebSocket API.

```text
┌──────────────────────────────────────────────────────────────┐
│  VOX Personalis                              ● live           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Current utterance                                      │  │
│  │                                                        │  │
│  │  Stage A → "I want to talk about kubernets uh..."     │  │
│  │  Stage B → "I want to talk about Kubernetes..."       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  History                                                     │
│  ─────────────────────────────────────────────────────────  │
│  "Can you tell me about your experience with Kubernetes?"   │
│  "Yes, I have three years working with Kubernetes."         │
│  "We use it for container orchestration at my company."     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Stage A styling:** `color: #888; font-style: italic` — signals "processing"
**Stage B styling:** `color: #fff; font-weight: 600` — signals "committed"
**Transition:** Stage A fades out, Stage B fades in (150ms CSS transition)

______________________________________________________________________

### Daemon CLI

File: `scripts/vox_daemon/cli.py`

```bash
python -m scripts.vox_daemon \
  --moonshine-model ./out/moonshine_finetune/checkpoint \
  --gemma-model ./models/gemma-4-e2b-Q4_K_M.gguf \
  --max-utterance-sec 3.0 \
  --silence-ms 1000 \
  --vad-threshold 0.5 \
  --port 8765 \
  --debug            # prints Stage A and Stage B to stdout per segment
```

`--moonshine-model` accepts either a HuggingFace model ID
(`UsefulSensors/moonshine-streaming-small`) or a local directory path
(fine-tuned PEFT-merged checkpoint). Default: `UsefulSensors/moonshine-streaming-small`.

**Startup sequence:**

1. Load Moonshine model via Transformers (fails fast if model ID/path invalid)
1. Load SileroVAD model
1. Spawn Gemma subprocess, verify it responds to a test prompt
1. Create WebSocket server on `--port`
1. Serve `static/index.html` at root
1. `open http://localhost:{port}` (macOS)
1. Start mic capture → SileroVAD loop

______________________________________________________________________

## Module Structure

```text
scripts/
  serving/
    vad.py              ← UNTOUCHED (webrtcvad, S1 code — do not modify)
    api.py              ← UNTOUCHED (FastAPI, Phase 2)
  baseline_eval/
    normalization.py    ← REUSE: textnorm_v2 / create_normalizer (no changes)
  feedback_finetune/
    manifest.py         ← REUSE: adapt for Moonshine after M1 (no changes now)
  vox_daemon/           ← NEW: Phase B daemon
    __main__.py
    cli.py
    consumer.py         # TextConsumer Protocol + WebConsumer
    moonshine.py        # Moonshine v2 Streaming (Transformers, MPS)
    silero_vad.py       # SileroVAD segmenter (replaces webrtcvad for S2)
    gemma.py            # Gemma 4 GGUF subprocess
    ws_server.py        # WebSocket broadcast server
    static/
      index.html        # Chrome frontend (single file)
  moonshine_eval/       ← NEW: M0 + M2 evaluation
    __main__.py
    cli.py
    eval.py             # WER eval — NEW code, uses Transformers API
                        # Reuses textnorm_v2 from baseline_eval/normalization.py
                        # Does NOT adapt/modify existing baseline_eval code
  moonshine_finetune/   ← NEW: M1 fine-tuning
    __main__.py
    cli.py
    train.py            # PyTorch + PEFT/LoRA on MPS (not mlx-tune)
    eval.py             # checkpoint evaluation
```

**Existing code policy:** All S1 code under `scripts/` remains untouched. S2
creates new modules alongside existing ones. Shared utilities (e.g.,
`textnorm_v2`) are imported, not copied or modified.

**Existing reuse (import only, no modifications):**

- `scripts/baseline_eval/normalization.py` — `textnorm_v2` for WER
- `scripts/feedback_finetune/manifest.py` — adapt for Moonshine after M1

______________________________________________________________________

## Evaluation Specs

### M0: Moonshine v2 Streaming Baseline WER

**Important:** M0 benchmarks BOTH small and medium models on the same val set.
The output determines which model proceeds to M1 fine-tuning.

```bash
# Benchmark moonshine-streaming-small
python -m scripts.moonshine_eval \
  --manifest <PATH_TO_VAL_MANIFEST> \
  --model UsefulSensors/moonshine-streaming-small \
  --norm-version 2     # textnorm_v2, same as all prior evals

# Benchmark moonshine-streaming-medium
python -m scripts.moonshine_eval \
  --manifest <PATH_TO_VAL_MANIFEST> \
  --model UsefulSensors/moonshine-streaming-medium \
  --norm-version 2
```

**Manifest path:** The user will provide the path to the val set directory
containing audio clips and a CSV file. The manifest CSV must have columns for
audio file paths and reference transcripts. Verify column names match before
running.

**Inference API (Transformers):**

```python
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = MoonshineStreamingForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float32
).to(device)

# Per audio file:
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt").to(device)
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=256)
transcript = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

**Output (per model):**

- WER on val set (compare to Whisper OOB and fine-tuned Whisper 34.05%)
- Latency per segment (p50, p95) at 1s, 2s, 3s audio lengths
- Decision: if OOB WER > 50%, fine-tuning is mandatory before demo

**Model selection decision:**

```text
M0 result:
  medium WER meaningfully better (>1% gap) → use medium for M1
  gap < 1% WER → use small for M1 (faster iteration, lower resource)
  both WER > 50% → fine-tuning is critical; pick whichever is lower
```

### M1: Moonshine v2 Streaming Fine-tuning

Uses PyTorch + PEFT/LoRA on MPS backend. This is the same LoRA approach as S1
Whisper fine-tuning but with different target layer names (Moonshine v2 streaming
architecture, not Whisper). The target layers must be discovered at implementation
time by inspecting the model's named modules.

**Architecture difference from Whisper:** Moonshine v2 streaming uses an ergodic
encoder with sliding-window attention and a context adapter. The S1 Whisper LoRA
targeted `q_proj` + `v_proj` — Moonshine's equivalent attention projection names
will differ. Run `model.named_modules()` to find them.

**Data pipeline difference from Whisper:** Whisper uses its own log-mel
spectrogram feature extractor. Moonshine v2 streaming uses `AutoProcessor` from
Transformers, which handles feature extraction differently. The training data
loader must use the Moonshine processor, not Whisper's.

#### S1 Fine-Tuning Baseline (carry-over config)

The S1 Whisper fine-tuning ran M3→M7 with controlled ablations. The winning
config from M5+M7 (34.05% val WER) is the starting point for Moonshine v2.
Use these values as defaults. Only re-tune if the M1 gate fails.

**S1 proven hyperparameters (carry over as-is):**

| Parameter               | S1 Value | Source                                 |
| ----------------------- | -------- | -------------------------------------- |
| LoRA rank (r)           | 16       | M3 ablation: 16 beat 8 on ~700 samples |
| LoRA alpha              | 32       | 2x rank, standard PEFT default         |
| LoRA dropout            | 0.15     | M4 ablation: 0.15 beat 0.1, adopted    |
| Learning rate           | 1e-4     | Stable across M3–M7, robust for PEFT   |
| Batch size              | 4        | Hardware-constrained (M4 Pro memory)   |
| Gradient accumulation   | 4        | Effective batch = 16                   |
| Max epochs              | 3        | All runs converged by epoch 3          |
| Early stopping patience | 2        | Conservative; never triggered in S1    |
| Warmup steps            | 100      | ~7% of total steps                     |
| Weight decay            | 0.0      | Never tuned; dropout sufficed          |
| Bias                    | "none"   | Standard PEFT default                  |

**Needs fresh discovery (Moonshine-specific):**

| Parameter      | Why                                                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| target_modules | Moonshine v2 streaming has different attention layer names than Whisper's `q_proj`+`v_proj`. Run `model.named_modules()` to discover. |
| task_type      | Whisper used encoder-decoder. Moonshine streaming may need `SEQ_2_SEQ_LM` instead of `CAUSAL_LM`. Verify from model architecture.     |
| MPS batch size | Moonshine's memory footprint differs from Whisper. May need to adjust batch_size or gradient_accumulation if OOM on MPS.              |

**What we are NOT doing in S2 Phase A:**

- No correction-then-reinforce cycle (S1-M7 style feedback loop). S2 Phase A
  is a single fine-tuning pass on the existing dataset.
- No hyperparameter ablation unless M1 gate fails. The S1 config is proven on
  this dataset size (~700 samples); only re-tune if results demand it.

**If M1 gate fails (WER >= 34.05% after 3 epochs):**

1. Check target_modules — try including more layers (all attention projections)
1. Try learning rate 5e-5 (S1 tested but found no improvement; may differ)
1. Extend to 5 epochs with patience=3
1. If still failing: the model may need more data, not more tuning

#### CLI

```bash
python -m scripts.moonshine_finetune \
  --manifest <PATH_TO_TRAIN_MANIFEST> \
  --base-model UsefulSensors/moonshine-streaming-{small or medium} \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.15 \
  --lr 1e-4 \
  --batch-size 4 \
  --grad-accum 4 \
  --max-epochs 3 \
  --patience 2 \
  --warmup-steps 100 \
  --output ./out/moonshine_finetune/
```

#### Fine-tuning recipe

```python
from transformers import (
    AutoProcessor,
    MoonshineStreamingForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import get_peft_model, LoraConfig
import torch

device = "mps"
model = MoonshineStreamingForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float32
).to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Step 1: Discover LoRA target layers (run once, then hardcode)
# for name, module in model.named_modules():
#     if "proj" in name: print(name)
# Expected: attention projection layers similar to q_proj/v_proj but
# with Moonshine-specific naming. Hardcode after discovery.

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["TBD — discover from model.named_modules()"],
    lora_dropout=0.15,
    bias="none",
    task_type="SEQ_2_SEQ_LM",  # verify: encoder-decoder like Whisper
)
model = get_peft_model(model, lora_config)

# Step 2: Training arguments (S1 proven defaults)
training_args = Seq2SeqTrainingArguments(
    output_dir="./out/moonshine_finetune/",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,     # effective batch = 16
    learning_rate=1e-4,
    num_train_epochs=3,
    warmup_steps=100,
    weight_decay=0.0,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    predict_with_generate=True,
    generation_max_length=256,
    fp16=False,                        # MPS does not support fp16 training
)

# Step 3: Trainer with early stopping
from transformers import EarlyStoppingCallback
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,       # Moonshine processor-prepared
    eval_dataset=val_dataset,
    processing_class=processor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
trainer.train()

# Step 4: Save PEFT adapter + merge into standalone checkpoint
model.save_pretrained("./out/moonshine_finetune/adapter")
merged = model.merge_and_unload()
merged.save_pretrained("./out/moonshine_finetune/checkpoint")
processor.save_pretrained("./out/moonshine_finetune/checkpoint")
```

#### Frozen config pattern

M1 must output a `frozen_config.json` alongside the checkpoint for
reproducibility (matching S1 pattern from `scripts/fine_tuning/training.py`).
This file records all hyperparameters, model ID, dataset path, and commit hash
so any future run can reproduce the exact training conditions.

```python
import json, subprocess
frozen = {
    "base_model": model_id,
    "lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.15,
    "target_modules": [...],  # discovered values
    "task_type": "SEQ_2_SEQ_LM",
    "lr": 1e-4, "batch_size": 4, "grad_accum": 4,
    "max_epochs": 3, "patience": 2, "warmup_steps": 100,
    "weight_decay": 0.0,
    "manifest": "<PATH_TO_TRAIN_MANIFEST>",
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"]).decode().strip(),
}
with open("./out/moonshine_finetune/frozen_config.json", "w") as f:
    json.dump(frozen, f, indent=2)
```

**Gate:** val WER < 34.05%. If not met after 3 epochs, follow the escalation
path above ("If M1 gate fails").

**Checkpoint output:** PEFT adapter at `adapter/`, merged standalone checkpoint
at `checkpoint/`, processor files at `checkpoint/`, frozen config at root.
All stored under `./out/moonshine_finetune/`.

### M2: Gemma 4 Correction Evaluation

Run Moonshine v2 (fine-tuned) against val set. For each transcription, run Gemma
correction. Measure:

1. WER of raw Moonshine output (Stage A baseline)
1. WER of Gemma-corrected output (Stage B)
1. False correction rate: cases where Gemma made it worse

```bash
python -m scripts.moonshine_eval \
  --manifest <PATH_TO_VAL_MANIFEST> \
  --model ./out/moonshine_finetune/checkpoint \
  --gemma-model ./models/gemma-4-e2b-Q4_K_M.gguf \
  --output ./out/moonshine_gemma_eval/
```

**Gate:** Stage B WER < Stage A WER AND false correction rate < 15%.
If false correction rate > 15%: adjust Gemma prompt (more conservative), re-run.
If Stage B WER >= Stage A WER: skip Gemma in demo (Moonshine alone is sufficient).

______________________________________________________________________

## Error / Rescue Registry (Layer 1)

| Error                                  | Rescue                                                                       |
| -------------------------------------- | ---------------------------------------------------------------------------- |
| Moonshine exception (any)              | Skip segment, log stderr, continue                                           |
| Moonshine returns empty string         | Suppress Stage A update                                                      |
| Gemma timeout (>1500ms)                | Emit Stage A text as Stage B                                                 |
| Gemma OOM / crash                      | Log + terminal warning; auto-restart up to 3 times; Stage A only after limit |
| Gemma malformed output (empty/garbled) | Emit Stage A as Stage B fallback                                             |
| WebSocket client disconnected          | Continue broadcasting; reconnect on next page load                           |
| Chrome not installed                   | Log warning; user opens <http://localhost:8765> manually                     |
| Port 8765 in use                       | CLI error on startup: "Port 8765 in use — use --port N"                      |

______________________________________________________________________

## Success Criteria (5-Day Demo)

| Metric                      | Target                                          | Milestone |
| --------------------------- | ----------------------------------------------- | --------- |
| Moonshine v2 OOB WER        | Measured and recorded (both small + medium)     | M0        |
| Moonshine v2 fine-tuned WER | < 34.05% (beats fine-tuned Whisper)             | M1        |
| Stage B WER vs Stage A      | Stage B ≤ Stage A                               | M2        |
| Stage A latency             | < 300ms end-of-VAD to Chrome display            | M3        |
| Stage B latency             | Stage B replaces Stage A within 800ms in Chrome | M3        |
| Demo stability              | Runs for 10 min without crash                   | M3        |

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
Max message size: ~500 bytes (VAD segments ≤3s → ≤50 words → ≤300 chars + envelope).
Named pipe path: `/tmp/vox_personalis_${UID}.pipe`

```text
Python  ──► {"type":"stage_a","text":"hello","uid":1}\n  ──► Swift injector
        ──► {"type":"stage_b","text":"Hello.","uid":1}\n ──► Swift injector
```

### Swift AX Injector Contract

```swift
// On receiving stage_b message:
func applyStageB(text: String, uid: Int) {
    guard uid == lastWrittenUID else { return }  // D6: stale guard
    guard !readFieldValue().isEmpty else { return }  // D3: Return race guard
    writeFieldValue(text)
}

// On receiving stage_a message:
func applyStageA(text: String, uid: Int) {
    lastWrittenUID = uid
    writeFieldValue(text)  // unconditional (field must be focused)
}
```

### Swap to AX

TextConsumer Protocol means swapping WebConsumer → AXConsumer is a config flag:

```bash
python -m scripts.vox_daemon --consumer ax  # Phase C
python -m scripts.vox_daemon --consumer web  # Phase B (default)
```

The ASR pipeline (Moonshine, Gemma, VAD) is unchanged. Only the consumer swaps.

### MVP 1 Pass Criterion

Swift CLI injects a hardcoded string into the Live Captions Type to Speak field
without manual focus intervention, on macOS 14.0+. If this fails: Chrome fallback
(WebConsumer) remains the production path and AX investigation continues in parallel.

______________________________________________________________________

## Pre-Implementation Blockers

| #   | Blocker                                                           | When     | Owner                                             |
| --- | ----------------------------------------------------------------- | -------- | ------------------------------------------------- |
| 1   | Moonshine v2 streaming loads via Transformers on MPS              | Day 0    | Verify with pip install + load test (both models) |
| 2   | SileroVAD loads via silero-vad                                    | Day 0    | Verify with pip install + load test               |
| 3   | Gemma 4 GGUF builds on macOS 14 via llama.cpp                     | Day 0    | cmake + Metal build                               |
| 4   | Gemma 4 p95 latency \<800ms on M4 Pro (E2B or 26B MoE)            | Day 0    | Benchmark before M3                               |
| 5   | PEFT/LoRA applies to MoonshineStreamingForConditionalGeneration   | M1 start | Verify target layer names via named_modules()     |
| 6   | Val set manifest path provided and compatible with moonshine_eval | M0 start | User provides path; check column names            |

**Dependency summary (pip install):**

```text
# Core (Day 0)
transformers             # Latest — must support MoonshineStreamingForConditionalGeneration
torch                    # PyTorch with MPS backend
torchaudio               # Audio processing
silero-vad               # Neural VAD (replaces webrtcvad for S2)
peft                     # LoRA fine-tuning (M1)
jiwer                    # WER computation (already in project)
sounddevice              # Mic capture (M3 daemon)
websockets               # WebSocket server (M3 daemon)

# Existing (already in pyproject.toml)
soundfile, librosa       # Audio I/O
```
