# S1-M6: Minimal Viable Serving (MVS)

## Purpose

Move Model v2 from offline evaluation into a real service.

This milestone answers one question:

> *Can we expose Model v2 as a real service with streaming transcription,
> measurable SLAs, and a minimal demo — treated as production in mindset,
> not a demo endpoint?*

**Architectural approach**: We use **batch-per-utterance transcription** (VAD
detects silence, buffer is transcribed once, result sent to client) rather than
streaming incremental output. This design is simpler to implement, fits MVS
scope, and better matches Whisper's batch processing architecture.

______________________________________________________________________

## Context

Model v2 (small.en + LoRA r=16, `textnorm_v2`) achieved 44.02% val WER —
18.35 absolute pts improvement over v1.1 (64.04%). The capacity hypothesis
(H2) from M4a is confirmed. The model is production-ready for inference.

**Expected behavior pattern**: Short utterances (primary use case) transcribe
reliably. Longer or more complex phrasing shows higher error rates. MVS testing
will verify this behavior holds during live serving.

MVS is a **single-speaker service** — the model is trained exclusively on one
person's voice. The service is demonstrated by the owner (screenshots, screen
recording) and shared with stakeholders for review. Stakeholders do not run
the service themselves; they observe the recording.

**Design trade-off (batch vs streaming)**: We considered streaming incremental
output (partial results as audio arrives, like Google Cloud Speech-to-Text or
OpenAI Realtime API). However, this would require continuous re-inference on
growing buffers (computationally expensive) and a different model architecture
(Whisper is batch-oriented). For MVP scope, batch-per-utterance is appropriate:
simpler to implement, faster to wrap up, and sufficient for a personal demo
service.

______________________________________________________________________

## Dependencies

- S1-M5 complete — Model v2 checkpoint at
  `out/capacity_scaling/20260304-103236/checkpoint`
- `configs/DECODE_V1.json` (frozen decode config)
- `webrtcvad` (already installed from S1-M0)

______________________________________________________________________

## Scope

### In Scope

- `WS /ws/transcribe` — real-time WebSocket streaming transcription endpoint
- `GET /metrics` — in-memory SLA metrics (JSON) + rolling JSONL log
- `GET /health` — startup/readiness check
- `GET /demo` — browser-based demo UI (single HTML/JS file)
- Model v2 warm-up at startup
- Dockerfile for cloud-readiness showcase
- Cloud architecture design section (provider-agnostic 12-factor)
- Production language trade-off analysis (Python → TypeScript/Rust path)

### Out of Scope

- Feedback collection or storage (M7 Feedback Tool)
- Ops/monitoring infrastructure (M8): no Prometheus, OpenTelemetry, or external
  metrics pipelines
- Authentication, multi-user support, rate limiting
- Test set evaluation (single-shot rule from S1-M3 applies)
- Provider-specific cloud deployment (Pulumi, Terraform, etc.)

______________________________________________________________________

## Design Decisions

Each decision below explains the "why" — following the format of prior specs
(e.g., S1-M3's "Why LoRA" section).

### Why WebSocket (not HTTP polling or gRPC)

WebSocket provides a persistent, bidirectional connection ideal for
continuous audio streaming. The client streams raw audio chunks; the server
asynchronously returns transcript segments. This bidirectional pattern maps
directly onto WebSocket's frame-based semantics. HTTP polling requires
a round-trip per chunk (significant latency overhead). gRPC streaming adds
protocol complexity without benefit for a single-developer project. WebSocket
is natively supported by browsers, `fastapi`, and all major cloud platforms.

### Why VAD-triggered segments (not fixed-length chunks)

Fixed-length chunking (e.g., every 3 seconds) splits sentences mid-word,
producing fragmented transcripts and confusing the model. VAD-triggered
transcription waits for a natural pause (silence) before sending buffered
audio to the model — producing complete utterances. This matches Whisper's
design intent (full-utterance inputs, not streaming frames). The trade-off
is variable latency (pause detection adds ~500ms to P50), which is acceptable
for a personal demo and aligns with how real humans experience speech breaks.

### Why batch-per-utterance (not streaming incremental output)

Streaming services like Google Cloud Speech-to-Text and OpenAI Realtime API
produce partial results as audio arrives ("I" → "I am" → "I am Skinner").
This requires continuous re-inference on growing buffers (computationally
expensive: multiple passes instead of one) and a streaming model architecture
(Whisper is batch-oriented). For MVS scope (MVP, personal demo, wrap-up
priority), batch-per-utterance is the right choice: simpler to implement,
matches the model's design, and sufficient for the use case. If true streaming
becomes necessary, it requires a different model architecture (Conformer,
Transducer) and is future scope.

### Why warm-up at startup

PyTorch JIT-compiles the model's compute graph on the first forward pass.
The first inference is typically 2–5× slower than subsequent ones. Running
a warm-up pass (silent audio) at startup forces this JIT compilation before
any real request, making P50/P95 latency measurements representative of
steady-state performance rather than cold-start artifacts. `/health` returns
503 during warm-up, preventing traffic from hitting a cold model.

### Why stateless service design

The service stores no session state in memory. Each WebSocket connection is
independent; metrics are persisted to a JSONL file externally. Stateless
design enables: (a) clean restarts without data loss, (b) horizontal scaling
with multiple instances (not needed for single-speaker, but required for
production readiness), (c) seamless local → cloud deployment with zero code
changes. This is the architectural foundation for any production system.

**Connection model:** MVS serves a single WebSocket connection (owner's demo)
because the model is trained exclusively for one speaker (personal voice, deaf
accent). Supporting multiple concurrent users would require per-user voice
recording, retraining, and model management — a different product. That
infrastructure is described in "Production Language Trade-Offs."

### Why `/health` and `/metrics` are separate endpoints

They serve different consumers with different requirements:

- **`/health`**: Polled by infrastructure (cloud platform, load balancer,
  Kubernetes). Answers one binary question: *"Is this process alive and
  ready for traffic?"* Must be fast (< 10ms), always available, and return
  200 (live) or 503 (starting/dead). Returns 503 during startup until
  warm-up completes.
- **`/metrics`**: Consumed by developer/operator. Answers: *"How is the
  service performing?"* Returns detailed SLA data (P50/P95 latency, failure
  rate, segments). Never polled by infrastructure automation; accessed
  manually or by a dashboard.

This separation follows single-responsibility principle and is standard
practice in production-minded services (Kubernetes probes, Prometheus
scraping, etc.).

______________________________________________________________________

## Service Endpoints

| Endpoint            | Protocol  | Purpose                           |
| ------------------- | --------- | --------------------------------- |
| `WS /ws/transcribe` | WebSocket | Real-time streaming transcription |
| `GET /metrics`      | HTTP      | In-memory SLA metrics (JSON)      |
| `GET /health`       | HTTP      | Startup/readiness check           |
| `GET /demo`         | HTTP      | Browser demo UI (static HTML/JS)  |

______________________________________________________________________

## WebSocket Streaming: `/ws/transcribe`

### Audio Format (Client → Server)

**Binary frames. Raw PCM: 16 kHz, 16-bit signed integer, mono.**

The demo UI uses the Web Audio API (`AudioWorklet`) to:

1. Capture microphone input (browser default sample rate, typically 48 kHz)
1. Resample to 16 kHz
1. Convert Float32 (JavaScript native) → Int16 (required by VAD and model)
1. Send binary frames over WebSocket

This format is required by `webrtcvad` (voice activity detection) and
by Whisper's audio pipeline.

### Session Lifecycle

```text
Client                          Server
  |--- open WebSocket ------------> |  connection established
  |--- binary PCM frames ---------> |  server buffers, runs VAD
  |                                 |  VAD: silence ≥ silence_threshold_ms
  |                                 |  → transcribe buffered audio
  |<-- JSON segment --------------- |  segment sent back (see below)
  |--- binary PCM frames ---------> |  new buffer starts
  |        ...                      |
  |--- {"type":"stop"} -----------> |  or client closes connection
  |<-- connection closed ---------- |  server cleans up, logs session stats
```

### VAD Configuration

| Parameter              | Value | Notes                                                        |
| ---------------------- | ----- | ------------------------------------------------------------ |
| `mode`                 | 3     | Most aggressive non-speech detection (same as S1-M0)         |
| `frame_size_ms`        | 30    | Required by webrtcvad                                        |
| `sample_rate_hz`       | 16000 | Model requirement                                            |
| `silence_threshold_ms` | 1000  | CLI: `--silence_ms`. 1s for this speaker. Range: 300–2000ms  |
| `min_utterance_ms`     | 300   | Skip transcription if buffer < 300ms (avoids spurious noise) |
| `max_utterance_sec`    | 30    | Cost protection: forces transcription at 30s buffer limit    |

**Detailed notes:**

- `silence_threshold_ms` (1000ms): Configurable via CLI (`--silence_ms`).
  1s chosen for this speaker—slightly more forgiving than OpenAI's 500ms
  default, accommodating natural speech patterns and longer pauses.

**Critical clarification:** `silence_threshold_ms` and `max_utterance_sec`
are independent controls:

- The first (1000ms) triggers transcription on natural pauses.
- The second (30s) is a safety cap on unbounded buffer growth.

### Server Response Per Segment (JSON)

```json
{
  "type": "transcript",
  "segment_id": 3,
  "transcript": "Turn on the lights.",
  "duration_sec": 2.43,
  "latency_ms": 312,
  "model_version": "v2"
}
```

- `transcript`: Whisper's natural output — capitalization and punctuation
  preserved exactly as the model produces it. `textnorm_v2` is NOT applied
  in the serving path. (See "Normalization Policy" section below.)
- `latency_ms`: Wall-clock time from end-of-utterance (VAD trigger) to
  this JSON sent to client.
- `model_version`: Identifier for reproduceability ("v2").

### Error Message Schema

```json
{
  "type": "error",
  "code": "max_utterance_exceeded",
  "message": "Buffer exceeded 30s — transcribing partial segment"
}
```

Error codes:

- `max_utterance_exceeded`: Buffer reached 30s limit before VAD pause.
- `inference_error`: Model inference failed (log details server-side).
- `inference_timeout`: Model inference exceeded time limit (> 30s).

______________________________________________________________________

## Normalization Policy

**Critical distinction:** `textnorm_v2` is an **evaluation-only** utility.
It is NOT applied in the serving path.

| Context                | Normalization | Output                  |
| ---------------------- | ------------- | ----------------------- |
| Training / WER scoring | `textnorm_v2` | `"turn on the lights"`  |
| **API response (M6)**  | **None**      | `"Turn on the lights."` |

The API returns Whisper's natural output exactly as the model produces it
(capitalization, punctuation preserved). Downstream processing (the future
"beautify" phase in M7) applies its own formatting as needed. This keeps the
serving layer clean and non-destructive.

______________________________________________________________________

## SLA Metrics: `GET /metrics`

### Latency Definition

Wall-clock time from end-of-utterance (VAD trigger or timeout) to JSON
segment sent to client. Measured server-side.

### Metrics Collected Per Segment

- `latency_ms`: P50, P95, min, max, mean
- `failure_rate`: failed segments / total segments attempted
- `total_segments`: count of utterances transcribed
- `total_audio_sec`: aggregate duration of audio processed
- `uptime_sec`: seconds since service started

### Response Schema

```json
{
  "uptime_sec": 3621,
  "total_segments": 47,
  "failed_segments": 1,
  "failure_rate": 0.021,
  "latency_ms": {
    "p50": 340,
    "p95": 820,
    "min": 210,
    "max": 1240,
    "mean": 410
  },
  "total_audio_sec": 183.4,
  "model_version": "v2"
}
```

### Storage — Two Layers

#### Layer 1: In-Memory Counters

- Exposed at `GET /metrics`
- Updated per segment
- Resets on service restart
- Used for live monitoring during demo

#### Layer 2: Rolling JSONL File

- Appended per segment to `out/serving/metrics.jsonl`
- Survives service restart
- Used for post-demo analysis and audit trail

JSONL entry schema:

```json
{
  "timestamp": "2026-03-10T14:23:01Z",
  "segment_id": 3,
  "duration_sec": 2.43,
  "latency_ms": 312,
  "status": "ok",
  "model_version": "v2"
}
```

______________________________________________________________________

## Health Check: `GET /health`

### Response (Service Ready)

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "v2"
}
```

HTTP `200` OK.

### Response (Startup in Progress)

```json
{
  "status": "starting",
  "model_loaded": false,
  "message": "Loading checkpoint..."
}
```

HTTP `503` Service Unavailable.

**Behavior:** Returns 503 during model load and warm-up. Only returns 200
after warm-up completes and model is ready for inference. Infrastructure
(load balancers, cloud platforms) poll this endpoint to determine readiness.

______________________________________________________________________

## Model Loading & Warm-Up

### Startup Sequence

1. **Load checkpoint** — Load Model v2 (small.en + LoRA r=16) from disk
   (fail fast with exit code 1 if not found)
1. **Warm-up inference** — Run one forward pass on 0.5s of silent audio
   (zeros) to force PyTorch JIT compilation
1. **Ready** — Set `model_loaded = True` and begin accepting traffic

### Startup Log Output

```text
[startup]  Loading Model v2 from checkpoint...              ✓ (4.2s)
[startup]  Running warm-up inference...                      ✓ (2.1s)
[startup]  Model ready. Serving on ws://127.0.0.1:8000
```

### Rationale

PyTorch JIT-compiles the model's compute graph on the first forward pass.
This warm-up forces the compilation before real traffic, ensuring steady-state
latency (P50/P95) is representative of post-warmup performance, not cold-start
artifacts (see "Design Decisions" section above).

______________________________________________________________________

## Demo UI: `GET /demo`

A single static `demo.html` file served by the API. No web framework.
Designed to be functional with clear status and metrics — sufficient for
stakeholder review (owner records the demo and shares screenshots/recordings).

### Layout

```text
┌─────────────────────────────────────────────────────────┐
│  VOX Personalis  ·  Model v2 (small.en + LoRA)          │
│  Val WER: 44.02%  ·  S1-M6 Minimal Viable Serving       │
├─────────────────────────────────────────────────────────┤
│  ● LISTENING       [■ Stop Recording]                   │
│  Status: Transcribing...                                │
├───────────────────────────────────────┬─────────────────┤
│  Live Transcript                      │  Session Metrics│
│  ─────────────────────────────────    │  ─────────────  │
│  "Turn on the lights."         312ms  │  P50:  340ms    │
│  "What's the weather today?"   287ms  │  P95:  820ms    │
│  "Play some music."            341ms  │  Segs: 3        │
│                                       │  Fail: 0.0%     │
│                                       │  Audio: 12.4s   │
└───────────────────────────────────────┴─────────────────┘
```

### Implementation Details

- **Single file**: `scripts/serving/static/demo.html` (HTML + CSS + JS inline,
  no external dependencies)
- **Audio capture**: Minimal pass-through `AudioWorklet` forwards raw Float32
  samples; main thread downsamples to 16 kHz Int16 PCM (proven Google Cloud /
  OpenAI pattern — avoids Chrome AudioWorklet input-zeroing edge cases)
- **Streaming**: WebSocket client to `/ws/transcribe`
- **Metrics display**: Polling `GET /metrics` every 5s, displays P50/P95
  latency, failure rate, and segment count

______________________________________________________________________

## Cloud & Production Readiness

### Cloud Architecture (Provider-Agnostic 12-Factor Design)

**What it means**: Provider-agnostic means the same code runs on any cloud
(AWS, Google Cloud, Azure, or local hardware) without changes. 12-factor is a
methodology for building apps that are portable, scalable, and maintainable —
emphasizing config via environment variables, stateless processes, and clean
containerization.

**Why we follow it**: MVS starts as a local personal demo but is designed for
eventual cloud sharing with stakeholders. 12-factor principles ensure that
deployment is simple (same Docker image, different env vars) and future-proof
(no vendor lock-in, no rewriting needed to move clouds).

#### Design Principles

| Principle                | MVS Implementation                             |
| ------------------------ | ---------------------------------------------- |
| Config via env variables | `MODEL_PATH`, `PORT`, `HOST`, `VAD_SILENCE_MS` |
| Stateless process        | No session state in memory; JSONL externalized |
| Single container         | API + model + demo UI in one Docker image      |
| Port binding             | One port; TLS termination by platform proxy    |
| Process independence     | Clean start/stop without data loss             |

#### Deployment Topology

```text
Local (Development)                  Cloud (Production Target)
─────────────────────────            ───────────────────────────────────
  [Browser / Owner]                    [Browser / Stakeholder]
     │ WebSocket                           │ WebSocket (wss://)
     ▼                                     ▼
  [Python FastAPI]                    [Platform Proxy (TLS termination)]
  scripts/serving/                         │
     ├── api.py (WebSocket)                ▼
     ├── model.py (Model v2)          [Docker Container]
     ├── vad.py (webrtcvad)           scripts/serving/ + Model v2 weights
     └── metrics.py                       (same code, different env vars)
```

The code is identical between local and cloud. Only environment variables
and TLS handling (provided by the platform) differ.

#### Dockerfile

Located at project root (`./Dockerfile`):

See [`Dockerfile`](../Dockerfile) at project root. Key features: non-root user,
stdlib-based `HEALTHCHECK`, `ENTRYPOINT`/`CMD` split for overridable defaults,
mount points for checkpoint and logs.

### Production Language Trade-Offs

MVS uses Python — the correct choice for an MVP with ML inference. However,
if this service were to scale beyond single-user, language trade-offs change.

| Language       | MVS | Production | Key Trade-off                               |
| -------------- | --- | ---------- | ------------------------------------------- |
| **Python**     | ✅  | ⚠️ GIL     | Best ML ecosystem; model code intact        |
| **TypeScript** | ❌  | ✅ API     | Many connections; Python sidecar for model  |
| **Rust**       | ❌  | ✅ Optimal | Best latency/memory; needs separate service |

**Trade-off details:**

- **Python**: Best ML ecosystem and model code stays intact. GIL and memory
  per request are constraints at production scale (> 10 concurrent users).
- **TypeScript**: Event loop handles many concurrent connections; model still
  served from Python sidecar (gRPC or REST).
- **Rust**: Lowest memory footprint and latency; requires separate Python or
  Triton model service.

#### Production Architecture Pattern (When to Migrate)

If and when scaling becomes necessary:

```text
[TypeScript or Rust — API Gateway]          [Python — Model Service]
   WebSocket handler                            FastAPI + TorchServe/Triton
   Audio buffering / VAD                        Model v2 inference
   Request routing / metrics collection    ←──  gRPC or REST internal
                                                Returns transcript
```

#### Decision Trigger for Migration

Migrate from Python monolith to decoupled architecture when:

- Serving latency SLA becomes critical (P95 < 200ms target)
- Concurrent users require horizontal scaling
- Memory footprint per request becomes a constraint

For a single-speaker personal service, Python is optimal and migration is
unnecessary.

______________________________________________________________________

## CLI Interface

```bash
python -m scripts.serving \
  --checkpoint "./out/capacity_scaling/20260304-103236/checkpoint" \
  --decode_config "./configs/DECODE_V1.json" \
  --host 127.0.0.1 \
  --port 8000 \
  --silence_ms 1000 \
  --max_utterance_sec 30 \
  --metrics_out "./out/serving/metrics.jsonl" \
  --verbose
```

### Arguments

| Argument              | Default                       | Description                         |
| --------------------- | ----------------------------- | ----------------------------------- |
| `--checkpoint`        | reqd                          | Path to Model v2 LoRA checkpoint    |
| `--decode_config`     | reqd                          | Path to `DECODE_V1.json`            |
| `--host`              | `127.0.0.1`                   | Bind address (`0.0.0.0` for cloud)  |
| `--port`              | `8000`                        | Port                                |
| `--silence_ms`        | `1000`                        | VAD silence (ms). Range: 300–2000   |
| `--max_utterance_sec` | `30`                          | Max buffer before forced transcribe |
| `--metrics_out`       | `./out/serving/metrics.jsonl` | JSONL log                           |
| `--no_warmup`         | False                         | Skip warm-up (testing only)         |
| `-v, --verbose`       | False                         | Detailed logging                    |

### Exit Codes

| Code | Meaning                                               |
| ---- | ----------------------------------------------------- |
| 0    | Clean shutdown (`SIGINT`, `SIGTERM`)                  |
| 1    | Fatal error (checkpoint not found, model load failed) |
| 130  | Keyboard interrupt (`Ctrl+C`)                         |

______________________________________________________________________

## Output Artifacts

All outputs written to repository as follows:

### Source Code (Part of Repository)

```text
scripts/serving/
├── __init__.py
├── __main__.py          # Entry: python -m scripts.serving
├── cli.py               # CLI parsing + pipeline orchestration
├── api.py               # FastAPI app, routes, WebSocket handler
├── model.py             # Model loading, warm-up, inference
├── vad.py               # webrtcvad wrapper (adapts S1-M0 VAD logic)
├── metrics.py           # In-memory counters + JSONL persistence
└── static/
    └── demo.html        # Demo UI (self-contained HTML/CSS/JS)

Dockerfile               # Cloud-readiness artifact (project root)
```

### Runtime Outputs (Produced When Service Runs)

```text
out/serving/
└── metrics.jsonl        # Rolling per-segment metrics log (append-only)
```

### Spec

```text
specs/S1-M6-minimal-viable-serving.md    # This specification
```

______________________________________________________________________

## Error Handling

| Scenario                        | Behavior                              | Exit |
| ------------------------------- | ------------------------------------- | ---- |
| Checkpoint not found            | Print path, fail fast                 | 1    |
| Model load OOM                  | Print error, suggest action           | 1    |
| Warm-up failure                 | Log error, exit                       | 1    |
| VAD buffer > 30s                | Transcribe partial, send error msg    | 0    |
| Inference error                 | Log, send WS error, increment counter | 0    |
| Inference timeout               | Send error, increment counter         | 0    |
| WebSocket disconnect mid-stream | Cleanup buffer, log session stats     | 0    |
| Keyboard interrupt              | Save state if needed, clean exit      | 130  |

______________________________________________________________________

## Package Dependencies

### New (add to `pyproject.toml`, `[serving]` extra)

| Package             | Version | Purpose                           |
| ------------------- | ------- | --------------------------------- |
| `fastapi`           | ≥ 0.100 | API framework + WebSocket support |
| `uvicorn[standard]` | ≥ 0.23  | ASGI server (includes WebSocket)  |

### Inherited (from S1-M0 through S1-M5)

- `webrtcvad` — VAD (already installed)
- `torch`, `transformers`, `peft` — Model inference
- `pandas`, `numpy` — Data manipulation
- `jiwer` — WER/CER (evaluation reference only)

### Installation

```bash
pip install -e ".[serving]"
```

______________________________________________________________________

## Completion Criteria

S1-M6 is complete when:

1. ✅ `WS /ws/transcribe` streams audio, detects utterance boundaries via VAD,
   and returns transcript segments with latency metadata
1. ✅ `GET /metrics` returns P50/P95 latency, failure rate, and segment count
1. ✅ `GET /health` returns 200 after warm-up, 503 during startup
1. ✅ `GET /demo` serves a functional browser UI with:
   - Live transcription working on owner's voice
   - Per-segment latency visible
   - Metrics panel (side or bottom split)
1. ✅ `out/serving/metrics.jsonl` written per segment (JSONL format)
1. ✅ `docker build` succeeds — Dockerfile is valid and builds successfully
1. ✅ Audio format verified — capture, resampling, and format conversion tested
   and documented on development hardware
1. ✅ End-to-end demo captured: screenshot or screen recording demonstrating
   live transcription

### No Performance Target

Unlike earlier milestones, MVS has no WER or latency target. The outcome
(good or bad) is documented. The project proceeds to Min Viable Serving
delivery and feedback collection (M7) regardless of performance.

M6 is about operational viability, not recognition quality optimization.

______________________________________________________________________

## Failure Modes if Skipped

If this milestone is skipped:

- Model remains offline (no serving capability)
- No SLA metrics for stakeholder review
- No demonstration of the system working in real-time
- No architecture foundation for future cloud deployment

______________________________________________________________________

## Implementation Notes

### Audio Buffering During Inference

While the model transcribes utterance N, new audio frames continue arriving.
The server must start a new VAD buffer for utterance N+1 immediately after
dispatching N for inference. This is the standard streaming ASR pattern
(used by Google Cloud Speech-to-Text and similar services). Implementation
detail is left to the engineer — the key invariant is: no audio frames are
dropped between utterances.

**Degradation condition**: For MVS scope, segment-level pseudo-streaming
(sequential buffering per utterance, no true overlapping async inference) is
acceptable. True overlapping asynchronous buffering is desirable but not
required if it threatens milestone completion.

### Audio Format Verification

**Required before completion**: Verify the audio capture and format conversion
chain on development hardware:

1. **Capture original audio** — Record 2-3 short commands through demo UI
1. **Verify capture format** — Check logs: confirm native sample rate (typically
   48 kHz on macOS, may vary by OS)
1. **Verify resampling** — Confirm Web Audio API successfully resamples to
   16 kHz without artifacts
1. **Verify model input** — Check that 16 kHz, 16-bit signed integer, mono PCM
   frames reach model correctly
1. **Document findings** — Record actual parameters and any issues encountered

If format conversion introduces problems, adjust resampling algorithm or codec
before proceeding to cloud deployment.

### Reuse from Prior Milestones

- **VAD integration**: Adapt VAD logic from S1-M0 `inventory.py`
- **Model loading**: Adapt checkpoint loading from S1-M3/M4/M5
- **Decoding**: Reuse `DECODE_V1.json` (frozen, no changes)

### Dockerfile Strategy

The Dockerfile should:

- Use `python:3.11-slim` (matches development environment)
- Install exactly the `[serving]` extras (not `[dev]`)
- Bundle model weights OR expect them at runtime via mounted volume
- For MVS: bundle weights in image (simplicity for demo)
- For cloud: externalize weights to a model registry (future iteration)

______________________________________________________________________

## Known Issues & Trade-Offs (After Implementation and Testing)

Documented during smoke testing and audit. None block MVP delivery.

| Issue                              | Severity   | Notes                                        |
| ---------------------------------- | ---------- | -------------------------------------------- |
| Global model state not thread-safe | Low        | Single-threaded ASGI; upgrade if scaling     |
| Metrics file write failures silent | Low        | Dockerfile creates `/var/log/vox` with perms |
| Server errors leak to client UI    | Low        | Trusted demo env; sanitize for multi-user    |
| Utterances lost on disconnect      | Expected   | VAD `min_utterance_ms=300` by design         |
| Metrics percentile off-by-one      | Cosmetic   | Negligible impact at small n                 |
| Linear resampling has aliasing     | Negligible | Whisper robust to minor distortion           |
| Segment recording not atomic       | Low        | Single WebSocket; add lock if multi-user     |
| AudioWorklet state not restored    | Low        | Explicit cleanup + retry recovers            |

**Summary**: All issues tracked and understood. No fixes needed for MVP. Clear upgrade path
for each if scaling beyond single-speaker service.

______________________________________________________________________

## References

- S1-M5 specification and output: Model v2 checkpoint, frozen_config.json
- S1-M4b specification: `textnorm_v2` implementation
- S1-M4 specification: DECODE_V1.json (frozen config)
- S1-M0 specification and implementation: webrtcvad integration
- [OpenAI Realtime API VAD documentation](https://developers.openai.com/api/docs/guides/realtime-vad/)
  — reference for VAD defaults and best practices
- [12-factor app methodology](https://12factor.net/) — cloud architecture
  principles
- [FastAPI WebSocket documentation](https://fastapi.tiangolo.com/advanced/websockets/)
  — implementation reference
