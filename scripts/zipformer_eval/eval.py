"""WER evaluation of a streaming Zipformer model via sherpa-onnx."""

import json
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import sherpa_onnx
import soundfile as sf
from jiwer import wer as compute_wer

from scripts.baseline_eval.normalization import create_normalizer


def _load_recognizer(model_dir: str, provider: str) -> sherpa_onnx.OnlineRecognizer:
    d = Path(model_dir)
    encoder = next(d.glob("encoder-*.onnx"))
    decoder = next(d.glob("decoder-*.onnx"))
    joiner = next(d.glob("joiner-*.onnx"))
    tokens = d / "tokens.txt"
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=str(encoder),
        decoder=str(decoder),
        joiner=str(joiner),
        tokens=str(tokens),
        provider=provider,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=1.2,
        rule2_min_trailing_silence=0.8,
        rule3_min_utterance_length=20.0,
    )


def _transcribe(recognizer: sherpa_onnx.OnlineRecognizer, audio_path: str) -> tuple[str, float]:
    """Transcribe a single audio file. Returns (transcript, latency_ms)."""
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio.astype(np.float32)

    t0 = time.perf_counter()
    stream = recognizer.create_stream()
    stream.accept_waveform(16000, audio)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    result = recognizer.get_result(stream)
    latency_ms = (time.perf_counter() - t0) * 1000

    # get_result returns str in sherpa-onnx 1.13
    text = result if isinstance(result, str) else result.text
    return text.strip(), latency_ms


def run_eval(
    manifest_csv: str,
    audio_root: str,
    model_dir: str,
    provider: str,
    split: str | None,
    output_dir: str,
    norm_version: int = 2,
) -> dict:
    normalize = create_normalizer(version=norm_version)
    audio_root = os.path.expanduser(audio_root)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)

    # Support both predictions CSVs (reference_raw) and manifest CSVs (transcript_raw)
    if "reference_raw" in df.columns:
        df = df.rename(columns={"reference_raw": "transcript_raw"})

    if split:
        if "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)
        else:
            print(
                f"Warning: no 'split' column in manifest — evaluating all {len(df)} rows",
                file=sys.stderr,
            )

    total = len(df)
    print(f"Loading model from {model_dir} ...")
    recognizer = _load_recognizer(model_dir, provider)
    print(f"Model loaded. Evaluating {total} clips (split={split or 'all'}) ...\n")

    rows = []
    references, hypotheses = [], []
    latencies = []

    for _, row in df.iterrows():
        file_name = row["file_name"]
        reference_raw = str(row["transcript_raw"])
        audio_path = os.path.join(audio_root, file_name)

        if not os.path.exists(audio_path):
            print(f"  MISSING: {audio_path}", file=sys.stderr)
            continue

        try:
            hyp_raw, lat_ms = _transcribe(recognizer, audio_path)
        except Exception as e:
            print(f"  ERROR on {file_name}: {e}", file=sys.stderr)
            continue

        ref_norm = normalize(reference_raw)
        hyp_norm = normalize(hyp_raw)
        clip_wer = compute_wer(ref_norm, hyp_norm) if ref_norm else 0.0

        references.append(ref_norm)
        hypotheses.append(hyp_norm)
        latencies.append(lat_ms)

        idx = len(references)
        running_wer = compute_wer(references, hypotheses) * 100

        # Live progress line
        status = f"[{idx:>4}/{total}] {file_name}"
        status += f"\n         ref: {reference_raw!r}"
        status += f"\n         hyp: {hyp_raw!r}"
        status += f"  |  clip WER: {clip_wer * 100:.1f}%"
        status += f"  |  running WER: {running_wer:.2f}%  |  {lat_ms:.0f}ms"
        print(status)

        rows.append(
            {
                "file_name": file_name,
                "split": row.get("split", split or "unknown"),
                "reference_raw": reference_raw,
                "hypothesis_raw": hyp_raw,
                "reference": ref_norm,
                "hypothesis": hyp_norm,
                "wer": round(clip_wer, 4),
                "latency_ms": round(lat_ms, 1),
            }
        )

    # Final metrics
    final_wer = compute_wer(references, hypotheses) if references else 1.0
    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    metrics = {
        "model_dir": model_dir,
        "split": split or "all",
        "num_clips": len(references),
        "wer": round(final_wer, 4),
        "wer_pct": round(final_wer * 100, 2),
        "baseline_to_beat_pct": 34.05,
        "beats_baseline": (final_wer * 100) < 34.05,
        "latency_p50_ms": round(p50_lat, 1),
        "latency_p95_ms": round(p95_lat, 1),
        "norm_version": norm_version,
        "provider": provider,
    }

    # Save outputs
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    gate = "YES ✓" if metrics["beats_baseline"] else "NO — fine-tuning required"
    print("\n" + "=" * 60)
    print(f"  WER:         {metrics['wer_pct']:.2f}%  (baseline: 34.05%)")
    print(f"  Beats gate:  {gate}")
    print(f"  Clips:       {metrics['num_clips']}")
    print(f"  Latency p50: {p50_lat:.0f}ms   p95: {p95_lat:.0f}ms")
    print(f"  Saved to:    {out_dir}")
    print("=" * 60)

    return metrics
