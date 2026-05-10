"""
M3 — Merged-model parity check on the S1 val split.

Why this script exists:
  After M1 merges the S1-M7 LoRA into whisper-small.en base weights via
  `merge_and_unload()`, we need to prove the merge didn't drift the model from the
  original LoRA-active forward pass. The S1-M7 baseline was 34.05% val WER on the
  361-clip val split with text-norm v2 (contraction-expanded). This script runs the
  merged checkpoint over the same 361 clips with the same normalizer and confirms
  WER stays within ±0.5% of 34.05%.

  We do NOT use mlx-whisper here, even though M1 produced an MLX checkpoint:
   - The merge is the only thing M3 is supposed to validate. Testing on HF directly
     keeps the test fast (no format-conversion drift to disentangle).
   - mlx-whisper's threading bug doesn't affect this script (we drive the model on
     the main thread), but using HF makes the comparison numerically equivalent to
     S1-M7's measurement path.

  Format-conversion drift (HF → CT2, HF → MLX) is covered separately by M4's
  Stage A eval, which runs the actual CT2 model the wlk server uses.

How it works:
  1. Load merged HF checkpoint from --merged-dir (default out/whisper_small_en_s1m7_merged).
  2. Load the dataset_v1 manifest, filter to split=='val' (361 rows).
  3. Remap stale /Users/skinner/ paths to /Users/skinnercheng/ (manifest was written
     on a different machine; see spec "Manifest path-fixup gotcha").
  4. For each clip: load audio at 16kHz, run WhisperProcessor + generate(), decode.
  5. Normalize ref + hyp with create_normalizer(version=2), compute per-row WER+CER
     via scripts.baseline_eval.metrics, aggregate.
  6. Write metrics.json + predictions.csv to results/S2-M3_whisper_merged_baseline/
     matching the M7_feedback_finetune/ schema for easy diff.

Run:
  python -m scripts.vox_daemon.eval_merged

Gate:
  Aggregate val WER within ±0.5% of 34.05%.
"""

import argparse
import json
import time
from pathlib import Path

import librosa
import pandas as pd  # type: ignore[import-untyped]
import torch
from tqdm import tqdm  # type: ignore[import-untyped]
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from scripts.baseline_eval.metrics import compute_aggregate_metrics, compute_sample_metrics
from scripts.baseline_eval.normalization import create_normalizer

S1_M7_WER = 0.3405
WER_GATE_TOL = 0.005  # ±0.5% absolute


def transcribe_one(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_path: str,
    device: str,
) -> str:
    # 16kHz mono float32 — Whisper's expected feature input.
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    # whisper-small.en doesn't accept language/task kwargs; generation_config from
    # the merged checkpoint already has forced English-transcription decoder tokens.
    with torch.no_grad():
        ids = model.generate(input_features=inputs)
    text: str = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--merged-dir",
        type=Path,
        default=Path("out/whisper_small_en_s1m7_merged"),
        help="Merged HF Transformers Whisper checkpoint directory (from M1)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("out/dataset_v1/20260206-142756/dataset_v1_manifest.csv"),
        help="dataset_v1 manifest CSV",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/S2-M3_whisper_merged_baseline"),
        help="Directory to write metrics.json + predictions.csv",
    )
    p.add_argument("--split", default="val", help="Manifest split to evaluate")
    p.add_argument(
        "--limit", type=int, default=None, help="Optional cap on rows for a quick smoke check"
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps"],
        help="Inference device (MPS preferred on M4 Pro; falls back to CPU)",
    )
    args = p.parse_args()

    # Device resolution. MPS is usable but historically flaky for Whisper generate()
    # — CPU is the safe default for an eval that must reproduce S1-M7 numbers.
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading merged model from {args.merged_dir} on {device}")
    # Transformers' generic _Wrapped overload doesn't accept str under our type pin;
    # at runtime str is the canonical input for from_pretrained.
    merged_path = str(args.merged_dir)
    model = WhisperForConditionalGeneration.from_pretrained(merged_path).to(device).eval()  # type: ignore[arg-type]
    processor = WhisperProcessor.from_pretrained(merged_path)  # type: ignore[arg-type]

    print(f"Reading manifest {args.manifest}")
    df = pd.read_csv(args.manifest)
    df = df[df["split"] == args.split].copy()
    # Manifest was written on a different machine — remap to the local user.
    df["audio_path_resolved"] = df["audio_path_resolved"].str.replace(
        "/Users/skinner/", "/Users/skinnercheng/", regex=False
    )
    if args.limit:
        df = df.head(args.limit)
    print(f"Evaluating {len(df)} clips from split={args.split!r}")

    normalize = create_normalizer(version=2)
    rows: list[dict] = []
    t0 = time.time()
    for r in tqdm(df.itertuples(index=False), total=len(df), desc="transcribe"):
        hyp_raw = transcribe_one(model, processor, r.audio_path_resolved, device)
        rows.append(
            {
                "file_name": r.file_name,
                "pair_sha256": r.pair_sha256,
                "split": r.split,
                "duration_sec": r.duration_sec,
                "duration_bin": r.duration_bin,
                "reference_raw": r.transcript_raw,
                "reference": normalize(r.transcript_raw),
                "hypothesis_raw": hyp_raw,
                "hypothesis": normalize(hyp_raw),
            }
        )
    elapsed = time.time() - t0
    print(
        f"Transcription done in {elapsed:.1f}s "
        f"({elapsed / len(df):.2f}s/clip, RTF={elapsed / df['duration_sec'].sum():.3f})"
    )

    pred_df = pd.DataFrame(rows)
    pred_df = compute_sample_metrics(pred_df)
    metrics = compute_aggregate_metrics(pred_df, [args.split])
    agg = metrics[args.split]
    delta = float(agg["wer"]) - S1_M7_WER
    metrics["comparison"] = {
        "s1_m7_reference_wer": S1_M7_WER,
        "merged_wer": float(agg["wer"]),
        "absolute_delta": round(delta, 4),
        "within_tolerance": bool(abs(delta) <= WER_GATE_TOL),
        "tolerance": WER_GATE_TOL,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.out_dir / "predictions.csv", index=False)
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {args.out_dir}/metrics.json + predictions.csv")
    print(
        f"  merged val WER = {agg['wer']:.4f}  (S1-M7 = {S1_M7_WER:.4f}, "
        f"delta = {metrics['comparison']['absolute_delta']:+.4f})"
    )
    if not metrics["comparison"]["within_tolerance"]:
        print(f"  GATE FAILED: |delta| > {WER_GATE_TOL}")
        return 1
    print("  GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
