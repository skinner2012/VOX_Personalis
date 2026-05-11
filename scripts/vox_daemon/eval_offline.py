"""
M4 — Gemma Stage B correction eval (offline).

Runs the 361-clip val split through two stages:
  Stage A: MLX Whisper (merged S1-M7 checkpoint) → raw transcript
  Stage B: Gemma 4 GemmaWorker.correct()        → polished transcript

Both hypotheses are normalized with create_normalizer(version=2) and evaluated
against the reference. The script produces:
  - predictions.csv : per-clip Stage A + Stage B hypotheses and WER
  - metrics.json    : aggregate WER for Stage A and Stage B, plus gate outcomes

Gates (from spec):
  1. Stage B aggregate WER ≤ Stage A aggregate WER
  2. False-correction rate < 15%   (clips where Stage B WER > Stage A WER)

Why MLX for Stage A here (not CT2): the live daemon serves via faster-whisper +
CT2, but MLX runs on the main thread and avoids the MLX thread-local Stream bug
that breaks it inside wlk's asyncio.to_thread workers. M3 verified that the HF
merged model sits at 34.05% WER; the MLX checkpoint is the same weights in a
different format, so numerical agreement with the S1-M7 reference is expected
within floating-point tolerances.

Run:
  python -m scripts.vox_daemon.eval_offline \\
    --manifest ./out/dataset_v1/20260206-142756/dataset_v1_manifest.csv \\
    --split val \\
    --whisper-model ./out/whisper_small_en_s1m7_merged_mlx \\
    --gemma-model ./models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \\
    --output ./results/S2-M4_gemma_correction_eval/
"""

import argparse
import json
import time
from pathlib import Path

import mlx_whisper  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from scripts.baseline_eval.metrics import compute_aggregate_metrics, compute_sample_metrics
from scripts.baseline_eval.normalization import create_normalizer
from scripts.vox_daemon.gemma import GemmaWorker

STAGE_A_WER_REFERENCE = 0.3405  # S1-M7 baseline (M3 confirmed merge parity)
FALSE_CORRECTION_GATE = 0.15  # < 15% of clips worsened by Gemma


def transcribe_mlx(audio_path: str, model_dir: str) -> str:
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_dir,
        verbose=False,
        temperature=0.0,
    )
    text: str = result.get("text", "") or ""
    return text.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("out/dataset_v1/20260206-142756/dataset_v1_manifest.csv"),
    )
    p.add_argument("--split", default="val")
    p.add_argument(
        "--whisper-model",
        type=str,
        default="out/whisper_small_en_s1m7_merged_mlx",
        dest="whisper_model",
        help="Path to MLX-format merged Whisper checkpoint",
    )
    p.add_argument(
        "--gemma-model",
        type=str,
        default="./models/gemma/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        dest="gemma_model",
    )
    p.add_argument(
        "--llama-cli",
        type=str,
        default="/Users/skinnercheng/llama.cpp/build/bin/llama-cli",
        dest="llama_cli",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/S2-M4_gemma_correction_eval"),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N clips (smoke check)",
    )
    p.add_argument(
        "--gemma-timeout",
        type=float,
        default=5.0,
        dest="gemma_timeout",
        help="Per-clip Gemma correction timeout in seconds",
    )
    p.add_argument(
        "--startup-timeout",
        type=float,
        default=180.0,
        dest="startup_timeout",
        help="Max seconds to wait for Gemma worker to load",
    )
    args = p.parse_args()

    # --- Load manifest ---
    print(f"Reading manifest {args.manifest}")
    df = pd.read_csv(args.manifest)
    df = df[df["split"] == args.split].copy()
    df["audio_path_resolved"] = df["audio_path_resolved"].str.replace(
        "/Users/skinner/", "/Users/skinnercheng/", regex=False
    )
    if args.limit:
        df = df.head(args.limit)
    print(f"Evaluating {len(df)} clips (split={args.split!r})")

    normalize = create_normalizer(version=2)

    # --- Start Gemma worker ---
    print(f"Starting Gemma worker (model load ~40-50s) … {args.gemma_model}")
    t_gemma_start = time.monotonic()
    gemma = GemmaWorker(
        model_path=args.gemma_model,
        llama_cli=args.llama_cli,
        startup_timeout=args.startup_timeout,
    )
    print(f"Gemma worker ready in {time.monotonic() - t_gemma_start:.1f}s")

    # --- Per-clip eval ---
    rows: list[dict] = []
    t0 = time.monotonic()

    try:
        for r in tqdm(df.itertuples(index=False), total=len(df), desc="eval"):
            # Stage A: MLX Whisper transcription
            stage_a_raw = transcribe_mlx(r.audio_path_resolved, args.whisper_model)

            # Stage B: Gemma correction
            stage_b_raw = gemma.correct(stage_a_raw, timeout=args.gemma_timeout)

            rows.append(
                {
                    "file_name": r.file_name,
                    "pair_sha256": r.pair_sha256,
                    "split": r.split,
                    "duration_sec": r.duration_sec,
                    "duration_bin": r.duration_bin,
                    "reference_raw": r.transcript_raw,
                    "reference": normalize(r.transcript_raw),
                    "stage_a_raw": stage_a_raw,
                    "stage_a": normalize(stage_a_raw),
                    "stage_b_raw": stage_b_raw,
                    "stage_b": normalize(stage_b_raw),
                }
            )
    finally:
        gemma.close()

    elapsed = time.monotonic() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / len(rows):.2f}s/clip)")

    pred_df = pd.DataFrame(rows)

    # --- Stage A metrics ---
    stage_a_eval = pred_df[
        ["file_name", "pair_sha256", "split", "duration_sec", "duration_bin", "reference"]
    ].copy()
    stage_a_eval["hypothesis"] = pred_df["stage_a"]
    stage_a_eval = compute_sample_metrics(stage_a_eval)

    # --- Stage B metrics ---
    stage_b_eval = pred_df[
        ["file_name", "pair_sha256", "split", "duration_sec", "duration_bin", "reference"]
    ].copy()
    stage_b_eval["hypothesis"] = pred_df["stage_b"]
    stage_b_eval = compute_sample_metrics(stage_b_eval)

    # Attach per-clip WER back to pred_df for the output CSV
    pred_df["stage_a_wer"] = stage_a_eval["wer"].values
    pred_df["stage_a_word_insertions"] = stage_a_eval["word_insertions"].values
    pred_df["stage_a_word_deletions"] = stage_a_eval["word_deletions"].values
    pred_df["stage_a_word_substitutions"] = stage_a_eval["word_substitutions"].values
    pred_df["stage_b_wer"] = stage_b_eval["wer"].values
    pred_df["stage_b_word_insertions"] = stage_b_eval["word_insertions"].values
    pred_df["stage_b_word_deletions"] = stage_b_eval["word_deletions"].values
    pred_df["stage_b_word_substitutions"] = stage_b_eval["word_substitutions"].values
    pred_df["stage_b_improved"] = pred_df["stage_b_wer"] < pred_df["stage_a_wer"]
    pred_df["stage_b_worsened"] = pred_df["stage_b_wer"] > pred_df["stage_a_wer"]

    # --- Aggregate metrics ---
    stage_a_agg = compute_aggregate_metrics(stage_a_eval, [args.split])
    stage_b_agg = compute_aggregate_metrics(stage_b_eval, [args.split])

    stage_a_wer = float(stage_a_agg[args.split]["wer"])
    stage_b_wer = float(stage_b_agg[args.split]["wer"])
    false_correction_rate = float(pred_df["stage_b_worsened"].mean())
    n_clips = len(pred_df)
    n_worsened = int(pred_df["stage_b_worsened"].sum())
    n_improved = int(pred_df["stage_b_improved"].sum())

    gate1_pass = stage_b_wer <= stage_a_wer
    gate2_pass = false_correction_rate < FALSE_CORRECTION_GATE

    metrics: dict = {
        "stage_a": stage_a_agg[args.split],
        "stage_b": stage_b_agg[args.split],
        "comparison": {
            "stage_a_wer": stage_a_wer,
            "stage_b_wer": stage_b_wer,
            "wer_delta": round(stage_b_wer - stage_a_wer, 4),
            "stage_b_wer_le_stage_a_wer": gate1_pass,
        },
        "false_correction": {
            "rate": round(false_correction_rate, 4),
            "n_worsened": n_worsened,
            "n_improved": n_improved,
            "n_unchanged": n_clips - n_worsened - n_improved,
            "n_clips": n_clips,
            "gate_lt_15pct": gate2_pass,
        },
        "gates": {
            "stage_b_wer_le_stage_a": gate1_pass,
            "false_correction_lt_15pct": gate2_pass,
            "all_pass": gate1_pass and gate2_pass,
        },
        "runtime_sec": round(elapsed, 1),
    }

    # --- Write output ---
    args.output.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.output / "predictions.csv", index=False)
    with open(args.output / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nWrote {args.output}/metrics.json + predictions.csv")
    print(f"  Stage A WER : {stage_a_wer:.4f}")
    print(f"  Stage B WER : {stage_b_wer:.4f}  (delta {stage_b_wer - stage_a_wer:+.4f})")
    print(
        f"  False-correction rate: {false_correction_rate:.1%} "
        f"({n_worsened}/{n_clips} clips worsened)"
    )
    print(f"  GATE 1 (Stage B ≤ Stage A WER): {'PASS' if gate1_pass else 'FAIL'}")
    print(f"  GATE 2 (false-correction < 15%): {'PASS' if gate2_pass else 'FAIL'}")

    return 0 if (gate1_pass and gate2_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
