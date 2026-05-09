"""Convert S1 CSV manifests to lhotse CutSet format for icefall fine-tuning."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment

_FEEDBACK_ROOT = os.path.expanduser("~/Downloads/feedback")


def _resolve_path(row: pd.Series, audio_root: str) -> str:
    """Resolve audio path for a manifest row.

    Euphonia clips: constructed from audio_root + file_name.
    Feedback clips: audio_path_resolved with old home dir remapped to
                    ~/Downloads/feedback/<id>/audio.wav.
    """
    source = str(row.get("source", "euphonia"))
    if source == "feedback":
        # e.g. /Users/skinner/Projects/.../out/feedback/0042/audio.wav
        # → ~/Downloads/feedback/0042/audio.wav
        old_path = str(row["audio_path_resolved"])
        clip_id = Path(old_path).parent.name  # "0042"
        path = os.path.join(_FEEDBACK_ROOT, clip_id, "audio.wav")
    else:
        path = os.path.join(audio_root, str(row["file_name"]))

    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    return path


def _make_cut(path: str, file_name: str, transcript: str) -> MonoCut:
    recording = Recording.from_file(path)
    supervision = SupervisionSegment(
        id=file_name,
        recording_id=recording.id,
        start=0.0,
        duration=recording.duration,
        text=transcript,
        language="en",
    )
    return MonoCut(
        id=file_name,
        start=0.0,
        duration=recording.duration,
        channel=0,
        recording=recording,
        supervisions=[supervision],
    )


def convert(
    train_csv: str,
    val_csv: str,
    audio_root: str,
    output_dir: str,
) -> None:
    audio_root = os.path.expanduser(audio_root)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Train split (merged_manifest: audio_path_resolved uses old home dir) ---
    print("Loading train manifest...")
    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df["split"] == "train"].reset_index(drop=True)
    print(f"  {len(train_df)} train clips")

    train_cuts = []
    for i, row in train_df.iterrows():
        try:
            path = _resolve_path(row, audio_root)
            cut = _make_cut(path, str(row["file_name"]), str(row["transcript_raw"]))
            train_cuts.append(cut)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{len(train_df)}] train cuts built...")

    train_cutset = CutSet.from_cuts(train_cuts)
    train_path = out / "cuts_train.jsonl.gz"
    train_cutset.to_jsonl(str(train_path))
    print(f"  Saved {len(train_cuts)} train cuts → {train_path}")

    # --- Dev/val split (predictions CSV: only file_name + reference_raw) ---
    print("Loading val manifest...")
    val_df = pd.read_csv(val_csv)
    if "reference_raw" in val_df.columns:
        val_df = val_df.rename(columns={"reference_raw": "transcript_raw"})
    print(f"  {len(val_df)} val clips")

    val_cuts = []
    for _, row in val_df.iterrows():
        try:
            path = _resolve_path(row, audio_root)
            cut = _make_cut(path, str(row["file_name"]), str(row["transcript_raw"]))
            val_cuts.append(cut)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}", file=sys.stderr)

    val_cutset = CutSet.from_cuts(val_cuts)
    val_path = out / "cuts_dev.jsonl.gz"
    val_cutset.to_jsonl(str(val_path))
    print(f"  Saved {len(val_cuts)} dev cuts → {val_path}")

    print("\nDone.")
    print(f"  Train: {len(train_cuts)} cuts  →  {train_path}")
    print(f"  Dev:   {len(val_cuts)} cuts  →  {val_path}")
    print(f"\nNext: tar czf voice_data.tar.gz ~/Downloads/takeout-E407/ {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert S1 CSV manifests to lhotse CutSet format for icefall fine-tuning."
    )
    parser.add_argument(
        "--train-csv",
        default="./results/M7_feedback_finetune/merged_manifest.csv",
        help="Train manifest CSV (merged_manifest with feedback corrections).",
    )
    parser.add_argument(
        "--val-csv",
        default="./results/M7_feedback_finetune/predictions.csv",
        help="Val manifest CSV (predictions with reference_raw column).",
    )
    parser.add_argument(
        "--audio-root",
        default="~/Downloads/takeout-E407",
        help="Directory containing all .wav files.",
    )
    parser.add_argument(
        "--output",
        default="./out/lhotse_manifests",
        help="Output directory for cuts_train.jsonl.gz and cuts_dev.jsonl.gz.",
    )
    args = parser.parse_args()

    try:
        convert(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            audio_root=args.audio_root,
            output_dir=args.output,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 1

    return 0
