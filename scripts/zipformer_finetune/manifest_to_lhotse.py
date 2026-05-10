"""Convert S1 CSV manifests to lhotse CutSet format for icefall fine-tuning.

Outputs cuts WITH pre-computed 80-dim fbank features at 16kHz, matching
icefall's expected input format. Also creates the icefall-convention symlinks
so the cloud bootstrap can drop them straight in.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
from lhotse.features.kaldi.extractors import Fbank, FbankConfig

_FEEDBACK_ROOT = os.path.expanduser("~/Downloads/feedback")

# icefall expects fbank features in this directory layout, with these specific
# filenames (the LibriSpeech recipe is hardcoded around them). We map our cuts
# to those names via symlinks so finetune.py picks them up unchanged.
#
# IMPORTANT: macOS APFS is case-insensitive by default. Names like "cuts_DEV"
# collide with "cuts_dev" — creating the symlink would overwrite the real file.
# So we ONLY create the case-non-clashing symlinks here. The case-clashing ones
# (cuts_DEV.jsonl.gz) are created by bootstrap_a10.sh on the case-sensitive Linux
# cloud filesystem.
_ICEFALL_TRAIN_NAMES = [
    "cuts_S.jsonl.gz",  # GigaSpeech subset-S — script's default training cuts
    "librispeech_cuts_train-clean-100.jsonl.gz",
]
_ICEFALL_DEV_NAMES = [
    # "cuts_DEV.jsonl.gz" — created on cloud only (case-collision with cuts_dev)
    "librispeech_cuts_dev-clean.jsonl.gz",
    "librispeech_cuts_dev-other.jsonl.gz",
]


def _resolve_path(row: pd.Series, audio_root: str) -> str:
    """Resolve audio path for a manifest row.

    Euphonia clips: constructed from audio_root + file_name.
    Feedback clips: audio_path_resolved with old home dir remapped to
                    ~/Downloads/feedback/<id>/audio.wav.
    """
    source = str(row.get("source", "euphonia"))
    if source == "feedback":
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


def _build_cutset(df: pd.DataFrame, audio_root: str, label: str) -> CutSet:
    cuts = []
    total = len(df)
    for i, row in df.iterrows():
        try:
            path = _resolve_path(row, audio_root)
            cut = _make_cut(path, str(row["file_name"]), str(row["transcript_raw"]))
            cuts.append(cut)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}", file=sys.stderr)
        if (i + 1) % 500 == 0:
            print(f"  [{i + 1}/{total}] {label} cuts built...")
    return CutSet.from_cuts(cuts)


def _compute_features(
    cutset: CutSet,
    storage_path: Path,
    num_jobs: int,
    label: str,
) -> CutSet:
    """Resample to 16kHz and compute 80-dim fbank features.

    Features are stored as Lilcom-compressed chunks at storage_path. The returned
    cuts have absolute paths pointing at storage_path so that on the cloud, after
    extracting the tarball with `sudo tar xzf ... -C /`, the same paths resolve.
    """
    print(f"  Computing fbank features for {label} ({len(cutset)} cuts, {num_jobs} workers)...")
    storage_path.mkdir(parents=True, exist_ok=True)
    extractor = Fbank(FbankConfig(sampling_rate=16000, num_mel_bins=80))
    cuts_16k = cutset.resample(16000)
    cuts_with_features = cuts_16k.compute_and_store_features(
        extractor=extractor,
        storage_path=str(storage_path),
        num_jobs=num_jobs,
    )
    return cuts_with_features


def _create_icefall_symlinks(out_dir: Path) -> None:
    """Create icefall-convention symlinks for cuts files."""
    for name in _ICEFALL_TRAIN_NAMES:
        link = out_dir / name
        link.unlink(missing_ok=True)
        link.symlink_to("cuts_train.jsonl.gz")
    for name in _ICEFALL_DEV_NAMES:
        link = out_dir / name
        link.unlink(missing_ok=True)
        link.symlink_to("cuts_dev.jsonl.gz")


def convert(
    train_csv: str,
    val_csv: str,
    audio_root: str,
    output_dir: str,
    num_jobs: int,
) -> None:
    audio_root = os.path.expanduser(audio_root)
    out = Path(output_dir).resolve()  # absolute path so feature storage is portable
    out.mkdir(parents=True, exist_ok=True)

    # --- Train split (merged_manifest: audio_path_resolved uses old home dir) ---
    print("Loading train manifest...")
    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df["split"] == "train"].reset_index(drop=True)
    print(f"  {len(train_df)} train clips")
    train_cuts = _build_cutset(train_df, audio_root, "train")

    train_cuts = _compute_features(
        train_cuts,
        out / "fbank_storage_train",
        num_jobs=num_jobs,
        label="train",
    )
    train_path = out / "cuts_train.jsonl.gz"
    train_cuts.to_jsonl(str(train_path))
    print(f"  Saved {len(train_cuts)} train cuts (with features) → {train_path}")

    # --- Dev/val split (predictions CSV: only file_name + reference_raw) ---
    print("Loading val manifest...")
    val_df = pd.read_csv(val_csv)
    if "reference_raw" in val_df.columns:
        val_df = val_df.rename(columns={"reference_raw": "transcript_raw"})
    print(f"  {len(val_df)} val clips")
    val_cuts = _build_cutset(val_df, audio_root, "dev")

    val_cuts = _compute_features(
        val_cuts,
        out / "fbank_storage_dev",
        num_jobs=num_jobs,
        label="dev",
    )
    val_path = out / "cuts_dev.jsonl.gz"
    val_cuts.to_jsonl(str(val_path))
    print(f"  Saved {len(val_cuts)} dev cuts (with features) → {val_path}")

    # --- icefall-convention symlinks ---
    print("Creating icefall-convention symlinks...")
    _create_icefall_symlinks(out)
    for name in _ICEFALL_TRAIN_NAMES + _ICEFALL_DEV_NAMES:
        link = out / name
        if link.is_symlink():
            print(f"  {name} → {os.readlink(link)}")

    print("\nDone.")
    print(f"  Train: {len(train_cuts)} cuts  →  {train_path}")
    print(f"  Dev:   {len(val_cuts)} cuts  →  {val_path}")
    print("\nNext: tar czf ~/Downloads/voice_data_v2.tar.gz \\")
    print("        ~/Downloads/takeout-E407/ ~/Downloads/feedback/ \\")
    print(f"        {out}/")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert S1 CSV manifests to lhotse CutSet format with pre-computed "
            "16kHz fbank features for icefall fine-tuning."
        )
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
        help="Output directory for cuts + fbank feature storage.",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=4,
        help="Parallel workers for feature extraction (default: 4).",
    )
    args = parser.parse_args()

    try:
        convert(
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            audio_root=args.audio_root,
            output_dir=args.output,
            num_jobs=args.num_jobs,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 1

    return 0
