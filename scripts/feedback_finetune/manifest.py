"""Manifest generation from collected corrections for feedback fine-tuning."""

import json
import wave
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from dataset_versioning.hashing import (
    compute_audio_sha256,
    compute_pair_sha256,
    compute_transcript_sha256,
)

# Duration bin edges matching M1 convention: [0, 1, 3, 10, 30, inf]
_BIN_EDGES = [0, 1, 3, 10, 30, float("inf")]
_BIN_LABELS = ["0-1s", "1-3s", "3-10s", "10-30s", "30s+"]


def compute_duration_bin(duration_sec: float) -> str:
    """Return bin label for duration_sec using [0, 1, 3, 10, 30, inf] edges."""
    for i, edge in enumerate(_BIN_EDGES[1:], 1):
        if duration_sec < edge:
            return _BIN_LABELS[i - 1]
    return _BIN_LABELS[-1]


def _wav_duration_sec(wav_path: Path) -> float:
    """Read WAV header to get duration in seconds."""
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def scan_pending_corrections(feedback_dir: Path) -> list[Path]:
    """Return correction dirs that have not yet been consumed.

    A directory is pending if it contains audio.wav + correction.json
    and does NOT contain consumed.marker.

    Args:
        feedback_dir: Root feedback directory

    Returns:
        Sorted list of Path objects for each unconsumed correction directory
    """
    pending: list[Path] = []
    if not feedback_dir.exists():
        return pending
    for child in sorted(feedback_dir.iterdir()):
        if (
            child.is_dir()
            and (child / "audio.wav").exists()
            and (child / "correction.json").exists()
            and not (child / "consumed.marker").exists()
        ):
            pending.append(child)
    return pending


def generate_manifest_rows(correction_dirs: list[Path]) -> pd.DataFrame:
    """Build a manifest DataFrame from a list of correction directories.

    Produces the 7 required columns for load_manifest() compatibility:
    file_name, audio_path_resolved, transcript_raw, split,
    duration_sec, duration_bin, pair_sha256.

    Also adds a 'source' column set to 'feedback' for traceability.

    Args:
        correction_dirs: Unconsumed correction directories

    Returns:
        DataFrame with one row per correction (rows with unreadable audio dropped)
    """
    rows = []
    for correction_dir in correction_dirs:
        wav_path = correction_dir / "audio.wav"
        json_path = correction_dir / "correction.json"

        with json_path.open(encoding="utf-8") as f:
            meta = json.load(f)

        corrected_text = meta["corrected_text"]
        duration_sec = _wav_duration_sec(wav_path)
        audio_sha256 = compute_audio_sha256(wav_path)
        transcript_sha256 = compute_transcript_sha256(corrected_text)
        pair_sha256 = (
            compute_pair_sha256(audio_sha256, transcript_sha256)
            if audio_sha256 is not None
            else None
        )

        rows.append(
            {
                "file_name": correction_dir.name,
                "audio_path_resolved": str(wav_path.resolve()),
                "transcript_raw": corrected_text,
                "split": "train",
                "duration_sec": round(duration_sec, 3),
                "duration_bin": compute_duration_bin(duration_sec),
                "pair_sha256": pair_sha256,
                "source": "feedback",
            }
        )

    df = pd.DataFrame(rows)
    # Drop rows where audio was unreadable (pair_sha256 is None)
    if not df.empty:
        df = df.dropna(subset=["pair_sha256"]).reset_index(drop=True)
    return df


def merge_manifests(
    original_train_df: pd.DataFrame,
    corrections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate original training samples and correction samples.

    Adds a 'source' column ('original' or 'feedback') for traceability.

    Args:
        original_train_df: From load_manifest(original_manifest, 'train')
        corrections_df: From generate_manifest_rows()

    Returns:
        Merged DataFrame reset to contiguous index
    """
    orig = original_train_df.copy()
    if "source" not in orig.columns:
        orig["source"] = "original"

    combined = pd.concat([orig, corrections_df], ignore_index=True)
    return combined.reset_index(drop=True)


def mark_consumed(
    correction_dirs: list[Path],
    batch_id: str,
    output_checkpoint: str,
    eval_status: str,
) -> None:
    """Write consumed.marker JSON to each correction directory.

    Args:
        correction_dirs: Directories to mark as consumed
        batch_id: Batch timestamp identifier (e.g. "20260314_120000")
        output_checkpoint: Absolute path to the saved LoRA checkpoint
        eval_status: One of 'improved', 'not_improved', 'not_evaluated'
    """
    marker = {
        "consumed_at": datetime.now(UTC).isoformat(),
        "batch_id": batch_id,
        "output_checkpoint": output_checkpoint,
        "eval_status": eval_status,
    }
    for d in correction_dirs:
        with (d / "consumed.marker").open("w", encoding="utf-8") as f:
            json.dump(marker, f, indent=2)
