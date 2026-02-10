"""Whisper model loading and transcription."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import torch
import whisper  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]


def get_available_device(requested_device: str) -> str:
    """
    Check device availability and return the actual device to use.

    Args:
        requested_device: Requested device (cpu, mps, cuda)

    Returns:
        Actual device string to use
    """
    if requested_device == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print("Warning: MPS not available, falling back to CPU", file=sys.stderr)
        return "cpu"
    elif requested_device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("Warning: CUDA not available, falling back to CPU", file=sys.stderr)
        return "cpu"
    return "cpu"


def load_whisper_model(model_size: str, requested_device: str) -> tuple[whisper.Whisper, str]:
    """
    Load Whisper model on the specified device.

    Args:
        model_size: Model size (tiny.en, base.en, small.en, medium.en, large)
        requested_device: Requested device (cpu, mps, cuda)

    Returns:
        Tuple of (model, actual_device)

    Raises:
        Exception: If model loading fails
    """
    actual_device = get_available_device(requested_device)

    try:
        model = whisper.load_model(model_size, device=actual_device)
        return model, actual_device
    except RuntimeError as e:
        # Handle MPS-specific errors by falling back to CPU
        if actual_device == "mps" and "MPS" in str(e):
            print(f"Warning: MPS error ({e}), falling back to CPU", file=sys.stderr)
            model = whisper.load_model(model_size, device="cpu")
            return model, "cpu"
        raise


def transcribe_audio(
    model: whisper.Whisper,
    audio_path: str,
) -> str:
    """
    Transcribe a single audio file using Whisper.

    Args:
        model: Loaded Whisper model
        audio_path: Path to audio file

    Returns:
        Transcription text

    Raises:
        Exception: If transcription fails (handled by caller)

    Note:
        Inference parameters are hardcoded for reproducible evaluation:
        - temperature=0: Deterministic greedy decoding (no sampling)
        - language="en": Matches English-only model variants (tiny.en, base.en, etc.)
        - task="transcribe": Generate transcription (not translation)
        - beam_size=5: Standard beam search width for quality/speed balance
    """
    result = model.transcribe(
        audio_path,
        temperature=0,
        language="en",
        task="transcribe",
        beam_size=5,
        fp16=model.device != torch.device("cpu"),
    )
    result_text: str = result["text"].strip()
    return result_text


def transcribe_samples(
    df: pd.DataFrame,
    model: whisper.Whisper,
    normalizer: Callable[[str], str],
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Transcribe all samples in the dataframe.

    Args:
        df: DataFrame with audio_path_resolved, transcript_raw, etc.
        model: Loaded Whisper model
        normalizer: Text normalization function
        verbose: Show progress bar and per-sample WER

    Returns:
        Tuple of (predictions DataFrame, skipped samples dict)
    """
    results = []
    skipped_samples: dict[str, Any] = {
        "count": 0,
        "reasons": {},
        "files": [],
    }

    iterator = df.iterrows()
    if verbose:
        iterator = tqdm(iterator, total=len(df), desc="Transcribing", unit="sample")

    for _, row in iterator:
        audio_path = row["audio_path_resolved"]
        file_name = row["file_name"]

        # Check if audio file exists
        if not Path(audio_path).exists():
            skipped_samples["count"] += 1
            skipped_samples["reasons"]["file_not_found"] = (
                skipped_samples["reasons"].get("file_not_found", 0) + 1
            )
            skipped_samples["files"].append(file_name)
            if verbose:
                tqdm.write(f"  Skipped (not found): {file_name}")
            continue

        # Transcribe
        try:
            hypothesis_raw = transcribe_audio(model, audio_path)
        except Exception as e:
            skipped_samples["count"] += 1
            skipped_samples["reasons"]["transcription_error"] = (
                skipped_samples["reasons"].get("transcription_error", 0) + 1
            )
            skipped_samples["files"].append(file_name)
            if verbose:
                tqdm.write(f"  Skipped (error): {file_name}: {e}")
            continue

        # Normalize both reference and hypothesis
        reference_raw = row["transcript_raw"]
        reference = normalizer(reference_raw)
        hypothesis = normalizer(hypothesis_raw)

        # Store result
        result = {
            "file_name": file_name,
            "pair_sha256": row["pair_sha256"],
            "split": row["split"],
            "duration_sec": row["duration_sec"],
            "duration_bin": row["duration_bin"],
            "reference_raw": reference_raw,
            "hypothesis_raw": hypothesis_raw,
            "reference": reference,
            "hypothesis": hypothesis,
        }
        results.append(result)

        # Verbose: show per-sample progress
        if verbose and isinstance(iterator, tqdm):
            # Quick WER estimate for progress display (normalized word count comparison)
            ref_words = len(reference.split())
            hyp_words = len(hypothesis.split())
            if ref_words > 0:
                # Simple word difference ratio (not true WER, just for progress display)
                diff_ratio = abs(ref_words - hyp_words) / ref_words
                iterator.set_postfix({"last_diff": f"{diff_ratio:.0%}"})

    predictions_df = pd.DataFrame(results)
    return predictions_df, skipped_samples
