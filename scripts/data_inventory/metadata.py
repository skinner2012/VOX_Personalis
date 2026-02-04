"""Audio metadata extraction functions"""

import re
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]


def extract_audio_metadata(audio_path: Path) -> dict:
    """
    Extract audio metadata using soundfile.

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with audio metadata:
        - audio_read_ok: Boolean indicating if file was readable
        - duration_sec: Duration in seconds (None if unreadable)
        - sample_rate_hz: Sample rate in Hz (None if unreadable)
        - channels: Number of channels (None if unreadable)
        - format: Audio format string (None if unreadable)
        - bit_depth: Bit depth if available (None otherwise)
    """
    try:
        info = sf.info(audio_path)

        # Extract bit depth from subtype if available (e.g., "PCM_16" -> 16)
        subtype = getattr(info, "subtype", None)
        bit_depth = None
        if subtype:
            # Try to extract numeric bit depth from formats like "PCM_16", "PCM_24"
            match = re.search(r"_(\d+)$", subtype)
            if match:
                bit_depth = int(match.group(1))

        return {
            "audio_read_ok": True,
            "duration_sec": round(info.duration, 3),
            "sample_rate_hz": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "bit_depth": bit_depth,
        }

    except Exception:
        return {
            "audio_read_ok": False,
            "duration_sec": None,
            "sample_rate_hz": None,
            "channels": None,
            "format": None,
            "bit_depth": None,
        }


# RMS floor value in dBFS for near-silence
RMS_FLOOR_DB = -100.0


def compute_rms_db(audio: np.ndarray) -> float | None:
    """
    Compute RMS (Root Mean Square) in dBFS scale.

    Assumes audio is normalized to [-1.0, 1.0] range (soundfile default).

    Args:
        audio: Audio waveform as numpy array

    Returns:
        RMS in dBFS (decibels relative to full scale), or None if computation fails.
        Returns -100.0 for near-silence (RMS < 1e-10).
    """
    try:
        # Ensure audio is float
        if not np.issubdtype(audio.dtype, np.floating):
            audio = audio.astype(np.float64)

        # Handle multi-channel: convert to mono by averaging channels
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Compute RMS
        rms = np.sqrt(np.mean(audio**2))

        # Avoid log(0) by setting a floor
        if rms < 1e-10:
            return RMS_FLOOR_DB

        # Convert to dBFS (assuming audio is normalized to [-1, 1])
        # dBFS = 20 * log10(rms / reference)
        # For normalized audio, reference = 1.0
        db = 20 * np.log10(rms)

        return round(float(db), 2)

    except Exception:
        return None
