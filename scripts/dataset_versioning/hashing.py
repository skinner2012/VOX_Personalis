"""SHA256 hashing utilities for audio files, transcripts, and pairs."""

import hashlib
from pathlib import Path

# 8KB chunks for memory-efficient streaming
CHUNK_SIZE = 8192


def compute_audio_sha256(audio_path: Path) -> str | None:
    """
    Compute SHA256 hash of audio file using streaming.

    Args:
        audio_path: Path to audio file

    Returns:
        Lowercase hex string (64 chars) or None on error
    """
    try:
        hasher = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def compute_transcript_sha256(transcript: str) -> str:
    """
    Compute SHA256 hash of transcript text (UTF-8 encoded).

    Args:
        transcript: Transcript text (may be empty string)

    Returns:
        Lowercase hex string (64 chars)
    """
    return hashlib.sha256(transcript.encode("utf-8")).hexdigest()


def compute_pair_sha256(audio_sha256: str, transcript_sha256: str) -> str:
    """
    Compute combined hash: SHA256(audio_sha256 + transcript_sha256).

    Both input hashes must be lowercase hex strings.

    Args:
        audio_sha256: Audio file hash (64 char hex string)
        transcript_sha256: Transcript hash (64 char hex string)

    Returns:
        Lowercase hex string (64 chars)
    """
    combined = audio_sha256 + transcript_sha256
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def compute_all_hashes(audio_path: Path, transcript: str) -> tuple[str | None, str, str | None]:
    """
    Convenience function to compute all three hashes.

    Args:
        audio_path: Path to audio file
        transcript: Transcript text

    Returns:
        (audio_sha256, transcript_sha256, pair_sha256)
        audio_sha256 and pair_sha256 may be None if audio unreadable
    """
    audio_sha256 = compute_audio_sha256(audio_path)
    transcript_sha256 = compute_transcript_sha256(transcript)

    if audio_sha256 is not None:
        pair_sha256 = compute_pair_sha256(audio_sha256, transcript_sha256)
    else:
        pair_sha256 = None

    return audio_sha256, transcript_sha256, pair_sha256
