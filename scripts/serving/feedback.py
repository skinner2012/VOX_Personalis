"""Audio retention buffer and feedback store for correction collection.

AudioRetentionBuffer keeps raw PCM audio in memory for recent segments
so the POST /feedback endpoint can retrieve audio by segment_id.

FeedbackStore persists correction pairs to disk::

    feedback_dir/
        0001/
            audio.wav         -- 16 kHz 16-bit mono WAV
            correction.json   -- correction metadata
        0002/
            ...
"""

import json
import shutil
import uuid
import wave
from pathlib import Path

# Maximum in-memory segments before FIFO eviction
_MAX_SEGMENTS = 100

# WAV format constants — must match serving pipeline (VADSegmenter)
_SAMPLE_RATE = 16000
_SAMPLE_WIDTH = 2  # bytes per sample, 16-bit
_N_CHANNELS = 1  # mono


class AudioRetentionBuffer:
    """In-memory FIFO buffer of raw PCM audio, keyed by segment_id.

    Retains up to MAX_SEGMENTS segments. Once full, the oldest entry is
    evicted on each new insert.

    The 100-segment cap is not a practical concern: the correction UX is
    designed for immediate, in-the-moment fixes on the current or most recent
    segment — not for scrolling back through a long history. In normal use,
    far fewer than 100 segments accumulate before the user disconnects and
    the buffer resets. If a very old segment is evicted before correction,
    post_feedback returns 404 and the UI shows a brief red flash — acceptable
    degradation for an edge case that does not occur in practice.

    Thread-safety: designed for asyncio single-thread — no locking needed.
    """

    MAX_SEGMENTS = _MAX_SEGMENTS

    def __init__(self) -> None:
        self.session_id: uuid.UUID = uuid.uuid4()
        self._buf: dict[int, bytes] = {}
        self._order: list[int] = []  # insertion order for FIFO eviction

    def store(self, segment_id: int, audio_bytes: bytes) -> None:
        """Store audio for segment_id, evicting oldest entry if at capacity."""
        if segment_id in self._buf:
            return  # idempotent

        if len(self._order) >= self.MAX_SEGMENTS:
            oldest = self._order.pop(0)
            self._buf.pop(oldest, None)

        self._buf[segment_id] = audio_bytes
        self._order.append(segment_id)

    def pop(self, segment_id: int) -> bytes | None:
        """Return and remove audio for segment_id, or None if expired/unknown."""
        audio = self._buf.pop(segment_id, None)
        if audio is not None and segment_id in self._order:
            self._order.remove(segment_id)
        return audio

    def clear(self) -> None:
        """Discard all buffered audio (called on WebSocket disconnect)."""
        self._buf.clear()
        self._order.clear()


class FeedbackStore:
    """Persistent store for audio-correction pairs.

    IDs are zero-padded to 4 digits and auto-increment from the highest
    existing ID found in feedback_dir at construction time. Partial writes
    are cleaned up on exception so the directory is always consistent.
    """

    def __init__(self, feedback_dir: Path) -> None:
        self._dir = feedback_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._next_id = self._scan_highest_id() + 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        audio_bytes: bytes,
        segment_id: int,
        session_id: uuid.UUID,
        original_text: str,
        corrected_text: str,
        model_version: str = "v2",
    ) -> tuple[str, Path]:
        """Write audio.wav and correction.json for one correction pair.

        Args:
            audio_bytes: Raw 16 kHz 16-bit mono PCM bytes
            segment_id: Segment ID from the transcription WebSocket response
            session_id: WebSocket session UUID (from AudioRetentionBuffer)
            original_text: Whisper's original transcript
            corrected_text: User-provided corrected transcript
            model_version: Model version string (default: "v2")

        Returns:
            (feedback_id, entry_dir) — e.g. ("0001", Path(".../0001"))

        Raises:
            OSError: if writes fail (partial directory cleaned up first)
        """
        feedback_id = f"{self._next_id:04d}"
        entry_dir = self._dir / feedback_id
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Synchronous disk I/O is intentional: post_feedback has no await points,
        # so this runs atomically in asyncio (no interleaving risk). Blocking time
        # is ~1-2 ms for a typical utterance on SSD — negligible for a single-user
        # tool where the user takes hundreds of ms to type a correction. Use
        # asyncio.to_thread() here if this ever serves concurrent users.
        try:
            self._write_wav(entry_dir / "audio.wav", audio_bytes)
            self._write_json(
                entry_dir / "correction.json",
                feedback_id=feedback_id,
                segment_id=segment_id,
                session_id=str(session_id),
                original_text=original_text,
                corrected_text=corrected_text,
                model_version=model_version,
            )
        except Exception:
            shutil.rmtree(entry_dir, ignore_errors=True)
            raise

        self._next_id += 1
        return feedback_id, entry_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _scan_highest_id(self) -> int:
        """Return the highest integer directory name found, or 0 if none."""
        return max(
            (int(c.name) for c in self._dir.iterdir() if c.is_dir() and c.name.isdecimal()),
            default=0,
        )

    @staticmethod
    def _write_wav(path: Path, audio_bytes: bytes) -> None:
        """Write raw PCM bytes as a 16 kHz 16-bit mono WAV file."""
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(_N_CHANNELS)
            wf.setsampwidth(_SAMPLE_WIDTH)
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(audio_bytes)

    @staticmethod
    def _write_json(path: Path, **fields) -> None:
        """Write JSON file with the provided keyword fields."""
        with path.open("w", encoding="utf-8") as f:
            json.dump(fields, f, indent=2)
