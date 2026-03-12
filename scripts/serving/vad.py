"""Streaming Voice Activity Detection (VAD) segmenter for live audio.

This is a stateful, frame-by-frame segmenter — distinct from S1-M0's
batch `detect_silence_vad()` which processes complete audio files.

Audio format: 16 kHz, 16-bit signed integer (Int16), mono PCM.
Frame size: exactly 30 ms (480 samples × 2 bytes = 960 bytes per frame).
"""

import logging

import webrtcvad  # type: ignore[import-untyped]

log = logging.getLogger("serving.vad")


class VADSegmenter:
    """Accumulate live 30ms PCM frames and emit utterance buffers on silence.

    Usage:
        segmenter = VADSegmenter(silence_ms=1000)
        for frame in live_audio_stream:
            result = segmenter.feed(frame)
            if result is not None:
                transcribe(result)  # complete utterance ready
        # At end of session:
        remainder = segmenter.flush()
        if remainder:
            transcribe(remainder)
    """

    SAMPLE_RATE = 16000
    FRAME_MS = 30
    BYTES_PER_SAMPLE = 2  # int16
    FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480
    FRAME_BYTES = FRAME_SAMPLES * BYTES_PER_SAMPLE  # 960

    def __init__(
        self,
        *,
        mode: int = 3,
        silence_ms: int = 1000,
        min_utterance_ms: int = 300,
        max_utterance_sec: float = 30.0,
    ) -> None:
        """
        Args:
            mode: webrtcvad aggressiveness (0–3, 3 = most aggressive non-speech detection)
            silence_ms: consecutive silence duration (ms) that ends an utterance
            min_utterance_ms: minimum buffer length (ms) to transcribe (skip shorter clips)
            max_utterance_sec: force transcription when buffer reaches this duration
        """
        self._vad = webrtcvad.Vad(mode)
        self._silence_frames_threshold = silence_ms // self.FRAME_MS
        self._min_utterance_frames = min_utterance_ms // self.FRAME_MS
        self._max_utterance_frames = int(max_utterance_sec * 1000 / self.FRAME_MS)

        self._buffer: list[bytes] = []
        self._consecutive_silence: int = 0
        self._has_speech: bool = False

    def feed(self, frame: bytes) -> bytes | None:
        """Feed one 30ms PCM frame. Returns complete utterance bytes or None.

        Returns:
            bytes: buffered audio ready for transcription (utterance boundary detected)
            None: no utterance boundary yet
        Raises:
            ValueError: if frame is not exactly FRAME_BYTES bytes
        """
        if len(frame) != self.FRAME_BYTES:
            raise ValueError(f"Expected {self.FRAME_BYTES}-byte frame, got {len(frame)}")

        is_speech = self._vad.is_speech(frame, sample_rate=self.SAMPLE_RATE)
        self._buffer.append(frame)

        if is_speech:
            if not self._has_speech:
                log.debug("speech started at frame %d", len(self._buffer))
            self._has_speech = True
            self._consecutive_silence = 0
        else:
            self._consecutive_silence += 1
            if self._has_speech and self._consecutive_silence == 1:
                log.debug("silence begins after speech (buffer=%d frames)", len(self._buffer))

        # Max utterance duration — force transcription regardless of VAD
        if len(self._buffer) >= self._max_utterance_frames:
            return self._emit()

        # Silence boundary detected after speech
        if (
            self._has_speech
            and self._consecutive_silence >= self._silence_frames_threshold
            and len(self._buffer) >= self._min_utterance_frames
        ):
            return self._emit()

        return None

    def flush(self) -> bytes | None:
        """Force-return any buffered audio. Call at session end or disconnect."""
        if self._buffer and self._has_speech and len(self._buffer) >= self._min_utterance_frames:
            return self._emit()
        self._reset()
        return None

    def buffer_duration_ms(self) -> int:
        """Current buffer duration in milliseconds."""
        return len(self._buffer) * self.FRAME_MS

    def _emit(self) -> bytes:
        audio = b"".join(self._buffer)
        self._reset()
        return audio

    def _reset(self) -> None:
        self._buffer = []
        self._consecutive_silence = 0
        self._has_speech = False
