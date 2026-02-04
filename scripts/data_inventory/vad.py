"""Voice Activity Detection (VAD) silence detection"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import webrtcvad  # type: ignore[import-untyped]


def detect_silence_vad(audio_path: Path) -> dict:
    """
    Detect silence using webrtcvad (Voice Activity Detection).

    Implementation:
    1. Read audio with soundfile (returns normalized float64, range [-1.0, 1.0])
    2. Convert multi-channel audio to mono (if needed)
    3. Resample to 16kHz in memory (webrtcvad requirement, source file untouched)
    4. Convert normalized float64 to 16-bit PCM integers
    5. Process audio in 30ms frames using VAD mode=3 (aggressive non-speech detection)
    6. Track speech/silence frames and longest consecutive silence span
    7. Compute silence ratio and longest silence duration in seconds

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with VAD metrics:
        - silence_ratio_est: Ratio of non-speech frames (0-1), or None if audio
          too short (<30ms) or error occurred
        - longest_silence_sec_est: Longest consecutive silence in seconds, or None
          if audio too short (<30ms) or error occurred
    """
    try:
        # Read audio (soundfile returns normalized float64 by default, range [-1.0, 1.0])
        audio, sr = sf.read(audio_path)

        # Handle multi-channel: convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample to 16kHz in memory (webrtcvad requirement)
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        # Convert to 16-bit PCM (audio is normalized float64 from sf.read())
        audio_int16 = (audio_16k * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Initialize VAD (mode 3 = most aggressive)
        vad = webrtcvad.Vad(3)

        # Frame parameters (webrtcvad supports 10, 20, 30 ms frames)
        frame_duration_ms = 30
        frame_size = int(16000 * frame_duration_ms / 1000)  # 480 samples at 16kHz
        frame_bytes = frame_size * 2  # 2 bytes per int16 sample

        # Process frames
        num_frames = len(audio_bytes) // frame_bytes
        speech_frames = 0
        silence_frames = 0

        # Track consecutive silence for longest_silence calculation
        current_silence_frames = 0
        longest_silence_frames = 0

        for i in range(num_frames):
            start = i * frame_bytes
            end = start + frame_bytes
            frame = audio_bytes[start:end]

            # VAD returns True if speech is detected
            is_speech = vad.is_speech(frame, sample_rate=16000)

            if is_speech:
                speech_frames += 1
                # End of silence streak
                if current_silence_frames > longest_silence_frames:
                    longest_silence_frames = current_silence_frames
                current_silence_frames = 0
            else:
                silence_frames += 1
                current_silence_frames += 1

        # Check final silence streak
        if current_silence_frames > longest_silence_frames:
            longest_silence_frames = current_silence_frames

        # Compute metrics
        total_frames = speech_frames + silence_frames
        if total_frames == 0:
            # Audio too short to process any frames
            return {
                "silence_ratio_est": None,
                "longest_silence_sec_est": None,
            }

        silence_ratio = silence_frames / total_frames

        # Convert longest silence from frames to seconds
        longest_silence_sec = (longest_silence_frames * frame_duration_ms) / 1000.0

        return {
            "silence_ratio_est": round(silence_ratio, 4),
            "longest_silence_sec_est": round(longest_silence_sec, 3),
        }

    except Exception:
        return {
            "silence_ratio_est": None,
            "longest_silence_sec_est": None,
        }
