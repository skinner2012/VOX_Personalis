"""Model loading, warm-up, and inference for the serving layer.

Reuses load_checkpoint() from fine_tuning.models. Reads base_model_name
from checkpoint/adapter_config.json (saved automatically by PEFT).
"""

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Module-level state — set by load_for_serving(), read by api.py
_model: Any = None
_processor: Any = None
_decode_cfg: dict = {}
_device: str = "cpu"
_base_model_name: str = ""
model_loaded: bool = False


def load_for_serving(
    checkpoint_path: str,
    decode_config_path: str,
    device: str = "cpu",
) -> None:
    """Load Model v2 checkpoint and decode config into module-level state.

    Reads base_model_name from checkpoint/adapter_config.json (PEFT standard).
    Calls load_checkpoint() from fine_tuning.models to handle PEFT/device setup.

    Args:
        checkpoint_path: Path to LoRA adapter directory (checkpoint/)
        decode_config_path: Path to DECODE_V1.json
        device: Target device ('cpu', 'mps', 'cuda')

    Raises:
        FileNotFoundError: if checkpoint or adapter_config.json not found
        RuntimeError: if model load fails
    """
    global _model, _processor, _decode_cfg, _device, _base_model_name

    cp = Path(checkpoint_path)
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cp}")

    adapter_cfg_path = cp / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in: {cp}")

    with adapter_cfg_path.open() as f:
        adapter_cfg = json.load(f)

    base_model_name = adapter_cfg.get("base_model_name_or_path", "")
    if not base_model_name:
        raise RuntimeError("adapter_config.json missing 'base_model_name_or_path'")

    dc_path = Path(decode_config_path)
    if not dc_path.exists():
        raise FileNotFoundError(f"Decode config not found: {dc_path}")

    with dc_path.open() as f:
        _decode_cfg = json.load(f)

    # Reuse the existing checkpoint loader from the fine_tuning pipeline
    from fine_tuning.models import load_checkpoint

    _base_model_name = base_model_name
    _model, _processor, _device = load_checkpoint(
        checkpoint_path=str(cp),
        base_model_name=base_model_name,
        device=device,
    )


def warm_up() -> None:
    """Run one forward pass on 0.5s of silent audio to force PyTorch JIT compilation.

    Must be called after load_for_serving(). Sets model_loaded = True on success.
    """
    global model_loaded

    if _model is None or _processor is None:
        raise RuntimeError("load_for_serving() must be called before warm_up()")

    # 0.5 seconds of silence at 16kHz = 8000 zero samples (float32)
    silent_audio = np.zeros(8000, dtype=np.float32)
    _transcribe_audio(silent_audio)
    model_loaded = True


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe raw 16kHz 16-bit PCM bytes. Returns raw model output (no textnorm).

    Args:
        audio_bytes: Raw PCM — 16kHz, 16-bit signed integer, mono

    Returns:
        Whisper's natural output text (capitalization and punctuation preserved)
    """
    if _model is None or _processor is None:
        raise RuntimeError("Model not loaded — call load_for_serving() first")

    # Convert Int16 bytes → float32 numpy (normalize to [-1.0, 1.0])
    n_samples = len(audio_bytes) // 2
    samples = struct.unpack(f"{n_samples}h", audio_bytes)
    audio_np = np.array(samples, dtype=np.float32) / 32767.0

    return _transcribe_audio(audio_np)


def _transcribe_audio(audio_np: np.ndarray) -> str:
    """Run Whisper inference on a float32 numpy array (16kHz, mono)."""
    input_features = _processor.feature_extractor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features.to(_device)

    generate_kwargs: dict[str, Any] = {}

    # Only add language/task for multilingual models (skip for .en models)
    is_english_only = ".en" in _base_model_name
    if not is_english_only:
        if "language" in _decode_cfg:
            generate_kwargs["language"] = _decode_cfg["language"]
        if "task" in _decode_cfg:
            generate_kwargs["task"] = _decode_cfg["task"]

    beam_size = _decode_cfg.get("beam_size", 5)
    temperature = float(_decode_cfg.get("temperature", 0))

    if beam_size and beam_size > 1:
        generate_kwargs["num_beams"] = beam_size
    if temperature > 0:
        generate_kwargs["temperature"] = temperature

    with torch.no_grad():
        predicted_ids = _model.generate(input_features=input_features, **generate_kwargs)

    text: str = _processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()
