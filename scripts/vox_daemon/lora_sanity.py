"""
Sanity check: load the S1-M7 LoRA adapter on top of whisper-small.en and
transcribe one known val clip.

Why this script exists:
  Before investing time in merge_lora.py and MLX conversion, we confirm the
  LoRA checkpoint actually loads and decodes correctly. This is a one-shot gate:
  if the hypothesis doesn't match "can you play some music", something is wrong
  with the checkpoint path or the PEFT version.

How it works:
  1. Load the base model (whisper-small.en) from HuggingFace Hub.
  2. Wrap it with PeftModel, which injects the LoRA adapter weights from the
     checkpoint directory (adapter_config.json + adapter_model.safetensors).
  3. Load the first val clip from the Euphonia dataset, resample to 16kHz.
  4. Run Whisper's feature extraction and greedy decode.
  5. Print hypothesis vs reference — expect an exact match since this clip
     had WER=0.0 in the S1-M7 evaluation run.

Note: whisper-small.en is English-only, so language/task must NOT be passed
  to generate() — the model's generation_config already encodes English
  transcription as the forced decoder prompt.
"""

import librosa
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE = "openai/whisper-small.en"
LORA = "out/feedback_finetune/batch_20260317_110057/checkpoint"
WAV = "/Users/skinnercheng/Downloads/takeout-E407/euphonia_002f4f2d5ad6ecd94202d4ef92719c02.wav"
EXPECTED = "can you play some music"

base = WhisperForConditionalGeneration.from_pretrained(BASE)
proc = WhisperProcessor.from_pretrained(BASE)

# PeftModel wraps the base model and applies LoRA adapters during the forward pass.
# merge_and_unload() is NOT called here — this tests the LoRA in its adapter form
# (the same state it was trained in). M1 handles the permanent merge.
model = PeftModel.from_pretrained(base, LORA).eval()

audio, _ = librosa.load(WAV, sr=16000)
inputs = proc(audio, sampling_rate=16000, return_tensors="pt").input_features

with torch.no_grad():
    # Positional arg form of generate() was removed in newer PEFT; use keyword.
    ids = model.generate(input_features=inputs)

print("Hypothesis:", proc.batch_decode(ids, skip_special_tokens=True)[0])
print("Reference: ", EXPECTED)
