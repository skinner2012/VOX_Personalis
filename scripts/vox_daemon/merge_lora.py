"""
One-time prep: merge the S1-M7 LoRA adapter permanently into the whisper-small.en
base weights and save the result as a standard HuggingFace checkpoint.

Why merge instead of keeping the LoRA adapter separate?
  WhisperLiveKit's --lora-path flag only works with the PyTorch (native Whisper)
  backend. To get the 4-6x speedup from the MLX backend, we must pass a plain
  merged model via --model_dir. merge_and_unload() bakes the LoRA into the base
  weights (W_merged = W_base + alpha/rank * B @ A), producing a checkpoint that
  is numerically identical to the LoRA-active model under inference, with zero
  runtime overhead and no adapter handling required downstream.

How it works:
  1. Load whisper-small.en base weights from HuggingFace Hub.
  2. Wrap with PeftModel to inject the LoRA adapter.
  3. Call merge_and_unload() — permanently bakes adapter into base, returns a
     plain WhisperForConditionalGeneration with no PEFT dependency.
  4. Save the merged model + processor to the output directory in standard HF
     Transformers format (config.json, model.safetensors, tokenizer files, etc.).
  5. The output directory is then ready for mlx_whisper.convert (see M1 in the
     spec) to produce the MLX format used by WhisperLiveKit.

Run:
  python -m scripts.vox_daemon.merge_lora [--lora PATH] [--out PATH]
"""

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BASE_MODEL_ID = "openai/whisper-small.en"


def merge(lora_checkpoint: str, output_dir: str) -> None:
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
    processor = WhisperProcessor.from_pretrained(BASE_MODEL_ID)

    peft_model = PeftModel.from_pretrained(base, lora_checkpoint)

    # Permanently bake LoRA weights into the base — no adapter state retained.
    merged = peft_model.merge_and_unload()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    processor.save_pretrained(out)
    # Preserve generation_config so wlk/mlx-whisper inherit the correct forced
    # decoder tokens (English transcription) without needing --language/--task.
    merged.generation_config.save_pretrained(out)
    print(f"Saved merged HF model to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lora",
        default="out/feedback_finetune/batch_20260317_110057/checkpoint",
        help="Path to the PEFT LoRA checkpoint directory",
    )
    p.add_argument(
        "--out",
        default="out/whisper_small_en_s1m7_merged",
        help="Output directory for the merged HF model",
    )
    args = p.parse_args()
    merge(args.lora, args.out)


if __name__ == "__main__":
    main()
