"""
Convert a HuggingFace Transformers Whisper checkpoint to MLX-Whisper format.

Why this script exists:
  The `mlx-whisper` PyPI package (0.4.3, latest) ships only the runtime — no
  `convert` subcommand. The conversion tooling lives in the upstream
  `ml-explore/mlx-examples` GitHub repo (whisper/convert.py), which we'd need to
  vendor + patch (it saves output as `model.safetensors`, but
  `mlx_whisper.load_models` looks for `weights.safetensors` or `weights.npz`).

  Rather than vendoring, this script implements the same conversion in ~70 lines
  using the installed `mlx_whisper.whisper` and `mlx_whisper.torch_whisper`
  modules as a library, and writes the file with the exact name the runtime
  expects.

How it works:
  1. Load HF Transformers config.json + model.safetensors from --hf-dir.
  2. Translate HF dim keys to OpenAI Whisper dim keys (n_mels, n_audio_state,
     n_text_layer, etc.) — required by `mlx_whisper.whisper.ModelDimensions`.
  3. Remap weight keys: HF naming (`model.encoder.layers.N.self_attn.q_proj`)
     to MLX naming (`encoder.blocks.N.attn.query`). Drop the encoder positional
     embedding (MLX regenerates it as fixed sinusoids) and the proj_out weight
     (tied to token_embedding in MLX).
  4. Conv1d weights need axis swap: HF stores (out, in, kernel), MLX expects
     (out, kernel, in).
  5. Cast to fp16 (or fp32) and save as `weights.safetensors` + `config.json`
     in the output directory.

Run:
  python -m scripts.vox_daemon.hf_to_mlx \
    --hf-dir ./out/whisper_small_en_s1m7_merged \
    --mlx-dir ./out/whisper_small_en_s1m7_merged_mlx \
    --dtype float16
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx  # type: ignore[import-not-found]
from mlx.utils import tree_flatten  # type: ignore[import-not-found]
from mlx_whisper.whisper import ModelDimensions, Whisper  # type: ignore[import-untyped]
from safetensors.torch import load_file as safe_load

# Map HF Transformers config keys → MLX/OpenAI Whisper dim keys.
HF_TO_MLX_DIM_KEYS = {
    "num_mel_bins": "n_mels",
    "max_source_positions": "n_audio_ctx",
    "d_model": "n_audio_state",  # also n_text_state; MLX uses same value for both
    "encoder_attention_heads": "n_audio_head",
    "encoder_layers": "n_audio_layer",
    "vocab_size": "n_vocab",
    "max_target_positions": "n_text_ctx",
    "decoder_attention_heads": "n_text_head",
    "decoder_layers": "n_text_layer",
}


def hf_config_to_mlx_dims(hf_config: dict) -> dict:
    """Translate the dim-related fields from HF config to MLX config."""
    dims = {mlx_key: hf_config[hf_key] for hf_key, mlx_key in HF_TO_MLX_DIM_KEYS.items()}
    dims["n_text_state"] = hf_config["d_model"]
    return dims


def remap_hf_key(key: str) -> str:
    """Translate a HF Transformers Whisper state_dict key to MLX naming."""
    k = key
    k = k.replace("model.", "")
    k = k.replace(".layers.", ".blocks.")
    k = k.replace(".self_attn.", ".attn.")
    k = k.replace(".self_attn_layer_norm", ".attn_ln")
    k = k.replace(".encoder_attn.", ".cross_attn.")
    k = k.replace(".encoder_attn_layer_norm", ".cross_attn_ln")
    k = k.replace(".final_layer_norm", ".mlp_ln")
    k = k.replace(".q_proj", ".query")
    k = k.replace(".k_proj", ".key")
    k = k.replace(".v_proj", ".value")
    k = k.replace(".out_proj", ".out")
    # MLX uses mlp1/mlp2 (no Sequential), HF uses fc1/fc2 — same target.
    k = k.replace(".fc1", ".mlp1")
    k = k.replace(".fc2", ".mlp2")
    k = k.replace("embed_positions.weight", "positional_embedding")
    k = k.replace("decoder.embed_tokens", "decoder.token_embedding")
    k = k.replace("encoder.layer_norm", "encoder.ln_post")
    k = k.replace("decoder.layer_norm", "decoder.ln")
    return k


def convert(hf_dir: Path, mlx_dir: Path, dtype: mx.Dtype) -> None:
    with open(hf_dir / "config.json") as f:
        hf_config = json.load(f)

    weights = safe_load(str(hf_dir / "model.safetensors"))

    # proj_out is tied to decoder.token_embedding in MLX — drop the duplicate.
    weights.pop("proj_out.weight", None)
    # MLX regenerates encoder positional embedding from fixed sinusoids.
    weights.pop("model.encoder.embed_positions.weight", None)

    remapped: dict[str, mx.array] = {}
    for hf_key, tensor in weights.items():
        mlx_key = remap_hf_key(hf_key)
        # Conv1d: HF stores (out_ch, in_ch, kernel), MLX expects (out_ch, kernel, in_ch).
        if "conv" in mlx_key and tensor.ndim == 3:
            tensor = tensor.swapaxes(1, 2)
        remapped[mlx_key] = mx.array(tensor.detach().cpu().numpy()).astype(dtype)

    dims = hf_config_to_mlx_dims(hf_config)
    model = Whisper(ModelDimensions(**dims), dtype)
    model.load_weights(list(remapped.items()), strict=False)

    mlx_dir.mkdir(parents=True, exist_ok=True)

    # Save weights with the EXACT filename mlx_whisper.load_models expects.
    out_weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(mlx_dir / "weights.safetensors"), out_weights)

    # Save MLX-style config (with model_type marker for the runtime).
    mlx_config = asdict(model.dims)
    mlx_config["model_type"] = "whisper"
    with open(mlx_dir / "config.json", "w") as f:
        json.dump(mlx_config, f, indent=4)

    print(f"Saved MLX checkpoint to {mlx_dir}")
    print(f"  weights.safetensors: {(mlx_dir / 'weights.safetensors').stat().st_size / 1e6:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hf-dir",
        required=True,
        type=Path,
        help="HuggingFace Transformers Whisper checkpoint directory",
    )
    p.add_argument(
        "--mlx-dir", required=True, type=Path, help="Output directory for the MLX-format checkpoint"
    )
    p.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Output weight precision (fp16 saves memory + matches MLX default)",
    )
    args = p.parse_args()

    dtype = mx.float16 if args.dtype == "float16" else mx.float32
    convert(args.hf_dir, args.mlx_dir, dtype)


if __name__ == "__main__":
    main()
