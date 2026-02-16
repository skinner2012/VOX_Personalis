"""Model loading and LoRA configuration."""

# Reuse device detection from S1-M2
from baseline_eval.inference import get_available_device
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Map shorthand names to HuggingFace model IDs
MODEL_MAPPING: dict[str, str] = {
    "whisper-base.en": "openai/whisper-base.en",
    "base.en": "openai/whisper-base.en",
    "whisper-small.en": "openai/whisper-small.en",
    "small.en": "openai/whisper-small.en",
}


def load_whisper_model(
    model_name: str,
    device: str = "cpu",
) -> tuple[WhisperForConditionalGeneration, str]:
    """
    Load Whisper model from HuggingFace.

    Args:
        model_name: Model identifier (e.g., 'openai/whisper-base.en')
        device: Target device ('cpu', 'mps', 'cuda')

    Returns:
        Tuple of (model, actual_device)
    """
    # Detect available device
    actual_device = get_available_device(device)

    print(f"Loading Whisper model: {model_name}")
    print(f"Target device: {actual_device}")

    # Load model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Move to device (type: ignore due to incomplete transformers stubs)
    model = model.to(actual_device)  # type: ignore[arg-type]

    return model, actual_device


def apply_lora(
    model: WhisperForConditionalGeneration,
    rank: int = 8,
    alpha: int | None = None,
    dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> PeftModel:
    """
    Apply LoRA to Whisper model.

    Args:
        model: Base Whisper model
        rank: LoRA rank (8 or 16)
        alpha: LoRA alpha (default: 2 × rank)
        dropout: LoRA dropout rate
        target_modules: Modules to apply LoRA to (default: q_proj, v_proj)

    Returns:
        Model wrapped with PEFT LoRA
    """
    if alpha is None:
        alpha = 2 * rank

    if target_modules is None:
        # Target query and value projections in attention layers
        target_modules = ["q_proj", "v_proj"]

    print(f"Applying LoRA: rank={rank}, alpha={alpha}, dropout={dropout}")
    print(f"Target modules: {target_modules}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type="AUTOMATIC_SPEECH_RECOGNITION",
    )

    # Wrap model with PEFT
    peft_model: PeftModel = get_peft_model(model, lora_config)  # type: ignore[assignment]

    return peft_model


def count_trainable_params(model) -> dict[str, int]:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model (base or PEFT)

    Returns:
        Dictionary with trainable_params, total_params, trainable_pct
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    trainable_pct = (trainable_params / total_params) * 100

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct,
    }


def print_trainable_params(model) -> None:
    """
    Print trainable parameter summary.

    Args:
        model: PyTorch model
    """
    stats = count_trainable_params(model)

    print(f"Trainable params: {stats['trainable_params']:,} ({stats['trainable_pct']:.2f}%)")
    print(f"Total params: {stats['total_params']:,}")


def setup_model_and_processor(
    model_name: str,
    lora_rank: int,
    device: str = "cpu",
) -> tuple[PeftModel, WhisperProcessor, str]:
    """
    Load Whisper model, apply LoRA, and load processor.

    Args:
        model_name: HuggingFace model ID
        lora_rank: LoRA rank (8 or 16)
        device: Target device

    Returns:
        Tuple of (lora_model, processor, actual_device)
    """
    # Resolve model ID from shorthand or use as-is
    model_id = MODEL_MAPPING.get(model_name, model_name)

    if model_name not in MODEL_MAPPING and not model_name.startswith("openai/"):
        print(f"WARNING: Unknown model '{model_name}', passing through to HuggingFace")

    # Load base model
    base_model, actual_device = load_whisper_model(model_id, device)

    # Apply LoRA
    lora_model = apply_lora(base_model, rank=lora_rank)

    # Print trainable params
    print_trainable_params(lora_model)

    # Load processor (feature extractor + tokenizer)
    print(f"Loading WhisperProcessor for {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id)

    return lora_model, processor, actual_device


def load_checkpoint(
    checkpoint_path: str,
    base_model_name: str,
    device: str = "cpu",
) -> tuple[PeftModel, WhisperProcessor, str]:
    """
    Load fine-tuned LoRA checkpoint for evaluation.

    Args:
        checkpoint_path: Path to saved LoRA adapter
        base_model_name: Base model identifier
        device: Target device

    Returns:
        Tuple of (model, processor, actual_device)
    """
    # Resolve model ID from shorthand
    model_id = MODEL_MAPPING.get(base_model_name, base_model_name)

    # Detect device
    actual_device = get_available_device(device)

    print(f"Loading base model: {model_id}")
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    base_model = base_model.to(actual_device)  # type: ignore[arg-type]

    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    print("Setting model to evaluation mode")
    model.eval()

    # Load processor
    processor = WhisperProcessor.from_pretrained(model_id)

    return model, processor, actual_device
