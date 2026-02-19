"""Training loop with HuggingFace Trainer and custom WER callback."""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    WhisperProcessor,
)


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    logging_steps: int = 50
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    early_stopping_patience: int = 2
    fp16: bool = False
    device: str = "cpu"
    disable_tqdm: bool = False  # Set True to hide progress bars
    weight_decay: float = 0.0
    seed: int = 42


def set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_training_args(config: TrainingConfig) -> Seq2SeqTrainingArguments:
    """
    Create Seq2SeqTrainingArguments from config.

    Args:
        config: Training configuration

    Returns:
        Seq2SeqTrainingArguments for Trainer
    """
    return Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        # Training loop
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size * 2,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        # Optimizer
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        # Evaluation
        eval_strategy=config.evaluation_strategy,
        # Logging
        logging_dir=str(Path(config.output_dir) / "logs"),
        logging_steps=config.logging_steps,
        # Checkpointing
        save_strategy=config.save_strategy,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # Hardware
        use_cpu=config.device == "cpu",  # Force CPU when specified
        fp16=config.fp16,
        # Generation settings for Whisper
        predict_with_generate=True,
        generation_max_length=225,
        # Misc
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        # Progress bars
        disable_tqdm=config.disable_tqdm,
    )


class WERCallback(TrainerCallback):
    """
    Custom callback to compute and log WER at each evaluation.

    This callback runs inference on the evaluation dataset and computes
    WER using the same normalization as S1-M2 baseline.
    """

    def __init__(
        self,
        processor: WhisperProcessor,
        normalizer: Callable[[str], str],
        verbose: bool = True,
    ):
        """
        Initialize WER callback.

        Args:
            processor: WhisperProcessor for decoding
            normalizer: Text normalization function (textnorm_v1)
            verbose: Whether to print WER to console
        """
        self.processor = processor
        self.normalizer = normalizer
        self.verbose = verbose
        self.wer_history: list[dict[str, Any]] = []

    def on_evaluate(  # type: ignore[override]
        self,
        args: Seq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Log WER after evaluation."""
        if metrics is not None and "eval_wer" in metrics:
            wer = metrics["eval_wer"]
            epoch = state.epoch or 0

            self.wer_history.append(
                {
                    "epoch": epoch,
                    "wer": wer,
                    "step": state.global_step,
                }
            )

            if self.verbose:
                print(f"Epoch {epoch:.1f}: Val WER = {wer:.2%}")


def compute_metrics_fn(
    processor: WhisperProcessor,
    normalizer: Callable[[str], str],
) -> Callable[[Any], dict[str, float]]:
    """
    Create metrics computation function for Trainer.

    Args:
        processor: WhisperProcessor for decoding
        normalizer: Text normalization function

    Returns:
        Function that computes WER from predictions
    """
    from jiwer import wer as compute_wer

    def compute_metrics(eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Replace -100 (padding) with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id  # type: ignore[attr-defined]

        # Decode predictions and labels
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)  # type: ignore[attr-defined]
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)  # type: ignore[attr-defined]

        # Normalize both
        pred_norm = [normalizer(p) for p in pred_str]
        label_norm = [normalizer(label) for label in label_str]

        # Filter out empty references (can't compute WER)
        valid_pairs = [
            (p, ref) for p, ref in zip(pred_norm, label_norm, strict=True) if ref.strip()
        ]

        if not valid_pairs:
            return {"wer": 1.0}

        preds, labels = zip(*valid_pairs, strict=True)

        # Compute WER
        wer_score = compute_wer(list(labels), list(preds))

        return {"wer": wer_score}

    return compute_metrics


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator for speech-to-text with padding.

    Handles padding of input_features and labels separately.
    """

    processor: WhisperProcessor
    decoder_start_token_id: int | None = None

    def __post_init__(self):
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.processor.tokenizer.bos_token_id  # type: ignore[attr-defined]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Separate input_features and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input_features (type: ignore - transformers typing incomplete)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")  # type: ignore[attr-defined]

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")  # type: ignore[attr-defined]

        # Replace padding with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove bos token if present (Whisper uses decoder_start_token_id)
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch  # type: ignore[no-any-return]


def train_model(
    model: PeftModel,
    processor: WhisperProcessor,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    normalizer: Callable[[str], str],
    config: TrainingConfig,
    verbose: bool = True,
) -> tuple[PeftModel, dict]:
    """
    Train model with HuggingFace Trainer.

    Args:
        model: PEFT model with LoRA
        processor: WhisperProcessor
        train_dataset: Prepared training dataset
        eval_dataset: Prepared evaluation dataset
        normalizer: Text normalization function
        config: Training configuration
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_model, training_metrics)
    """
    # Create training arguments
    training_args = create_training_args(config)

    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,  # type: ignore[union-attr]
    )

    # Create callbacks
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_threshold=0.0,
        ),
        WERCallback(
            processor=processor,
            normalizer=normalizer,
            verbose=verbose,
        ),
    ]

    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(processor, normalizer),
        callbacks=callbacks,
    )

    # Track training time
    start_time = time.time()

    # Train
    if verbose:
        print("\nStarting training...")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Eval samples: {len(eval_dataset)}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Effective batch: {config.batch_size * config.gradient_accumulation_steps}")
        print(f"  Learning rate: {config.learning_rate}")
        print()

    train_result = trainer.train()

    training_time = time.time() - start_time

    # Get WER history from callback
    wer_callback = next((cb for cb in callbacks if isinstance(cb, WERCallback)), None)
    wer_history = wer_callback.wer_history if wer_callback else []

    # Compile training metrics
    training_metrics = {
        "training_time_sec": training_time,
        "epochs_trained": train_result.metrics.get("epoch", config.epochs),
        "train_loss": train_result.metrics.get("train_loss"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "wer_history": wer_history,
    }

    if verbose:
        print(f"\nTraining completed in {training_time / 60:.1f} minutes")
        print(f"Final train loss: {training_metrics['train_loss']:.4f}")

    return model, training_metrics


def save_checkpoint(
    model: PeftModel,
    processor: WhisperProcessor,
    output_dir: str | Path,
    training_config: dict | None = None,
) -> Path:
    """
    Save LoRA checkpoint and processor.

    Args:
        model: Trained PEFT model
        processor: WhisperProcessor
        output_dir: Output directory
        training_config: Optional training config to save

    Returns:
        Path to saved checkpoint directory
    """
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter
    model.save_pretrained(str(checkpoint_dir))

    # Save processor
    processor.save_pretrained(str(checkpoint_dir))

    # Save training config if provided
    if training_config:
        config_path = checkpoint_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(training_config, f, indent=2)

    print(f"Checkpoint saved to: {checkpoint_dir}")

    return checkpoint_dir
