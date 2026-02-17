"""Data loading and preprocessing for fine-tuning."""

from collections.abc import Callable
from pathlib import Path

import librosa
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, WhisperProcessor


def load_manifest(manifest_path: str | Path, split: str) -> pd.DataFrame:
    """
    Load Dataset v1 manifest and filter by split.

    Args:
        manifest_path: Path to dataset_v1_manifest.csv
        split: Split to filter ('train', 'val', or 'test')

    Returns:
        DataFrame with samples from the specified split

    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If split column missing or no samples found
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    # Load manifest
    df = pd.read_csv(manifest_path)

    # Validate required columns
    required_cols = [
        "file_name",
        "audio_path_resolved",
        "transcript_raw",
        "split",
        "duration_sec",
        "duration_bin",
        "pair_sha256",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Filter by split
    split_df = df[df["split"] == split].copy()

    if len(split_df) == 0:
        raise ValueError(f"No samples found for split='{split}'")

    return split_df


def create_hf_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert pandas DataFrame to HuggingFace Dataset.

    Args:
        df: DataFrame with audio_path_resolved column

    Returns:
        HuggingFace Dataset (audio loaded lazily during preprocessing)
    """
    # Convert to HF Dataset
    # Note: We load audio manually in preprocess_function to avoid
    # casting issues with large_string -> Audio struct
    dataset = Dataset.from_pandas(df, preserve_index=False)

    return dataset


def prepare_dataset(
    dataset: Dataset,
    processor: WhisperProcessor,
    normalizer: Callable[[str], str],
    verbose: bool = True,
) -> Dataset:
    """
    Prepare dataset for fine-tuning: extract features and tokenize labels.

    Args:
        dataset: HuggingFace Dataset with Audio column
        processor: WhisperProcessor for feature extraction and tokenization
        normalizer: Text normalization function (textnorm_v1 from S1-M2)
        verbose: Whether to show progress bar

    Returns:
        Dataset with input_features and labels columns
    """

    def preprocess_function(batch):
        """Process a batch of samples."""
        # Load audio with librosa (Whisper expects 16kHz)
        audio_path = batch["audio_path_resolved"]
        audio_array, sr = librosa.load(audio_path, sr=16000)

        # Extract mel spectrogram features
        input_features = processor.feature_extractor(
            audio_array,
            sampling_rate=sr,
        ).input_features[0]

        # Normalize transcript (same as S1-M2 baseline)
        transcript_norm = normalizer(batch["transcript_raw"])

        # Tokenize transcript for labels
        labels = processor.tokenizer(transcript_norm).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }

    # Apply preprocessing (map one sample at a time to avoid OOM)
    dataset = dataset.map(
        preprocess_function,
        remove_columns=dataset.column_names,
        desc="Preparing audio features" if verbose else None,
    )

    return dataset


def create_data_collator(processor: WhisperProcessor):
    """
    Create data collator for batching with padding.

    Args:
        processor: WhisperProcessor

    Returns:
        DataCollatorForSeq2Seq instance
    """
    return DataCollatorForSeq2Seq(
        tokenizer=processor.tokenizer,  # type: ignore[attr-defined]  # transformers typing incomplete
        model=None,  # Will be set by Trainer
        padding=True,
    )
