"""Evaluation and inference for fine-tuned models."""

from collections.abc import Callable

import pandas as pd
import torch

# Reuse metrics from S1-M2 baseline
from baseline_eval.metrics import (
    compute_sample_cer,
    compute_sample_wer,
)
from datasets import Dataset  # type: ignore[import-untyped]
from peft import PeftModel
from tqdm import tqdm
from transformers import WhisperProcessor


def run_inference(
    model: PeftModel,
    dataset: Dataset,
    processor: WhisperProcessor,
    normalizer: Callable[[str], str],
    device: str = "cpu",
    verbose: bool = True,
    beam_size: int = 1,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """
    Run inference on dataset and return predictions DataFrame.

    Args:
        model: Fine-tuned PEFT model
        dataset: HuggingFace Dataset with audio features
        processor: WhisperProcessor for decoding
        normalizer: Text normalization function
        device: Device to run on
        verbose: Whether to show progress bar
        beam_size: Number of beams for beam search (1 = greedy)
        temperature: Sampling temperature (0.0 = deterministic)

    Returns:
        DataFrame with predictions and metadata
    """
    model.eval()

    results = []
    iterator = tqdm(range(len(dataset)), desc="Inference") if verbose else range(len(dataset))

    # Build generation kwargs
    gen_kwargs: dict = {
        "max_length": 225,
        "num_beams": beam_size,
    }

    # Temperature handling: 0.0 means deterministic (greedy/beam), >0 enables sampling
    if temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature

    if verbose:
        print(f"  Decoding: beam_size={beam_size}, temperature={temperature}")

    for idx in iterator:
        sample = dataset[idx]

        # Prepare input
        input_features = torch.tensor(sample["input_features"]).unsqueeze(0)
        input_features = input_features.to(device)

        # Generate transcription
        # Note: language/task locked by forced_decoder_ids in English-only models
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,  # PEFT models require keyword args
                **gen_kwargs,
            )

        # Decode
        hypothesis_raw = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Normalize
        hypothesis = normalizer(hypothesis_raw)

        results.append(
            {
                "hypothesis_raw": hypothesis_raw,
                "hypothesis": hypothesis,
            }
        )

    return pd.DataFrame(results)


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    normalizer: Callable[[str], str],
) -> pd.DataFrame:
    """
    Compute metrics for predictions.

    Args:
        predictions_df: DataFrame with hypothesis column
        manifest_df: Original manifest with transcript_raw
        normalizer: Text normalization function

    Returns:
        DataFrame with predictions and metrics
    """
    # Ensure same length
    if len(predictions_df) != len(manifest_df):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions_df)}, manifest={len(manifest_df)}"
        )

    # Reset indices for alignment
    predictions_df = predictions_df.reset_index(drop=True)
    manifest_df = manifest_df.reset_index(drop=True)

    # Add manifest metadata
    result_df = manifest_df[
        ["file_name", "pair_sha256", "split", "duration_sec", "duration_bin"]
    ].copy()

    # Add reference
    result_df["reference_raw"] = manifest_df["transcript_raw"]
    result_df["reference"] = manifest_df["transcript_raw"].apply(normalizer)

    # Add hypothesis
    result_df["hypothesis_raw"] = predictions_df["hypothesis_raw"]
    result_df["hypothesis"] = predictions_df["hypothesis"]

    # Compute WER for each sample
    wer_results = result_df.apply(
        lambda row: compute_sample_wer(row["reference"], row["hypothesis"]),
        axis=1,
    )
    result_df["wer"] = wer_results.apply(lambda x: x["wer"])
    result_df["word_insertions"] = wer_results.apply(lambda x: x["word_insertions"])
    result_df["word_deletions"] = wer_results.apply(lambda x: x["word_deletions"])
    result_df["word_substitutions"] = wer_results.apply(lambda x: x["word_substitutions"])

    # Compute CER for each sample
    cer_results = result_df.apply(
        lambda row: compute_sample_cer(row["reference"], row["hypothesis"]),
        axis=1,
    )
    result_df["cer"] = cer_results.apply(lambda x: x["cer"])

    return result_df


def compute_aggregate_metrics(
    predictions_df: pd.DataFrame,
    split: str,
) -> dict:
    """
    Compute aggregate metrics for a split.

    Args:
        predictions_df: DataFrame with per-sample metrics
        split: Split name (val or test)

    Returns:
        Dictionary with aggregate metrics
    """
    df = predictions_df[predictions_df["split"] == split]

    if len(df) == 0:
        return {}

    # Total word counts for proper WER aggregation
    total_words = df["reference"].apply(lambda x: len(x.split())).sum()
    total_errors = (
        df["word_insertions"].sum() + df["word_deletions"].sum() + df["word_substitutions"].sum()
    )

    # Aggregate WER (not mean of per-sample WERs)
    aggregate_wer = total_errors / total_words if total_words > 0 else 0

    # Total chars for CER
    total_chars = df["reference"].apply(len).sum()

    return {
        "sample_count": len(df),
        "total_duration_sec": df["duration_sec"].sum(),
        "total_words": total_words,
        "total_chars": total_chars,
        "wer": aggregate_wer,
        "cer": df["cer"].mean(),  # Mean of per-sample CER
        "insertions": int(df["word_insertions"].sum()),
        "deletions": int(df["word_deletions"].sum()),
        "substitutions": int(df["word_substitutions"].sum()),
    }


def compare_to_baseline(
    finetuned_metrics: dict,
    baseline_metrics: dict,
    split: str = "val",
) -> dict:
    """
    Compute improvement over baseline.

    Args:
        finetuned_metrics: Fine-tuned model metrics
        baseline_metrics: S1-M2 baseline metrics
        split: Split to compare

    Returns:
        Dictionary with comparison metrics
    """
    ft_wer = finetuned_metrics.get("wer", 0)

    # Get baseline WER from baseline_metrics.json structure
    baseline_agg = baseline_metrics.get("aggregate", {})
    bl_wer = baseline_agg.get(split, {}).get("wer", 0)

    # Compute improvements
    absolute_improvement = bl_wer - ft_wer
    relative_improvement_pct = (absolute_improvement / bl_wer) * 100 if bl_wer > 0 else 0

    return {
        "baseline_wer": bl_wer,
        "finetuned_wer": ft_wer,
        "absolute_improvement": absolute_improvement,
        "relative_improvement_pct": relative_improvement_pct,
        "improved": ft_wer < bl_wer,
    }


def run_full_evaluation(
    model: PeftModel,
    processor: WhisperProcessor,
    manifest_df: pd.DataFrame,
    prepared_dataset: Dataset,
    normalizer: Callable[[str], str],
    baseline_metrics: dict,
    split: str,
    device: str = "cpu",
    verbose: bool = True,
    beam_size: int = 1,
    temperature: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Run full evaluation pipeline.

    Args:
        model: Fine-tuned model
        processor: WhisperProcessor
        manifest_df: Original manifest DataFrame
        prepared_dataset: Prepared HuggingFace dataset
        normalizer: Text normalization function
        baseline_metrics: S1-M2 baseline metrics dict
        split: Split being evaluated
        device: Device to run on
        verbose: Show progress
        beam_size: Number of beams for beam search
        temperature: Sampling temperature

    Returns:
        Tuple of (predictions_df, evaluation_metrics)
    """
    if verbose:
        print(f"\nRunning evaluation on {split} split ({len(manifest_df)} samples)...")

    # Run inference
    predictions_df = run_inference(
        model=model,
        dataset=prepared_dataset,
        processor=processor,
        normalizer=normalizer,
        device=device,
        verbose=verbose,
        beam_size=beam_size,
        temperature=temperature,
    )

    # Compute metrics
    predictions_df = evaluate_predictions(
        predictions_df=predictions_df,
        manifest_df=manifest_df,
        normalizer=normalizer,
    )

    # Aggregate metrics
    aggregate = compute_aggregate_metrics(predictions_df, split)

    # Compare to baseline
    comparison = compare_to_baseline(aggregate, baseline_metrics, split)

    # Combine metrics
    evaluation_metrics = {
        "split": split,
        "aggregate": aggregate,
        "comparison": comparison,
    }

    if verbose:
        print(f"\n{split.upper()} Results:")
        print(f"  WER: {aggregate['wer']:.2%}")
        print(f"  CER: {aggregate['cer']:.2%}")
        print(f"  Baseline WER: {comparison['baseline_wer']:.2%}")
        print(f"  Improvement: {comparison['relative_improvement_pct']:.1f}%")

    return predictions_df, evaluation_metrics
