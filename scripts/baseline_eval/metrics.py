"""WER/CER metrics computation and aggregation."""

from typing import Any

import jiwer
import pandas as pd  # type: ignore[import-untyped]


def compute_sample_wer(reference: str, hypothesis: str) -> dict[str, Any]:
    """
    Compute WER and error breakdown for a single sample.

    Args:
        reference: Normalized reference text
        hypothesis: Normalized hypothesis text

    Returns:
        Dictionary with wer, insertions, deletions, substitutions
    """
    # Handle edge cases
    if not reference and not hypothesis:
        return {
            "wer": 0.0,
            "word_insertions": 0,
            "word_deletions": 0,
            "word_substitutions": 0,
            "word_hits": 0,
            "word_count_ref": 0,
        }

    if not reference:
        # No reference but has hypothesis: 100% error (all insertions)
        hyp_words = len(hypothesis.split())
        return {
            "wer": 1.0,
            "word_insertions": hyp_words,
            "word_deletions": 0,
            "word_substitutions": 0,
            "word_hits": 0,
            "word_count_ref": 0,
        }

    if not hypothesis:
        # Has reference but no hypothesis: 100% error (all deletions)
        ref_words = len(reference.split())
        return {
            "wer": 1.0,
            "word_insertions": 0,
            "word_deletions": ref_words,
            "word_substitutions": 0,
            "word_hits": 0,
            "word_count_ref": ref_words,
        }

    # Compute WER with jiwer
    output = jiwer.process_words(reference, hypothesis)

    ref_word_count = len(reference.split())

    return {
        "wer": output.wer,
        "word_insertions": output.insertions,
        "word_deletions": output.deletions,
        "word_substitutions": output.substitutions,
        "word_hits": output.hits,
        "word_count_ref": ref_word_count,
    }


def compute_sample_cer(reference: str, hypothesis: str) -> dict[str, Any]:
    """
    Compute CER for a single sample.

    Args:
        reference: Normalized reference text
        hypothesis: Normalized hypothesis text

    Returns:
        Dictionary with cer and char_count_ref
    """
    # Handle edge cases
    if not reference and not hypothesis:
        return {"cer": 0.0, "char_count_ref": 0}

    if not reference:
        return {"cer": 1.0, "char_count_ref": 0}

    if not hypothesis:
        return {"cer": 1.0, "char_count_ref": len(reference)}

    # Compute CER with jiwer
    output = jiwer.process_characters(reference, hypothesis)

    return {
        "cer": output.cer,
        "char_count_ref": len(reference),
    }


def compute_sample_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute WER and CER metrics for each sample in the dataframe.

    Args:
        df: DataFrame with 'reference' and 'hypothesis' columns (normalized)

    Returns:
        DataFrame with added metric columns
    """
    result_df = df.copy()

    # Compute WER metrics
    wer_results = df.apply(
        lambda row: compute_sample_wer(row["reference"], row["hypothesis"]),
        axis=1,
    )
    wer_df = pd.DataFrame(wer_results.tolist())
    result_df = pd.concat([result_df, wer_df], axis=1)

    # Compute CER metrics
    cer_results = df.apply(
        lambda row: compute_sample_cer(row["reference"], row["hypothesis"]),
        axis=1,
    )
    cer_df = pd.DataFrame(cer_results.tolist())
    result_df = pd.concat([result_df, cer_df], axis=1)

    return result_df


def _calculate_wer(subset_df: pd.DataFrame) -> float:
    """Calculate aggregate WER from error counts."""
    total_errors = (
        subset_df["word_substitutions"].sum()
        + subset_df["word_deletions"].sum()
        + subset_df["word_insertions"].sum()
    )
    total_ref_words = subset_df["word_count_ref"].sum()
    return total_errors / total_ref_words if total_ref_words > 0 else 0.0


def _calculate_cer(subset_df: pd.DataFrame) -> float:
    """Calculate weighted aggregate CER."""
    total_chars = subset_df["char_count_ref"].sum()
    if total_chars > 0:
        result: float = (subset_df["cer"] * subset_df["char_count_ref"]).sum() / total_chars
        return result
    return 0.0


def _compute_bin_metrics(split_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Compute metrics grouped by duration bin."""
    by_duration_bin = {}
    for bin_name, bin_df in split_df.groupby("duration_bin"):
        if len(bin_df) == 0:
            continue

        by_duration_bin[str(bin_name)] = {
            "sample_count": len(bin_df),
            "wer": round(_calculate_wer(bin_df), 4),
            "cer": round(_calculate_cer(bin_df), 4),
        }

    return by_duration_bin


def compute_aggregate_metrics(df: pd.DataFrame, splits: list[str]) -> dict[str, Any]:
    """
    Compute aggregate metrics by split and duration bin.

    Args:
        df: DataFrame with computed WER/CER per sample
        splits: List of splits to include

    Returns:
        Dictionary with aggregate metrics per split and duration bin
    """
    metrics: dict[str, Any] = {}

    for split in splits:
        split_df = df[df["split"] == split]

        if len(split_df) == 0:
            continue

        # Compute split-level aggregates
        total_ref_words = split_df["word_count_ref"].sum()
        total_chars = split_df["char_count_ref"].sum()

        metrics[split] = {
            "sample_count": len(split_df),
            "total_duration_sec": round(split_df["duration_sec"].sum(), 2),
            "total_words": int(total_ref_words),
            "total_chars": int(total_chars),
            "wer": round(_calculate_wer(split_df), 4),
            "cer": round(_calculate_cer(split_df), 4),
            "insertions": int(split_df["word_insertions"].sum()),
            "deletions": int(split_df["word_deletions"].sum()),
            "substitutions": int(split_df["word_substitutions"].sum()),
        }

        # Add metrics by duration bin
        metrics[split]["by_duration_bin"] = _compute_bin_metrics(split_df)

    return metrics
