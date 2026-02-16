"""Error-type and duration slicing for fine-tuned model analysis."""

from pathlib import Path

import pandas as pd


def _compute_slice_wer(df_slice: pd.DataFrame) -> float:
    """Compute aggregate WER for a DataFrame slice."""
    total_words = df_slice["reference"].str.split().str.len().sum()
    total_errors = (
        df_slice["word_insertions"].sum()
        + df_slice["word_deletions"].sum()
        + df_slice["word_substitutions"].sum()
    )
    return total_errors / total_words if total_words > 0 else 0.0


def load_baseline_predictions(baseline_predictions_path: str | Path) -> pd.DataFrame:
    """
    Load baseline predictions CSV from S1-M2.

    Args:
        baseline_predictions_path: Path to baseline_predictions.csv

    Returns:
        DataFrame with baseline predictions
    """
    baseline_predictions_path = Path(baseline_predictions_path)

    if not baseline_predictions_path.exists():
        raise FileNotFoundError(f"Baseline predictions not found: {baseline_predictions_path}")

    df = pd.read_csv(baseline_predictions_path)

    # Validate required columns
    required_cols = [
        "pair_sha256",
        "wer",
        "word_insertions",
        "word_deletions",
        "word_substitutions",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Baseline predictions missing columns: {missing}")

    return df


def classify_error_type(row: pd.Series) -> str:
    """
    Classify a sample by its dominant error type.

    Args:
        row: DataFrame row with error columns

    Returns:
        Error type classification string
    """
    wer = row.get("wer", 0)
    insertions = row.get("word_insertions", 0)
    deletions = row.get("word_deletions", 0)
    substitutions = row.get("word_substitutions", 0)

    total_errors = insertions + deletions + substitutions

    # Low error threshold (baseline WER < 10%)
    if wer < 0.1:
        return "low_error"

    # High error threshold (baseline WER > 50%)
    if wer > 0.5:
        return "high_error"

    # No errors - shouldn't happen but handle it
    if total_errors == 0:
        return "low_error"

    # Classify by dominant error type (>50% of errors)
    if deletions / total_errors > 0.5:
        return "deletion_heavy"
    elif substitutions / total_errors > 0.5:
        return "substitution_heavy"
    elif insertions / total_errors > 0.5:
        return "insertion_heavy"
    else:
        return "mixed"


def add_error_type_classification(
    predictions_df: pd.DataFrame,
    baseline_predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add error type classification from baseline predictions.

    Args:
        predictions_df: Fine-tuned model predictions
        baseline_predictions_df: S1-M2 baseline predictions

    Returns:
        predictions_df with added baseline_error_type column
    """
    # Classify baseline errors
    baseline_predictions_df = baseline_predictions_df.copy()
    baseline_predictions_df["baseline_error_type"] = baseline_predictions_df.apply(
        classify_error_type, axis=1
    )

    # Also store baseline WER for comparison
    baseline_subset = baseline_predictions_df[["pair_sha256", "baseline_error_type", "wer"]].rename(
        columns={"wer": "baseline_wer"}
    )

    # Join with predictions
    result_df = predictions_df.merge(
        baseline_subset,
        on="pair_sha256",
        how="left",
    )

    # Fill missing (samples not in baseline)
    result_df["baseline_error_type"] = result_df["baseline_error_type"].fillna("unknown")
    result_df["baseline_wer"] = result_df["baseline_wer"].fillna(0)

    return result_df


def compute_duration_slice_metrics(
    predictions_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute metrics by duration bin.

    Args:
        predictions_df: DataFrame with predictions and duration_bin

    Returns:
        Dictionary mapping duration_bin to metrics
    """
    slice_metrics = {}

    for duration_bin in predictions_df["duration_bin"].unique():
        df_slice = predictions_df[predictions_df["duration_bin"] == duration_bin]

        if len(df_slice) == 0:
            continue

        slice_metrics[str(duration_bin)] = {
            "sample_count": len(df_slice),
            "wer": _compute_slice_wer(df_slice),
            "cer": df_slice["cer"].mean(),
        }

    return slice_metrics


def compute_error_type_slice_metrics(
    predictions_df: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute metrics by baseline error type.

    Args:
        predictions_df: DataFrame with predictions and baseline_error_type

    Returns:
        Dictionary mapping error_type to metrics
    """
    if "baseline_error_type" not in predictions_df.columns:
        return {}

    slice_metrics = {}

    for error_type in predictions_df["baseline_error_type"].unique():
        if pd.isna(error_type) or error_type == "unknown":
            continue

        df_slice = predictions_df[predictions_df["baseline_error_type"] == error_type]

        if len(df_slice) == 0:
            continue

        aggregate_wer = _compute_slice_wer(df_slice)

        # Also compute baseline WER for comparison
        baseline_wer = (
            df_slice["baseline_wer"].mean() if "baseline_wer" in df_slice.columns else 0.0
        )

        slice_metrics[error_type] = {
            "sample_count": len(df_slice),
            "wer": aggregate_wer,
            "cer": df_slice["cer"].mean(),
            "baseline_wer": baseline_wer,
            "improvement": baseline_wer - aggregate_wer,
        }

    return slice_metrics


def compute_all_slice_metrics(
    predictions_df: pd.DataFrame,
    baseline_predictions_path: str | Path | None = None,
) -> dict:
    """
    Compute all slice metrics (duration + error type).

    Args:
        predictions_df: DataFrame with predictions
        baseline_predictions_path: Path to baseline predictions CSV

    Returns:
        Dictionary with all slice metrics
    """
    result = {
        "by_duration_bin": compute_duration_slice_metrics(predictions_df),
    }

    # Add error type slices if baseline predictions available
    if baseline_predictions_path:
        try:
            baseline_df = load_baseline_predictions(baseline_predictions_path)
            predictions_df = add_error_type_classification(predictions_df, baseline_df)
            result["by_error_type"] = compute_error_type_slice_metrics(predictions_df)
        except Exception as e:
            print(f"WARNING: Could not compute error type slices: {e}")
            result["by_error_type"] = {}

    return result
