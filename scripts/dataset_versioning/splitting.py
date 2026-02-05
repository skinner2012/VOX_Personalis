"""Duration-stratified train/val/test split assignment for Dataset v1."""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

# Default duration bin edges (in seconds) per spec
# Bins: (0, 1], (1, 3], (3, 10], (10, 30], (30, inf]
DEFAULT_DURATION_BIN_EDGES = [0, 1, 3, 10, 30, float("inf")]


def assign_duration_bins(df: pd.DataFrame, bin_edges: list[float] | None = None) -> pd.DataFrame:
    """
    Add duration_bin column based on duration_sec.

    Args:
        df: DataFrame with duration_sec column
        bin_edges: Custom bin edges (default: [0, 1, 3, 10, 30, inf])

    Returns:
        DataFrame with duration_bin column added
    """
    if bin_edges is None:
        bin_edges = DEFAULT_DURATION_BIN_EDGES

    # Generate labels from edges
    labels = []
    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if right == float("inf"):
            labels.append(f"({left}, inf]")
        else:
            labels.append(f"({left}, {int(right)}]")

    result = df.copy()
    result["duration_bin"] = pd.cut(
        result["duration_sec"],
        bins=bin_edges,
        labels=labels,
        right=True,  # (left, right] intervals
    )

    return result


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign train/val/test splits stratified by duration bin.

    Within each bin:
    1. Sort by pair_sha256 (deterministic ordering)
    2. Assign based on sorted index position

    Args:
        df: DataFrame with duration_bin and pair_sha256 columns
        train_ratio: Train split proportion (default: 0.8)
        val_ratio: Validation split proportion (default: 0.1)
        test_ratio: Test split proportion (default: 0.1)
        seed: Random seed (documented for reproducibility, not used in sorting)

    Returns:
        DataFrame with split column added ('train', 'val', 'test')
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    result = df.copy()
    result["split"] = None

    # Process each duration bin separately
    for bin_label in result["duration_bin"].unique():
        if pd.isna(bin_label):
            continue

        # Get samples in this bin
        bin_mask = result["duration_bin"] == bin_label
        bin_df = result[bin_mask].copy()

        if len(bin_df) == 0:
            continue

        # Sort by pair_sha256 for deterministic ordering
        bin_df = bin_df.sort_values("pair_sha256")
        bin_indices = bin_df.index.tolist()

        n = len(bin_indices)
        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        # n_test = n - n_train - n_val  # Remainder goes to test

        # Assign splits based on position in sorted order
        for i, idx in enumerate(bin_indices):
            if i < n_train:
                result.loc[idx, "split"] = "train"
            elif i < n_train + n_val:
                result.loc[idx, "split"] = "val"
            else:
                result.loc[idx, "split"] = "test"

    return result


def get_split_statistics(df: pd.DataFrame) -> dict:
    """
    Compute statistics for each split.

    Args:
        df: DataFrame with split and duration_sec columns

    Returns:
        Dictionary with counts and durations per split
    """
    stats: dict = {
        "split_counts": {},
        "split_durations_sec": {},
        "split_durations_hours": {},
        "split_duration_distributions": {},
    }

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        count = len(split_df)
        duration_sec = split_df["duration_sec"].sum()

        stats["split_counts"][split] = count
        stats["split_durations_sec"][split] = float(duration_sec)
        stats["split_durations_hours"][split] = float(duration_sec / 3600)

        # Duration distribution within this split
        if "duration_bin" in split_df.columns:
            bin_counts = split_df["duration_bin"].value_counts().to_dict()
            # Convert keys to strings for JSON serialization
            stats["split_duration_distributions"][split] = {
                str(k): int(v) for k, v in bin_counts.items()
            }

    return stats
