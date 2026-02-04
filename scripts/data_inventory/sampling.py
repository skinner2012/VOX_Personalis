"""Stratified sampling by duration"""

import numpy as np
import pandas as pd  # type: ignore[import-untyped]


def stratified_sample_by_duration(
    df: pd.DataFrame, sample_n: int, seed: int, stratify_bins: list[float] | None = None
) -> pd.DataFrame:
    """
    Perform stratified sampling by duration bins.

    Allocation strategy:
    - 0-1s: 10% of samples
    - 1-3s: 20% of samples
    - 3-10s: 40% of samples
    - 10-30s: 20% of samples
    - >30s: 10% of samples

    If a bin has fewer files than allocated samples, all files from that bin
    are included and remaining samples are distributed proportionally to other bins.

    Args:
        df: DataFrame with 'duration_sec', 'audio_read_ok', and 'file_name' columns
        sample_n: Total number of samples to draw
        seed: Random seed for reproducibility
        stratify_bins: Custom bin edges (default: [1, 3, 10, 30])

    Returns:
        Sampled DataFrame, sorted by file_name for determinism.
        Returns empty DataFrame if no valid data or on error.

    Note:
        If sample_n >= available valid files, returns all files without stratification.
    """
    try:
        if stratify_bins is None:
            stratify_bins = [1, 3, 10, 30]

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Filter to only readable audio files with valid duration
        df_valid = df[df["audio_read_ok"]].copy()
        df_valid = df_valid[df_valid["duration_sec"].notna()]

        if len(df_valid) == 0:
            return pd.DataFrame()

        # If sample_n is larger than available data, return all (stratification not needed)
        if sample_n >= len(df_valid):
            return df_valid.sort_values("file_name")

        # Create duration bins
        bin_edges = [0] + stratify_bins + [float("inf")]
        df_valid["_duration_bin"] = pd.cut(
            df_valid["duration_sec"], bins=bin_edges, include_lowest=True
        )

        # Define allocation percentages
        allocations = {
            f"({bin_edges[i]}, {bin_edges[i + 1]}]": pct
            for i, pct in enumerate([0.10, 0.20, 0.40, 0.20, 0.10])
        }

        # Calculate target samples per bin
        target_samples = {}
        for bin_label, pct in allocations.items():
            target_samples[bin_label] = int(sample_n * pct)

        # First pass: sample from bins that have enough data
        sampled_dfs = []
        remaining_to_allocate = 0
        bin_index = 0

        for bin_label, n_target in target_samples.items():
            bin_df = df_valid[df_valid["_duration_bin"].astype(str) == bin_label]
            bin_size = len(bin_df)

            if bin_size >= n_target:
                # Sufficient data in this bin
                sampled = bin_df.sample(n=n_target, random_state=seed + bin_index)
                sampled_dfs.append(sampled)
            else:
                # Insufficient data: take all and track deficit
                sampled_dfs.append(bin_df)
                remaining_to_allocate += n_target - bin_size

            bin_index += 1

        # Second pass: redistribute remaining samples to bins with extra capacity
        if remaining_to_allocate > 0:
            # Find bins with surplus capacity
            bins_with_surplus = []
            for bin_label, n_target in target_samples.items():
                bin_df = df_valid[df_valid["_duration_bin"].astype(str) == bin_label]
                bin_size = len(bin_df)
                if bin_size > n_target:
                    surplus = bin_size - n_target
                    bins_with_surplus.append((bin_label, surplus, bin_df))

            # Redistribute proportionally
            if bins_with_surplus:
                total_surplus = sum(surplus for _, surplus, _ in bins_with_surplus)
                surplus_index = 0

                for _bin_label, surplus, bin_df in bins_with_surplus:
                    # Calculate additional samples for this bin
                    extra_samples = int(remaining_to_allocate * (surplus / total_surplus))

                    if extra_samples > 0 and extra_samples <= surplus:
                        # Remove already sampled rows from this bin
                        already_sampled = pd.concat(sampled_dfs)
                        bin_unsampled = bin_df[~bin_df.index.isin(already_sampled.index)]

                        if len(bin_unsampled) >= extra_samples:
                            extra = bin_unsampled.sample(
                                n=extra_samples, random_state=seed + 100 + surplus_index
                            )
                            sampled_dfs.append(extra)

                    surplus_index += 1

        # Combine all sampled data
        result = pd.concat(sampled_dfs).drop_duplicates()

        # If we still don't have enough, sample more randomly
        if len(result) < sample_n:
            unsampled = df_valid[~df_valid.index.isin(result.index)]
            needed = sample_n - len(result)
            if len(unsampled) >= needed:
                extra = unsampled.sample(n=needed, random_state=seed + 200)
                result = pd.concat([result, extra])

        # Limit to exact sample_n (in case we oversampled)
        if len(result) > sample_n:
            result = result.sample(n=sample_n, random_state=seed + 300)

        # Sort by file_name for determinism
        result = result.drop(columns=["_duration_bin"]).sort_values("file_name")

        return result

    except Exception:
        # Return empty DataFrame on any error
        return pd.DataFrame()
