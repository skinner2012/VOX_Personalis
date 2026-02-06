"""Temporal clustering detection for Dataset v1."""

import pandas as pd  # type: ignore[import-untyped]

# Default session gap threshold (1 minute in milliseconds)
DEFAULT_SESSION_GAP_MS = 60000

# Minimum timestamp coverage to enable temporal check
MIN_TIMESTAMP_COVERAGE = 0.5  # 50%


def detect_temporal_clusters(
    df: pd.DataFrame, gap_ms: int = DEFAULT_SESSION_GAP_MS
) -> pd.DataFrame | None:
    """
    Detect recording session clusters based on timestamp gaps.

    A new session starts when the gap between consecutive recordings
    exceeds gap_ms milliseconds.

    Args:
        df: DataFrame with timestamp_ms column
        gap_ms: Maximum gap (in ms) within a session (default: 60000 = 1 minute)

    Returns:
        DataFrame with session_cluster_id column, or None if <50% have timestamps
    """
    if "timestamp_ms" not in df.columns:
        return None

    # Check timestamp coverage
    non_null_count = df["timestamp_ms"].notna().sum()
    coverage = non_null_count / len(df)

    if coverage < MIN_TIMESTAMP_COVERAGE:
        return None

    result = df.copy()

    # Sort by timestamp
    result = result.sort_values("timestamp_ms")

    # Compute gaps between consecutive samples
    timestamp_diff = result["timestamp_ms"].diff()

    # Assign cluster IDs: increment when gap exceeds threshold (vectorized)
    # Create boolean mask: True when new session starts (first row or gap > threshold)
    new_session = timestamp_diff.isna() | (timestamp_diff > gap_ms)
    # Cumulative sum gives each session a unique ID starting from 1
    result["session_cluster_id"] = new_session.cumsum()

    # Restore original order
    result = result.sort_index()

    return result


def find_clusters_crossing_splits(df: pd.DataFrame) -> int:
    """
    Count session clusters that span train/test boundary.

    A cluster "crosses" if it contains samples in both train and test splits.

    Args:
        df: DataFrame with session_cluster_id and split columns

    Returns:
        Count of clusters containing both train and test samples
    """
    if "session_cluster_id" not in df.columns:
        return 0

    crossing_count = 0

    for cluster_id in df["session_cluster_id"].unique():
        cluster_df = df[df["session_cluster_id"] == cluster_id]
        splits_in_cluster = set(cluster_df["split"].unique())

        # Check if both train and test are present
        if "train" in splits_in_cluster and "test" in splits_in_cluster:
            crossing_count += 1

    return crossing_count


def temporal_leakage_report(df: pd.DataFrame, verbose: bool = False) -> dict:
    """
    Generate temporal clustering analysis report.

    Args:
        df: DataFrame with timestamp_ms and split columns
        verbose: Print progress messages

    Returns:
        Dictionary with temporal analysis results
    """
    report = {
        "temporal_check_status": "skipped_insufficient_timestamps",
        "temporal_clusters_crossing_splits": None,
        "total_clusters": None,
        "timestamp_coverage_pct": 0.0,
    }

    if "timestamp_ms" not in df.columns:
        if verbose:
            print("Temporal check skipped: no timestamp_ms column")
        return report

    # Calculate coverage
    non_null_count = df["timestamp_ms"].notna().sum()
    coverage = non_null_count / len(df) if len(df) > 0 else 0
    report["timestamp_coverage_pct"] = round(coverage * 100, 2)

    if coverage < MIN_TIMESTAMP_COVERAGE:
        if verbose:
            print(
                f"Temporal check skipped: only {coverage * 100:.1f}% of samples have timestamps "
                f"(minimum: {MIN_TIMESTAMP_COVERAGE * 100:.0f}%)"
            )
        return report

    # Perform temporal clustering
    clustered_df = detect_temporal_clusters(df)

    if clustered_df is None:
        return report

    # Count clusters and crossings
    total_clusters = clustered_df["session_cluster_id"].nunique()
    crossing_count = find_clusters_crossing_splits(clustered_df)

    report["temporal_check_status"] = "completed"
    report["total_clusters"] = total_clusters
    report["temporal_clusters_crossing_splits"] = crossing_count

    if verbose:
        print(f"Temporal clustering: {total_clusters} sessions detected")
        if crossing_count > 0:
            print(f"WARNING: {crossing_count} sessions cross train/test boundary")
        else:
            print("No temporal leakage detected")

    return report
