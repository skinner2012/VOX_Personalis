"""Cleaning rules and duplicate detection for Dataset v1."""

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from .hashing import compute_all_hashes

# Exclusion reason constants (order matters per spec)
REASON_AUDIO_UNREADABLE = "audio_unreadable"
REASON_DURATION_INVALID = "duration_invalid"
REASON_TRANSCRIPT_BLANK = "transcript_blank"
REASON_DUPLICATE_PAIR = "duplicate_audio_transcript"


def compute_hashes_for_dataframe(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add hash columns to DataFrame.

    Args:
        df: DataFrame with audio_path_resolved and transcript_raw columns
        verbose: Show progress bar

    Returns:
        DataFrame with audio_sha256, transcript_sha256, pair_sha256 columns
    """
    audio_hashes = []
    transcript_hashes = []
    pair_hashes = []

    iterator = df.iterrows()
    if verbose:
        iterator = tqdm(iterator, total=len(df), desc="Computing hashes", unit="file")

    for _, row in iterator:
        audio_path = Path(row["audio_path_resolved"])
        transcript = row.get("transcript_raw", "") or ""

        audio_sha, transcript_sha, pair_sha = compute_all_hashes(audio_path, transcript)

        audio_hashes.append(audio_sha)
        transcript_hashes.append(transcript_sha)
        pair_hashes.append(pair_sha)

    result = df.copy()
    result["audio_sha256"] = audio_hashes
    result["transcript_sha256"] = transcript_hashes
    result["pair_sha256"] = pair_hashes

    return result


def apply_cleaning_rules(
    df: pd.DataFrame, verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply cleaning rules and separate included/excluded samples.

    Exclusion rules (in order per spec):
    1. audio_read_ok = False
    2. duration_sec is NULL or <= 0
    3. transcript_is_blank = True
    4. Duplicate audio-transcript pairs (keep first by manifest_row_index)

    Args:
        df: DataFrame with inventory columns + computed hashes
        verbose: Print progress

    Returns:
        (included_df, excluded_df) with excluded_reason column in excluded_df
    """
    # Track exclusions
    excluded_records = []
    remaining_mask = pd.Series([True] * len(df), index=df.index)

    # Rule 1: Audio unreadable
    rule1_mask = df["audio_read_ok"] == False  # noqa: E712
    for idx in df[remaining_mask & rule1_mask].index:
        excluded_records.append(
            {
                "file_name": df.loc[idx, "file_name"],
                "manifest_row_index": df.loc[idx, "manifest_row_index"],
                "excluded_reason": REASON_AUDIO_UNREADABLE,
                "audio_sha256": df.loc[idx, "audio_sha256"],
                "transcript_sha256": df.loc[idx, "transcript_sha256"],
            }
        )
    remaining_mask = remaining_mask & ~rule1_mask

    # Rule 2: Duration invalid (NULL or <= 0)
    rule2_mask = df["duration_sec"].isna() | (df["duration_sec"] <= 0)
    for idx in df[remaining_mask & rule2_mask].index:
        excluded_records.append(
            {
                "file_name": df.loc[idx, "file_name"],
                "manifest_row_index": df.loc[idx, "manifest_row_index"],
                "excluded_reason": REASON_DURATION_INVALID,
                "audio_sha256": df.loc[idx, "audio_sha256"],
                "transcript_sha256": df.loc[idx, "transcript_sha256"],
            }
        )
    remaining_mask = remaining_mask & ~rule2_mask

    # Rule 3: Transcript blank
    rule3_mask = df["transcript_is_blank"] == True  # noqa: E712
    for idx in df[remaining_mask & rule3_mask].index:
        excluded_records.append(
            {
                "file_name": df.loc[idx, "file_name"],
                "manifest_row_index": df.loc[idx, "manifest_row_index"],
                "excluded_reason": REASON_TRANSCRIPT_BLANK,
                "audio_sha256": df.loc[idx, "audio_sha256"],
                "transcript_sha256": df.loc[idx, "transcript_sha256"],
            }
        )
    remaining_mask = remaining_mask & ~rule3_mask

    # Rule 4: Duplicate pair_sha256 (keep first by manifest_row_index)
    remaining_df = df[remaining_mask].copy()
    remaining_df = remaining_df.sort_values("manifest_row_index")

    # Find duplicates (keep='first' marks all but first occurrence as duplicate)
    duplicate_mask = remaining_df.duplicated(subset=["pair_sha256"], keep="first")
    duplicate_indices = remaining_df[duplicate_mask].index

    for idx in duplicate_indices:
        excluded_records.append(
            {
                "file_name": df.loc[idx, "file_name"],
                "manifest_row_index": df.loc[idx, "manifest_row_index"],
                "excluded_reason": REASON_DUPLICATE_PAIR,
                "audio_sha256": df.loc[idx, "audio_sha256"],
                "transcript_sha256": df.loc[idx, "transcript_sha256"],
            }
        )
        remaining_mask[idx] = False

    # Build result DataFrames
    included_df = df[remaining_mask].copy()
    excluded_df = pd.DataFrame(excluded_records)

    if verbose:
        print(
            f"Cleaning: {len(df)} input -> {len(included_df)} included, {len(excluded_df)} excluded"
        )

    return included_df, excluded_df


def detect_duplicate_audio_different_transcript(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag samples with same audio but different transcripts.

    These are kept but flagged with duplicate_audio_flag=True for manual review.

    Args:
        df: DataFrame with audio_sha256 and transcript_sha256

    Returns:
        DataFrame with duplicate_audio_flag column added
    """
    result = df.copy()

    # Group by audio_sha256 and check if there are multiple unique transcript_sha256
    audio_groups = result.groupby("audio_sha256")["transcript_sha256"].nunique()
    audio_with_multiple_transcripts = audio_groups[audio_groups > 1].index

    result["duplicate_audio_flag"] = result["audio_sha256"].isin(audio_with_multiple_transcripts)

    return result
