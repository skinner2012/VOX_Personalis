"""Phase 4: Audio quality correlation with WER."""

from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats  # type: ignore[import-untyped]

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_POOR_SILENCE_RATIO = 0.4
_POOR_RMS_DB = -40.0


def correlate_audio_quality(
    df: pd.DataFrame, inventory_df: pd.DataFrame, top_n: int = 50
) -> dict[str, Any]:
    """
    Phase 4 — Join val predictions with M0 inventory metrics by file_name and compute:
      - Join match rate
      - Poor-audio overlap in top-N worst samples
      - WER vs silence_ratio_est / rms_db_est correlation coefficients
    """
    joined = df.merge(
        inventory_df[["file_name", "silence_ratio_est", "rms_db_est"]],
        on="file_name",
        how="left",
    )
    matched = joined["silence_ratio_est"].notna().sum()
    match_rate = round(matched / len(df), 4) if len(df) > 0 else 0.0

    warn_low_match = match_rate < 0.8

    # Use only matched rows for correlation
    matched_df = joined.dropna(subset=["silence_ratio_est", "rms_db_est"])

    # Poor-audio overlap in top-N worst
    worst_n = df.nlargest(top_n, "wer")["file_name"]
    worst_joined = joined[joined["file_name"].isin(worst_n)].dropna(
        subset=["silence_ratio_est", "rms_db_est"]
    )
    poor_audio_mask = (worst_joined["silence_ratio_est"] > _POOR_SILENCE_RATIO) | (
        worst_joined["rms_db_est"] < _POOR_RMS_DB
    )
    poor_audio_count = int(poor_audio_mask.sum())
    poor_audio_pct = round(poor_audio_count / top_n, 4) if top_n > 0 else 0.0

    # Correlation coefficients
    wer_vs_silence: dict[str, Any] = {}
    wer_vs_rms: dict[str, Any] = {}

    if len(matched_df) >= 3:
        wer_vals = matched_df["wer"].to_numpy()
        silence_vals = matched_df["silence_ratio_est"].to_numpy()
        rms_vals = matched_df["rms_db_est"].to_numpy()

        if _HAS_SCIPY:
            pr, _ = _scipy_stats.pearsonr(wer_vals, silence_vals)
            sr, _ = _scipy_stats.spearmanr(wer_vals, silence_vals)
            pr_rms, _ = _scipy_stats.pearsonr(wer_vals, rms_vals)
            sr_rms, _ = _scipy_stats.spearmanr(wer_vals, rms_vals)
            wer_vs_silence = {"pearson": round(float(pr), 4), "spearman": round(float(sr), 4)}
            wer_vs_rms = {"pearson": round(float(pr_rms), 4), "spearman": round(float(sr_rms), 4)}
        else:
            pr = float(np.corrcoef(wer_vals, silence_vals)[0, 1])
            pr_rms = float(np.corrcoef(wer_vals, rms_vals)[0, 1])
            wer_vs_silence = {
                "pearson": round(pr, 4),
                "spearman": None,
                "note": "scipy unavailable",
            }
            wer_vs_rms = {
                "pearson": round(pr_rms, 4),
                "spearman": None,
                "note": "scipy unavailable",
            }

    return {
        "join_match_rate": match_rate,
        "low_match_warning": warn_low_match,
        "matched_sample_count": int(matched),
        "poor_audio_in_top50_worst": poor_audio_count,
        "poor_audio_in_top50_worst_pct": poor_audio_pct,
        "wer_vs_silence_ratio_corr": wer_vs_silence,
        "wer_vs_rms_db_corr": wer_vs_rms,
    }
