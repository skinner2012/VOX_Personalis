"""Phase 1: Error decomposition, concentration, and hallucination analysis."""

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from baseline_eval.error_analysis import extract_alignments


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _dominant_error(ins: int, dels: int, subs: int) -> str:
    m = max(ins, dels, subs)
    if m == 0:
        return "none"
    if ins == m:
        return "insertion"
    if dels == m:
        return "deletion"
    return "substitution"


def enrich_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used across all analysis phases."""
    df = df.copy()
    df["word_count_ref"] = df["reference"].apply(_word_count)
    df["hypothesis_word_count"] = df["hypothesis"].apply(_word_count)
    df["dominant_error"] = df.apply(
        lambda r: _dominant_error(
            r["word_insertions"], r["word_deletions"], r["word_substitutions"]
        ),
        axis=1,
    )
    df["hallucination"] = (df["hypothesis_word_count"] > 1.5 * df["word_count_ref"]) | (
        df["word_insertions"] > df["word_count_ref"]
    )
    return df


def compute_global_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Phase 1.1 — Aggregate WER, CER, and error type counts."""
    total_ins = int(df["word_insertions"].sum())
    total_del = int(df["word_deletions"].sum())
    total_sub = int(df["word_substitutions"].sum())
    total_errors = total_ins + total_del + total_sub

    total_ref_words = int(df["word_count_ref"].sum())
    agg_wer = total_errors / total_ref_words if total_ref_words > 0 else 0.0

    ref_chars = df["reference"].apply(lambda t: len(t) if t else 0)
    total_chars = int(ref_chars.sum())
    agg_cer = float((df["cer"] * ref_chars).sum() / total_chars) if total_chars > 0 else 0.0

    return {
        "wer": round(agg_wer, 4),
        "cer": round(agg_cer, 4),
        "total_reference_words": total_ref_words,
        "total_errors": total_errors,
        "insertions": total_ins,
        "deletions": total_del,
        "substitutions": total_sub,
        "insertion_pct": round(total_ins / total_errors, 4) if total_errors else 0.0,
        "deletion_pct": round(total_del / total_errors, 4) if total_errors else 0.0,
        "substitution_pct": round(total_sub / total_errors, 4) if total_errors else 0.0,
    }


def compute_error_concentration(df: pd.DataFrame) -> dict[str, Any]:
    """Phase 1.2 — Error concentration: what share of errors come from worst N% of samples."""
    sample_errors = (
        df["word_insertions"] + df["word_deletions"] + df["word_substitutions"]
    ).to_numpy()
    total_errors = int(sample_errors.sum())

    sorted_errors = np.sort(sample_errors)[::-1]
    cumulative = np.cumsum(sorted_errors)

    n = len(df)
    top10_n = max(1, round(n * 0.1))
    top20_n = max(1, round(n * 0.2))

    top10_share = float(cumulative[top10_n - 1] / total_errors) if total_errors > 0 else 0.0
    top20_share = float(cumulative[top20_n - 1] / total_errors) if total_errors > 0 else 0.0

    return {
        "top_10pct_error_share": round(top10_share, 4),
        "top_20pct_error_share": round(top20_share, 4),
        "zero_wer_sample_count": int((df["wer"] == 0.0).sum()),
        "catastrophic_sample_count": int((df["wer"] > 1.0).sum()),
        "top10_n_samples": top10_n,
        "top20_n_samples": top20_n,
    }


def compute_by_duration_bin(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Phase 1.3 — Per-duration-bin metrics with error type breakdown."""
    result: dict[str, dict[str, Any]] = {}
    for bin_name, bin_df in df.groupby("duration_bin"):
        if bin_df.empty:
            continue

        total_ref = int(bin_df["word_count_ref"].sum())
        total_ins = int(bin_df["word_insertions"].sum())
        total_del = int(bin_df["word_deletions"].sum())
        total_sub = int(bin_df["word_substitutions"].sum())
        total_errors = total_ins + total_del + total_sub
        agg_wer = total_errors / total_ref if total_ref > 0 else 0.0

        ref_chars = bin_df["reference"].apply(lambda t: len(t) if t else 0)
        total_chars = int(ref_chars.sum())
        agg_cer = float((bin_df["cer"] * ref_chars).sum() / total_chars) if total_chars > 0 else 0.0

        result[str(bin_name)] = {
            "sample_count": len(bin_df),
            "total_words": total_ref,
            "total_errors": total_errors,
            "wer": round(agg_wer, 4),
            "cer": round(agg_cer, 4),
            "insertions": total_ins,
            "deletions": total_del,
            "substitutions": total_sub,
            "insertion_pct": round(total_ins / total_errors, 4) if total_errors else 0.0,
            "deletion_pct": round(total_del / total_errors, 4) if total_errors else 0.0,
            "substitution_pct": round(total_sub / total_errors, 4) if total_errors else 0.0,
            "mean_wer": round(float(bin_df["wer"].mean()), 4),
            "median_wer": round(float(bin_df["wer"].median()), 4),
        }
    return result


def compute_hallucination_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Phase 1.4 — Hallucination-heavy samples: count, common tokens, duration bin correlation."""
    hall_df = df[df["hallucination"]]

    inserted_tokens: Counter[str] = Counter()
    for ref, hyp in zip(hall_df["reference"], hall_df["hypothesis"], strict=True):
        for align in extract_alignments(ref, hyp):
            if align["type"] == "insertion":
                inserted_tokens[align["hyp_token"]] += 1

    bin_counts = hall_df["duration_bin"].value_counts().to_dict() if not hall_df.empty else {}

    return {
        "count": int(df["hallucination"].sum()),
        "percentage": round(float(df["hallucination"].mean()), 4),
        "common_inserted_tokens": [
            {"token": t, "count": c} for t, c in inserted_tokens.most_common(20)
        ],
        "by_duration_bin": {str(k): int(v) for k, v in bin_counts.items()},
    }


def get_worst_samples(df: pd.DataFrame, baseline_df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Phase 1.5 — Top-N highest-WER samples with baseline comparison."""
    baseline_wer_map = baseline_df.set_index("pair_sha256")["wer"]
    worst = df.nlargest(top_n, "wer").copy()

    worst_ref_words = worst["reference"].apply(_word_count)
    worst_hyp_words = worst["hypothesis"].apply(_word_count)
    worst_hall = (worst_hyp_words > 1.5 * worst_ref_words) | (
        worst["word_insertions"] > worst_ref_words
    )
    worst_dominant = worst.apply(
        lambda r: _dominant_error(
            r["word_insertions"], r["word_deletions"], r["word_substitutions"]
        ),
        axis=1,
    )
    worst_baseline_wer = worst["pair_sha256"].map(baseline_wer_map)

    return pd.DataFrame(
        {
            "file_name": worst["file_name"].values,
            "pair_sha256": worst["pair_sha256"].values,
            "duration_sec": worst["duration_sec"].values,
            "duration_bin": worst["duration_bin"].values,
            "reference": worst["reference"].values,
            "hypothesis": worst["hypothesis"].values,
            "wer": worst["wer"].values,
            "word_insertions": worst["word_insertions"].values,
            "word_deletions": worst["word_deletions"].values,
            "word_substitutions": worst["word_substitutions"].values,
            "dominant_error": worst_dominant.values,
            "hallucination": worst_hall.values,
            "baseline_wer": worst_baseline_wer.values,
            "wer_improvement": (worst_baseline_wer - worst["wer"]).values,
        }
    )
