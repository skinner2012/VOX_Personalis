"""Phase 2: Comparative analysis — what each training stage fixed vs what persists."""

from typing import Any

import pandas as pd

from .decomposition import _dominant_error


def build_comparative_df(
    v1_1_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    v1_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Join val predictions from all available models by pair_sha256.

    Returns per-sample comparison across model stages (Phase 2, Deliverable 2a).
    """
    base = v1_1_df[
        [
            "file_name",
            "pair_sha256",
            "duration_sec",
            "reference",
            "wer",
            "word_insertions",
            "word_deletions",
            "word_substitutions",
        ]
    ].rename(
        columns={
            "wer": "v1_1_wer",
            "word_insertions": "v1_1_ins",
            "word_deletions": "v1_1_del",
            "word_substitutions": "v1_1_sub",
        }
    )

    bl = baseline_df[
        ["pair_sha256", "wer", "word_insertions", "word_deletions", "word_substitutions"]
    ].rename(
        columns={
            "wer": "baseline_wer",
            "word_insertions": "bl_ins",
            "word_deletions": "bl_del",
            "word_substitutions": "bl_sub",
        }
    )

    comp = base.merge(bl, on="pair_sha256", how="inner")

    if v1_df is not None:
        v1 = v1_df[
            ["pair_sha256", "wer", "word_insertions", "word_deletions", "word_substitutions"]
        ].rename(
            columns={
                "wer": "v1_wer",
                "word_insertions": "v1_ins",
                "word_deletions": "v1_del",
                "word_substitutions": "v1_sub",
            }
        )
        comp = comp.merge(v1, on="pair_sha256", how="left")
    else:
        comp["v1_wer"] = None
        comp["v1_ins"] = None
        comp["v1_del"] = None
        comp["v1_sub"] = None

    comp["improvement_total"] = comp["baseline_wer"] - comp["v1_1_wer"]
    comp["improvement_m3"] = (comp["baseline_wer"] - comp["v1_wer"]) if v1_df is not None else None
    comp["improvement_m4"] = (comp["v1_wer"] - comp["v1_1_wer"]) if v1_df is not None else None
    comp["persistent_failure"] = (comp["baseline_wer"] > 0.5) & (comp["v1_1_wer"] > 0.5)

    comp["baseline_dominant_error"] = comp.apply(
        lambda r: _dominant_error(int(r["bl_ins"]), int(r["bl_del"]), int(r["bl_sub"])), axis=1
    )
    comp["v1_1_dominant_error"] = comp.apply(
        lambda r: _dominant_error(int(r["v1_1_ins"]), int(r["v1_1_del"]), int(r["v1_1_sub"])),
        axis=1,
    )
    comp["v1_dominant_error"] = comp.apply(  # type: ignore[call-overload]  # pandas-stubs overloads don't accept Optional[str] return
        lambda r: (
            _dominant_error(int(r["v1_ins"]), int(r["v1_del"]), int(r["v1_sub"]))
            if r["v1_ins"] is not None and not pd.isna(r["v1_ins"])
            else None
        ),
        axis=1,
    )

    return comp[
        [
            "file_name",
            "pair_sha256",
            "duration_sec",
            "reference",
            "baseline_wer",
            "v1_wer",
            "v1_1_wer",
            "improvement_total",
            "improvement_m3",
            "improvement_m4",
            "baseline_dominant_error",
            "v1_dominant_error",
            "v1_1_dominant_error",
            "persistent_failure",
        ]
    ]


def compute_improvement_distribution(comp_df: pd.DataFrame, has_v1: bool) -> dict[str, Any]:
    """Phase 2.1 — Per-transition improvement/regression/unchanged counts."""
    transitions: dict[str, Any] = {}

    def _transition_stats(delta: pd.Series, name: str) -> dict[str, Any]:
        valid = delta.dropna()
        improved = valid[valid > 0]
        regressed = valid[valid < 0]
        unchanged = valid[valid == 0]
        return {
            "improved_count": int(len(improved)),
            "improved_mean": round(float(improved.mean()), 4) if not improved.empty else 0.0,
            "regressed_count": int(len(regressed)),
            "regressed_mean": round(float(regressed.mean()), 4) if not regressed.empty else 0.0,
            "unchanged_count": int(len(unchanged)),
        }

    transitions["baseline_to_v1_1"] = _transition_stats(comp_df["improvement_total"], "total")
    if has_v1:
        transitions["baseline_to_v1"] = _transition_stats(comp_df["improvement_m3"], "m3")
        transitions["v1_to_v1_1"] = _transition_stats(comp_df["improvement_m4"], "m4")

    return transitions


def compute_error_type_migration(
    baseline_df: pd.DataFrame,
    v1_1_df: pd.DataFrame,
    v1_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Phase 2.2 — Track how error types shifted across model stages."""

    def _totals(df: pd.DataFrame) -> dict[str, int]:
        return {
            "insertions": int(df["word_insertions"].sum()),
            "deletions": int(df["word_deletions"].sum()),
            "substitutions": int(df["word_substitutions"].sum()),
        }

    result: dict[str, Any] = {
        "baseline": _totals(baseline_df),
        "v1_1": _totals(v1_1_df),
    }
    if v1_df is not None:
        result["v1"] = _totals(v1_df)

    result["insertion_reduction_baseline_to_v1_1"] = (
        result["baseline"]["insertions"] - result["v1_1"]["insertions"]
    )
    if v1_df is not None:
        result["insertion_reduction_baseline_to_v1"] = (
            result["baseline"]["insertions"] - result["v1"]["insertions"]
        )
        result["insertion_reduction_v1_to_v1_1"] = (
            result["v1"]["insertions"] - result["v1_1"]["insertions"]
        )

    return result


def find_persistent_errors(comp_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2.3 — Samples with WER > 0.5 in both baseline and v1.1."""
    return comp_df[comp_df["persistent_failure"]][
        [
            "file_name",
            "pair_sha256",
            "reference",
            "baseline_wer",
            "v1_wer",
            "v1_1_wer",
            "baseline_dominant_error",
            "v1_1_dominant_error",
        ]
    ].copy()
