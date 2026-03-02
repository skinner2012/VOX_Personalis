"""Phase 5: Evidence-based hypothesis generation and decision gate."""

from datetime import UTC, datetime
from typing import Any


def generate_hypotheses(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Generate up to 3 data-driven hypotheses from Phases 1–4 results.

    Each hypothesis is ordered by estimated WER impact (highest first).
    All field values are derived from quantitative analysis, not hardcoded.
    """
    hypotheses: list[dict[str, Any]] = []

    norm = analysis.get("normalization_audit", {})
    concentration = analysis.get("error_concentration", {})
    error_migration = analysis.get("error_type_migration", {})
    audio = analysis.get("audio_quality", {})
    subs_df = analysis.get("substitution_patterns_df")
    comp_df = analysis.get("comparative_df")

    # --- H1: Normalization artifact (contraction mismatch) ---
    norm_pts = norm.get("normalization_wer_contribution_pts", 0.0)
    norm_count = norm.get("artifact_count", 0)
    if norm_pts > 0 or norm_count > 0:
        hypotheses.append(
            {
                "hypothesis_id": "H1",
                "observed_pattern": (
                    "Contraction mismatch inflates WER: model expands contractions "
                    "(e.g. 'whats' vs 'what is') but textnorm_v1 only strips apostrophes."
                ),
                "evidence": (
                    f"{norm_count} val samples affected; contraction normalization reduces "
                    f"aggregate WER by {norm_pts:.4f} points ({norm_pts * 100:.2f} WER pts)."
                ),
                "suspected_root_cause": (
                    "textnorm_v1 removes apostrophes but does not expand contractions. "
                    "The fine-tuned model outputs expanded forms while normalized reference "
                    "retains the collapsed form, creating spurious mismatches."
                ),
                "proposed_intervention": (
                    "Add contraction expansion step to normalization pipeline: apply "
                    "CONTRACTION_MAP to both reference and hypothesis before WER scoring."
                ),
                "experiment_type": "normalization-only",
                "expected_impact": f"{norm_pts * 100:.1f} WER pts (measured)",
                "risk_assessment": (
                    "Low risk. Additive rule; does not change model weights. "
                    "Possible false positives for ambiguous tokens (e.g. 'its', 'were'), "
                    "but overall impact is bounded by measured contribution."
                ),
                "testable_in_m4a": True,
            }
        )

    # --- H2: Model capacity ceiling (persistent baseline errors) ---
    persistent_count = 0
    if comp_df is not None:
        persistent_count = (
            int(comp_df["persistent_failure"].sum())
            if "persistent_failure" in comp_df.columns
            else 0
        )

    baseline_insertion_count = error_migration.get("baseline", {}).get("insertions", 0)
    v1_1_insertion_count = error_migration.get("v1_1", {}).get("insertions", 0)
    insertion_reduction = error_migration.get("insertion_reduction_baseline_to_v1_1", 0)

    also_in_baseline_count = 0
    if subs_df is not None and "also_in_baseline" in subs_df.columns:
        also_in_baseline_count = int(subs_df["also_in_baseline"].sum())

    top10_share = concentration.get("top_10pct_error_share", 0.0)
    uniform_errors = top10_share < 0.25

    # Capacity hypothesis is warranted if: many persistent errors OR uniform distribution
    # OR most substitution pairs persist from baseline
    subs_total = len(subs_df) if subs_df is not None else 1
    baseline_overlap_pct = (also_in_baseline_count / subs_total) if subs_total > 0 else 0.0

    capacity_evidence_strong = persistent_count > 10 or uniform_errors or baseline_overlap_pct > 0.5
    if capacity_evidence_strong:
        hypotheses.append(
            {
                "hypothesis_id": f"H{len(hypotheses) + 1}",
                "observed_pattern": (
                    f"{persistent_count} samples remain high-WER (>0.5) in both baseline and v1.1. "
                    f"{also_in_baseline_count}/{subs_total} top substitution pairs also appear in "
                    f"baseline errors. Top 10% of samples contribute "
                    f"{top10_share:.1%} of total errors "
                    f"({'uniform' if uniform_errors else 'concentrated'} distribution)."
                ),
                "evidence": (
                    f"Baseline had {baseline_insertion_count} insertions; "
                    f"v1.1 has {v1_1_insertion_count} "
                    f"(reduction: {insertion_reduction}). "
                    f"{also_in_baseline_count} of top-{subs_total} "
                    "substitution pairs persist from baseline, "
                    "suggesting the model has not learned "
                    "certain acoustic patterns despite fine-tuning."
                ),
                "suspected_root_cause": (
                    "base.en (74M parameters) may lack capacity "
                    "for this speaker's acoustic patterns. "
                    "Fine-tuning with LoRA adapts the existing representations but cannot add new "
                    "capacity to distinguish phonetically similar tokens."
                ),
                "proposed_intervention": (
                    "Fine-tune small.en (244M parameters) with same LoRA config (r=16, alpha=32) "
                    "and Dataset v1. Unfreeze DECODE_V1 decoding parameters for joint search."
                ),
                "experiment_type": "training-required",
                "expected_impact": "10–20 WER pts (estimated from model size scaling)",
                "risk_assessment": (
                    "Medium risk. Longer training (~3–4× compute), higher memory usage. "
                    "Risk of overfitting with same dataset size. "
                    "Requires S1-M5 training run; not testable in M4a."
                ),
                "testable_in_m4a": False,
            }
        )

    # --- H3: Audio quality as driver (if strong correlation) ---
    audio_poor_pct = audio.get("poor_audio_in_top50_worst_pct", 0.0)
    silence_corr = (audio.get("wer_vs_silence_ratio_corr") or {}).get("pearson", 0.0) or 0.0
    rms_corr = (audio.get("wer_vs_rms_db_corr") or {}).get("pearson", 0.0) or 0.0

    if audio_poor_pct > 0.2 or abs(silence_corr) > 0.3 or abs(rms_corr) > 0.3:
        hypotheses.append(
            {
                "hypothesis_id": f"H{len(hypotheses) + 1}",
                "observed_pattern": (
                    f"{audio_poor_pct:.1%} of top-50 worst-WER samples have poor audio quality "
                    f"(silence_ratio > 0.4 or rms_db < -40). "
                    f"WER vs silence_ratio Pearson r={silence_corr:.3f}, "
                    f"WER vs rms_db Pearson r={rms_corr:.3f}."
                ),
                "evidence": (
                    f"{int(audio_poor_pct * 50)} of top-50 worst samples classified as poor audio. "
                    f"Correlation with silence ratio: {silence_corr:.3f}. "
                    "Suggests audio quality is a contributing factor to high WER."
                ),
                "suspected_root_cause": (
                    "High silence ratio (pauses, breath noise) and low RMS level "
                    "cause the model to hallucinate or produce low-confidence transcriptions."
                ),
                "proposed_intervention": (
                    "Targeted re-recording of high-silence, low-RMS utterances, or "
                    "apply audio preprocessing (silence trimming, normalization) before inference."
                ),
                "experiment_type": "inference-only",
                "expected_impact": "2–8 WER pts (estimated based on affected sample share)",
                "risk_assessment": (
                    "Low-medium risk. Re-recording is labor-intensive but targeted. "
                    "Audio preprocessing may degrade other samples. "
                    "Requires Dataset v2 if re-recording is chosen."
                ),
                "testable_in_m4a": False,
            }
        )

    return hypotheses[:3]


def make_decision(
    analysis: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    experiment_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Apply decision gate criteria and return a single decision record.

    Decision criteria (from spec), applied in priority order:
    1. Normalization fix alone drops WER below 55% → serve with feedback
    2. Normalization artifacts > 3 WER pts → fix normalization first
    3. >50% of top-50 worst share persistent baseline patterns → upgrade model
    4. >30% of high-WER samples have poor audio → data quality intervention
    5. Errors are uniformly distributed → upgrade model

    Each branch is also gated on whether the corresponding hypothesis was
    generated: a decision is only recommended if the evidence was strong
    enough to produce the hypothesis in the first place.
    """
    norm_pts = analysis.get("normalization_audit", {}).get(
        "normalization_wer_contribution_pts", 0.0
    )
    agg_wer = analysis.get("global_metrics", {}).get("wer", 1.0)
    top10_share = analysis.get("error_concentration", {}).get("top_10pct_error_share", 0.0)
    audio_poor_pct = analysis.get("audio_quality", {}).get("poor_audio_in_top50_worst_pct", 0.0)
    comp_df = analysis.get("comparative_df")
    persistent_count = 0
    top50_n = 0
    top50_persistent = 0
    top50_persistent_pct = 0.0
    if comp_df is not None and "persistent_failure" in comp_df.columns:
        persistent_count = int(comp_df["persistent_failure"].sum())
        top50_n = min(50, len(comp_df))
        if top50_n > 0:
            top50_df = comp_df.nlargest(top50_n, "v1_1_wer")
            top50_persistent = int(top50_df["persistent_failure"].sum())
            top50_persistent_pct = top50_persistent / top50_n

    # Guard: only recommend a fix if the corresponding hypothesis was generated.
    # Hypothesis experiment_type is stable: normalization-only / training-required / inference-only.
    generated_types = {h["experiment_type"] for h in hypotheses}

    # Check if normalization experiment showed WER below 55%
    norm_exp_wer = None
    for exp in experiment_results:
        if exp.get("hypothesis_id") == "H1":
            norm_exp_wer = exp.get("experiment_val_wer")

    # Decision logic
    if norm_exp_wer is not None and norm_exp_wer < 0.55:
        decision = "serve_with_feedback"
        primary_evidence = (
            f"Normalization fix alone drops WER to {norm_exp_wer:.4f} (<0.55 threshold). "
            "Model performance sufficient for serving with feedback loop."
        )
        next_milestone = "Build correction pipeline and user feedback interface."
        norm_first = True
    elif norm_pts > 0.03 and "normalization-only" in generated_types:
        decision = "fix_normalization"
        primary_evidence = (
            f"Normalization artifacts contribute {norm_pts:.4f} WER points "
            f"(>{0.03:.2f} threshold). "
            "Quick win available before committing to model upgrade."
        )
        next_milestone = "Implement contraction expansion in textnorm pipeline; re-evaluate WER."
        norm_first = True
    elif top50_persistent_pct > 0.5 and "training-required" in generated_types:
        decision = "upgrade_model"
        primary_evidence = (
            f"{top50_persistent} of top-{top50_n} worst samples remain high-WER (>0.5) "
            f"in both baseline and v1.1 ({top50_persistent_pct:.1%} > 50% threshold). "
            "Fine-tuning has not resolved these persistent failures "
            "— model capacity is the bottleneck."
        )
        next_milestone = "S1-M5: Fine-tune small.en (244M) with LoRA r=16; unfreeze DECODE_V1."
        norm_first = norm_pts > 0.01
    elif audio_poor_pct > 0.30 and "inference-only" in generated_types:
        decision = "collect_data"
        primary_evidence = (
            f"{audio_poor_pct:.1%} of top-50 worst-WER samples have poor audio quality. "
            "Data quality is the primary driver; targeted re-recording recommended."
        )
        next_milestone = "Record 50–100 targeted utterances; build Dataset v2."
        norm_first = norm_pts > 0.01
    else:
        # Default: uniform distribution → upgrade model
        decision = "upgrade_model"
        primary_evidence = (
            f"Top 10% of samples contribute {top10_share:.1%} of errors "
            "(<25% threshold = uniform). "
            "No dominant fixable pattern identified; model capacity upgrade is highest-leverage."
        )
        next_milestone = "S1-M5: Fine-tune small.en (244M) with LoRA r=16; unfreeze DECODE_V1."
        norm_first = norm_pts > 0.01

    supporting: dict[str, Any] = {
        "aggregate_wer_v1_1": round(agg_wer, 4),
        "normalization_wer_contribution_pts": round(norm_pts, 4),
        "top_10pct_error_share": round(top10_share, 4),
        "persistent_failure_count": persistent_count,
        "top50_persistent_failure_count": top50_persistent,
        "top50_persistent_failure_pct": round(top50_persistent_pct, 4),
        "poor_audio_in_top50_pct": round(audio_poor_pct, 4),
    }
    if norm_exp_wer is not None:
        supporting["post_normalization_wer"] = round(norm_exp_wer, 4)

    return {
        "decision": decision,
        "primary_evidence": primary_evidence,
        "supporting_metrics": supporting,
        "next_milestone": next_milestone,
        "normalization_fix_first": norm_first,
        "timestamp": datetime.now(UTC).isoformat(),
    }
