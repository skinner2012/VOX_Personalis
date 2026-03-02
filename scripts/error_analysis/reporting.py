"""Write all S1-M4a output artifacts to disk."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# 1. error_distribution_report.json
# ---------------------------------------------------------------------------


def write_error_distribution_report(out_dir: Path, analysis: dict[str, Any]) -> None:
    report = {
        "model_version": "v1.1",
        "val_sample_count": analysis["val_sample_count"],
        "aggregate": {
            "wer": analysis["global_metrics"]["wer"],
            "cer": analysis["global_metrics"]["cer"],
            "total_reference_words": analysis["global_metrics"]["total_reference_words"],
            "total_errors": analysis["global_metrics"]["total_errors"],
            "insertions": analysis["global_metrics"]["insertions"],
            "deletions": analysis["global_metrics"]["deletions"],
            "substitutions": analysis["global_metrics"]["substitutions"],
        },
        "error_concentration": {
            "top_10pct_error_share": analysis["error_concentration"]["top_10pct_error_share"],
            "top_20pct_error_share": analysis["error_concentration"]["top_20pct_error_share"],
            "zero_wer_sample_count": analysis["error_concentration"]["zero_wer_sample_count"],
            "catastrophic_sample_count": analysis["error_concentration"][
                "catastrophic_sample_count"
            ],
        },
        "by_duration_bin": analysis["by_duration_bin"],
        "hallucination": {
            "count": analysis["hallucination"]["count"],
            "percentage": analysis["hallucination"]["percentage"],
            "common_inserted_tokens": analysis["hallucination"]["common_inserted_tokens"],
        },
        "normalization_audit": {
            "artifact_count": analysis["normalization_audit"]["artifact_count"],
            "estimated_wer_contribution_pts": analysis["normalization_audit"][
                "normalization_wer_contribution_pts"
            ],
            "contraction_pairs": analysis["normalization_audit"]["contraction_pairs"],
        },
        "audio_quality_overlap": {
            "join_match_rate": analysis["audio_quality"]["join_match_rate"],
            "poor_audio_in_top50_worst": analysis["audio_quality"]["poor_audio_in_top50_worst"],
            "poor_audio_in_top50_worst_pct": analysis["audio_quality"][
                "poor_audio_in_top50_worst_pct"
            ],
            "wer_vs_silence_ratio_corr": analysis["audio_quality"].get("wer_vs_silence_ratio_corr"),
            "wer_vs_rms_db_corr": analysis["audio_quality"].get("wer_vs_rms_db_corr"),
        },
    }
    _write_json(out_dir / "error_distribution_report.json", report)


# ---------------------------------------------------------------------------
# 2. worst_samples.csv
# ---------------------------------------------------------------------------


def write_worst_samples(out_dir: Path, worst_df: pd.DataFrame) -> None:
    _write_csv(out_dir / "worst_samples.csv", worst_df)


# ---------------------------------------------------------------------------
# 3. substitution_patterns.csv
# ---------------------------------------------------------------------------


def write_substitution_patterns(out_dir: Path, subs_df: pd.DataFrame) -> None:
    _write_csv(out_dir / "substitution_patterns.csv", subs_df)


# ---------------------------------------------------------------------------
# 4. comparative_analysis.csv
# ---------------------------------------------------------------------------


def write_comparative_analysis(out_dir: Path, comp_df: pd.DataFrame) -> None:
    _write_csv(out_dir / "comparative_analysis.csv", comp_df)


# ---------------------------------------------------------------------------
# 5. hypotheses.json
# ---------------------------------------------------------------------------


def write_hypotheses(out_dir: Path, hypotheses: list[dict[str, Any]]) -> None:
    _write_json(out_dir / "hypotheses.json", hypotheses)


# ---------------------------------------------------------------------------
# 6. m4a_experiment_log.csv
# ---------------------------------------------------------------------------


def write_experiment_log(out_dir: Path, experiments: list[dict[str, Any]]) -> None:
    _write_csv(out_dir / "m4a_experiment_log.csv", pd.DataFrame(experiments))


# ---------------------------------------------------------------------------
# 7. error_analysis_report.md
# ---------------------------------------------------------------------------


def write_markdown_report(out_dir: Path, analysis: dict[str, Any]) -> None:
    agg = analysis["global_metrics"]
    conc = analysis["error_concentration"]
    hall = analysis["hallucination"]
    norm = analysis["normalization_audit"]
    audio = analysis["audio_quality"]
    by_bin = analysis["by_duration_bin"]
    hypotheses = analysis.get("hypotheses", [])
    experiments = analysis.get("experiments", [])
    decision = analysis.get("decision", {})
    imp_dist = analysis.get("improvement_distribution", {})
    migration = analysis.get("error_type_migration", {})
    short_utt = analysis.get("short_utterances", {})
    long_utt = analysis.get("long_utterances", {})
    persistent_df = analysis.get("persistent_errors_df")
    subs_df = analysis.get("substitution_patterns_df")
    domain_failures_df = analysis.get("domain_failures_df")

    lines: list[str] = []
    a = lines.append

    # --- 1. Executive Summary ---
    a("# S1-M4a Error Analysis Report")
    a("")
    a("## 1. Executive Summary")
    a("")
    a(f"**Model v1.1 val WER:** {agg['wer']:.4f} ({agg['wer'] * 100:.1f}%)  ")
    a(f"**Model v1.1 val CER:** {agg['cer']:.4f} ({agg['cer'] * 100:.1f}%)  ")
    a(f"**Val samples:** {analysis['val_sample_count']}  ")
    a(f"**Total reference words:** {agg['total_reference_words']}  ")
    a(
        f"**Total errors:** {agg['total_errors']} "
        f"(I:{agg['insertions']} D:{agg['deletions']} S:{agg['substitutions']})"
    )
    a("")
    dec_str = decision.get("decision", "N/A")
    a(f"**Decision:** `{dec_str}`")
    a(f"> {decision.get('primary_evidence', '')}")
    a("")

    # --- 2. Error Distribution ---
    a("## 2. Error Distribution")
    a("")
    a("### 2.1 Global Metrics")
    a("")
    a("| Metric | Value |")
    a("|--------|-------|")
    a(f"| WER | {agg['wer']:.4f} |")
    a(f"| CER | {agg['cer']:.4f} |")
    a(f"| Insertions | {agg['insertions']} ({agg['insertion_pct'] * 100:.1f}%) |")
    a(f"| Deletions | {agg['deletions']} ({agg['deletion_pct'] * 100:.1f}%) |")
    a(f"| Substitutions | {agg['substitutions']} ({agg['substitution_pct'] * 100:.1f}%) |")
    a("")
    a("### 2.2 Error Concentration")
    a("")
    a(
        f"- Top 10% worst samples ({conc['top10_n_samples']} samples): "
        f"**{conc['top_10pct_error_share'] * 100:.1f}%** of total errors"
    )
    a(
        f"- Top 20% worst samples ({conc['top20_n_samples']} samples): "
        f"**{conc['top_20pct_error_share'] * 100:.1f}%** of total errors"
    )
    a(f"- Zero-WER samples: {conc['zero_wer_sample_count']}")
    a(f"- Catastrophic (WER > 1.0): {conc['catastrophic_sample_count']}")
    a("")
    if conc["top_10pct_error_share"] > 0.40:
        a("_Interpretation: Errors are **concentrated** — targeted intervention may help._")
    elif conc["top_10pct_error_share"] < 0.25:
        a(
            "_Interpretation: Errors are **uniformly distributed** "
            "— model capacity or systematic issue._"
        )
    else:
        a("_Interpretation: Errors are **moderately concentrated**._")
    a("")
    a("### 2.3 By Duration Bin")
    a("")
    a("| Bin | Samples | WER | CER | Ins% | Del% | Sub% | Mean WER | Median WER |")
    a("|-----|---------|-----|-----|------|------|------|----------|------------|")
    for bin_name, bm in sorted(by_bin.items()):
        a(
            f"| {bin_name} | {bm['sample_count']} | {bm['wer']:.4f} | {bm['cer']:.4f} | "
            f"{bm['insertion_pct'] * 100:.1f}% | {bm['deletion_pct'] * 100:.1f}% | "
            f"{bm['substitution_pct'] * 100:.1f}% | {bm['mean_wer']:.4f} | {bm['median_wer']:.4f} |"
        )
    a("")
    a("### 2.4 Hallucination Analysis")
    a("")
    a(f"Hallucination-heavy samples: **{hall['count']}** ({hall['percentage'] * 100:.1f}%)")
    if hall["common_inserted_tokens"]:
        top5 = hall["common_inserted_tokens"][:5]
        token_str = ", ".join(f"`{t['token']}` ({t['count']})" for t in top5)
        a(f"Top inserted tokens: {token_str}")
    if hall["by_duration_bin"]:
        a(f"By duration bin: {hall['by_duration_bin']}")
    a("")

    # --- 3. Comparative Analysis ---
    a("## 3. Comparative Analysis (baseline → v1 → v1.1)")
    a("")
    bl_ins = migration.get("baseline", {}).get("insertions", "N/A")
    v1_ins = migration.get("v1", {}).get("insertions", "N/A")
    v1_1_ins = migration.get("v1_1", {}).get("insertions", "N/A")
    a(f"**Insertion reduction:** baseline={bl_ins} → v1={v1_ins} → v1.1={v1_1_ins}")
    a("")
    a("### Improvement Distribution (baseline → v1.1)")
    total_trans = imp_dist.get("baseline_to_v1_1", {})
    a(
        f"- Improved: {total_trans.get('improved_count', 0)} samples "
        f"(mean +{total_trans.get('improved_mean', 0):.4f} WER)"
    )
    a(
        f"- Regressed: {total_trans.get('regressed_count', 0)} samples "
        f"(mean {total_trans.get('regressed_mean', 0):.4f} WER)"
    )
    a(f"- Unchanged: {total_trans.get('unchanged_count', 0)} samples")
    a("")
    if persistent_df is not None:
        a(
            f"**Persistent failures** (WER > 0.5 in both baseline and v1.1): "
            f"{len(persistent_df)} samples"
        )
    a("")

    # --- 4. Systematic Patterns ---
    a("## 4. Systematic Patterns")
    a("")
    a("### 4.1 Substitution Patterns")
    a("")
    if subs_df is not None and not subs_df.empty:
        a(f"Top {min(10, len(subs_df))} substitution pairs (of {len(subs_df)} total):")
        a("")
        a("| Reference | Hypothesis | Count | In Baseline? |")
        a("|-----------|-----------|-------|--------------|")
        for _, row in subs_df.head(10).iterrows():
            in_bl = "Yes" if row["also_in_baseline"] else "No"
            ref_tok = row["reference_token"]
            hyp_tok = row["hypothesis_token"]
            a(f"| `{ref_tok}` | `{hyp_tok}` | {row['count']} | {in_bl} |")
    else:
        a("No substitution patterns extracted.")
    a("")
    a("### 4.2 Normalization Artifacts")
    a("")
    a(f"- Samples affected: **{norm['artifact_count']}**")
    a(f"- WER contribution: **{norm['normalization_wer_contribution_pts'] * 100:.2f} WER pts**")
    if norm["contraction_pairs"]:
        a("")
        a("Top detected contraction pairs:")
        a("| Reference Token | Hypothesis Token | Count |")
        a("|----------------|-----------------|-------|")
        for pair in norm["contraction_pairs"][:10]:
            a(f"| `{pair['reference_token']}` | `{pair['hypothesis_token']}` | {pair['count']} |")
    a("")
    a("### 4.3 Short Utterance Instability")
    a("")
    a(f"Short utterances (≤3 ref words): **{short_utt.get('sample_count', 0)}** samples  ")
    a(
        f"Mean WER: {short_utt.get('mean_wer', 0):.4f} vs longer utterances: "
        f"{short_utt.get('mean_wer_longer_utterances', 0):.4f}  "
    )
    a(f"Error share of total: {short_utt.get('error_share', 0) * 100:.1f}%")
    a("")
    a("### 4.4 Long Utterance Analysis")
    a("")
    a(f"Long utterances (≥10 ref words): **{long_utt.get('sample_count', 0)}** samples  ")
    a(
        f"Mean WER: {long_utt.get('mean_wer', 0):.4f}, "
        f"Median WER: {long_utt.get('median_wer', 0):.4f}  "
    )
    a(f"Medium utterances mean WER: {long_utt.get('medium_mean_wer', 0):.4f}  ")
    systematically = long_utt.get("systematically_higher", False)
    a(f"Systematically higher WER than medium: **{systematically}**")
    a("")
    a("### 4.5 Domain-Specific Token Failures")
    a("")
    if domain_failures_df is not None and not domain_failures_df.empty:
        a(f"Tokens with error rate > 50%: **{len(domain_failures_df)}**")
        a("")
        a("| Token | Error Rate | Occurrences | Category |")
        a("|-------|-----------|-------------|----------|")
        for _, row in domain_failures_df.head(20).iterrows():
            a(
                f"| `{row['token']}` | {row['error_rate']:.1%} "
                f"| {row['total_occurrences']} | {row['category']} |"
            )
    else:
        a("No tokens with error rate > 50% detected (uniform errors across vocabulary).")
    a("")

    # --- 5. Audio Quality Correlation ---
    a("## 5. Audio Quality Correlation")
    a("")
    a(
        f"Join match rate: {audio['join_match_rate'] * 100:.1f}% "
        f"({audio['matched_sample_count']} samples matched)"
    )
    if audio.get("low_match_warning"):
        a("WARNING: match rate < 80% — results may be incomplete.")
    a(
        f"Poor-audio samples in top-50 worst WER: **{audio['poor_audio_in_top50_worst']}** "
        f"({audio['poor_audio_in_top50_worst_pct'] * 100:.1f}%)"
    )
    sr_corr = audio.get("wer_vs_silence_ratio_corr") or {}
    rms_corr = audio.get("wer_vs_rms_db_corr") or {}
    a(
        f"WER vs silence_ratio: Pearson={sr_corr.get('pearson', 'N/A')}, "
        f"Spearman={sr_corr.get('spearman', 'N/A')}"
    )
    a(
        f"WER vs rms_db: Pearson={rms_corr.get('pearson', 'N/A')}, "
        f"Spearman={rms_corr.get('spearman', 'N/A')}"
    )
    a("")

    # --- 6. Hypotheses ---
    a("## 6. Hypotheses")
    a("")
    for h in hypotheses:
        a(f"### {h['hypothesis_id']}: {h['observed_pattern'][:60]}...")
        a("")
        a(f"**Evidence:** {h['evidence']}")
        a(f"**Root cause:** {h['suspected_root_cause']}")
        a(f"**Intervention:** {h['proposed_intervention']}")
        a(f"**Experiment type:** {h['experiment_type']}  ")
        a(f"**Expected impact:** {h['expected_impact']}  ")
        a(f"**Risk:** {h['risk_assessment']}  ")
        a(f"**Testable in M4a:** {h['testable_in_m4a']}")
        a("")

    # --- 7. Follow-Up Experiment Results ---
    a("## 7. Follow-Up Experiment Results")
    a("")
    if experiments:
        a("| ID | Hypothesis | Type | v1.1 WER | Exp WER | Delta | Decision |")
        a("|----|-----------|------|----------|---------|-------|----------|")
        for exp in experiments:
            a(
                f"| {exp['experiment_id']} | {exp['hypothesis_id']} | "
                f"{exp['experiment_type']} | {exp['model_v1_1_val_wer']:.4f} | "
                f"{exp['experiment_val_wer']:.4f} | {exp['val_wer_delta']:+.4f} | "
                f"{exp['decision']} |"
            )
        a("")
        for exp in experiments:
            a(f"**{exp['experiment_id']}** — {exp['intervention']}  ")
            a(f"Notes: {exp['notes']}")
            a("")
    else:
        a("No experiments recorded.")
        a("")

    # --- 8. Decision Gate Outcome ---
    a("## 8. Decision Gate Outcome")
    a("")
    a(f"**Decision:** `{decision.get('decision', 'N/A')}`")
    a(f"**Primary evidence:** {decision.get('primary_evidence', '')}")
    a(f"**Normalization fix first:** {decision.get('normalization_fix_first', False)}")
    a(f"**Next milestone:** {decision.get('next_milestone', '')}")
    a("")

    # --- 9. Recommendations ---
    a("## 9. Recommendations")
    a("")
    dec_val = decision.get("decision", "")
    if dec_val == "fix_normalization":
        a("1. **Immediate:** Add contraction expansion to textnorm pipeline and re-score.")
        a("2. **If WER drops below 55%:** Consider serving with feedback loop.")
        a("3. **If WER remains above 55% after normalization fix:** Proceed to S1-M5 (small.en).")
    elif dec_val == "upgrade_model":
        norm_first = decision.get("normalization_fix_first", False)
        if norm_first:
            a("1. **Quick win:** Apply normalization fix first to recover estimated WER points.")
        a(
            f"{'2' if norm_first else '1'}. **S1-M5:** "
            "Fine-tune small.en (244M) with LoRA r=16, alpha=32."
        )
        a(
            f"{'3' if norm_first else '2'}. **Unfreeze DECODE_V1** "
            "decoding parameters during S1-M5 training."
        )
    elif dec_val == "serve_with_feedback":
        a("1. **Deploy** current model with correction interface.")
        a("2. **Collect user corrections** to build continuous improvement pipeline.")
        a("3. **Monitor WER** on production traffic for drift detection.")
    elif dec_val == "collect_data":
        a("1. **Audit** high-silence / low-RMS recordings for re-recording candidates.")
        a("2. **Record 50–100** targeted utterances following DATASET-VERSIONING-STRATEGY.md.")
        a("3. **Build Dataset v2** and retrain with combined data.")
    a("")

    (out_dir / "error_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 8. decision.json
# ---------------------------------------------------------------------------


def write_decision(out_dir: Path, decision: dict[str, Any]) -> None:
    _write_json(out_dir / "decision.json", decision)


# ---------------------------------------------------------------------------
# Convenience: write all outputs
# ---------------------------------------------------------------------------


def write_all_outputs(out_dir: Path, analysis: dict[str, Any], verbose: bool = False) -> None:
    """Write all 8 output artifacts to out_dir."""
    write_error_distribution_report(out_dir, analysis)
    write_worst_samples(out_dir, analysis["worst_samples_df"])
    write_substitution_patterns(out_dir, analysis["substitution_patterns_df"])
    write_comparative_analysis(out_dir, analysis["comparative_df"])
    write_hypotheses(out_dir, analysis["hypotheses"])
    write_experiment_log(out_dir, analysis["experiments"])
    write_markdown_report(out_dir, analysis)
    write_decision(out_dir, analysis["decision"])

    if verbose:
        artifacts = [
            "error_distribution_report.json",
            "worst_samples.csv",
            "substitution_patterns.csv",
            "comparative_analysis.csv",
            "hypotheses.json",
            "m4a_experiment_log.csv",
            "error_analysis_report.md",
            "decision.json",
        ]
        for name in artifacts:
            print(f"  - {name}")
