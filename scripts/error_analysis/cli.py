"""CLI parsing and pipeline orchestration for S1-M4a Error Analysis."""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .audio_correlation import correlate_audio_quality
from .comparative import (
    build_comparative_df,
    compute_error_type_migration,
    compute_improvement_distribution,
    find_persistent_errors,
)
from .decomposition import (
    compute_by_duration_bin,
    compute_error_concentration,
    compute_global_metrics,
    compute_hallucination_analysis,
    enrich_predictions,
    get_worst_samples,
)
from .hypotheses import generate_hypotheses, make_decision
from .patterns import (
    analyze_long_utterances,
    analyze_short_utterances,
    detect_normalization_artifacts,
    find_domain_specific_failures,
    mine_substitution_patterns,
)
from .reporting import write_all_outputs

_REQUIRED_PRED_COLS = {
    "file_name",
    "pair_sha256",
    "split",
    "duration_sec",
    "duration_bin",
    "reference",
    "hypothesis",
    "wer",
    "cer",
    "word_insertions",
    "word_deletions",
    "word_substitutions",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VOX Personalis S1-M4a — Error Analysis & Targeted Improvement Hypotheses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m scripts.error_analysis \\
    --v1_1_predictions "./out/model_improvement/experiments/training_2/val_predictions.csv" \\
    --baseline_predictions "./out/baseline_eval/20260210-220646/baseline_predictions.csv" \\
    --manifest_path "./out/dataset_v1/*/dataset_v1_manifest.csv" \\
    --inventory_path "./out/inventory/20260205-142601/inventory_files.csv" \\
    --verbose
""",
    )
    parser.add_argument(
        "--v1_1_predictions", required=True, help="Path to Model v1.1 val predictions CSV"
    )
    parser.add_argument(
        "--baseline_predictions", required=True, help="Path to baseline val predictions CSV"
    )
    parser.add_argument(
        "--manifest_path", required=True, help="Path to Dataset v1 manifest CSV (glob supported)"
    )
    parser.add_argument("--inventory_path", required=True, help="Path to M0 inventory CSV")
    parser.add_argument(
        "--v1_predictions", default=None, help="Path to Model v1 val predictions CSV (optional)"
    )
    parser.add_argument(
        "--out_dir",
        default="./out/error_analysis",
        help="Output directory (default: ./out/error_analysis)",
    )
    parser.add_argument(
        "--top_n_worst",
        type=int,
        default=50,
        help="Number of worst samples to report (default: 50)",
    )
    parser.add_argument(
        "--top_n_subs",
        type=int,
        default=30,
        help="Number of substitution pairs to report (default: 30)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed logging")
    return parser.parse_args()


def _load_predictions(path: str, name: str) -> pd.DataFrame | None:
    """Load a predictions CSV; return None on missing/invalid file (caller decides severity)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"Fatal error: Cannot read {name}: {e}", file=sys.stderr)
        return None

    missing = _REQUIRED_PRED_COLS - set(df.columns)
    if missing:
        print(
            f"Fatal error: {name} is missing required columns: {missing}\n"
            f"Expected: {sorted(_REQUIRED_PRED_COLS)}",
            file=sys.stderr,
        )
        return None
    return df


def _resolve_glob(path: str) -> Path | None:
    """Resolve a path that may contain glob wildcards; return first match."""
    p = Path(path)
    if "*" in str(p):
        matches = sorted(p.parent.glob(p.name))
        return matches[0] if matches else None
    return p if p.exists() else None


def _build_experiment_log(
    norm_audit: dict,
    agg_wer: float,
    hypotheses: list[dict],
    timestamp: str,
) -> list[dict]:
    """
    Phase 6 — Build experiment log from normalization-only experiment results.

    The contraction normalization experiment is derived from detect_normalization_artifacts()
    which already computes the re-scored WER. We record it as m4a_exp_1.
    """
    experiments: list[dict[str, Any]] = []
    norm_pts = norm_audit.get("normalization_wer_contribution_pts", 0.0)
    exp_wer = agg_wer - norm_pts

    # Find H1 (normalization hypothesis)
    h1 = next((h for h in hypotheses if h.get("testable_in_m4a")), None)
    if h1 is None:
        return experiments

    supported = norm_pts > 0.005  # > 0.5 WER pt is meaningful
    decision_str = (
        "VALIDATED"
        if norm_pts > 0.01
        else ("PARTIALLY_VALIDATED" if norm_pts > 0.001 else "REJECTED")
    )

    experiments.append(
        {
            "experiment_id": "m4a_exp_1",
            "hypothesis_id": h1["hypothesis_id"],
            "timestamp": timestamp,
            "intervention": (
                "Apply contraction expansion (CONTRACTION_MAP) to both reference and hypothesis "
                "before WER scoring. Re-score all val predictions."
            ),
            "experiment_type": "normalization-only",
            "model_v1_1_val_wer": round(agg_wer, 4),
            "experiment_val_wer": round(exp_wer, 4),
            "val_wer_delta": round(agg_wer - exp_wer, 4),
            "hypothesis_supported": supported,
            "decision": decision_str,
            "notes": (
                f"{norm_audit.get('artifact_count', 0)} samples affected. "
                f"WER reduction: {norm_pts * 100:.2f} pts. "
                f"{len(norm_audit.get('contraction_pairs', []))} "
                "distinct contraction pairs detected."
            ),
        }
    )
    return experiments


def run_pipeline(args: argparse.Namespace) -> int:
    """Main analysis pipeline. Returns exit code."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    log = print if args.verbose else (lambda *_: None)
    log(f"Output directory: {out_dir}")

    # --- Load required inputs ---
    log("Loading Model v1.1 predictions...")
    v1_1_df = _load_predictions(args.v1_1_predictions, "v1_1_predictions")
    if v1_1_df is None:
        print(f"Fatal error: v1_1_predictions not found: {args.v1_1_predictions}", file=sys.stderr)
        return 1

    log("Loading baseline predictions...")
    baseline_df = _load_predictions(args.baseline_predictions, "baseline_predictions")
    if baseline_df is None:
        print(
            f"Fatal error: baseline_predictions not found: {args.baseline_predictions}",
            file=sys.stderr,
        )
        return 1

    manifest_path = _resolve_glob(args.manifest_path)
    if manifest_path is None:
        print(f"Fatal error: Manifest not found: {args.manifest_path}", file=sys.stderr)
        return 1
    try:
        pd.read_csv(manifest_path)
    except Exception as e:
        print(f"Fatal error: Cannot read manifest: {e}", file=sys.stderr)
        return 1

    inventory_path = Path(args.inventory_path)
    if not inventory_path.exists():
        print(f"Fatal error: Inventory not found: {args.inventory_path}", file=sys.stderr)
        return 1
    try:
        inventory_df = pd.read_csv(inventory_path)
    except Exception as e:
        print(f"Fatal error: Cannot read inventory: {e}", file=sys.stderr)
        return 1

    # --- Load optional v1 predictions ---
    v1_df: pd.DataFrame | None = None
    if args.v1_predictions:
        v1_df = _load_predictions(args.v1_predictions, "v1_predictions")
        if v1_df is None:
            print(
                "Warning: v1_predictions not found or invalid — skipping three-way comparison.",
                file=sys.stderr,
            )

    # --- Filter to val split ---
    v1_1_val = v1_1_df[v1_1_df["split"] == "val"].copy()
    baseline_val = baseline_df[baseline_df["split"] == "val"].copy()
    v1_val = v1_df[v1_df["split"] == "val"].copy() if v1_df is not None else None

    if v1_1_val.empty:
        print("Fatal error: No val samples found in v1_1_predictions", file=sys.stderr)
        return 2

    if baseline_val.empty:
        print("Fatal error: No val samples found in baseline_predictions", file=sys.stderr)
        return 2

    log(
        f"Val samples — v1.1: {len(v1_1_val)}, baseline: {len(baseline_val)}"
        + (f", v1: {len(v1_val)}" if v1_val is not None else "")
    )

    # --- Validate pair_sha256 overlap ---
    v1_1_keys = set(v1_1_val["pair_sha256"])
    baseline_keys = set(baseline_val["pair_sha256"])
    overlap = v1_1_keys & baseline_keys
    if not overlap:
        print(
            "Fatal error: No pair_sha256 overlap between v1_1 and baseline predictions",
            file=sys.stderr,
        )
        return 2
    if len(overlap) < len(v1_1_keys) * 0.8:
        print(
            f"Warning: Low pair_sha256 overlap ({len(overlap)}/{len(v1_1_keys)})", file=sys.stderr
        )

    # --- Enrich v1.1 predictions with derived columns ---
    log("Enriching predictions...")
    v1_1_val = enrich_predictions(v1_1_val)
    if v1_val is not None:
        v1_val = enrich_predictions(v1_val)
    baseline_val = enrich_predictions(baseline_val)

    # --- Phase 1: Error Decomposition ---
    log("Phase 1: Error decomposition...")
    global_metrics = compute_global_metrics(v1_1_val)
    error_concentration = compute_error_concentration(v1_1_val)
    by_duration_bin = compute_by_duration_bin(v1_1_val)
    hallucination = compute_hallucination_analysis(v1_1_val)
    worst_df = get_worst_samples(v1_1_val, baseline_val, top_n=args.top_n_worst)

    # --- Phase 2: Comparative Analysis ---
    log("Phase 2: Comparative analysis...")
    comp_df = build_comparative_df(v1_1_val, baseline_val, v1_val)
    has_v1 = v1_val is not None
    improvement_distribution = compute_improvement_distribution(comp_df, has_v1)
    error_migration = compute_error_type_migration(baseline_val, v1_1_val, v1_val)
    persistent_df = find_persistent_errors(comp_df)

    # --- Phase 3: Systematic Pattern Analysis ---
    log("Phase 3: Pattern analysis...")
    subs_df = mine_substitution_patterns(v1_1_val, baseline_val, top_n=args.top_n_subs)
    norm_audit = detect_normalization_artifacts(v1_1_val)
    short_utt = analyze_short_utterances(v1_1_val)
    long_utt = analyze_long_utterances(v1_1_val)
    domain_failures_df = find_domain_specific_failures(v1_1_val)

    # --- Phase 4: Audio Quality Correlation ---
    log("Phase 4: Audio quality correlation...")
    audio_quality = correlate_audio_quality(v1_1_val, inventory_df, top_n=args.top_n_worst)
    if audio_quality["low_match_warning"]:
        print(
            f"Warning: Audio inventory join rate "
            f"{audio_quality['join_match_rate'] * 100:.0f}% < 80%",
            file=sys.stderr,
        )

    # --- Phase 5: Hypotheses ---
    log("Phase 5: Generating hypotheses...")
    analysis_for_hypotheses: dict = {
        "global_metrics": global_metrics,
        "normalization_audit": norm_audit,
        "error_concentration": error_concentration,
        "improvement_distribution": improvement_distribution,
        "error_type_migration": error_migration,
        "audio_quality": audio_quality,
        "substitution_patterns_df": subs_df,
        "comparative_df": comp_df,
    }
    hypotheses = generate_hypotheses(analysis_for_hypotheses)

    # --- Phase 6: Follow-up experiments ---
    log("Phase 6: Recording experiment results...")
    iso_ts = datetime.now(UTC).isoformat()
    experiments = _build_experiment_log(norm_audit, global_metrics["wer"], hypotheses, iso_ts)

    # --- Decision gate ---
    log("Computing decision gate...")
    decision = make_decision(analysis_for_hypotheses, hypotheses, experiments)

    # --- Assemble full analysis dict for reporting ---
    analysis: dict = {
        "val_sample_count": len(v1_1_val),
        "global_metrics": global_metrics,
        "error_concentration": error_concentration,
        "by_duration_bin": by_duration_bin,
        "hallucination": hallucination,
        "normalization_audit": norm_audit,
        "audio_quality": audio_quality,
        "improvement_distribution": improvement_distribution,
        "error_type_migration": error_migration,
        "short_utterances": short_utt,
        "long_utterances": long_utt,
        "worst_samples_df": worst_df,
        "substitution_patterns_df": subs_df,
        "comparative_df": comp_df,
        "persistent_errors_df": persistent_df,
        "domain_failures_df": domain_failures_df,
        "hypotheses": hypotheses,
        "experiments": experiments,
        "decision": decision,
    }

    # --- Write outputs ---
    log("Writing outputs...")
    write_all_outputs(out_dir, analysis, verbose=args.verbose)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("S1-M4a ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Val samples:     {len(v1_1_val)}")
    print(f"Model v1.1 WER:  {global_metrics['wer']:.4f} ({global_metrics['wer'] * 100:.1f}%)")
    print(f"Model v1.1 CER:  {global_metrics['cer']:.4f} ({global_metrics['cer'] * 100:.1f}%)")
    print(
        f"Errors:          I={global_metrics['insertions']} "
        f"D={global_metrics['deletions']} S={global_metrics['substitutions']}"
    )
    print(f"Top-10% share:   {error_concentration['top_10pct_error_share'] * 100:.1f}% of errors")
    print(f"Norm artifact:   {norm_audit['normalization_wer_contribution_pts'] * 100:.2f} WER pts")
    print(f"Persistent fail: {len(persistent_df)} samples")
    print(f"Hypotheses:      {len(hypotheses)}")
    print(f"Decision:        {decision['decision']}")
    print("=" * 60)
    print(f"\nOutputs written to: {out_dir}")
    if args.verbose:
        pass  # artifact list already printed by write_all_outputs

    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_pipeline(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1
