"""Controlled experiment log — tracks single-variable ablation results."""

import csv
from pathlib import Path

# Schema for controlled_experiment_log.csv
CONTROLLED_LOG_COLUMNS = [
    "experiment_id",  # C1, C2, B1, B2, B3
    "timestamp",  # ISO 8601 start timestamp
    "category",  # C (inference hygiene) or B (training regularization)
    "description",  # One-line description of what changed
    "variable_name",  # Exact variable changed
    "baseline_value",  # Model v1 value
    "experiment_value",  # New value tested
    "model_v1_val_wer",  # Model v1 val WER (0.6823)
    "experiment_val_wer",  # This experiment's val WER (C-category: mean of 3 runs)
    "val_wer_delta",  # Absolute improvement (positive = better, v1 - experiment)
    "relative_improvement",  # (delta / model_v1_val_wer) * 100
    "hypothesis_supported",  # True/False
    "decision",  # ADOPT / INVESTIGATE / REJECT
    "decision_rationale",  # Brief explanation
    "training_time_sec",  # Duration (0 for inference-only C-category)
    "checkpoint_path",  # Path to experiment checkpoint (empty for C-category)
    "notes",  # Additional observations
]

# Decision thresholds per spec
_ADOPT_THRESHOLD = 2.0  # >= 2.0 abs pts improvement
_INVESTIGATE_THRESHOLD = 0.5  # 0.5–1.9 abs pts


def compute_decision(
    val_wer_delta: float,
    insertion_delta: int = 0,
    slice_regression_max: float = 0.0,
) -> tuple[str, str]:
    """
    Compute ADOPT/INVESTIGATE/REJECT decision per spec thresholds.

    Args:
        val_wer_delta: WER improvement in WER ratio units (positive = better).
                       e.g. 0.6478 - 0.6404 = 0.0074. Converted to absolute pts
                       internally (×100) before comparing against thresholds.
        insertion_delta: Change in insertions vs Model v1 (positive = more insertions)
        slice_regression_max: Worst slice regression in absolute WER pts

    Returns:
        Tuple of (decision, rationale)
    """
    # Convert WER ratio to absolute percentage points for threshold comparison
    val_wer_delta_pts = val_wer_delta * 100

    if insertion_delta > 0:
        return "REJECT", "Insertions increased vs Model v1"
    if slice_regression_max > 3.0:
        return "REJECT", f"Slice regression {slice_regression_max:.1f} pts > 3.0 threshold"
    if val_wer_delta_pts >= _ADOPT_THRESHOLD:
        return (
            "ADOPT",
            f"{val_wer_delta_pts:.2f} abs pts improvement >= {_ADOPT_THRESHOLD} threshold",
        )
    if val_wer_delta_pts >= _INVESTIGATE_THRESHOLD:
        return "INVESTIGATE", f"{val_wer_delta_pts:.2f} abs pts improvement (0.5–1.9 range)"
    return (
        "REJECT",
        f"{val_wer_delta_pts:.2f} abs pts improvement < {_INVESTIGATE_THRESHOLD} threshold",
    )


def create_c_category_entry(
    experiment_id: str,
    timestamp: str,
    description: str,
    variable_name: str,
    baseline_value: str,
    experiment_value: str,
    model_v1_val_wer: float,
    wer_runs: list[float],
    notes: str = "",
) -> dict:
    """
    Create a controlled experiment log entry for C-category (inference hygiene).

    C-category success criterion is reproducibility (variance < 0.2 abs pts),
    not WER delta. val_wer_delta is set to 0.0.

    Args:
        wer_runs: List of WER values from the n reproducibility runs

    Returns:
        Entry dict matching CONTROLLED_LOG_COLUMNS schema
    """
    mean_wer = sum(wer_runs) / len(wer_runs)
    variance = max(wer_runs) - min(wer_runs)
    passed = variance < 0.2

    decision = "ADOPT" if passed else "INVESTIGATE"
    rationale = (
        f"Variance {variance:.4f} abs pts ({'< 0.2 PASS' if passed else '>= 0.2 FAIL'})"
        f", {len(wer_runs)} runs, mean WER {mean_wer:.4f}"
    )

    return {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "category": "C",
        "description": description,
        "variable_name": variable_name,
        "baseline_value": baseline_value,
        "experiment_value": experiment_value,
        "model_v1_val_wer": round(model_v1_val_wer, 4),
        "experiment_val_wer": round(mean_wer, 4),
        "val_wer_delta": 0.0,  # C-category: no WER delta claim
        "relative_improvement": 0.0,
        "hypothesis_supported": passed,
        "decision": decision,
        "decision_rationale": rationale,
        "training_time_sec": 0,
        "checkpoint_path": "",
        "notes": notes,
    }


def create_b_category_entry(
    experiment_id: str,
    timestamp: str,
    description: str,
    variable_name: str,
    baseline_value: str,
    experiment_value: str,
    model_v1_val_wer: float,
    experiment_val_wer: float,
    training_time_sec: float,
    checkpoint_path: str,
    insertion_delta: int = 0,
    slice_regression_max: float = 0.0,
    notes: str = "",
) -> dict:
    """
    Create a controlled experiment log entry for B-category (training regularization).

    Args:
        model_v1_val_wer: Model v1 val WER (locked reference, 0.6823)
        experiment_val_wer: This experiment's val WER

    Returns:
        Entry dict matching CONTROLLED_LOG_COLUMNS schema
    """
    val_wer_delta = model_v1_val_wer - experiment_val_wer
    relative_improvement = (val_wer_delta / model_v1_val_wer * 100) if model_v1_val_wer > 0 else 0.0
    decision, rationale = compute_decision(val_wer_delta, insertion_delta, slice_regression_max)

    return {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "category": "B",
        "description": description,
        "variable_name": variable_name,
        "baseline_value": baseline_value,
        "experiment_value": experiment_value,
        "model_v1_val_wer": round(model_v1_val_wer, 4),
        "experiment_val_wer": round(experiment_val_wer, 4),
        "val_wer_delta": round(val_wer_delta, 4),
        "relative_improvement": round(relative_improvement, 2),
        "hypothesis_supported": val_wer_delta >= _INVESTIGATE_THRESHOLD,
        "decision": decision,
        "decision_rationale": rationale,
        "training_time_sec": int(training_time_sec),
        "checkpoint_path": checkpoint_path,
        "notes": notes,
    }


def append_controlled_experiment_log(log_path: str | Path, entry: dict) -> Path:
    """
    Append an entry to the controlled experiment log CSV.

    Creates the file with headers if it does not exist.

    Args:
        log_path: Path to controlled_experiment_log.csv
        entry: Dict matching CONTROLLED_LOG_COLUMNS schema

    Returns:
        Path to the log file
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONTROLLED_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({col: entry.get(col, "") for col in CONTROLLED_LOG_COLUMNS})

    return log_path


def read_controlled_experiment_log(log_path: str | Path) -> list[dict]:
    """
    Read all entries from the controlled experiment log.

    Returns:
        List of entry dicts, empty list if file does not exist
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []

    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_adopt_experiments(log_path: str | Path) -> list[dict]:
    """
    Return all ADOPT entries from the experiment log, sorted by val_wer_delta descending.

    Returns:
        List of ADOPT entry dicts, best improvement first
    """
    entries = read_controlled_experiment_log(log_path)
    adopted = [e for e in entries if e.get("decision") == "ADOPT"]
    return sorted(adopted, key=lambda e: float(e.get("val_wer_delta", 0)), reverse=True)


def get_best_b_experiment(log_path: str | Path) -> dict | None:
    """
    Return the best B-category experiment for assembly consideration.

    Prefers ADOPT over INVESTIGATE. Returns None if no ADOPT or INVESTIGATE
    B-experiment exists. Used by assembly when no B-category experiment reached
    the ADOPT threshold but INVESTIGATE results are still worth including.

    Returns:
        Best B-category entry dict, or None
    """
    entries = read_controlled_experiment_log(log_path)
    b_entries = [e for e in entries if e.get("category") == "B"]

    for decision in ("ADOPT", "INVESTIGATE"):
        candidates = [e for e in b_entries if e.get("decision") == decision]
        if candidates:
            return max(candidates, key=lambda e: float(e.get("val_wer_delta", 0)))

    return None
