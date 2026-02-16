"""Test audit log management for single-shot test evaluation policy."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Audit log schema - consistent column ordering
AUDIT_LOG_COLUMNS: list[str] = [
    "run_id",
    "model",
    "test_run_timestamp",
    "git_commit_sha",
    "config_hash",
    "val_wer",
    "test_wer",
    "justification",
]


def get_git_commit_sha() -> str:
    """
    Get current git commit SHA.

    Returns:
        40-character git SHA or 'unknown' if not in a git repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,  # Project root
        )
        if result.returncode == 0:
            return result.stdout.strip()[:40]
    except (FileNotFoundError, OSError):
        # git not installed or permission issues - fallback to unknown
        pass
    return "unknown"


def compute_config_hash(config: dict) -> str:
    """
    Compute SHA256 hash of training configuration.

    Args:
        config: Configuration dictionary

    Returns:
        64-character hex SHA256 hash
    """
    # Sort keys for deterministic hashing
    config_json = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode()).hexdigest()


def create_audit_entry(
    run_id: str,
    model: str,
    val_wer: float,
    test_wer: float,
    config: dict,
    justification: str,
) -> dict:
    """
    Create test audit log entry.

    Args:
        run_id: Unique run identifier (timestamp)
        model: Model name (e.g., whisper-base.en)
        val_wer: Validation WER that justified test evaluation
        test_wer: Final test WER
        config: Training configuration
        justification: Reason for test evaluation

    Returns:
        Dictionary with audit entry fields
    """
    return {
        "run_id": run_id,
        "model": model,
        "test_run_timestamp": datetime.now(UTC).isoformat(),
        "git_commit_sha": get_git_commit_sha(),
        "config_hash": compute_config_hash(config),
        "val_wer": val_wer,
        "test_wer": test_wer,
        "justification": justification,
    }


def check_test_contamination(
    audit_log_path: str | Path,
    config_hash: str,
) -> tuple[bool, list[dict]]:
    """
    Check if configuration has already been evaluated on test set.

    Args:
        audit_log_path: Path to test_audit_log.csv
        config_hash: Hash of current configuration

    Returns:
        Tuple of (is_contaminated, previous_entries)
        where previous_entries are rows with matching config_hash
    """
    audit_log_path = Path(audit_log_path)

    if not audit_log_path.exists():
        return False, []

    df = pd.read_csv(audit_log_path)

    if "config_hash" not in df.columns:
        return False, []

    matches = df[df["config_hash"] == config_hash]

    if len(matches) > 0:
        return True, matches.to_dict("records")

    return False, []


def append_test_audit_log(
    audit_log_path: str | Path,
    entry: dict,
) -> None:
    """
    Append entry to test audit log.

    Args:
        audit_log_path: Path to test_audit_log.csv
        entry: Audit entry dictionary
    """
    audit_log_path = Path(audit_log_path)

    # Create or append
    if audit_log_path.exists():
        df = pd.read_csv(audit_log_path)
        new_row = pd.DataFrame([entry])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    # Ensure columns exist
    for col in AUDIT_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Save with consistent column order
    df[AUDIT_LOG_COLUMNS].to_csv(audit_log_path, index=False)


def enforce_test_policy(
    audit_log_path: str | Path,
    config: dict,
    verbose: bool = True,
) -> bool:
    """
    Enforce single-shot test policy.

    Checks if configuration has been tested before and warns user.
    Does NOT block - just warns and logs.

    Args:
        audit_log_path: Path to test_audit_log.csv
        config: Training configuration
        verbose: Print warnings

    Returns:
        True if this is first test for this config, False if repeat
    """
    config_hash = compute_config_hash(config)
    is_contaminated, previous = check_test_contamination(audit_log_path, config_hash)

    if is_contaminated:
        if verbose:
            print("\n" + "=" * 60)
            print("WARNING: TEST SET POLICY VIOLATION DETECTED")
            print("=" * 60)
            print("This configuration has been evaluated on test set before!")
            print(f"Config hash: {config_hash[:16]}...")
            print(f"Previous evaluations: {len(previous)}")
            for prev in previous:
                print(
                    f"  - {prev.get('test_run_timestamp', 'unknown')}: "
                    f"WER={prev.get('test_wer', 'N/A')}"
                )
            print("\nThis evaluation will be logged but results may be contaminated.")
            print("=" * 60 + "\n")
        return False

    return True
