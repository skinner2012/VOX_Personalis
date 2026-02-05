"""Split validation and quality checks for Dataset v1."""

from dataclasses import dataclass, field

import pandas as pd  # type: ignore[import-untyped]

# Minimum thresholds from spec
MIN_TRAIN_SAMPLES = 100
MIN_TRAIN_DURATION_SEC = 10 * 60  # 10 minutes
MIN_VAL_TEST_SAMPLES = 20
MIN_VAL_TEST_DURATION_SEC = 2 * 60  # 2 minutes

# Distribution balance threshold
DISTRIBUTION_BALANCE_THRESHOLD_PCT = 20.0


@dataclass
class ValidationResult:
    """Result of split validation checks."""

    passed: bool = True
    sample_validation_passed: bool = True
    duration_validation_passed: bool = True
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_split_sizes(df: pd.DataFrame, allow_small_splits: bool = False) -> ValidationResult:
    """
    Validate minimum sample counts and durations per split.

    Args:
        df: DataFrame with split and duration_sec columns
        allow_small_splits: If True, downgrade errors to warnings

    Returns:
        ValidationResult with pass/fail status and messages
    """
    result = ValidationResult()

    # Count samples per split
    split_counts = df["split"].value_counts().to_dict()
    train_count = split_counts.get("train", 0)
    val_count = split_counts.get("val", 0)
    test_count = split_counts.get("test", 0)

    # Sum durations per split
    split_durations = df.groupby("split")["duration_sec"].sum().to_dict()
    train_duration = split_durations.get("train", 0)
    val_duration = split_durations.get("val", 0)
    test_duration = split_durations.get("test", 0)

    # Check sample counts
    sample_errors = []
    if train_count < MIN_TRAIN_SAMPLES:
        sample_errors.append(
            f"Train split has {train_count} samples, minimum is {MIN_TRAIN_SAMPLES}"
        )
    if val_count < MIN_VAL_TEST_SAMPLES:
        sample_errors.append(
            f"Val split has {val_count} samples, minimum is {MIN_VAL_TEST_SAMPLES}"
        )
    if test_count < MIN_VAL_TEST_SAMPLES:
        sample_errors.append(
            f"Test split has {test_count} samples, minimum is {MIN_VAL_TEST_SAMPLES}"
        )

    # Check durations
    duration_errors = []
    if train_duration < MIN_TRAIN_DURATION_SEC:
        duration_errors.append(
            f"Train split has {train_duration / 60:.1f} min, "
            f"minimum is {MIN_TRAIN_DURATION_SEC / 60:.0f} min"
        )
    if val_duration < MIN_VAL_TEST_DURATION_SEC:
        duration_errors.append(
            f"Val split has {val_duration / 60:.1f} min, "
            f"minimum is {MIN_VAL_TEST_DURATION_SEC / 60:.0f} min"
        )
    if test_duration < MIN_VAL_TEST_DURATION_SEC:
        duration_errors.append(
            f"Test split has {test_duration / 60:.1f} min, "
            f"minimum is {MIN_VAL_TEST_DURATION_SEC / 60:.0f} min"
        )

    # Determine pass/fail
    if sample_errors:
        result.sample_validation_passed = False
    if duration_errors:
        result.duration_validation_passed = False

    all_issues = sample_errors + duration_errors

    if allow_small_splits:
        # Downgrade to warnings
        result.warnings.extend(all_issues)
        result.passed = True
    else:
        result.errors.extend(all_issues)
        if all_issues:
            result.passed = False

    return result


def check_distribution_balance(
    df: pd.DataFrame, threshold_pct: float = DISTRIBUTION_BALANCE_THRESHOLD_PCT
) -> list[str]:
    """
    Check if duration distributions are balanced across splits.

    Compares duration bin proportions in val/test against train as reference.
    Flags bins where relative difference exceeds threshold_pct.

    Example:
        Train: 30% in bin "(1,3]", Val: 40% in same bin
        → Difference: |40-30|/30 * 100 = 33.3% → Warning if threshold=20%

    Args:
        df: DataFrame with split and duration_bin columns
        threshold_pct: Maximum acceptable relative deviation (default: 20%)

    Returns:
        List of warning messages (empty if balanced)
    """
    warnings = []

    if "duration_bin" not in df.columns:
        warnings.append(
            "Cannot check distribution balance: duration_bin column missing "
            "(this indicates a pipeline error)"
        )
        return warnings

    # Get train distribution as reference
    train_df = df[df["split"] == "train"]
    train_total = len(train_df)

    if train_total == 0:
        return ["Train split is empty, cannot check distribution balance"]

    train_bin_props = train_df["duration_bin"].value_counts(normalize=True).to_dict()

    # Compare val and test to train
    for split in ["val", "test"]:
        split_df = df[df["split"] == split]
        split_total = len(split_df)

        if split_total == 0:
            warnings.append(f"{split.capitalize()} split is empty")
            continue

        split_bin_props = split_df["duration_bin"].value_counts(normalize=True).to_dict()

        # Check bins that exist in train
        for bin_label, train_prop in train_bin_props.items():
            split_prop = split_bin_props.get(bin_label, 0)

            if train_prop > 0:
                diff_pct = abs(split_prop - train_prop) / train_prop * 100
                if diff_pct > threshold_pct:
                    warnings.append(
                        f"Duration bin '{bin_label}' differs by {diff_pct:.1f}% "
                        f"between train ({train_prop * 100:.1f}%) "
                        f"and {split} ({split_prop * 100:.1f}%)"
                    )

        # Check bins that exist in val/test but not in train
        for bin_label, split_prop in split_bin_props.items():
            if bin_label not in train_bin_props and split_prop > 0:
                warnings.append(
                    f"Duration bin '{bin_label}' exists in {split} ({split_prop * 100:.1f}%) "
                    f"but not in train (0.0%)"
                )

    return warnings
