"""Data integrity checks for manifest and audio files"""

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]


def check_manifest_validity(df: pd.DataFrame, file_col: str, text_col: str) -> dict:
    """
    Check manifest CSV for data integrity issues.

    Args:
        df: Manifest DataFrame
        file_col: Name of file column
        text_col: Name of transcript column

    Returns:
        Dictionary with integrity check results:
        - num_rows: Total number of rows
        - duplicate_file_count: Number of duplicate file_name entries
        - empty_transcript_count: Number of empty/null transcripts
        - empty_filename_count: Number of empty/null file_names

        Returns None values if check fails (e.g., missing columns).
    """
    try:
        num_rows = len(df)

        # Check for duplicate file names
        duplicate_file_count = df[file_col].duplicated().sum()

        # Check for empty/null transcripts
        empty_transcript_count = (df[text_col].isna() | (df[text_col] == "")).sum()

        # Check for empty/null file names
        empty_filename_count = (df[file_col].isna() | (df[file_col] == "")).sum()

        return {
            "num_rows": num_rows,
            "duplicate_file_count": int(duplicate_file_count),
            "empty_transcript_count": int(empty_transcript_count),
            "empty_filename_count": int(empty_filename_count),
        }

    except Exception:
        # Return safe defaults indicating check failed
        return {
            "num_rows": 0,
            "duplicate_file_count": None,
            "empty_transcript_count": None,
            "empty_filename_count": None,
        }


def check_file_existence(
    df: pd.DataFrame, data_dir: Path, file_col: str, audio_glob: str = "**/*"
) -> tuple[list[str], list[str]]:
    """
    Check which audio files exist and identify missing/extra files.

    Matching behavior: Files are matched by both full relative path and filename.
    For example, if manifest contains "file.wav" and disk has "audio/file.wav",
    they will be considered a match.

    Args:
        df: Manifest DataFrame
        data_dir: Directory containing audio files
        file_col: Name of file column in manifest
        audio_glob: Glob pattern for finding audio files (default: **/* for recursive)

    Returns:
        Tuple of (missing_files, extra_files):
        - missing_files: Files in manifest but not found on disk
        - extra_files: Files on disk but not in manifest

        Returns ([], []) if check fails (e.g., missing column, directory not found).
    """
    try:
        data_dir = Path(data_dir)

        # Validate data_dir exists
        if not data_dir.exists():
            return [], []

        # Get all file names from manifest
        manifest_files = set(df[file_col].dropna().astype(str).tolist())

        # Find all audio files on disk
        audio_extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus"}
        disk_files = set()

        for path in data_dir.glob(audio_glob):
            if path.is_file() and path.suffix.lower() in audio_extensions:
                # Get relative path from data_dir
                try:
                    rel_path = path.relative_to(data_dir)
                    disk_files.add(str(rel_path))
                except ValueError:
                    # Path is not relative to data_dir
                    disk_files.add(path.name)

        # Also check for files just by name (not full relative path)
        disk_files_by_name = {Path(f).name for f in disk_files}

        # Find missing files (in manifest but not on disk)
        missing_files = []
        for manifest_file in manifest_files:
            # Check both full path and just filename
            if (
                manifest_file not in disk_files
                and Path(manifest_file).name not in disk_files_by_name
            ):
                missing_files.append(manifest_file)

        # Find extra files (on disk but not in manifest)
        manifest_names = {Path(f).name for f in manifest_files}
        extra_files = []
        for disk_file in disk_files:
            disk_name = Path(disk_file).name
            # Check if this file or its name appears in manifest
            if disk_file not in manifest_files and disk_name not in manifest_names:
                extra_files.append(disk_file)

        # Sort for deterministic output
        missing_files.sort()
        extra_files.sort()

        return missing_files, extra_files

    except Exception:
        # Return empty lists on any error
        return [], []
