"""Error pattern extraction and analysis."""

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import jiwer
import pandas as pd  # type: ignore[import-untyped]


def extract_alignments(reference: str, hypothesis: str) -> list[dict[str, Any]]:
    """
    Extract word-level alignments between reference and hypothesis.

    Args:
        reference: Normalized reference text
        hypothesis: Normalized hypothesis text

    Returns:
        List of alignment operations with tokens
    """
    if not reference and not hypothesis:
        return []

    if not reference:
        # All insertions
        return [
            {"type": "insertion", "ref_token": "", "hyp_token": word} for word in hypothesis.split()
        ]

    if not hypothesis:
        # All deletions
        return [
            {"type": "deletion", "ref_token": word, "hyp_token": ""} for word in reference.split()
        ]

    # Use jiwer to get word-level alignment
    output = jiwer.process_words(reference, hypothesis)

    alignments = []
    for chunk in output.alignments[0]:  # [0] because single reference/hypothesis pair
        if chunk.type == "equal":
            # Correct match - skip for error analysis
            continue
        elif chunk.type == "substitute":
            ref_words = reference.split()[chunk.ref_start_idx : chunk.ref_end_idx]
            hyp_words = hypothesis.split()[chunk.hyp_start_idx : chunk.hyp_end_idx]
            # Pair up words (may have different counts in substitution spans)
            for i in range(max(len(ref_words), len(hyp_words))):
                ref_word = ref_words[i] if i < len(ref_words) else ""
                hyp_word = hyp_words[i] if i < len(hyp_words) else ""
                if ref_word and hyp_word:
                    alignments.append(
                        {
                            "type": "substitution",
                            "ref_token": ref_word,
                            "hyp_token": hyp_word,
                        }
                    )
                elif ref_word:
                    alignments.append(
                        {
                            "type": "deletion",
                            "ref_token": ref_word,
                            "hyp_token": "",
                        }
                    )
                else:
                    alignments.append(
                        {
                            "type": "insertion",
                            "ref_token": "",
                            "hyp_token": hyp_word,
                        }
                    )
        elif chunk.type == "delete":
            ref_words = reference.split()[chunk.ref_start_idx : chunk.ref_end_idx]
            for word in ref_words:
                alignments.append(
                    {
                        "type": "deletion",
                        "ref_token": word,
                        "hyp_token": "",
                    }
                )
        elif chunk.type == "insert":
            hyp_words = hypothesis.split()[chunk.hyp_start_idx : chunk.hyp_end_idx]
            for word in hyp_words:
                alignments.append(
                    {
                        "type": "insertion",
                        "ref_token": "",
                        "hyp_token": word,
                    }
                )

    return alignments


def extract_error_patterns(df: pd.DataFrame, top_n: int = 50) -> dict[str, list[dict]]:
    """
    Extract error patterns from predictions.

    Returns top N patterns per error type (substitution, deletion, insertion).

    Args:
        df: DataFrame with 'reference', 'hypothesis', 'file_name' columns
        top_n: Number of top patterns per error type

    Returns:
        Dictionary with keys 'substitutions', 'deletions', 'insertions',
        each containing a list of pattern dicts with count and examples
    """
    # Track patterns with example files
    substitutions: dict[tuple[str, str], dict] = defaultdict(lambda: {"count": 0, "files": []})
    deletions: dict[str, dict] = defaultdict(lambda: {"count": 0, "files": []})
    insertions: dict[str, dict] = defaultdict(lambda: {"count": 0, "files": []})

    # Process all rows using zip for better performance
    for ref, hyp, file_name in zip(df["reference"], df["hypothesis"], df["file_name"], strict=True):
        alignments = extract_alignments(ref, hyp)

        for alignment in alignments:
            error_type = alignment["type"]
            ref_token = alignment["ref_token"]
            hyp_token = alignment["hyp_token"]

            if error_type == "substitution":
                key = (ref_token, hyp_token)
                substitutions[key]["count"] += 1
                if len(substitutions[key]["files"]) < 5:
                    substitutions[key]["files"].append(file_name)

            elif error_type == "deletion":
                deletions[ref_token]["count"] += 1
                if len(deletions[ref_token]["files"]) < 5:
                    deletions[ref_token]["files"].append(file_name)

            elif error_type == "insertion":
                insertions[hyp_token]["count"] += 1
                if len(insertions[hyp_token]["files"]) < 5:
                    insertions[hyp_token]["files"].append(file_name)

    # Sort by count and take top N
    def format_patterns(
        patterns: dict, ref_extractor: Callable, hyp_extractor: Callable
    ) -> list[dict]:
        """Format error patterns sorted by frequency."""
        sorted_items = sorted(patterns.items(), key=lambda x: x[1]["count"], reverse=True)
        return [
            {
                "reference_token": ref_extractor(key),
                "hypothesis_token": hyp_extractor(key),
                "count": val["count"],
                "example_files": ",".join(val["files"]),
            }
            for key, val in sorted_items[:top_n]
        ]

    return {
        "substitutions": format_patterns(
            substitutions, ref_extractor=lambda k: k[0], hyp_extractor=lambda k: k[1]
        ),
        "deletions": format_patterns(
            deletions, ref_extractor=lambda k: k, hyp_extractor=lambda k: ""
        ),
        "insertions": format_patterns(
            insertions, ref_extractor=lambda k: "", hyp_extractor=lambda k: k
        ),
    }
