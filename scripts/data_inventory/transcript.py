"""Transcript analysis functions"""

import re


def analyze_transcript(text: str | None) -> dict:
    """
    Analyze transcript text and extract metrics.

    Args:
        text: Transcript text (may be None or empty)

    Returns:
        Dictionary with transcript metrics:
        - transcript_len_chars: Character count
        - transcript_len_words: Word count (alphanumeric sequences)
        - transcript_is_blank: Boolean indicating if blank/empty
        - transcript_has_non_ascii_ratio: Ratio of non-ASCII characters (0-1)
    """
    try:
        if text is None or (isinstance(text, str) and text.strip() == ""):
            return {
                "transcript_len_chars": 0,
                "transcript_len_words": 0,
                "transcript_is_blank": True,
                "transcript_has_non_ascii_ratio": 0.0,
            }

        text = str(text)  # Ensure string type

        # Check if blank first (early return optimization)
        if text.strip() == "":
            return {
                "transcript_len_chars": 0,
                "transcript_len_words": 0,
                "transcript_is_blank": True,
                "transcript_has_non_ascii_ratio": 0.0,
            }

        # Character count
        char_count = len(text)

        # Word count (alphanumeric sequences with optional apostrophes)
        words = re.findall(r"\b[\w']+\b", text)
        word_count = len(words)

        # Non-ASCII ratio
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / char_count

        return {
            "transcript_len_chars": char_count,
            "transcript_len_words": word_count,
            "transcript_is_blank": False,
            "transcript_has_non_ascii_ratio": round(non_ascii_ratio, 4),
        }

    except Exception:
        # Return safe defaults on any error
        return {
            "transcript_len_chars": 0,
            "transcript_len_words": 0,
            "transcript_is_blank": True,
            "transcript_has_non_ascii_ratio": 0.0,
        }
