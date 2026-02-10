"""Text normalization for WER/CER computation."""

from collections.abc import Callable

from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase


def create_normalizer() -> Callable[[str], str]:
    """
    Create a text normalizer following the spec.

    Normalization pipeline (applied in order):
    1. Convert to lowercase
    2. Remove all punctuation: .,!?;:\"'-
    3. Collapse multiple spaces to single space
    4. Strip leading/trailing whitespace

    Returns:
        A callable that normalizes text strings
    """
    normalizer = Compose(
        [
            ToLowerCase(),
            RemovePunctuation(),
            RemoveMultipleSpaces(),
            Strip(),
        ]
    )

    def normalize(text: str) -> str:
        """Normalize text for WER comparison."""
        if not text:
            return ""
        result: str = normalizer(text)
        return result

    return normalize
