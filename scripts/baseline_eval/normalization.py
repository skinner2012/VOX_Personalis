"""Text normalization for WER/CER computation."""

from collections.abc import Callable

from jiwer import Compose, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase

# Contraction map: contracted form (after apostrophe removal) → expanded form.
# English contractions are a closed linguistic class — the set is finite and stable.
# Keys are post-apostrophe-removal forms (e.g. "don't" → "dont").
CONTRACTION_MAP: dict[str, str] = {
    "whats": "what is",
    "dont": "do not",
    "im": "i am",
    "cant": "cannot",
    "wont": "will not",
    "theyre": "they are",
    "youre": "you are",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
    "doesnt": "does not",
    "didnt": "did not",
    "hasnt": "has not",
    "havent": "have not",
    "hadnt": "had not",
    "couldnt": "could not",
    "wouldnt": "would not",
    "shouldnt": "should not",
    "thats": "that is",
    "theres": "there is",
    "heres": "here is",
    "shes": "she is",
    "whos": "who is",
    "hows": "how is",
    "wheres": "where is",
    "whens": "when is",
    "youve": "you have",
    "youll": "you will",
    "theyll": "they will",
    "youd": "you would",
    "theyd": "they would",
    "ive": "i have",
    "hes": "he is",
}


def expand_contractions(text: str) -> str:
    """Replace each token matching a contraction key with its expanded form."""
    return " ".join(CONTRACTION_MAP.get(w, w) for w in text.split())


def create_normalizer(version: int = 1) -> Callable[[str], str]:
    """
    Create a text normalizer for WER/CER computation.

    version=1 (textnorm_v1):
        1. Convert to lowercase
        2. Remove all punctuation
        3. Collapse multiple spaces to single space
        4. Strip leading/trailing whitespace

    version=2 (textnorm_v2):
        1. Convert to lowercase
        2. Remove all punctuation (apostrophes stripped here)
        3. Expand contractions via CONTRACTION_MAP
        4. Collapse multiple spaces to single space
        5. Strip leading/trailing whitespace
    """
    pre = Compose([ToLowerCase(), RemovePunctuation()])
    post = Compose([RemoveMultipleSpaces(), Strip()])
    expand = version >= 2

    def normalize(text: str) -> str:
        """Normalize text for WER comparison."""
        if not text:
            return ""
        result: str = pre(text)
        if expand:
            result = expand_contractions(result)
        result = post(result)
        return result

    return normalize
