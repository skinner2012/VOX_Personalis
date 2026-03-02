"""Phase 3: Substitution pattern mining, normalization audit, utterance length analysis."""

from typing import Any

import jiwer
import pandas as pd
from baseline_eval.error_analysis import extract_error_patterns

# Contraction map: contracted form (after apostrophe removal) → expanded form.
# Hardcoded because English contractions are a closed linguistic class — the set is finite
# and stable. Keys are post-apostrophe-removal forms (e.g. "don't" → "dont"). Only
# unambiguous pairs are included: keys that cannot be mistaken for a standalone English word
# (e.g. "were", "its", "lets" were excluded because they are valid words in other contexts).
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


def _expand_contractions(text: str, contraction_map: dict[str, str]) -> str:
    """Replace each token that matches a contraction with its expanded form."""
    return " ".join(contraction_map.get(w, w) for w in text.split())


def _wer_score(reference: str, hypothesis: str) -> float:
    """Compute WER for a single pair; returns 0 if both empty."""
    if not reference and not hypothesis:
        return 0.0
    if not reference or not hypothesis:
        return 1.0
    return float(jiwer.process_words(reference, hypothesis).wer)


def mine_substitution_patterns(
    df: pd.DataFrame, baseline_df: pd.DataFrame, top_n: int = 30
) -> pd.DataFrame:
    """Phase 3.1 — Top substitution pairs with baseline cross-check."""
    v1_1_patterns = extract_error_patterns(df, top_n=top_n)
    baseline_patterns = extract_error_patterns(baseline_df, top_n=50)

    baseline_subs: set[tuple[str, str]] = {
        (p["reference_token"], p["hypothesis_token"]) for p in baseline_patterns["substitutions"]
    }

    rows = []
    for p in v1_1_patterns["substitutions"][:top_n]:
        key = (p["reference_token"], p["hypothesis_token"])
        rows.append(
            {
                "reference_token": p["reference_token"],
                "hypothesis_token": p["hypothesis_token"],
                "count": p["count"],
                "also_in_baseline": key in baseline_subs,
                "example_files": p["example_files"],
            }
        )

    return pd.DataFrame(rows)


def detect_normalization_artifacts(df: pd.DataFrame) -> dict[str, Any]:
    """
    Phase 3.2 — Measure WER inflation from contraction mismatches.

    Expands contractions in both reference and hypothesis, re-scores WER,
    and compares to original to isolate normalization contribution.
    Contribution is reported as corpus-level WER points (consistent with
    aggregate WER from compute_global_metrics).
    """
    pair_counts: dict[tuple[str, str], int] = {}
    artifact_count = 0
    total_ref_words = 0
    total_original_errors = 0.0
    total_norm_errors = 0.0
    has_word_count = "word_count_ref" in df.columns

    for _, row in df.iterrows():
        ref_orig = row["reference"]
        hyp_orig = row["hypothesis"]
        ref_norm = _expand_contractions(ref_orig, CONTRACTION_MAP)
        hyp_norm = _expand_contractions(hyp_orig, CONTRACTION_MAP)

        word_count = (
            int(row["word_count_ref"])
            if has_word_count
            else (len(ref_orig.split()) if ref_orig else 0)
        )
        wer_orig = float(row["wer"])
        total_ref_words += word_count
        total_original_errors += wer_orig * word_count

        if ref_norm == ref_orig and hyp_norm == hyp_orig:
            total_norm_errors += wer_orig * word_count
            continue

        wer_norm = _wer_score(ref_norm, hyp_norm)
        total_norm_errors += wer_norm * word_count

        if abs(wer_norm - wer_orig) > 1e-6:
            artifact_count += 1

        ref_words = ref_orig.split()
        hyp_words = hyp_orig.split()
        for rw, hw in zip(ref_words, hyp_words, strict=False):
            r_exp = CONTRACTION_MAP.get(rw, rw)
            h_exp = CONTRACTION_MAP.get(hw, hw)
            if r_exp != rw or h_exp != hw:
                key = (rw, hw)
                pair_counts[key] = pair_counts.get(key, 0) + 1

    corpus_wer_orig = total_original_errors / total_ref_words if total_ref_words > 0 else 0.0
    corpus_wer_norm = total_norm_errors / total_ref_words if total_ref_words > 0 else 0.0
    contribution = corpus_wer_orig - corpus_wer_norm

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "artifact_count": artifact_count,
        "normalization_wer_contribution_pts": round(contribution, 4),
        "contraction_pairs": [
            {"reference_token": k[0], "hypothesis_token": k[1], "count": v}
            for k, v in sorted_pairs[:20]
        ],
    }


def analyze_short_utterances(df: pd.DataFrame) -> dict[str, Any]:
    """Phase 3.3 — Short utterances (≤3 reference words): WER inflation from single errors."""
    short = df[df["word_count_ref"] <= 3]
    long_rest = df[df["word_count_ref"] > 3]

    total_errors = int(
        (short["word_insertions"] + short["word_deletions"] + short["word_substitutions"]).sum()
    )
    all_errors = int(
        (df["word_insertions"] + df["word_deletions"] + df["word_substitutions"]).sum()
    )

    return {
        "sample_count": len(short),
        "mean_wer": round(float(short["wer"].mean()), 4) if not short.empty else 0.0,
        "median_wer": round(float(short["wer"].median()), 4) if not short.empty else 0.0,
        "error_share": round(total_errors / all_errors, 4) if all_errors > 0 else 0.0,
        "mean_wer_longer_utterances": (
            round(float(long_rest["wer"].mean()), 4) if not long_rest.empty else 0.0
        ),
    }


def analyze_long_utterances(df: pd.DataFrame) -> dict[str, Any]:
    """Phase 3.4 — Long utterances (≥10 reference words): WER comparison to medium."""
    long = df[df["word_count_ref"] >= 10]
    medium = df[(df["word_count_ref"] >= 4) & (df["word_count_ref"] < 10)]

    return {
        "sample_count": len(long),
        "mean_wer": round(float(long["wer"].mean()), 4) if not long.empty else 0.0,
        "median_wer": round(float(long["wer"].median()), 4) if not long.empty else 0.0,
        "medium_sample_count": len(medium),
        "medium_mean_wer": round(float(medium["wer"].mean()), 4) if not medium.empty else 0.0,
        "systematically_higher": (
            float(long["wer"].mean()) > float(medium["wer"].mean())
            if not long.empty and not medium.empty
            else False
        ),
    }


# Function words used to classify tokens in domain-specific failure analysis.
# Hardcoded because function words (articles, prepositions, auxiliary verbs, pronouns,
# conjunctions) are a closed grammatical class in English — new entries essentially never
# appear. Values are drawn directly from standard linguistic classification.
_FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "by",
    "from",
    "that",
    "this",
    "it",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "and",
    "or",
    "but",
    "not",
    "no",
}
# Voice-assistant command verbs used to classify tokens in domain-specific failure analysis.
# Unlike the two sets above, this list is project-specific and open-ended — it reflects the
# expected command vocabulary of VOX Personalis and was chosen heuristically from common
# voice-UI verbs. If the product's command vocabulary grows significantly, consider deriving
# this set from the dataset inventory rather than maintaining it here.
_COMMAND_VERBS = {
    "play",
    "stop",
    "pause",
    "resume",
    "turn",
    "set",
    "show",
    "tell",
    "get",
    "call",
    "open",
    "close",
    "start",
    "end",
    "cancel",
    "add",
    "remove",
    "delete",
    "search",
    "find",
    "go",
    "navigate",
    "send",
    "read",
}


def _classify_token(token: str) -> str:
    if token in _FUNCTION_WORDS:
        return "function_word"
    if token in _COMMAND_VERBS:
        return "command_verb"
    if token.replace(".", "").replace(",", "").isdigit():
        return "number"
    if token[0].isupper() if token else False:
        return "proper_noun"
    return "other"


def find_domain_specific_failures(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 3.5 — Tokens with error rate > 50% across the val set."""
    from baseline_eval.error_analysis import extract_alignments

    token_correct: dict[str, int] = {}
    token_error: dict[str, int] = {}

    for ref, hyp in zip(df["reference"], df["hypothesis"], strict=True):
        alignments = extract_alignments(ref, hyp)
        error_ref_tokens = {a["ref_token"] for a in alignments if a["ref_token"]}
        for word in ref.split():
            if word not in token_correct:
                token_correct[word] = 0
                token_error[word] = 0
            if word in error_ref_tokens:
                token_error[word] += 1
            else:
                token_correct[word] += 1

    rows = []
    for token, errors in token_error.items():
        total = errors + token_correct.get(token, 0)
        if total < 2:
            continue
        error_rate = errors / total
        if error_rate > 0.5:
            rows.append(
                {
                    "token": token,
                    "error_count": errors,
                    "total_occurrences": total,
                    "error_rate": round(error_rate, 4),
                    "category": _classify_token(token),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["token", "error_count", "total_occurrences", "error_rate", "category"]
        )
    return pd.DataFrame(rows).sort_values("error_rate", ascending=False).reset_index(drop=True)
