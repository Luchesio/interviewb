"""
Soft-skill analysis helpers.

Runs entirely in-process so it adds < 2 ms latency per answer.
Sentiment uses TextBlob; install with:  pip install textblob
Filler detection is rule-based.
"""

import re
from typing import Tuple

# ── Filler-word corpus ────────────────────────────────────────────────────────
FILLER_WORDS: set[str] = {
    "um", "uh", "er", "ah", "like", "you know", "i mean", "sort of",
    "kind of", "basically", "literally", "actually", "honestly",
    "right", "okay so", "so yeah", "you see", "well", "anyway",
}

_MULTI_WORD_FILLERS = {f for f in FILLER_WORDS if " " in f}
_SINGLE_WORD_FILLERS = {f for f in FILLER_WORDS if " " not in f}


def detect_filler_words(text: str) -> Tuple[int, list[str]]:
    """
    Return (count, list_of_found_fillers) for the given transcribed text.
    """
    if not text:
        return 0, []

    normalised = text.lower()
    found: list[str] = []

    # Multi-word first (order matters for count accuracy)
    for phrase in _MULTI_WORD_FILLERS:
        occurrences = len(re.findall(r'\b' + re.escape(phrase) + r'\b', normalised))
        found.extend([phrase] * occurrences)

    # Single-word
    tokens = re.findall(r'\b\w+\b', normalised)
    for token in tokens:
        if token in _SINGLE_WORD_FILLERS:
            found.append(token)

    return len(found), sorted(set(found))


def analyse_sentiment(text: str) -> Tuple[str, float]:
    """
    Return (label, polarity_score) where label is positive | neutral | negative
    and polarity_score ∈ [-1.0, +1.0].

    Falls back gracefully if TextBlob is not installed.
    """
    if not text:
        return "neutral", 0.0

    try:
        from textblob import TextBlob  # type: ignore
        polarity: float = TextBlob(text).sentiment.polarity
    except ImportError:
        # Lightweight keyword fallback
        pos_words = {"great", "excellent", "good", "well", "confident",
                     "experienced", "strong", "achieved", "improved", "built"}
        neg_words = {"bad", "poor", "weak", "failed", "struggled",
                     "unable", "difficult", "problem", "issue", "mistake"}
        words = set(text.lower().split())
        pos_count = len(words & pos_words)
        neg_count = len(words & neg_words)
        polarity = (pos_count - neg_count) / max(pos_count + neg_count, 1) * 0.5

    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return label, round(polarity, 4)


def estimate_confidence(text: str, filler_count: int) -> str:
    """
    Heuristic confidence label based on answer length and filler density.
    """
    if not text:
        return "low"

    words = text.split()
    word_count = len(words)
    density = filler_count / max(word_count, 1)

    if word_count < 20 or density > 0.15:
        return "low"
    if word_count >= 60 and density < 0.05:
        return "high"
    return "medium"


def analyse_answer(text: str | None) -> dict:
    """
    Convenience wrapper used by interview_service – returns a dict ready to
    merge into an Answer model.
    """
    if not text:
        return {
            "filler_word_count": 0,
            "filler_words_found": [],
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "confidence_level": "low",
        }

    filler_count, filler_list = detect_filler_words(text)
    sentiment_label, sentiment_score = analyse_sentiment(text)
    confidence = estimate_confidence(text, filler_count)

    return {
        "filler_word_count": filler_count,
        "filler_words_found": filler_list,
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "confidence_level": confidence,
    }