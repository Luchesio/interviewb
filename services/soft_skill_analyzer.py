"""
Soft-skill analysis helpers — enhanced with NLP (Feature 8).

Tier 1 (always runs, < 2 ms): rule-based filler detection, sentiment, confidence
Tier 2 (async, optional): Hugging Face transformers for leadership / teamwork /
                           problem-solving detection.

Install extras:
    pip install textblob transformers torch

The HF pipeline is loaded lazily on first call and cached in-process.
If transformers is not installed, Tier 2 returns None gracefully.
"""

import re
import logging
import asyncio
from functools import lru_cache
from typing import Tuple, Optional

log = logging.getLogger(__name__)

# ── Filler-word corpus ────────────────────────────────────────────────────────
FILLER_WORDS: set[str] = {
    "um", "uh", "er", "ah", "like", "you know", "i mean", "sort of",
    "kind of", "basically", "literally", "actually", "honestly",
    "right", "okay so", "so yeah", "you see", "well", "anyway",
}

_MULTI_WORD_FILLERS  = {f for f in FILLER_WORDS if " " in f}
_SINGLE_WORD_FILLERS = {f for f in FILLER_WORDS if " " not in f}

# The filler corpus, TextBlob sentiment, and confidence heuristics below are all
# English-specific. Running them on other languages produces meaningless filler
# counts and sentiment, which then pollute the report. We only compute these
# metrics for the languages they're actually valid for.
_METRICS_SUPPORTED_LANGS = {"en"}


def metrics_supported(language: str) -> bool:
    return (language or "en").lower() in _METRICS_SUPPORTED_LANGS

# ── NLP label → soft-skill mapping ────────────────────────────────────────────
# Maps zero-shot candidate labels to soft-skill dimensions
_LEADERSHIP_PHRASES = [
    "led", "managed", "directed", "owned", "drove", "mentored",
    "coordinated", "organized", "initiated", "spearheaded",
]
_TEAMWORK_PHRASES = [
    "collaborated", "worked with", "team", "together", "partnered",
    "supported", "helped", "contributed", "alongside",
]
_PROBLEM_SOLVING_PHRASES = [
    "solved", "fixed", "debugged", "optimized", "refactored",
    "improved", "resolved", "designed solution", "figured out",
]


# ── Tier 1: fast rule-based analysis ──────────────────────────────────────────

def detect_filler_words(text: str) -> Tuple[int, list[str]]:
    if not text:
        return 0, []

    normalised = text.lower()
    found: list[str] = []

    for phrase in _MULTI_WORD_FILLERS:
        occurrences = len(re.findall(r'\b' + re.escape(phrase) + r'\b', normalised))
        found.extend([phrase] * occurrences)

    tokens = re.findall(r'\b\w+\b', normalised)
    for token in tokens:
        if token in _SINGLE_WORD_FILLERS:
            found.append(token)

    return len(found), sorted(set(found))


def analyse_sentiment(text: str) -> Tuple[str, float]:
    if not text:
        return "neutral", 0.0

    try:
        from textblob import TextBlob  # type: ignore
        polarity: float = TextBlob(text).sentiment.polarity
    except ImportError:
        pos_words = {"great", "excellent", "good", "well", "confident",
                     "experienced", "strong", "achieved", "improved", "built"}
        neg_words = {"bad", "poor", "weak", "failed", "struggled",
                     "unable", "difficult", "problem", "issue", "mistake"}
        words     = set(text.lower().split())
        pos_count = len(words & pos_words)
        neg_count = len(words & neg_words)
        polarity  = (pos_count - neg_count) / max(pos_count + neg_count, 1) * 0.5

    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return label, round(polarity, 4)


def estimate_confidence(text: str, filler_count: int) -> str:
    if not text:
        return "low"
    words      = text.split()
    word_count = len(words)
    density    = filler_count / max(word_count, 1)

    if word_count < 20 or density > 0.15:
        return "low"
    if word_count >= 60 and density < 0.05:
        return "high"
    return "medium"


# ── Tier 2: NLP phrase heuristic (fast, no heavy model required) ──────────────

def _score_phrase_presence(text: str, phrases: list[str]) -> int:
    """Return a 0-100 score based on how many indicator phrases appear."""
    if not text:
        return 0
    lower = text.lower()
    hits  = sum(1 for p in phrases if p in lower)
    return min(100, int((hits / max(len(phrases), 1)) * 300))  # scale generously


def _extract_key_phrases(text: str, all_phrase_lists: list[list[str]]) -> list[str]:
    if not text:
        return []
    lower  = text.lower()
    found  = []
    for phrases in all_phrase_lists:
        for p in phrases:
            if p in lower and p not in found:
                found.append(p)
    return found[:8]


def analyse_nlp_soft_skills(text: str) -> Optional[dict]:
    """
    Fast heuristic NLP scoring for leadership, teamwork, problem-solving.
    Falls back to zero-shot HuggingFace classification if available.
    Returns dict or None if text is empty.
    """
    if not text or len(text.strip()) < 20:
        return None

    leadership_score     = _score_phrase_presence(text, _LEADERSHIP_PHRASES)
    teamwork_score       = _score_phrase_presence(text, _TEAMWORK_PHRASES)
    problem_solving_score = _score_phrase_presence(text, _PROBLEM_SOLVING_PHRASES)
    key_phrases          = _extract_key_phrases(
        text, [_LEADERSHIP_PHRASES, _TEAMWORK_PHRASES, _PROBLEM_SOLVING_PHRASES]
    )

    # Optional: upgrade with HuggingFace zero-shot classification
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore

        @lru_cache(maxsize=1)
        def _get_classifier():
            log.info("Loading HuggingFace zero-shot classifier (first call only)…")
            return hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,   # CPU; set to 0 for GPU
            )

        classifier = _get_classifier()
        candidate_labels = ["leadership", "teamwork", "problem solving", "communication"]
        result = classifier(text[:512], candidate_labels=candidate_labels, multi_label=True)

        scores_map = dict(zip(result["labels"], result["scores"]))
        leadership_score      = int(scores_map.get("leadership",       0) * 100)
        teamwork_score        = int(scores_map.get("teamwork",         0) * 100)
        problem_solving_score = int(scores_map.get("problem solving",  0) * 100)
        communication_cues    = [l for l, s in scores_map.items() if s > 0.5]

        return {
            "leadership_score":      leadership_score,
            "teamwork_score":        teamwork_score,
            "problem_solving_score": problem_solving_score,
            "communication_cues":    communication_cues,
            "key_phrases":           key_phrases,
            "source":                "huggingface_zero_shot",
        }
    except ImportError:
        pass
    except Exception as exc:
        log.warning("HuggingFace NLP failed, using heuristic fallback: %s", exc)

    return {
        "leadership_score":      leadership_score,
        "teamwork_score":        teamwork_score,
        "problem_solving_score": problem_solving_score,
        "communication_cues":    [],
        "key_phrases":           key_phrases,
        "source":                "heuristic",
    }


def analyse_answer(text: str | None, language: str = "en") -> dict:
    """
    Convenience wrapper used by interview_service.
    Returns a dict ready to merge into an Answer model.
    Tier 2 NLP results are stored separately and merged into report at generation time.

    The metrics are English-only. For other languages we store neutral
    placeholders instead of misleading English-derived numbers; the report
    generator is told these weren't computed (see generate_report).
    """
    if not text:
        return {
            "filler_word_count":  0,
            "filler_words_found": [],
            "sentiment":          "neutral",
            "sentiment_score":    0.0,
            "confidence_level":   "low",
        }

    if not metrics_supported(language):
        return {
            "filler_word_count":  0,
            "filler_words_found": [],
            "sentiment":          "neutral",
            "sentiment_score":    0.0,
            "confidence_level":   "medium",
        }

    filler_count, filler_list     = detect_filler_words(text)
    sentiment_label, sentiment_sc = analyse_sentiment(text)
    confidence                    = estimate_confidence(text, filler_count)

    return {
        "filler_word_count":  filler_count,
        "filler_words_found": filler_list,
        "sentiment":          sentiment_label,
        "sentiment_score":    sentiment_sc,
        "confidence_level":   confidence,
    }


def aggregate_nlp_metrics(answers: list[dict], language: str = "en") -> Optional[dict]:
    """
    Run NLP analysis across all answers and return aggregated metrics.
    Called once at report generation time. English-only — returns None for
    other languages so the report doesn't present invalid soft-skill scores.
    """
    if not metrics_supported(language):
        return None

    all_text = " ".join(
        a.get("answer", "") or ""
        for a in answers
        if not a.get("skip") and a.get("answer")
    )
    if not all_text.strip():
        return None

    return analyse_nlp_soft_skills(all_text)