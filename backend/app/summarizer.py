"""Extractive summarization: sentence scoring by word frequency (no ML deps)."""

from __future__ import annotations

import re
from collections import Counter

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been
    before being below between both but by can cannot could did do does doing down
    during each few for from further had has have having he her here hers herself
    him himself his how i if in into is it its itself just like me more most my
    myself no nor not now of off on once only or other ought our ours ourselves
    out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up
    very was we were what when where which while who whom why with would you
    your yours yourself yourselves will also may one two first new used using
    """.split()
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]*")

LENGTH_SENTENCES = {"short": 3, "medium": 6, "detailed": 12}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping only non-trivial ones."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 1]


def _content_words(sentence: str) -> list[str]:
    return [w for w in _WORD.findall(sentence.lower()) if w not in STOPWORDS]


def extractive_summarize(text: str, length: str = "medium") -> tuple[str, int]:
    """Return (summary, sentences_used) using word-frequency sentence scoring."""
    sentences = split_sentences(text)
    if not sentences:
        return "", 0

    freq: Counter[str] = Counter()
    for s in sentences:
        freq.update(_content_words(s))
    if not freq:
        # Degenerate case: no content words; return leading sentences.
        n = min(LENGTH_SENTENCES[length], len(sentences))
        return " ".join(sentences[:n]), n

    # Score = sum of word frequencies, normalized by sqrt of sentence length
    # to avoid over-favoring very long sentences.
    scored: list[tuple[float, int]] = []
    for i, s in enumerate(sentences):
        words = _content_words(s)
        if not words:
            continue
        score = sum(freq[w] for w in words) / (len(words) ** 0.5)
        scored.append((score, i))

    n = min(LENGTH_SENTENCES[length], len(sentences))
    top = sorted(scored, key=lambda t: (-t[0], t[1]))[:n]
    chosen = sorted(i for _, i in top)
    return " ".join(sentences[i] for i in chosen), len(chosen)


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """Return the top-N most frequent content words as keywords."""
    freq: Counter[str] = Counter()
    for m in _WORD.finditer(text.lower()):
        w = m.group(0)
        if w not in STOPWORDS and len(w) > 3:
            freq[w] += 1
    return [w for w, _ in freq.most_common(top_n)]
