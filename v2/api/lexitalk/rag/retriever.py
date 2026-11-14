"""Retrieval with a transparent, documented relevance score.

Replaces the original heuristic (L2 → 1/(1+d) blended with fuzzy matching and
hard-coded invoice/aws/h1b boosts) with:

    score = 0.85 * cosine_similarity + 0.15 * query_term_coverage

Both terms are in [0, 1], so the score is too. It is still a heuristic and is
labelled as such in the UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .ports import Embedder, Hit, VectorIndex

_WORD = re.compile(r"[a-z0-9§]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "does",
        "do",
        "can",
        "under",
        "about",
        "this",
        "that",
    ]
)


def term_coverage(query: str, text: str) -> float:
    terms = {t for t in _WORD.findall(query.lower()) if t not in _STOP and len(t) > 1}
    if not terms:
        return 0.0
    body = set(_WORD.findall(text.lower()))
    return len(terms & body) / len(terms)


@dataclass(frozen=True)
class ScoredHit:
    hit: Hit
    score: float


class Retriever:
    def __init__(self, embedder: Embedder, index: VectorIndex, min_score: float = 0.35) -> None:
        self.embedder = embedder
        self.index = index
        self.min_score = min_score

    def retrieve(self, query: str, k: int) -> list[ScoredHit]:
        if not query.strip() or self.index.count() == 0:
            return []
        emb = self.embedder.encode([query])[0]
        hits = self.index.query(emb, k=k)
        scored = [
            ScoredHit(h, round(0.85 * max(0.0, min(1.0, h.similarity)) + 0.15 * term_coverage(query, h.chunk.text), 4))
            for h in hits
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored
