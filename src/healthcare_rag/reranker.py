"""Second-stage cross-encoder re-ranking.

The hybrid retriever (TF-IDF + dense, fused with RRF) is a fast *first stage*
that scores the query and each document independently. A cross-encoder is a
slower but more accurate *second stage*: it reads the query and a candidate
document *together* and outputs a single relevance score, letting it reason
about how query terms interact with document terms.

Typical usage is two-stage: pull a generous candidate pool from the hybrid
retriever, then re-rank just those candidates and keep the best few.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from healthcare_rag.config import RERANK_MODEL_NAME

if TYPE_CHECKING:
    from healthcare_rag.retriever import SearchResult


class CrossEncoderReranker:
    """Re-rank candidate passages with a FastEmbed ONNX cross-encoder.

    The underlying model is loaded lazily on first use so importing this
    module (e.g. during tests) never triggers a model download.
    """

    def __init__(self, model_name: str = RERANK_MODEL_NAME):
        self.model_name = model_name
        self._encoder = None

    def _model(self):
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._encoder = TextCrossEncoder(model_name=self.model_name)
        return self._encoder

    def score(self, query: str, documents: List[str]) -> List[float]:
        """Return one relevance score per document (higher = more relevant)."""
        if not documents:
            return []
        return [float(s) for s in self._model().rerank(query, documents)]

    def rerank(
        self,
        query: str,
        hits: List["SearchResult"],
        top_k: Optional[int] = None,
    ) -> List["SearchResult"]:
        """Score each hit against the query, sort by relevance, and truncate.

        Mutates each hit's ``score_rerank`` field and returns the hits sorted
        in descending relevance order. If ``top_k`` is given, only the best
        ``top_k`` are returned.
        """
        if not hits:
            return []

        scores = self.score(query, [h.text for h in hits])
        for hit, s in zip(hits, scores):
            hit.score_rerank = s

        reordered = sorted(hits, key=lambda h: h.score_rerank, reverse=True)
        return reordered[:top_k] if top_k is not None else reordered
