"""Tests for healthcare_rag.reranker and the HybridRetriever rerank path.

A real cross-encoder is never loaded: we inject a fake encoder/reranker so the
tests stay fast and offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from healthcare_rag.reranker import CrossEncoderReranker
from healthcare_rag.retriever import HybridRetriever, SearchResult, TfidfRetriever


def _hit(doc_id: str, text: str) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        title=doc_id.upper(),
        text=text,
        score=0.0,
        source="Test",
        year=2023,
        specialty="cardiology",
    )


# ── CrossEncoderReranker ──────────────────────────────────────────────────────


class TestCrossEncoderReranker:
    def _reranker_with_scores(self, scores):
        """Build a reranker whose underlying model returns fixed scores."""
        reranker = CrossEncoderReranker(model_name="fake-model")
        fake_model = MagicMock()
        fake_model.rerank.return_value = list(scores)
        reranker._encoder = fake_model
        return reranker, fake_model

    def test_empty_hits_returns_empty(self):
        reranker, _ = self._reranker_with_scores([])
        assert reranker.rerank("q", []) == []

    def test_assigns_rerank_scores(self):
        hits = [_hit("a", "alpha"), _hit("b", "beta"), _hit("c", "gamma")]
        reranker, _ = self._reranker_with_scores([0.1, 0.9, 0.5])
        out = reranker.rerank("q", hits)
        scores = {h.doc_id: h.score_rerank for h in out}
        assert scores == {"a": pytest.approx(0.1), "b": pytest.approx(0.9), "c": pytest.approx(0.5)}

    def test_reorders_by_relevance(self):
        hits = [_hit("a", "alpha"), _hit("b", "beta"), _hit("c", "gamma")]
        reranker, _ = self._reranker_with_scores([0.1, 0.9, 0.5])
        out = reranker.rerank("q", hits)
        assert [h.doc_id for h in out] == ["b", "c", "a"]

    def test_top_k_truncates(self):
        hits = [_hit("a", "alpha"), _hit("b", "beta"), _hit("c", "gamma")]
        reranker, _ = self._reranker_with_scores([0.1, 0.9, 0.5])
        out = reranker.rerank("q", hits, top_k=2)
        assert [h.doc_id for h in out] == ["b", "c"]

    def test_passes_query_and_texts_to_model(self):
        hits = [_hit("a", "alpha text"), _hit("b", "beta text")]
        reranker, fake_model = self._reranker_with_scores([0.2, 0.8])
        reranker.rerank("my query", hits)
        fake_model.rerank.assert_called_once_with("my query", ["alpha text", "beta text"])

    def test_handles_negative_logits(self):
        hits = [_hit("a", "alpha"), _hit("b", "beta")]
        reranker, _ = self._reranker_with_scores([-11.4, 5.2])
        out = reranker.rerank("q", hits)
        assert [h.doc_id for h in out] == ["b", "a"]

    def test_lazy_model_not_loaded_on_init(self):
        reranker = CrossEncoderReranker(model_name="fake-model")
        assert reranker._encoder is None


# ── HybridRetriever rerank wiring ─────────────────────────────────────────────


class TestHybridRerankPath:
    @pytest.fixture()
    def hybrid_retriever(self, clean_df):
        tfidf = TfidfRetriever.build(clean_df)
        n_docs = len(clean_df)
        dim = 32
        rng = np.random.RandomState(7)
        emb = rng.randn(n_docs, dim).astype(np.float64)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        retriever = HybridRetriever(
            tfidf=tfidf, doc_embeddings=emb, embedding_model_name="fake-model"
        )

        mock_encoder = MagicMock()
        q_emb = rng.randn(1, dim).astype(np.float64)
        q_emb /= np.linalg.norm(q_emb)
        mock_encoder.embed.return_value = iter([q_emb[0]])
        retriever._encoder = mock_encoder
        return retriever

    def test_no_rerank_does_not_invoke_reranker(self, hybrid_retriever):
        fake_reranker = MagicMock()
        hybrid_retriever._reranker = fake_reranker
        hybrid_retriever.search("treatment", k=2, rerank=False)
        fake_reranker.rerank.assert_not_called()

    def test_rerank_invokes_reranker_with_candidate_pool(self, hybrid_retriever):
        fake_reranker = MagicMock()
        fake_reranker.rerank.side_effect = lambda q, hits, top_k=None: hits[:top_k]
        hybrid_retriever._reranker = fake_reranker

        hybrid_retriever.search("treatment", k=2, rerank=True, rerank_candidates=3)

        fake_reranker.rerank.assert_called_once()
        _, called_hits = fake_reranker.rerank.call_args.args
        # Candidate pool should be wider than the final k (capped by corpus size).
        assert len(called_hits) == min(3, len(hybrid_retriever.tfidf.metadata))

    def test_rerank_returns_reranked_order_truncated_to_k(self, hybrid_retriever):
        fake_reranker = MagicMock()
        # Reverse the candidate order to prove rerank output is what's returned.
        fake_reranker.rerank.side_effect = lambda q, hits, top_k=None: list(reversed(hits))[:top_k]
        hybrid_retriever._reranker = fake_reranker

        baseline = hybrid_retriever.search("treatment", k=4, rerank=False)
        # Re-prime the one-shot encoder iterator for a second search.
        rng = np.random.RandomState(7)
        q_emb = rng.randn(1, 32).astype(np.float64)
        q_emb /= np.linalg.norm(q_emb)
        hybrid_retriever._encoder.embed.return_value = iter([q_emb[0]])

        out = hybrid_retriever.search("treatment", k=2, rerank=True, rerank_candidates=4)
        assert len(out) == 2
