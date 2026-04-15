"""Tests for healthcare_rag.retriever — TF-IDF, hybrid search, metadata filters."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from healthcare_rag.retriever import (
    HybridRetriever,
    SearchResult,
    TfidfRetriever,
    _apply_filters,
    _rows_to_results,
)


# ── _apply_filters ───────────────────────────────────────────────────────────


class TestApplyFilters:
    @pytest.fixture()
    def meta(self):
        return pd.DataFrame(
            {
                "doc_id": ["a", "b", "c", "d"],
                "specialty": ["cardiology", "psychiatry", "cardiology", "endocrinology"],
                "year": [2019, 2020, 2022, 2021],
            }
        )

    def test_no_filters_returns_all(self, meta):
        idx = _apply_filters(meta)
        assert len(idx) == len(meta)

    def test_specialty_filter(self, meta):
        idx = _apply_filters(meta, specialty="cardiology")
        assert list(idx) == [0, 2]

    def test_specialty_case_insensitive(self, meta):
        idx = _apply_filters(meta, specialty="Cardiology")
        assert list(idx) == [0, 2]

    def test_year_range_filter(self, meta):
        idx = _apply_filters(meta, year_range=(2020, 2021))
        assert set(idx) == {1, 3}

    def test_combined_filters(self, meta):
        idx = _apply_filters(meta, specialty="cardiology", year_range=(2020, 2025))
        assert list(idx) == [2]

    def test_no_match_returns_empty(self, meta):
        idx = _apply_filters(meta, specialty="oncology")
        assert len(idx) == 0


# ── TfidfRetriever ───────────────────────────────────────────────────────────


class TestTfidfRetriever:
    def test_build_and_search(self, clean_df):
        retriever = TfidfRetriever.build(clean_df)
        results = retriever.search("ACE inhibitors hypertension", k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_top_result_relevance(self, clean_df):
        retriever = TfidfRetriever.build(clean_df)
        results = retriever.search("ACE inhibitors hypertension", k=1)
        assert results[0].doc_id == "t-001"

    def test_results_sorted_descending(self, clean_df):
        retriever = TfidfRetriever.build(clean_df)
        results = retriever.search("treatment for depression", k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_specialty_filter(self, clean_df):
        retriever = TfidfRetriever.build(clean_df)
        results = retriever.search("treatment", k=10, specialty="psychiatry")
        assert all(r.specialty == "psychiatry" for r in results)

    def test_search_with_year_range(self, clean_df):
        retriever = TfidfRetriever.build(clean_df)
        results = retriever.search("treatment", k=10, year_range=(2021, 2025))
        assert all(2021 <= r.year <= 2025 for r in results)

    def test_save_and_load_roundtrip(self, clean_df, tmp_path):
        path = tmp_path / "index.pkl"
        original = TfidfRetriever.build(clean_df)
        original.save(path)

        loaded = TfidfRetriever.load(path)
        results_orig = original.search("hypertension", k=2)
        results_load = loaded.search("hypertension", k=2)

        assert [r.doc_id for r in results_orig] == [r.doc_id for r in results_load]

    def test_specialty_field_populated(self, clean_df):
        retriever = TfidfRetriever.build(clean_df)
        results = retriever.search("diabetes", k=1)
        assert results[0].specialty == "endocrinology"


# ── _rows_to_results ─────────────────────────────────────────────────────────


class TestRowsToResults:
    def test_maps_fields_correctly(self):
        meta = pd.DataFrame(
            {
                "doc_id": ["d1"],
                "title": ["Title One"],
                "text": ["Some text"],
                "source": ["Journal"],
                "year": [2023],
                "specialty": ["neuro"],
            }
        )
        indices = np.array([0])
        fused = np.array([0.5])
        sparse = np.array([0.3])
        dense = np.array([0.7])

        results = _rows_to_results(meta, indices, fused, sparse, dense)
        r = results[0]
        assert r.doc_id == "d1"
        assert r.title == "Title One"
        assert r.score == pytest.approx(0.5)
        assert r.score_sparse == pytest.approx(0.3)
        assert r.score_dense == pytest.approx(0.7)
        assert r.specialty == "neuro"


# ── HybridRetriever ──────────────────────────────────────────────────────────


class TestHybridRetriever:
    """Tests for HybridRetriever using a mock dense encoder to avoid model downloads."""

    @pytest.fixture()
    def hybrid_retriever(self, clean_df):
        tfidf = TfidfRetriever.build(clean_df)
        n_docs = len(clean_df)
        dim = 32
        rng = np.random.RandomState(42)
        fake_embeddings = rng.randn(n_docs, dim).astype(np.float64)
        norms = np.linalg.norm(fake_embeddings, axis=1, keepdims=True)
        fake_embeddings /= norms

        retriever = HybridRetriever(
            tfidf=tfidf,
            doc_embeddings=fake_embeddings,
            embedding_model_name="fake-model",
        )

        mock_encoder = MagicMock()
        q_emb = rng.randn(1, dim).astype(np.float64)
        q_emb /= np.linalg.norm(q_emb)
        mock_encoder.embed.return_value = iter([q_emb[0]])
        retriever._encoder = mock_encoder

        return retriever

    def test_search_returns_results(self, hybrid_retriever):
        results = hybrid_retriever.search("hypertension treatment", k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_have_both_scores(self, hybrid_retriever):
        results = hybrid_retriever.search("hypertension treatment", k=1)
        r = results[0]
        assert r.score_sparse is not None
        assert r.score_dense is not None

    def test_rrf_score_positive(self, hybrid_retriever):
        results = hybrid_retriever.search("treatment", k=3)
        for r in results:
            assert r.score > 0

    def test_specialty_filter_applied(self, hybrid_retriever):
        results = hybrid_retriever.search("treatment", k=10, specialty="cardiology")
        assert all(r.specialty == "cardiology" for r in results)

    def test_year_range_filter_applied(self, hybrid_retriever):
        results = hybrid_retriever.search("treatment", k=10, year_range=(2021, 2025))
        assert all(2021 <= r.year <= 2025 for r in results)

    def test_save_and_load_roundtrip(self, hybrid_retriever, tmp_path):
        path = tmp_path / "hybrid.pkl"
        hybrid_retriever.save(path)

        loaded = HybridRetriever.load(path)
        assert loaded.embedding_model_name == hybrid_retriever.embedding_model_name
        assert loaded.doc_embeddings.shape == hybrid_retriever.doc_embeddings.shape

    def test_load_rejects_non_hybrid_index(self, clean_df, tmp_path):
        path = tmp_path / "tfidf.pkl"
        TfidfRetriever.build(clean_df).save(path)
        with pytest.raises(ValueError, match="not a hybrid index"):
            HybridRetriever.load(path)
