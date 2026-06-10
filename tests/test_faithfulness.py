"""Tests for healthcare_rag.faithfulness.

A deterministic topic-based fake embedder stands in for the dense model: each
text maps to a one-hot vector over a small topic vocabulary, so a claim and a
passage about the same topic have cosine 1.0 and different topics have cosine 0.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from healthcare_rag.faithfulness import (
    FaithfulnessReport,
    check_faithfulness,
    extract_citations,
    split_sentences,
    strip_citations,
)
from healthcare_rag.retriever import SearchResult

TOPICS = ["hypertension", "diabetes", "depression", "stroke"]


def fake_embed(texts):
    vecs = []
    for t in texts:
        v = np.zeros(len(TOPICS), dtype=np.float64)
        low = t.lower()
        for i, topic in enumerate(TOPICS):
            if topic in low:
                v[i] = 1.0
        if v.sum() == 0:
            v[0] = 1e-9  # avoid a zero vector
        vecs.append(v)
    return vecs


def _hit(doc_id: str, text: str) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        title=doc_id,
        text=text,
        score=0.0,
        source="Test",
        year=2023,
        specialty="cardiology",
    )


# ── parsing helpers ───────────────────────────────────────────────────────────


class TestParsing:
    def test_split_sentences(self):
        assert split_sentences("First sentence. Second one! Third?") == [
            "First sentence.",
            "Second one!",
            "Third?",
        ]

    def test_split_handles_newlines(self):
        assert split_sentences("Line one.\n\nLine two.") == ["Line one.", "Line two."]

    def test_extract_citations(self):
        assert extract_citations("ACE inhibitors help [1] and also [2].") == [1, 2]

    def test_extract_multidigit_citation(self):
        assert extract_citations("see [12]") == [12]

    def test_extract_none(self):
        assert extract_citations("no citations here") == []

    def test_strip_citations(self):
        assert strip_citations("ACE inhibitors [1] work [2].") == "ACE inhibitors  work ."


# ── check_faithfulness ────────────────────────────────────────────────────────


class TestCheckFaithfulness:
    def test_supported_citation(self):
        hits = [_hit("d1", "ACE inhibitors are first-line for hypertension.")]
        answer = "ACE inhibitors are recommended for hypertension [1]."
        report = check_faithfulness(answer, hits, embed_fn=fake_embed)
        assert report.citation_coverage == pytest.approx(1.0)
        assert report.citation_support == pytest.approx(1.0)
        assert report.mean_support == pytest.approx(1.0)

    def test_unsupported_citation_offtopic(self):
        hits = [_hit("d1", "ACE inhibitors are first-line for hypertension.")]
        # Claim is about diabetes but cites the hypertension passage.
        answer = "Metformin is preferred for diabetes management [1]."
        report = check_faithfulness(answer, hits, embed_fn=fake_embed)
        assert report.citation_coverage == pytest.approx(1.0)
        assert report.citation_support == pytest.approx(0.0)
        assert report.mean_support < 0.5

    def test_invalid_citation_flagged(self):
        hits = [_hit("d1", "ACE inhibitors for hypertension.")]
        answer = "Anticoagulants reduce stroke risk [5]."
        report = check_faithfulness(answer, hits, embed_fn=fake_embed)
        assert report.invalid_citations == 1
        assert report.invalid_citation_rate == pytest.approx(1.0)

    def test_uncited_claim_lowers_coverage(self):
        hits = [_hit("d1", "ACE inhibitors for hypertension.")]
        answer = "ACE inhibitors treat hypertension [1]. Diabetes also needs management here."
        report = check_faithfulness(answer, hits, embed_fn=fake_embed)
        assert report.n_claims == 2
        assert report.citation_coverage == pytest.approx(0.5)

    def test_abstention_skips_scoring(self):
        hits = [_hit("d1", "ACE inhibitors for hypertension.")]
        answer = "INSUFFICIENT EVIDENCE: the corpus lacks information on this topic."
        spy = MagicMock(side_effect=fake_embed)
        report = check_faithfulness(answer, hits, embed_fn=spy)
        assert report.abstained is True
        spy.assert_not_called()

    def test_no_citations_skips_embedding(self):
        hits = [_hit("d1", "ACE inhibitors for hypertension.")]
        answer = "ACE inhibitors are commonly used to treat hypertension in adults."
        spy = MagicMock(side_effect=fake_embed)
        report = check_faithfulness(answer, hits, embed_fn=spy)
        spy.assert_not_called()
        assert report.citation_coverage == pytest.approx(0.0)
        assert report.n_claims == 1

    def test_threshold_gates_support(self):
        hits = [_hit("d1", "ACE inhibitors for hypertension.")]
        answer = "ACE inhibitors treat hypertension [1]."
        # Even a perfect cosine of 1.0 fails an impossible threshold.
        report = check_faithfulness(answer, hits, embed_fn=fake_embed, threshold=1.5)
        assert report.citation_support == pytest.approx(0.0)

    def test_multiple_citations_uses_best(self):
        hits = [
            _hit("d1", "Content about diabetes management."),
            _hit("d2", "ACE inhibitors are first-line for hypertension."),
        ]
        # Cites both; the hypertension passage [2] supports the claim.
        answer = "ACE inhibitors are first-line for hypertension [1][2]."
        report = check_faithfulness(answer, hits, embed_fn=fake_embed)
        assert report.citation_support == pytest.approx(1.0)

    def test_report_type(self):
        hits = [_hit("d1", "ACE inhibitors for hypertension.")]
        report = check_faithfulness("ACE inhibitors treat hypertension [1].", hits, embed_fn=fake_embed)
        assert isinstance(report, FaithfulnessReport)
