"""Tests for healthcare_rag.generator — context building, abstention, offline fallback."""

from __future__ import annotations

import pytest

from healthcare_rag.generator import (
    ABSTENTION_MARKER,
    LOW_EVIDENCE_THRESHOLD,
    _as_abstention,
    _build_context,
    for_visitors,
    generate_answer,
    is_abstention,
    stream_answer,
)
from healthcare_rag.retriever import SearchResult


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_hit(
    doc_id="d1",
    title="Test Title",
    text="Some evidence text.",
    year=2023,
    source="Journal",
    specialty="general",
    score=0.5,
    score_sparse=0.4,
    score_dense=0.85,
):
    return SearchResult(
        doc_id=doc_id,
        title=title,
        text=text,
        score=score,
        source=source,
        year=year,
        specialty=specialty,
        score_sparse=score_sparse,
        score_dense=score_dense,
    )


# ── _build_context ───────────────────────────────────────────────────────────


class TestBuildContext:
    def test_includes_numbered_passages(self):
        hits = [_make_hit(doc_id="d1"), _make_hit(doc_id="d2")]
        ctx = _build_context(hits)
        assert "[1]" in ctx
        assert "[2]" in ctx

    def test_includes_title_and_year(self):
        hit = _make_hit(title="My Study", year=2021)
        ctx = _build_context([hit])
        assert "My Study" in ctx
        assert "2021" in ctx

    def test_low_evidence_warning_when_below_threshold(self):
        hit = _make_hit(score_dense=LOW_EVIDENCE_THRESHOLD - 0.05)
        ctx = _build_context([hit])
        assert "NOTE:" in ctx or "low" in ctx.lower()

    def test_no_warning_when_above_threshold(self):
        hit = _make_hit(score_dense=LOW_EVIDENCE_THRESHOLD + 0.05)
        ctx = _build_context([hit])
        assert "NOTE:" not in ctx

    def test_text_truncated_at_1300_chars(self):
        long_text = "x" * 2000
        hit = _make_hit(text=long_text)
        ctx = _build_context([hit])
        assert "x" * 1301 not in ctx


# ── generate_answer (offline fallback, no API key) ───────────────────────────


class TestGenerateAnswerOffline:
    """Extractive list used when the local model file is not configured."""

    @pytest.fixture(autouse=True)
    def _no_local_model(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_PATH", raising=False)

    def test_abstains_on_low_similarity(self):
        hits = [_make_hit(score_dense=LOW_EVIDENCE_THRESHOLD - 0.1)]
        answer = generate_answer("irrelevant question", hits)
        assert answer.upper().startswith(ABSTENTION_MARKER.upper())

    def test_extracts_on_high_similarity(self):
        hits = [_make_hit(score_dense=LOW_EVIDENCE_THRESHOLD + 0.05)]
        answer = generate_answer("relevant question", hits)
        assert not answer.upper().startswith(ABSTENTION_MARKER.upper())
        assert "[1]" in answer

    def test_abstention_is_visitor_facing(self):
        hits = [_make_hit(score_dense=0.50)]
        answer = generate_answer("off-topic question", hits)
        assert "close match" in answer
        assert "API" not in answer
        assert "threshold" not in answer.lower()

    def test_extractive_baseline_includes_titles(self):
        hits = [
            _make_hit(title="Study Alpha", score_dense=0.90),
            _make_hit(title="Study Beta", doc_id="d2", score_dense=0.88),
        ]
        answer = generate_answer("question", hits)
        assert "Study Alpha" in answer
        assert "Study Beta" in answer

    def test_cites_close_passages_and_skips_a_weak_one(self):
        hits = [
            _make_hit(title="Study Alpha", score_dense=0.90, text="ACE inhibitors help."),
            _make_hit(
                title="Study Beta",
                doc_id="d2",
                score_dense=0.86,
                text="ARBs are an alternative.",
            ),
            _make_hit(
                title="Off Topic",
                doc_id="d3",
                score_dense=0.70,
                text="Unrelated aspergillosis findings.",
            ),
        ]
        answer = generate_answer("question", hits)
        assert "[1]" in answer
        assert "[2]" in answer
        assert "[3]" not in answer
        assert "Off Topic" not in answer

    def test_visitor_view_hides_the_eval_token(self):
        hits = [_make_hit(score_dense=0.50)]
        shown = for_visitors(generate_answer("off-topic question", hits))
        assert ABSTENTION_MARKER not in shown
        assert "close match" in shown
        assert "won't guess" in shown

    def test_leaked_marker_is_a_no_close_match(self):
        leaked = (
            "These articles do not contain relevant information "
            f"[{ABSTENTION_MARKER}]"
        )
        assert is_abstention(leaked) is True
        shown = for_visitors(leaked)
        assert ABSTENTION_MARKER not in shown
        assert "relevant information" not in shown
        assert "close match" in shown
        cleaned = _as_abstention(leaked)
        assert cleaned.upper().startswith(ABSTENTION_MARKER.upper())
        assert "relevant information" not in cleaned


# ── stream_answer ────────────────────────────────────────────────────────────


class TestStreamAnswerOffline:
    @pytest.fixture(autouse=True)
    def _no_local_model(self, monkeypatch):
        monkeypatch.delenv("LOCAL_LLM_PATH", raising=False)

    def test_yields_offline_answer_in_one_chunk(self):
        hits = [_make_hit(score_dense=0.90)]
        chunks = list(stream_answer("question", hits))
        assert len(chunks) == 1
        # Streamed text should match the non-streaming offline answer.
        assert chunks[0] == generate_answer("question", hits)

    def test_offline_abstains_on_low_similarity(self):
        hits = [_make_hit(score_dense=LOW_EVIDENCE_THRESHOLD - 0.1)]
        full = "".join(stream_answer("irrelevant", hits))
        assert full.upper().startswith(ABSTENTION_MARKER.upper())


class TestLocalRefusal:
    def test_mixed_refusal_is_rewritten(self, monkeypatch):
        monkeypatch.setattr(
            "healthcare_rag.generator.local_generator_ready", lambda: True
        )

        class _Fake:
            def create_chat_completion(self, **_kwargs):
                text = (
                    "These articles do not contain relevant information "
                    f"[{ABSTENTION_MARKER}]"
                )
                yield {"choices": [{"delta": {"content": text}}]}

        monkeypatch.setattr("healthcare_rag.generator._local_llm", lambda: _Fake())
        hits = [_make_hit(score_dense=0.90)]
        answer = generate_answer("unrelated question", hits)
        assert answer.upper().startswith(ABSTENTION_MARKER.upper())
        assert "relevant information" not in answer
        shown = for_visitors(answer)
        assert ABSTENTION_MARKER not in shown
        assert "close match" in shown
