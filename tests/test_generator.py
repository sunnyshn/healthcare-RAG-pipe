"""Tests for healthcare_rag.generator — context building, abstention, offline fallback."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from healthcare_rag.generator import (
    ABSTENTION_MARKER,
    LOW_EVIDENCE_THRESHOLD,
    _build_context,
    generate_answer,
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
    """Tests that run without an OPENAI_API_KEY — exercises the extractive fallback."""

    @pytest.fixture(autouse=True)
    def _clear_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def test_abstains_on_low_similarity(self):
        hits = [_make_hit(score_dense=LOW_EVIDENCE_THRESHOLD - 0.1)]
        answer = generate_answer("irrelevant question", hits)
        assert answer.upper().startswith(ABSTENTION_MARKER.upper())

    def test_extracts_on_high_similarity(self):
        hits = [_make_hit(score_dense=LOW_EVIDENCE_THRESHOLD + 0.05)]
        answer = generate_answer("relevant question", hits)
        assert not answer.upper().startswith(ABSTENTION_MARKER.upper())
        assert "Top evidence" in answer

    def test_abstention_mentions_threshold(self):
        hits = [_make_hit(score_dense=0.50)]
        answer = generate_answer("off-topic question", hits)
        assert str(LOW_EVIDENCE_THRESHOLD) in answer

    def test_extractive_baseline_includes_titles(self):
        hits = [
            _make_hit(title="Study Alpha", score_dense=0.90),
            _make_hit(title="Study Beta", doc_id="d2", score_dense=0.88),
        ]
        answer = generate_answer("question", hits)
        assert "Study Alpha" in answer
        assert "Study Beta" in answer


# ── generate_answer (with mocked OpenAI) ─────────────────────────────────────


class TestGenerateAnswerLLM:
    def test_calls_openai_with_system_and_user_messages(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ACE inhibitors are recommended [1]."

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        with patch(
            "healthcare_rag.generator.openai.OpenAI",
            create=True,
            return_value=mock_client_instance,
        ):
            hits = [_make_hit(score_dense=0.90)]
            answer = generate_answer("hypertension treatment?", hits)

        call_kwargs = mock_client_instance.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "ACE inhibitors" in answer

    def test_system_prompt_mentions_abstention_marker(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer."

        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response

        with patch(
            "healthcare_rag.generator.openai.OpenAI",
            create=True,
            return_value=mock_client_instance,
        ):
            generate_answer("test?", [_make_hit()])

        call_kwargs = mock_client_instance.chat.completions.create.call_args.kwargs
        system_msg = call_kwargs["messages"][0]["content"]
        assert ABSTENTION_MARKER in system_msg


# ── stream_answer ────────────────────────────────────────────────────────────


def _chunk(content):
    """Build a fake streaming chunk with a single choice delta."""
    ch = MagicMock()
    ch.choices = [MagicMock()]
    ch.choices[0].delta.content = content
    return ch


class TestStreamAnswerOffline:
    @pytest.fixture(autouse=True)
    def _clear_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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


class TestStreamAnswerLLM:
    def test_streams_and_joins_deltas(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        fake_stream = [
            _chunk("ACE inhibitors "),
            _chunk("are recommended "),
            _chunk(None),  # keep-alive / role chunk with no text
            _chunk("[1]."),
        ]
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = fake_stream

        with patch(
            "healthcare_rag.generator.openai.OpenAI",
            create=True,
            return_value=mock_client_instance,
        ):
            full = "".join(stream_answer("hypertension?", [_make_hit(score_dense=0.9)]))

        assert full == "ACE inhibitors are recommended [1]."
        call_kwargs = mock_client_instance.chat.completions.create.call_args.kwargs
        assert call_kwargs["stream"] is True

    def test_handles_chunk_without_choices(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-fake")

        empty = MagicMock()
        empty.choices = []
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = [empty, _chunk("ok")]

        with patch(
            "healthcare_rag.generator.openai.OpenAI",
            create=True,
            return_value=mock_client_instance,
        ):
            full = "".join(stream_answer("q", [_make_hit(score_dense=0.9)]))

        assert full == "ok"
