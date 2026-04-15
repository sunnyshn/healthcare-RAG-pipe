"""Tests for eval.py helper functions — keyword hit detection, abstention checks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from eval import _keyword_hit, is_abstention
from healthcare_rag.generator import ABSTENTION_MARKER
from healthcare_rag.retriever import SearchResult


# ── helpers ──────────────────────────────────────────────────────────────────


def _hit(text: str) -> SearchResult:
    return SearchResult(
        doc_id="x",
        title="t",
        text=text,
        score=0.5,
        source="s",
        year=2023,
        specialty="general",
    )


# ── _keyword_hit ─────────────────────────────────────────────────────────────


class TestKeywordHit:
    def test_found_in_top_1(self):
        hits = [_hit("ACE inhibitors are effective"), _hit("unrelated text")]
        assert _keyword_hit(hits, "ACE", k=1) == 1

    def test_not_in_top_1_but_in_top_3(self):
        hits = [_hit("unrelated"), _hit("unrelated"), _hit("ACE inhibitors work")]
        assert _keyword_hit(hits, "ACE", k=1) == 0
        assert _keyword_hit(hits, "ACE", k=3) == 1

    def test_case_insensitive(self):
        hits = [_hit("ace inhibitors")]
        assert _keyword_hit(hits, "ACE", k=1) == 1

    def test_missing_keyword(self):
        hits = [_hit("nothing relevant here")]
        assert _keyword_hit(hits, "ACE", k=1) == 0

    def test_partial_keyword_match(self):
        hits = [_hit("anticoagulation therapy recommended")]
        assert _keyword_hit(hits, "anticoagul", k=1) == 1

    def test_empty_hits(self):
        assert _keyword_hit([], "ACE", k=3) == 0


# ── is_abstention ────────────────────────────────────────────────────────────


class TestIsAbstention:
    def test_detects_abstention_marker(self):
        answer = f"{ABSTENTION_MARKER} not enough evidence for this question."
        assert is_abstention(answer) is True

    def test_case_insensitive(self):
        answer = ABSTENTION_MARKER.lower() + " some explanation"
        assert is_abstention(answer) is True

    def test_normal_answer_not_abstention(self):
        answer = "ACE inhibitors are recommended for hypertension [1]."
        assert is_abstention(answer) is False

    def test_marker_in_middle_not_counted(self):
        answer = f"The model should say {ABSTENTION_MARKER} when unsure."
        assert is_abstention(answer) is False

    def test_whitespace_stripped(self):
        answer = f"   \n{ABSTENTION_MARKER} explanation"
        assert is_abstention(answer) is True
