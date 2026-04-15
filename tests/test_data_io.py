"""Tests for healthcare_rag.data_io — JSONL reading, normalization, chunking."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from healthcare_rag.data_io import (
    REQUIRED_COLUMNS,
    chunk_text,
    expand_chunks,
    normalize_documents,
    read_jsonl,
)


# ── read_jsonl ───────────────────────────────────────────────────────────────


class TestReadJsonl:
    def test_reads_all_rows(self, corpus_jsonl):
        df = read_jsonl(corpus_jsonl)
        assert len(df) == 5  # includes the duplicate

    def test_has_required_columns(self, corpus_jsonl):
        df = read_jsonl(corpus_jsonl)
        for col in REQUIRED_COLUMNS:
            assert col in df.columns

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "sparse.jsonl"
        rec = {
            "doc_id": "x",
            "title": "t",
            "text": "hello",
            "year": 2024,
            "source": "s",
            "specialty": "general",
        }
        with path.open("w") as f:
            f.write("\n")
            f.write(json.dumps(rec) + "\n")
            f.write("\n\n")
        df = read_jsonl(path)
        assert len(df) == 1

    def test_raises_on_missing_column(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"doc_id": "x", "title": "t"}) + "\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            read_jsonl(path)


# ── normalize_documents ──────────────────────────────────────────────────────


class TestNormalizeDocuments:
    def test_deduplicates_by_doc_id(self, sample_df):
        out = normalize_documents(sample_df)
        assert out["doc_id"].is_unique

    def test_removes_duplicate(self, sample_df):
        out = normalize_documents(sample_df)
        assert len(out) == 4

    def test_collapses_whitespace(self):
        df = pd.DataFrame(
            [
                {
                    "doc_id": "ws",
                    "title": "t",
                    "text": "  lots   of   spaces  \n\tnewlines  ",
                    "year": 2024,
                    "source": "s",
                    "specialty": "general",
                }
            ]
        )
        out = normalize_documents(df)
        assert "  " not in out.iloc[0]["text"]
        assert "\n" not in out.iloc[0]["text"]
        assert not out.iloc[0]["text"].startswith(" ")

    def test_resets_index(self, sample_df):
        out = normalize_documents(sample_df)
        assert list(out.index) == list(range(len(out)))


# ── chunk_text ───────────────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "one two three four"
        chunks = list(chunk_text(text, chunk_size=10, overlap=2))
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_long_text(self):
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        chunks = list(chunk_text(text, chunk_size=8, overlap=2))
        assert len(chunks) > 1

    def test_overlap_creates_shared_words(self):
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        chunks = list(chunk_text(text, chunk_size=10, overlap=3))
        if len(chunks) >= 2:
            words_a = set(chunks[0].split())
            words_b = set(chunks[1].split())
            assert len(words_a & words_b) >= 1

    def test_empty_text_returns_empty(self):
        assert list(chunk_text("")) == []
        assert list(chunk_text("   ")) == []

    def test_all_words_covered(self):
        words = [f"w{i}" for i in range(50)]
        text = " ".join(words)
        chunks = list(chunk_text(text, chunk_size=15, overlap=4))
        recovered = set()
        for c in chunks:
            recovered.update(c.split())
        assert recovered == set(words)


# ── expand_chunks ────────────────────────────────────────────────────────────


class TestExpandChunks:
    def test_preserves_parent_doc_id(self, clean_df):
        expanded = expand_chunks(clean_df, chunk_size=5, overlap=1)
        assert "parent_doc_id" in expanded.columns
        assert set(expanded["parent_doc_id"]) == set(clean_df["doc_id"])

    def test_chunk_ids_are_unique(self, clean_df):
        expanded = expand_chunks(clean_df, chunk_size=5, overlap=1)
        assert expanded["doc_id"].is_unique

    def test_chunk_id_format(self, clean_df):
        expanded = expand_chunks(clean_df, chunk_size=5, overlap=1)
        for _, row in expanded.iterrows():
            assert row["doc_id"].startswith(row["parent_doc_id"] + "_c")

    def test_metadata_carried_forward(self, clean_df):
        expanded = expand_chunks(clean_df, chunk_size=5, overlap=1)
        for col in ("title", "year", "source", "specialty"):
            assert col in expanded.columns

    def test_large_chunk_size_keeps_one_per_doc(self, clean_df):
        expanded = expand_chunks(clean_df, chunk_size=9999, overlap=0)
        assert len(expanded) == len(clean_df)
