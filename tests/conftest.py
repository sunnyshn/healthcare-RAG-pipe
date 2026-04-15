"""Shared fixtures for the healthcare-RAG-pipe test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the src package is importable regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Minimal corpus data — five lightweight records that cover the fields
# every module expects, including edge cases (missing specialty, duplicate).
# ---------------------------------------------------------------------------

SAMPLE_RECORDS = [
    {
        "doc_id": "t-001",
        "title": "Hypertension basics",
        "text": "ACE inhibitors are first-line therapy for hypertension in diabetic patients.",
        "year": 2021,
        "source": "Test Journal",
        "specialty": "cardiology",
    },
    {
        "doc_id": "t-002",
        "title": "Atrial fibrillation review",
        "text": "Anticoagulation therapy reduces stroke risk in atrial fibrillation patients.",
        "year": 2020,
        "source": "Test Journal",
        "specialty": "cardiology",
    },
    {
        "doc_id": "t-003",
        "title": "Depression treatment",
        "text": "SSRIs are commonly used as first-line treatment for major depressive disorder.",
        "year": 2019,
        "source": "Test Journal",
        "specialty": "psychiatry",
    },
    {
        "doc_id": "t-004",
        "title": "Diabetes management",
        "text": "Metformin is the preferred initial pharmacotherapy for type 2 diabetes.",
        "year": 2022,
        "source": "Test Journal",
        "specialty": "endocrinology",
    },
    {
        "doc_id": "t-001",
        "title": "Hypertension basics (duplicate)",
        "text": "ACE inhibitors are first-line therapy for hypertension in diabetic patients.",
        "year": 2021,
        "source": "Test Journal",
        "specialty": "cardiology",
    },
]


@pytest.fixture()
def sample_records():
    """Return raw sample records (list of dicts), including one duplicate."""
    return [r.copy() for r in SAMPLE_RECORDS]


@pytest.fixture()
def sample_df():
    """Return a DataFrame of the sample records (with the duplicate)."""
    return pd.DataFrame(SAMPLE_RECORDS)


@pytest.fixture()
def clean_df():
    """Return a deduplicated, normalized DataFrame (4 unique docs)."""
    from healthcare_rag.data_io import normalize_documents

    return normalize_documents(pd.DataFrame(SAMPLE_RECORDS))


@pytest.fixture()
def corpus_jsonl(tmp_path, sample_records):
    """Write sample records to a temporary JSONL file and return its path."""
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for rec in sample_records:
            f.write(json.dumps(rec) + "\n")
    return path
