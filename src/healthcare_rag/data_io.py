import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ["doc_id", "title", "text", "year", "source", "specialty"]


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def normalize_documents(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = out["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out = out.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)
    return out


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 75) -> Iterable[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

