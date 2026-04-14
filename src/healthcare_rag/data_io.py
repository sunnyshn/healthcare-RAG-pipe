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


def expand_chunks(
    df: pd.DataFrame, chunk_size: int = 450, overlap: int = 75
) -> pd.DataFrame:
    """Expand each document row into one row per text chunk.

    The original ``doc_id`` is preserved in a ``parent_doc_id`` column and
    each chunk receives a unique ``doc_id`` of the form ``{parent_id}_c{n}``.
    All other metadata columns are carried forward unchanged.
    """
    meta_cols = [c for c in df.columns if c != "text"]
    rows = []
    for _, record in df.iterrows():
        chunks = list(chunk_text(record["text"], chunk_size=chunk_size, overlap=overlap))
        parent_id = str(record["doc_id"])
        for n, chunk in enumerate(chunks):
            new_row = {col: record[col] for col in meta_cols}
            new_row["parent_doc_id"] = parent_id
            new_row["doc_id"] = f"{parent_id}_c{n}"
            new_row["text"] = chunk
            rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)

