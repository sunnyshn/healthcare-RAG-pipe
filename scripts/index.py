import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import INDEX_PATH, PROCESSED_PATH
from healthcare_rag.retriever import HybridRetriever


def main() -> None:
    docs = pd.read_csv(PROCESSED_PATH)
    retriever = HybridRetriever.build(docs)
    retriever.save(INDEX_PATH)
    print(f"Saved hybrid index (TF-IDF + dense, RRF fusion) to {INDEX_PATH}")


if __name__ == "__main__":
    main()

