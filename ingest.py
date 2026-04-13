import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import PROCESSED_PATH, RAW_PATH
from healthcare_rag.data_io import normalize_documents, read_jsonl


def main() -> None:
    df = read_jsonl(RAW_PATH)
    cleaned = normalize_documents(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    print(f"Wrote {len(cleaned)} documents to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()

