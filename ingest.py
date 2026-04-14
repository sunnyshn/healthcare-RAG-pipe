import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import PROCESSED_PATH, RAW_PATH
from healthcare_rag.data_io import expand_chunks, normalize_documents, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and clean the raw corpus.")
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Expand documents into overlapping text chunks before writing.",
    )
    parser.add_argument("--chunk-size", type=int, default=450, metavar="N")
    parser.add_argument("--overlap", type=int, default=75, metavar="N")
    args = parser.parse_args()

    df = read_jsonl(RAW_PATH)
    cleaned = normalize_documents(df)

    if args.chunk:
        cleaned = expand_chunks(cleaned, chunk_size=args.chunk_size, overlap=args.overlap)
        print(f"Chunking enabled: {len(df)} docs -> {len(cleaned)} chunks")

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_PATH, index=False)
    print(f"Wrote {len(cleaned)} rows to {PROCESSED_PATH}")


if __name__ == "__main__":
    main()

