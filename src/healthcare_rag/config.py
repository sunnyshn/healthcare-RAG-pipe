from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "sample_corpus.jsonl"
PROCESSED_PATH = DATA_DIR / "processed" / "documents.csv"
INDEX_PATH = DATA_DIR / "index" / "hybrid_index.pkl"
# Legacy TF-IDF-only index (older runs); `scripts/index.py` now writes hybrid_index.pkl
LEGACY_TFIDF_INDEX_PATH = DATA_DIR / "index" / "tfidf_index.pkl"
EVAL_PATH = DATA_DIR / "processed" / "eval_results.csv"

