import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "sample_corpus.jsonl"
PROCESSED_PATH = DATA_DIR / "processed" / "documents.csv"
INDEX_PATH = DATA_DIR / "index" / "hybrid_index.pkl"
# Legacy TF-IDF-only index (older runs); `index.py` now writes hybrid_index.pkl
LEGACY_TFIDF_INDEX_PATH = DATA_DIR / "index" / "tfidf_index.pkl"
EVAL_PATH = DATA_DIR / "processed" / "eval_results.csv"

# ---------------------------------------------------------------------------
# Retrieval / reranking
# ---------------------------------------------------------------------------

# Dense embedding model used to build the hybrid index (FastEmbed / ONNX).
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Cross-encoder used for optional second-stage re-ranking (FastEmbed / ONNX).
RERANK_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"

# How many stage-1 candidates to pull before cross-encoder re-ranking.
# Larger = better recall into the reranker, but slower.
RERANK_CANDIDATES = 20

# ---------------------------------------------------------------------------
# Experiment tracking (MLflow)
# ---------------------------------------------------------------------------

# Where MLflow writes runs. Defaults to a local ./mlruns directory at the repo
# root; override with the MLFLOW_TRACKING_URI env var (e.g. a remote server).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri())
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "healthcare-rag-eval")

