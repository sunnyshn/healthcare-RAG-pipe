import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import INDEX_PATH
from healthcare_rag.generator import generate_answer
from healthcare_rag.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True, help="Clinical style question to answer")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    retriever = HybridRetriever.load(INDEX_PATH)
    hits = retriever.search(args.question, k=args.top_k)
    answer = generate_answer(args.question, hits)

    print("\n=== Answer ===")
    print(answer)
    print("\n=== Retrieved Sources ===")
    for i, h in enumerate(hits, start=1):
        extra = ""
        if h.score_sparse is not None and h.score_dense is not None:
            extra = f" | cos_sparse={h.score_sparse:.3f} cos_dense={h.score_dense:.3f}"
        print(
            f"[{i}] {h.title} | {h.source} ({h.year}) | rrf={h.score:.4f}{extra} | doc_id={h.doc_id}"
        )


if __name__ == "__main__":
    main()

