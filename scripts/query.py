import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import INDEX_PATH
from healthcare_rag.generator import generate_answer
from healthcare_rag.retriever import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI query for the Healthcare Evidence Assistant.")
    parser.add_argument("--question", required=True, help="Clinical style question to answer")
    parser.add_argument("--top_k", type=int, default=3, metavar="K")
    parser.add_argument(
        "--specialty",
        default=None,
        metavar="SPECIALTY",
        help="Filter results to a specific specialty (case-insensitive, e.g. cardiology)",
    )
    parser.add_argument(
        "--year_min",
        type=int,
        default=None,
        metavar="YEAR",
        help="Earliest publication year to include",
    )
    parser.add_argument(
        "--year_max",
        type=int,
        default=None,
        metavar="YEAR",
        help="Latest publication year to include",
    )
    args = parser.parse_args()

    year_range: Optional[Tuple[int, int]] = None
    if args.year_min is not None or args.year_max is not None:
        lo = args.year_min if args.year_min is not None else 0
        hi = args.year_max if args.year_max is not None else 9999
        year_range = (lo, hi)

    retriever = HybridRetriever.load(INDEX_PATH)
    hits = retriever.search(
        args.question,
        k=args.top_k,
        specialty=args.specialty,
        year_range=year_range,
    )

    if not hits:
        print("No documents matched the specified filters.")
        return

    answer = generate_answer(args.question, hits)

    print("\n=== Answer ===")
    print(answer)
    print("\n=== Retrieved Sources ===")
    for i, h in enumerate(hits, start=1):
        extra = ""
        if h.score_sparse is not None and h.score_dense is not None:
            extra = f" | cos_sparse={h.score_sparse:.3f} cos_dense={h.score_dense:.3f}"
        specialty_tag = f" | specialty={h.specialty}" if h.specialty else ""
        print(
            f"[{i}] {h.title} | {h.source} ({h.year}){specialty_tag}"
            f" | rrf={h.score:.4f}{extra} | doc_id={h.doc_id}"
        )


if __name__ == "__main__":
    main()

