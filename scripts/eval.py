import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import EVAL_PATH, INDEX_PATH
from healthcare_rag.generator import ABSTENTION_MARKER, LOW_EVIDENCE_THRESHOLD, generate_answer
from healthcare_rag.retriever import HybridRetriever

# ---------------------------------------------------------------------------
# Retrieval eval — questions that SHOULD be answered from the corpus.
# ---------------------------------------------------------------------------

ANSWERABLE_SET = [
    {
        "question": "first line treatment for stage 1 hypertension with diabetes",
        "expected_keyword": "ACE",
    },
    {
        "question": "how to reduce stroke risk in atrial fibrillation",
        "expected_keyword": "anticoagul",
    },
    {
        "question": "initial treatment for major depressive disorder adults",
        "expected_keyword": "SSRI",
    },
]

# ---------------------------------------------------------------------------
# Abstention eval — questions OUTSIDE the corpus; the model should abstain.
# ---------------------------------------------------------------------------

UNANSWERABLE_SET = [
    "What is the recommended chemotherapy regimen for acute myeloid leukemia?",
    "How should a pediatric asthma exacerbation be managed in the emergency department?",
    "What are the first-line treatments for Alzheimer's disease?",
    "What surgical options exist for lumbar spinal stenosis?",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _keyword_hit(hits, keyword: str, k: int) -> int:
    joined = " ".join(h.text for h in hits[:k]).lower()
    return int(keyword.lower() in joined)


def is_abstention(answer: str) -> bool:
    """Return True if the answer begins with the abstention marker."""
    return answer.strip().upper().startswith(ABSTENTION_MARKER.upper())


def _corpus_covers(
    question: str, retriever: HybridRetriever, k: int = 3, rerank: bool = False
) -> bool:
    """Return True if the corpus likely covers this question.

    Uses the same dense-similarity threshold as the generator's low-confidence
    warning, so the eval and the model stay in sync as the corpus grows.
    """
    hits = retriever.search(question, k=k, rerank=rerank)
    max_dense = max(
        (h.score_dense for h in hits if h.score_dense is not None),
        default=0.0,
    )
    return max_dense >= LOW_EVIDENCE_THRESHOLD


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and abstention quality.")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder re-ranking during retrieval",
    )
    args = parser.parse_args()

    retriever = HybridRetriever.load(INDEX_PATH)
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    if args.rerank:
        print("Cross-encoder re-ranking: ENABLED")

    # ── Part 1: Retrieval quality ────────────────────────────────────────────
    _section("Part 1: Retrieval quality (recall@k)")

    retrieval_rows = []
    for item in ANSWERABLE_SET:
        hits = retriever.search(item["question"], k=3, rerank=args.rerank)
        keyword = item["expected_keyword"]
        retrieval_rows.append(
            {
                "question": item["question"],
                "expected_keyword": keyword,
                "recall_at_1": _keyword_hit(hits, keyword, k=1),
                "recall_at_3": _keyword_hit(hits, keyword, k=3),
            }
        )

    retrieval_df = pd.DataFrame(retrieval_rows)
    retrieval_summary = pd.DataFrame(
        [
            {
                "question": "--- MEAN ---",
                "expected_keyword": "",
                "recall_at_1": retrieval_df["recall_at_1"].mean(),
                "recall_at_3": retrieval_df["recall_at_3"].mean(),
            }
        ]
    )
    print(
        pd.concat([retrieval_df, retrieval_summary], ignore_index=True).to_string(
            index=False,
            columns=["question", "expected_keyword", "recall_at_1", "recall_at_3"],
        )
    )

    # ── Part 2: Abstention rate ──────────────────────────────────────────────
    _section("Part 2: Abstention rate (requires OPENAI_API_KEY)")

    abstention_rows = []

    mode = "LLM (gpt-4.1-mini)" if has_api_key else "heuristic offline (dense similarity)"
    print(f"  Abstention marker : '{ABSTENTION_MARKER}'")
    print(f"  Coverage threshold: dense cosine >= {LOW_EVIDENCE_THRESHOLD}")
    print(f"  Generation mode   : {mode}\n")

    if True:

        # Filter out questions the corpus now covers so the metric stays valid.
        still_unanswerable = []
        covered = []
        for question in UNANSWERABLE_SET:
            if _corpus_covers(question, retriever, rerank=args.rerank):
                covered.append(question)
            else:
                still_unanswerable.append(question)

        if covered:
            print(
                f"  NOTE: {len(covered)} question(s) skipped — corpus now covers these topics.\n"
                "  Add new out-of-scope questions to UNANSWERABLE_SET to keep this eval meaningful:\n"
            )
            for q in covered:
                print(f"    - {q}")
            print()

        if not still_unanswerable:
            print(
                "  All unanswerable questions are now covered by the corpus.\n"
                "  Add new out-of-scope questions to UNANSWERABLE_SET in eval.py."
            )
        else:
            print(f"  Testing {len(still_unanswerable)} out-of-scope question(s)...\n")
            for question in still_unanswerable:
                hits = retriever.search(question, k=3, rerank=args.rerank)
                answer = generate_answer(question, hits)
                abstained = is_abstention(answer)
                abstention_rows.append(
                    {
                        "question": question,
                        "abstained": int(abstained),
                        "answer_preview": answer[:120].replace("\n", " "),
                    }
                )
                status = "ABSTAINED" if abstained else "ANSWERED (check for hallucination)"
                print(f"  [{status}] {question[:70]}")

            abstention_df = pd.DataFrame(abstention_rows)
            rate = abstention_df["abstained"].mean()
            print(
                f"\n  Abstention rate: {rate:.0%} "
                f"({int(abstention_df['abstained'].sum())}/{len(still_unanswerable)} questions)"
            )
            if rate < 1.0:
                print(
                    "  WARNING: Model gave confident answers to out-of-scope questions.\n"
                    "  Review 'answer_preview' in the CSV for potential hallucinations."
                )

    # ── Save results ─────────────────────────────────────────────────────────
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Combine retrieval rows + abstention rows into one CSV.
    retrieval_df["eval_type"] = "answerable"
    retrieval_df["abstained"] = None
    retrieval_df["answer_preview"] = None

    if abstention_rows:
        abstention_df["eval_type"] = "unanswerable"
        abstention_df["expected_keyword"] = None
        abstention_df["recall_at_1"] = None
        abstention_df["recall_at_3"] = None
        combined = pd.concat([retrieval_df, abstention_df], ignore_index=True)
    else:
        combined = retrieval_df

    combined.to_csv(EVAL_PATH, index=False)
    print(f"\nFull results saved to {EVAL_PATH}")


if __name__ == "__main__":
    main()

