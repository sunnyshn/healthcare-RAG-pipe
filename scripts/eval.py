import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import (
    EMBEDDING_MODEL_NAME,
    EVAL_PATH,
    INDEX_PATH,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RERANK_CANDIDATES,
    RERANK_MODEL_NAME,
)
from healthcare_rag.faithfulness import DEFAULT_SUPPORT_THRESHOLD, check_faithfulness
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


def _log_mlflow(params: dict, metrics: dict, artifact_path, run_name: str) -> None:
    """Log one eval run to MLflow. No-op (with a note) if MLflow is unavailable."""
    try:
        import mlflow
    except ImportError:
        print("\n  [MLflow] not installed — skipping experiment logging (pip install mlflow).")
        return

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        # Only log real numbers; skip metrics that weren't computed this run.
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        if artifact_path is not None and Path(artifact_path).exists():
            mlflow.log_artifact(str(artifact_path))
    print(f"\n  [MLflow] Logged run '{run_name}' to {MLFLOW_TRACKING_URI}")
    print(f"  [MLflow] View with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


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
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable logging this run to MLflow",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional MLflow run name (defaults to a config-based name)",
    )
    args = parser.parse_args()

    retriever = HybridRetriever.load(INDEX_PATH)
    has_api_key = bool(os.getenv("OPENAI_API_KEY"))

    if args.rerank:
        print("Cross-encoder re-ranking: ENABLED")

    # Params and metrics accumulated across the eval parts for MLflow logging.
    params = {
        "rerank": args.rerank,
        "top_k": 3,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "rerank_model": RERANK_MODEL_NAME if args.rerank else "none",
        "rerank_candidates": RERANK_CANDIDATES if args.rerank else 0,
        "low_evidence_threshold": LOW_EVIDENCE_THRESHOLD,
        "support_threshold": DEFAULT_SUPPORT_THRESHOLD,
        "generation_mode": "llm" if has_api_key else "offline_heuristic",
        "corpus_size": int(len(retriever.tfidf.metadata)),
        "n_answerable": len(ANSWERABLE_SET),
        "n_unanswerable": len(UNANSWERABLE_SET),
    }
    metrics = {
        "recall_at_1": None,
        "recall_at_3": None,
        "abstention_rate": None,
        "citation_coverage": None,
        "citation_support": None,
        "invalid_citation_rate": None,
        "mean_support": None,
    }

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
    metrics["recall_at_1"] = retrieval_df["recall_at_1"].mean()
    metrics["recall_at_3"] = retrieval_df["recall_at_3"].mean()
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
            metrics["abstention_rate"] = rate
            print(
                f"\n  Abstention rate: {rate:.0%} "
                f"({int(abstention_df['abstained'].sum())}/{len(still_unanswerable)} questions)"
            )
            if rate < 1.0:
                print(
                    "  WARNING: Model gave confident answers to out-of-scope questions.\n"
                    "  Review 'answer_preview' in the CSV for potential hallucinations."
                )

    # ── Part 3: Citation faithfulness ────────────────────────────────────────
    _section("Part 3: Citation faithfulness (requires OPENAI_API_KEY)")
    print(f"  Support threshold : dense cosine >= {DEFAULT_SUPPORT_THRESHOLD}")
    print("  Coverage = claims that cite a source | Support = citations on-topic\n")

    faithfulness_rows = []
    if not has_api_key:
        print(
            "  Skipped: no OPENAI_API_KEY set. The offline extractive baseline does not\n"
            "  emit [n] citations, so faithfulness cannot be measured. Set the key to enable."
        )
    else:

        def _embed_fn(texts):
            encoder = retriever._encoder_model()
            return [np.asarray(v, dtype=np.float64) for v in encoder.embed(list(texts))]

        for item in ANSWERABLE_SET:
            question = item["question"]
            hits = retriever.search(question, k=3, rerank=args.rerank)
            answer = generate_answer(question, hits)
            report = check_faithfulness(answer, hits, embed_fn=_embed_fn)

            if report.abstained:
                print(f"  [ABSTAINED] {question[:66]}")
                continue

            faithfulness_rows.append(
                {
                    "question": question,
                    "citation_coverage": report.citation_coverage,
                    "citation_support": report.citation_support,
                    "invalid_citation_rate": report.invalid_citation_rate,
                    "mean_support": report.mean_support,
                }
            )
            flag = "" if report.invalid_citations == 0 else f"  !! {report.invalid_citations} invalid cite(s)"
            print(
                f"  coverage={report.citation_coverage:.0%} "
                f"support={report.citation_support:.0%} "
                f"mean_sim={report.mean_support:.2f}{flag}  | {question[:50]}"
            )

        if faithfulness_rows:
            faithfulness_df = pd.DataFrame(faithfulness_rows)
            metrics["citation_coverage"] = faithfulness_df["citation_coverage"].mean()
            metrics["citation_support"] = faithfulness_df["citation_support"].mean()
            metrics["invalid_citation_rate"] = faithfulness_df["invalid_citation_rate"].mean()
            metrics["mean_support"] = faithfulness_df["mean_support"].mean()
            print(
                f"\n  MEAN  coverage={faithfulness_df['citation_coverage'].mean():.0%} "
                f"support={faithfulness_df['citation_support'].mean():.0%} "
                f"invalid_rate={faithfulness_df['invalid_citation_rate'].mean():.0%} "
                f"mean_sim={faithfulness_df['mean_support'].mean():.2f}"
            )
            if faithfulness_df["citation_support"].mean() < 1.0:
                print(
                    "  WARNING: some cited passages are weakly related to their claims.\n"
                    "  Review answers for citations that don't actually support the statement."
                )

    # ── Save results ─────────────────────────────────────────────────────────
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Combine retrieval rows + abstention rows into one CSV.
    retrieval_df["eval_type"] = "answerable"
    retrieval_df["abstained"] = None
    retrieval_df["answer_preview"] = None

    frames = [retrieval_df]
    if abstention_rows:
        abstention_df["eval_type"] = "unanswerable"
        abstention_df["expected_keyword"] = None
        abstention_df["recall_at_1"] = None
        abstention_df["recall_at_3"] = None
        frames.append(abstention_df)
    if faithfulness_rows:
        faithfulness_df["eval_type"] = "faithfulness"
        frames.append(faithfulness_df)

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else retrieval_df

    combined.to_csv(EVAL_PATH, index=False)
    print(f"\nFull results saved to {EVAL_PATH}")

    # ── Experiment tracking ──────────────────────────────────────────────────
    if not args.no_mlflow:
        run_name = args.run_name or ("rerank" if args.rerank else "baseline")
        _log_mlflow(params, metrics, EVAL_PATH, run_name)


if __name__ == "__main__":
    main()

