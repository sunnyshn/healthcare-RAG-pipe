import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import EVAL_PATH, INDEX_PATH
from healthcare_rag.retriever import HybridRetriever

EVAL_SET = [
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


def _keyword_hit(hits, keyword: str, k: int) -> int:
    joined = " ".join(h.text for h in hits[:k]).lower()
    return int(keyword.lower() in joined)


def main() -> None:
    retriever = HybridRetriever.load(INDEX_PATH)
    rows = []
    for item in EVAL_SET:
        hits = retriever.search(item["question"], k=3)
        keyword = item["expected_keyword"]
        rows.append(
            {
                "question": item["question"],
                "expected_keyword": keyword,
                "recall_at_1": _keyword_hit(hits, keyword, k=1),
                "recall_at_3": _keyword_hit(hits, keyword, k=3),
            }
        )

    out = pd.DataFrame(rows)

    summary = pd.DataFrame(
        [
            {
                "question": "--- MEAN ---",
                "expected_keyword": "",
                "recall_at_1": out["recall_at_1"].mean(),
                "recall_at_3": out["recall_at_3"].mean(),
            }
        ]
    )
    display = pd.concat([out, summary], ignore_index=True)

    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(EVAL_PATH, index=False)

    print(f"\nEvaluation results saved to {EVAL_PATH}")
    print(
        display.to_string(
            index=False,
            columns=["question", "expected_keyword", "recall_at_1", "recall_at_3"],
        )
    )


if __name__ == "__main__":
    main()

