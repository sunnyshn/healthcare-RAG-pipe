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


def main() -> None:
    retriever = HybridRetriever.load(INDEX_PATH)
    rows = []
    for item in EVAL_SET:
        hits = retriever.search(item["question"], k=3)
        joined = " ".join([h.text for h in hits]).lower()
        keyword_hit = item["expected_keyword"].lower() in joined
        rows.append(
            {
                "question": item["question"],
                "expected_keyword": item["expected_keyword"],
                "keyword_hit_at_3": int(keyword_hit),
            }
        )

    out = pd.DataFrame(rows)
    out["mean_keyword_hit"] = out["keyword_hit_at_3"].mean()
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(EVAL_PATH, index=False)
    print(f"Saved evaluation report to {EVAL_PATH}")
    print(out)


if __name__ == "__main__":
    main()

