import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag.config import INDEX_PATH
from healthcare_rag.generator import generate_answer
from healthcare_rag.retriever import HybridRetriever

st.set_page_config(page_title="Healthcare Evidence Assistant", layout="wide")
st.title("Healthcare Evidence Assistant")
st.caption("Educational decision support demo. Not for diagnosis.")

question = st.text_input(
    "Ask a clinical question",
    value="What is first-line treatment for stage 1 hypertension in adults with diabetes?",
)
top_k = st.slider("Top-k retrieval", min_value=1, max_value=6, value=3)

if st.button("Run"):
    retriever = HybridRetriever.load(INDEX_PATH)
    hits = retriever.search(question, k=top_k)
    answer = generate_answer(question, hits)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved evidence")
    for i, h in enumerate(hits, start=1):
        with st.expander(f"[{i}] {h.title} ({h.year}) | RRF={h.score:.4f}"):
            st.write(f"**source:** {h.source}")
            st.write(f"**doc_id:** {h.doc_id}")
            if h.score_sparse is not None and h.score_dense is not None:
                st.caption(
                    f"cosine TF-IDF={h.score_sparse:.3f} · cosine dense={h.score_dense:.3f}"
                )
            st.write(h.text)

