import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from healthcare_rag.config import INDEX_PATH, PROCESSED_PATH, RAW_PATH
from healthcare_rag.data_io import normalize_documents, read_jsonl
from healthcare_rag.generator import generate_answer
from healthcare_rag.retriever import HybridRetriever

import fetch_pubmed as fp

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Healthcare Evidence Assistant", layout="wide")
st.title("Healthcare Evidence Assistant")
st.caption("Educational decision support demo. Not for diagnosis.")

# ---------------------------------------------------------------------------
# Cached retriever — call load_retriever.clear() after an index rebuild
# ---------------------------------------------------------------------------


@st.cache_resource
def load_retriever() -> HybridRetriever:
    return HybridRetriever.load(INDEX_PATH)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ask, tab_corpus = st.tabs(["Ask", "Manage Corpus"])

# ===========================  ASK TAB  =====================================

with tab_ask:
    if not INDEX_PATH.exists():
        st.warning(
            "No search index found. Go to the **Manage Corpus** tab and click "
            "**Rebuild Index** to build one."
        )
        st.stop()

    retriever = load_retriever()

    _specialties = sorted(
        retriever.tfidf.metadata["specialty"].dropna().unique().tolist()
    )
    _specialty_options = ["(all)"] + _specialties

    col_q, col_k = st.columns([4, 1])
    with col_q:
        question = st.text_input(
            "Ask a clinical question",
            value="What is first-line treatment for stage 1 hypertension in adults with diabetes?",
        )
    with col_k:
        top_k = st.slider("Top-k", min_value=1, max_value=10, value=3)

    col_spec, col_yr = st.columns(2)
    with col_spec:
        specialty_sel = st.selectbox("Filter by specialty", options=_specialty_options)
    with col_yr:
        all_years = sorted(
            retriever.tfidf.metadata["year"].dropna().astype(int).unique().tolist()
        )
        if len(all_years) >= 2:
            year_range = st.slider(
                "Year range",
                min_value=int(all_years[0]),
                max_value=int(all_years[-1]),
                value=(int(all_years[0]), int(all_years[-1])),
            )
        else:
            year_range = None

    if st.button("Run", type="primary"):
        specialty_filter = None if specialty_sel == "(all)" else specialty_sel
        yr_filter = tuple(year_range) if year_range else None

        with st.spinner("Retrieving evidence..."):
            hits = retriever.search(
                question, k=top_k, specialty=specialty_filter, year_range=yr_filter
            )

        if not hits:
            st.warning(
                "No documents matched the selected filters. Try broadening your criteria."
            )
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(question, hits)

            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved evidence")
            for i, h in enumerate(hits, start=1):
                label = f"[{i}] {h.title} ({h.year})"
                if h.specialty:
                    label += f" · {h.specialty}"
                label += f" | RRF={h.score:.4f}"
                with st.expander(label):
                    st.write(f"**source:** {h.source}")
                    st.write(f"**doc_id:** {h.doc_id}")
                    if h.score_sparse is not None and h.score_dense is not None:
                        st.caption(
                            f"cosine TF-IDF={h.score_sparse:.3f} "
                            f"· cosine dense={h.score_dense:.3f}"
                        )
                    st.write(h.text)


# =======================  MANAGE CORPUS TAB  ================================

with tab_corpus:

    # ── Corpus stats ────────────────────────────────────────────────────────
    st.subheader("Current corpus")

    if RAW_PATH.exists():
        existing_ids = fp._load_existing_ids(RAW_PATH)
        total_docs = len(existing_ids)

        raw_df = pd.read_json(RAW_PATH, lines=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total documents", total_docs)
        col_b.metric(
            "Specialties",
            raw_df["specialty"].nunique() if "specialty" in raw_df.columns else "—",
        )
        col_c.metric(
            "Year range",
            f"{int(raw_df['year'].min())} – {int(raw_df['year'].max())}"
            if "year" in raw_df.columns and not raw_df["year"].isna().all()
            else "—",
        )

        with st.expander("Specialty breakdown"):
            st.dataframe(
                raw_df["specialty"].value_counts().rename_axis("specialty").reset_index(name="count"),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No corpus file found yet. Fetch some documents below to get started.")
        existing_ids = set()

    st.divider()

    # ── Fetch from PubMed ────────────────────────────────────────────────────
    st.subheader("Fetch from PubMed")

    with st.form("pubmed_form"):
        f_col1, f_col2 = st.columns([3, 1])
        with f_col1:
            pm_query = st.text_input(
                "Search query",
                placeholder='e.g. "sepsis antibiotic treatment guidelines"',
            )
        with f_col2:
            pm_max = st.number_input("Max results", min_value=1, max_value=500, value=50, step=10)

        pm_specialty = st.text_input(
            "Specialty label",
            placeholder="e.g. cardiology, oncology, psychiatry",
        )
        fetch_submitted = st.form_submit_button("Fetch abstracts", type="primary")

    if fetch_submitted:
        if not pm_query.strip():
            st.error("Please enter a search query.")
        else:
            specialty_label = pm_specialty.strip() or "general"

            key = fp._api_key()
            rate_msg = (
                "NCBI API key detected (10 req/s)."
                if key
                else "No NCBI_API_KEY — using public rate limit (3 req/s)."
            )
            st.caption(rate_msg)

            status = st.status(
                f'Searching PubMed for "{pm_query}"...', expanded=True
            )

            with status:
                try:
                    pmids = fp.esearch(pm_query, int(pm_max))
                    st.write(f"Found **{len(pmids)}** matching PMIDs.")

                    if not pmids:
                        status.update(label="No results found.", state="error")
                    else:
                        delay = fp._rate_delay()
                        batches = [
                            pmids[i : i + fp.FETCH_BATCH]
                            for i in range(0, len(pmids), fp.FETCH_BATCH)
                        ]
                        docs = []
                        prog = st.progress(0, text="Fetching abstracts...")
                        for n, batch in enumerate(batches, start=1):
                            xml_text = fp._efetch_xml(batch)
                            docs.extend(fp._parse_articles(xml_text))
                            prog.progress(
                                n / len(batches),
                                text=f"Batch {n}/{len(batches)} — {len(docs)} abstracts so far",
                            )
                            time.sleep(delay)
                        prog.empty()

                        new_docs = [d for d in docs if d["doc_id"] not in existing_ids]
                        skipped = len(docs) - len(new_docs)

                        if skipped:
                            st.write(f"Skipped **{skipped}** duplicate(s) already in corpus.")

                        if new_docs:
                            RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
                            with RAW_PATH.open("a", encoding="utf-8") as f:
                                import json
                                for doc in new_docs:
                                    doc["specialty"] = specialty_label
                                    f.write(json.dumps(doc) + "\n")
                            st.write(
                                f"Appended **{len(new_docs)}** new document(s) "
                                f"to corpus as specialty **{specialty_label}**."
                            )
                            status.update(
                                label=f"Fetched {len(new_docs)} new documents. Rebuild the index below.",
                                state="complete",
                            )
                        else:
                            status.update(
                                label="No new documents — all results already in corpus.",
                                state="complete",
                            )

                except Exception as exc:
                    status.update(label=f"Fetch failed: {exc}", state="error")
                    st.exception(exc)

    st.divider()

    # ── Rebuild index ────────────────────────────────────────────────────────
    st.subheader("Rebuild index")
    st.caption(
        "Run this after fetching new documents. "
        "Ingest cleans the corpus; Index builds the hybrid TF-IDF + dense retrieval index."
    )

    rb_col1, rb_col2 = st.columns(2)
    with rb_col1:
        chunk_enabled = st.checkbox("Enable chunking", value=False)
    with rb_col2:
        chunk_size = st.number_input(
            "Chunk size (words)", min_value=50, max_value=1000, value=450, step=50,
            disabled=not chunk_enabled,
        )

    if st.button("Rebuild index", type="primary"):
        if not RAW_PATH.exists():
            st.error("No corpus file found. Fetch documents first.")
        else:
            with st.status("Rebuilding index...", expanded=True) as rebuild_status:
                try:
                    # -- Ingest --
                    st.write("Running ingest (normalize + dedup)...")
                    from healthcare_rag.data_io import expand_chunks

                    df = read_jsonl(RAW_PATH)
                    cleaned = normalize_documents(df)
                    if chunk_enabled:
                        cleaned = expand_chunks(cleaned, chunk_size=int(chunk_size))
                        st.write(
                            f"Chunking: {len(df)} docs → {len(cleaned)} chunks."
                        )
                    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
                    cleaned.to_csv(PROCESSED_PATH, index=False)
                    st.write(f"Wrote **{len(cleaned)}** rows to processed CSV.")

                    # -- Index --
                    st.write("Building hybrid index (TF-IDF + dense embeddings)...")
                    new_retriever = HybridRetriever.build(cleaned)
                    new_retriever.save(INDEX_PATH)
                    st.write(f"Saved index to `{INDEX_PATH}`.")

                    # Clear cached retriever so the Ask tab picks up the new index.
                    load_retriever.clear()

                    rebuild_status.update(
                        label="Index rebuilt successfully. Switch to the Ask tab to query.",
                        state="complete",
                    )
                    st.success(
                        f"Index rebuilt with **{len(cleaned)}** documents. "
                        "The Ask tab now uses the updated index."
                    )

                except Exception as exc:
                    rebuild_status.update(label=f"Rebuild failed: {exc}", state="error")
                    st.exception(exc)
