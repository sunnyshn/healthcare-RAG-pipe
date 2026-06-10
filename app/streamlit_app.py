import math
import re
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
from healthcare_rag.faithfulness import check_faithfulness
from healthcare_rag.generator import ABSTENTION_MARKER, stream_answer
from healthcare_rag.retriever import HybridRetriever

import fetch_pubmed as fp

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Healthcare Evidence Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* Tighten the top padding so the hero sits higher */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; }

/* Hero header */
.hero {
    background: linear-gradient(120deg, #2E7D6B 0%, #3A9B82 100%);
    border-radius: 16px;
    padding: 1.6rem 1.9rem;
    color: #ffffff;
    margin-bottom: 1.4rem;
    box-shadow: 0 4px 18px rgba(46, 125, 107, 0.22);
}
.hero h1 { color: #ffffff; font-size: 1.9rem; margin: 0 0 0.25rem 0; font-weight: 700; }
.hero p { color: #E6F2EE; font-size: 0.98rem; margin: 0; }
.hero .pill {
    display: inline-block; margin-top: 0.8rem; padding: 0.28rem 0.7rem;
    background: rgba(255,255,255,0.18); border-radius: 999px;
    font-size: 0.78rem; letter-spacing: 0.02em;
}

/* Answer card */
.answer-card {
    background: #ffffff; border: 1px solid #E2EAE8; border-left: 5px solid #2E7D6B;
    border-radius: 12px; padding: 1.3rem 1.5rem; margin: 0.4rem 0 1.2rem 0;
    box-shadow: 0 2px 10px rgba(26, 43, 41, 0.05);
    font-size: 1.02rem; line-height: 1.6; color: #1A2B29;
}
.answer-card.abstain { border-left-color: #C9821B; background: #FFF9F0; }

/* Inline citation chips like [1], [2] inside the answer */
.answer-card .cite {
    background: #E0EFEA; color: #226152; border-radius: 6px;
    padding: 0 0.32rem; font-weight: 600; font-size: 0.9em;
    text-decoration: none;
}
a.cite:hover { background: #C8E2D9; text-decoration: none; }

/* Blinking caret while the answer streams in */
.type-cursor {
    display: inline-block; margin-left: 1px; color: #2E7D6B;
    animation: blink 1s step-start infinite; font-weight: 700;
}
@keyframes blink { 50% { opacity: 0; } }

/* Linked evidence title */
.ev-title-link {
    font-weight: 650; color: #15302C; font-size: 1.02rem; text-decoration: none;
}
.ev-title-link:hover { color: #2E7D6B; text-decoration: underline; }

/* Evidence card */
.ev-card {
    background: #ffffff; border: 1px solid #E2EAE8; border-radius: 12px;
    padding: 1rem 1.2rem; margin-bottom: 0.85rem;
    box-shadow: 0 1px 6px rgba(26, 43, 41, 0.04);
}
.ev-head { display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap; }
.ev-num {
    background: #2E7D6B; color: #fff; border-radius: 6px;
    padding: 0.05rem 0.5rem; font-weight: 700; font-size: 0.85rem;
}
.ev-title { font-weight: 650; color: #15302C; font-size: 1.02rem; }
.ev-meta { color: #6B7C79; font-size: 0.82rem; margin: 0.35rem 0 0.5rem 0; }
.ev-text { color: #2B3D3A; font-size: 0.92rem; line-height: 1.55; }

/* Badges */
.badge {
    display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px;
    font-size: 0.74rem; font-weight: 600; margin-right: 0.3rem;
}
.badge-spec { background: #E0EFEA; color: #226152; }
.badge-year { background: #EDEAF6; color: #5B4B9B; }

/* Score bars */
.score-row { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.3rem; }
.score-label { width: 92px; font-size: 0.76rem; color: #6B7C79; }
.score-track {
    flex: 1; height: 7px; background: #EDF1F0; border-radius: 999px; overflow: hidden;
}
.score-fill { height: 100%; border-radius: 999px; }
.score-fill.sparse { background: #6FB1A0; }
.score-fill.dense { background: #5B4B9B; }
.score-fill.rerank { background: #C9821B; }
.score-val { width: 46px; text-align: right; font-size: 0.76rem; color: #44524F; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def load_retriever() -> HybridRetriever:
    """Load the hybrid retriever once and cache it across reruns.

    Call ``load_retriever.clear()`` after rebuilding the index.
    """
    return HybridRetriever.load(INDEX_PATH)


def _source_url(hit) -> str | None:
    """Return a public URL for a hit when we can build one (PubMed PMIDs)."""
    if hit.doc_id.startswith("pmid-"):
        pmid = hit.doc_id.split("pmid-", 1)[1]
        if pmid.isdigit():
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return None


def _format_citations(text: str, hits=None) -> str:
    """Wrap inline [n] citations in chips, linking to the source when available."""
    safe = (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    urls = {}
    if hits:
        for i, h in enumerate(hits, start=1):
            url = _source_url(h)
            if url:
                urls[i] = url

    def _repl(match):
        n = int(match.group(1))
        url = urls.get(n)
        if url:
            return f'<a class="cite" href="{url}" target="_blank" rel="noopener">[{n}]</a>'
        return f'<span class="cite">[{n}]</span>'

    safe = re.sub(r"\[(\d+)\]", _repl, safe)
    return safe.replace("\n", "<br>")


def _score_bar(label: str, value: float, kind: str, fill: float | None = None) -> str:
    """Render a labeled score bar.

    ``value`` is the number shown on the right; ``fill`` (0-1) controls the bar
    width. When ``fill`` is None the value itself is clamped to [0, 1] and used.
    """
    frac = float(value) if fill is None else float(fill)
    pct = max(0.0, min(1.0, frac)) * 100
    return (
        f'<div class="score-row">'
        f'<span class="score-label">{label}</span>'
        f'<span class="score-track"><span class="score-fill {kind}" '
        f'style="width:{pct:.0f}%"></span></span>'
        f'<span class="score-val">{value:.2f}</span>'
        f"</div>"
    )


def _answer_card_html(answer: str, hits=None, streaming: bool = False) -> str:
    is_abstain = (
        not streaming
        and answer.strip().upper().startswith(ABSTENTION_MARKER.upper())
    )
    cls = "answer-card abstain" if is_abstain else "answer-card"
    if streaming:
        heading, icon = "Generating…", "✍️"
    else:
        heading = "Insufficient evidence" if is_abstain else "Answer"
        icon = "⚠️" if is_abstain else "✅"
    body = _format_citations(answer, hits)
    cursor = '<span class="type-cursor">▌</span>' if streaming else ""
    return (
        f'<div class="{cls}"><div style="font-weight:700;margin-bottom:0.5rem;">'
        f"{icon} {heading}</div>{body}{cursor}</div>"
    )


def stream_answer_into(placeholder, question, hits) -> str:
    """Stream the answer into a placeholder, keeping the styled card. Returns full text."""
    acc = ""
    for chunk in stream_answer(question, hits):
        acc += chunk
        placeholder.markdown(_answer_card_html(acc, hits, streaming=True), unsafe_allow_html=True)
    placeholder.markdown(_answer_card_html(acc, hits, streaming=False), unsafe_allow_html=True)
    return acc


def render_grounding(report) -> None:
    """Show a compact citation-faithfulness summary for a generated answer."""
    if report.abstained or report.n_claims == 0:
        return

    cov = report.citation_coverage
    sup = report.citation_support
    invalid = report.invalid_citations

    cols = st.columns(3)
    cols[0].metric("Claims cited", f"{cov:.0%}", help="Share of factual claims that cite a source")
    cols[1].metric(
        "Citations on-topic",
        f"{sup:.0%}",
        help="Share of cited claims whose passage is semantically supporting",
    )
    cols[2].metric(
        "Invalid citations",
        invalid,
        delta=None if invalid == 0 else "check answer",
        delta_color="inverse",
        help="Citation numbers pointing to a passage that wasn't retrieved",
    )

    cited = report.cited_claims
    if cited:
        with st.expander("Per-claim grounding detail"):
            for s in cited:
                mark = "✅" if s.supported else "⚠️"
                score = f"{s.best_support:.2f}" if s.best_support is not None else "—"
                cite_str = ", ".join(f"[{c}]" for c in s.valid_citations)
                st.markdown(
                    f"{mark} **sim {score}** {cite_str} — {strip_for_display(s.sentence)}"
                )


def strip_for_display(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def render_evidence(hits) -> None:
    for i, h in enumerate(hits, start=1):
        spec_badge = (
            f'<span class="badge badge-spec">{h.specialty}</span>' if h.specialty else ""
        )
        year_badge = f'<span class="badge badge-year">{h.year}</span>'
        text = h.text[:600] + ("…" if len(h.text) > 600 else "")
        text = text.replace("<", "&lt;").replace(">", "&gt;")

        bars = ""
        if h.score_sparse is not None and h.score_dense is not None:
            bars = (
                _score_bar("keyword", h.score_sparse, "sparse")
                + _score_bar("semantic", h.score_dense, "dense")
            )
        if h.score_rerank is not None:
            # Cross-encoder scores are logits; squash to [0,1] for the bar width.
            fill = 1.0 / (1.0 + math.exp(-h.score_rerank))
            bars += _score_bar("re-rank", h.score_rerank, "rerank", fill=fill)

        url = _source_url(h)
        title_safe = h.title.replace("<", "&lt;").replace(">", "&gt;")
        if url:
            title_html = (
                f'<a class="ev-title-link" href="{url}" target="_blank" '
                f'rel="noopener">{title_safe} ↗</a>'
            )
            source_html = f'<a href="{url}" target="_blank" rel="noopener">{h.source}</a>'
        else:
            title_html = f'<span class="ev-title">{title_safe}</span>'
            source_html = h.source

        st.markdown(
            f'<div class="ev-card">'
            f'<div class="ev-head"><span class="ev-num">{i}</span>'
            f"{title_html}</div>"
            f'<div class="ev-meta">{spec_badge}{year_badge}'
            f'&nbsp;{source_html} &middot; <code>{h.doc_id}</code> &middot; RRF {h.score:.4f}</div>'
            f'<div class="ev-text">{text}</div>'
            f"{bars}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero">'
    "<h1>🩺 Healthcare Evidence Assistant</h1>"
    "<p>Retrieval-augmented answers grounded in PubMed literature, with built-in "
    "safeguards that refuse to answer rather than hallucinate.</p>"
    '<span class="pill">Educational decision support &middot; Not for diagnosis or emergency use</span>'
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ask, tab_corpus = st.tabs(["🔍  Ask", "📚  Manage Corpus"])

# ===========================  ASK TAB  =====================================

with tab_ask:
    if not INDEX_PATH.exists():
        st.warning(
            "No search index found. Go to the **Manage Corpus** tab and click "
            "**Rebuild Index** to build one."
        )
        st.stop()

    retriever = load_retriever()
    meta = retriever.tfidf.metadata

    # ── Sidebar: retrieval settings ──────────────────────────────────────────
    with st.sidebar:
        st.header("Retrieval settings")

        top_k = st.slider("Number of sources (top-k)", 1, 10, 3)

        rerank = st.toggle(
            "Cross-encoder re-ranking",
            value=False,
            help="Two-stage retrieval: pull a wider candidate pool, then re-rank "
            "with a cross-encoder for more precise top results (slower).",
        )

        specialties = sorted(meta["specialty"].dropna().unique().tolist())
        specialty_sel = st.selectbox("Specialty", options=["(all)"] + specialties)

        all_years = sorted(meta["year"].dropna().astype(int).unique().tolist())
        if len(all_years) >= 2:
            year_range = st.slider(
                "Publication year",
                min_value=int(all_years[0]),
                max_value=int(all_years[-1]),
                value=(int(all_years[0]), int(all_years[-1])),
            )
        else:
            year_range = None

        st.divider()
        st.caption(
            f"📁 {len(meta)} indexed passages\n\n"
            f"🏷️ {len(specialties)} specialties"
        )

    # ── Main query area ──────────────────────────────────────────────────────
    question = st.text_input(
        "Ask a clinical question",
        value="What is first-line treatment for stage 1 hypertension in adults with diabetes?",
        placeholder="e.g. How is atrial fibrillation stroke risk managed?",
    )

    run = st.button("Search evidence", type="primary", use_container_width=True)

    if run:
        specialty_filter = None if specialty_sel == "(all)" else specialty_sel
        yr_filter = tuple(year_range) if year_range else None

        spinner_msg = "Retrieving + re-ranking evidence…" if rerank else "Retrieving evidence…"
        with st.spinner(spinner_msg):
            hits = retriever.search(
                question,
                k=top_k,
                specialty=specialty_filter,
                year_range=yr_filter,
                rerank=rerank,
            )

        if not hits:
            st.warning(
                "No documents matched the selected filters. Try broadening your criteria."
            )
        else:
            answer_placeholder = st.empty()
            answer = stream_answer_into(answer_placeholder, question, hits)

            # Citation faithfulness check (reuses the retriever's dense encoder).
            is_abstain = answer.strip().upper().startswith(ABSTENTION_MARKER.upper())
            if not is_abstain:
                def _embed_fn(texts):
                    enc = retriever._encoder_model()
                    return [list(v) for v in enc.embed(list(texts))]

                with st.spinner("Checking citation grounding…"):
                    report = check_faithfulness(answer, hits, embed_fn=_embed_fn)
                render_grounding(report)

            st.markdown("##### 📄 Retrieved evidence")
            render_evidence(hits)


# =======================  MANAGE CORPUS TAB  ================================

with tab_corpus:
    st.subheader("Current corpus")

    if RAW_PATH.exists():
        existing_ids = fp._load_existing_ids(RAW_PATH)
        total_docs = len(existing_ids)
        raw_df = pd.read_json(RAW_PATH, lines=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("📄 Documents", total_docs)
        col_b.metric(
            "🏷️ Specialties",
            raw_df["specialty"].nunique() if "specialty" in raw_df.columns else "—",
        )
        col_c.metric(
            "📅 Year range",
            f"{int(raw_df['year'].min())}–{int(raw_df['year'].max())}"
            if "year" in raw_df.columns and not raw_df["year"].isna().all()
            else "—",
        )

        if "specialty" in raw_df.columns:
            counts = (
                raw_df["specialty"].value_counts().rename_axis("specialty").reset_index(name="count")
            )
            st.bar_chart(counts, x="specialty", y="count", color="#2E7D6B", height=240)
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

            status = st.status(f'Searching PubMed for "{pm_query}"…', expanded=True)

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
                        prog = st.progress(0, text="Fetching abstracts…")
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
                                f"as specialty **{specialty_label}**."
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
            "Chunk size (words)",
            min_value=50,
            max_value=1000,
            value=450,
            step=50,
            disabled=not chunk_enabled,
        )

    if st.button("Rebuild index", type="primary"):
        if not RAW_PATH.exists():
            st.error("No corpus file found. Fetch documents first.")
        else:
            with st.status("Rebuilding index…", expanded=True) as rebuild_status:
                try:
                    st.write("Running ingest (normalize + dedup)…")
                    from healthcare_rag.data_io import expand_chunks

                    df = read_jsonl(RAW_PATH)
                    cleaned = normalize_documents(df)
                    if chunk_enabled:
                        cleaned = expand_chunks(cleaned, chunk_size=int(chunk_size))
                        st.write(f"Chunking: {len(df)} docs → {len(cleaned)} chunks.")
                    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
                    cleaned.to_csv(PROCESSED_PATH, index=False)
                    st.write(f"Wrote **{len(cleaned)}** rows to processed CSV.")

                    st.write("Building hybrid index (TF-IDF + dense embeddings)…")
                    new_retriever = HybridRetriever.build(cleaned)
                    new_retriever.save(INDEX_PATH)
                    st.write(f"Saved index to `{INDEX_PATH}`.")

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
