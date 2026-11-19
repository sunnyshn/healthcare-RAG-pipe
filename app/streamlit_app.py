from __future__ import annotations

import os
import re
import sys
import time
import uuid
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from healthcare_rag.config import INDEX_PATH, PROCESSED_PATH, RAW_PATH
from healthcare_rag.data_io import normalize_documents, read_jsonl
from healthcare_rag.faithfulness import check_faithfulness
from healthcare_rag.generator import ABSTENTION_MARKER, for_visitors, stream_answer
from healthcare_rag.monitoring import (
    QueryHealth,
    health_alerts,
    log_query,
    monitoring_available,
    monitoring_enabled,
    recent_runs,
    retrieval_scores,
    rolling_health,
    summarize,
)
from healthcare_rag.retriever import HybridRetriever

import fetch_pubmed as fp

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Healthcare Evidence Assistant",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bridge Streamlit Cloud secrets to env vars so generator.py (which reads
# os.getenv) works in deployment without code changes. We only touch
# st.secrets when a secrets.toml actually exists — otherwise Streamlit surfaces
# a "No secrets found" error (e.g. on Hugging Face Spaces, where the key, if
# any, is injected as an environment variable instead).
if not os.getenv("OPENAI_API_KEY"):
    _secret_paths = (
        Path.home() / ".streamlit" / "secrets.toml",
        ROOT / ".streamlit" / "secrets.toml",
    )
    if any(p.exists() for p in _secret_paths):
        try:
            if "OPENAI_API_KEY" in st.secrets:
                os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])
        except Exception:
            pass  # Malformed/empty secrets file — fall back to env vars.

LLM_ENABLED = bool(os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap">
<style>
html, body, .stApp, [class*="css"] {
    font-family: "Figtree", "Segoe UI", sans-serif;
    color: #163028;
}
.stApp {
    background:
        radial-gradient(980px 720px at 0% -4%, rgba(58, 132, 90, 0.92), transparent 58%),
        radial-gradient(900px 720px at 100% 108%, rgba(255, 248, 236, 0.96), transparent 54%),
        radial-gradient(620px 460px at 72% 6%, rgba(168, 204, 164, 0.75), transparent 58%),
        linear-gradient(152deg, #4f946c 0%, #9ec8ae 36%, #dcead6 64%, #f7f3ea 100%);
    background-attachment: fixed;
}
h1, h2, h3, h4, .hero h1, .stat-value, .section-label, .feature-card h4,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
    font-family: "Figtree", "Segoe UI", sans-serif;
    font-weight: 600;
    letter-spacing: -0.035em;
    line-height: 1.15;
    color: #12261E;
}
p, li, label, .stMarkdown, .stCaption { color: rgba(22, 48, 40, 0.78); }

.block-container {
    background: rgba(255, 252, 246, 0.58);
    backdrop-filter: blur(22px) saturate(140%);
    -webkit-backdrop-filter: blur(22px) saturate(140%);
    border: 1px solid rgba(255, 255, 255, 0.72);
    border-radius: 32px;
    box-shadow: 0 24px 60px rgba(40, 72, 52, 0.1);
    padding: 1.8rem 2rem 2.4rem;
    margin-top: 1.15rem;
    max-width: 1040px;
}
footer, #MainMenu, .stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden;
    height: 0;
}
header[data-testid="stHeader"] { background: transparent; height: 0; }

section[data-testid="stSidebar"] { background: transparent !important; }
[data-testid="stSidebarContent"] {
    background: rgba(255, 252, 246, 0.52);
    backdrop-filter: blur(24px) saturate(140%);
    -webkit-backdrop-filter: blur(24px) saturate(140%);
    border: 1px solid rgba(255, 255, 255, 0.7);
    border-radius: 28px;
    margin: 14px 8px 14px 14px;
    box-shadow: 0 24px 60px rgba(40, 72, 52, 0.1);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    border-bottom: 1px solid rgba(22, 48, 40, 0.12);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    height: auto;
    padding: 0.4rem 0.85rem 0.7rem;
    font-size: 0.95rem;
    font-weight: 500;
    color: rgba(22, 48, 40, 0.55);
}
.stTabs [aria-selected="true"] {
    color: #12261E;
    font-weight: 600;
    border-bottom: 2px solid #1F6B4A;
}

.stTextInput input, .stNumberInput input, .stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 14px !important;
    border: 1px solid rgba(22, 48, 40, 0.12) !important;
    background: rgba(255, 255, 255, 0.72) !important;
    color: #163028 !important;
}
.stTextInput input { padding-top: 0.75rem; padding-bottom: 0.75rem; font-size: 1rem; }
.stTextInput input::placeholder { color: rgba(22, 48, 40, 0.42); }
.stTextInput input:focus {
    border-color: rgba(31, 107, 74, 0.45) !important;
    box-shadow: 0 0 0 3px rgba(31, 107, 74, 0.12);
}
div[data-testid="stButton"] > button { border-radius: 999px; font-weight: 600; }
div[data-testid="stButton"] > button[kind="primary"] {
    background: rgba(26, 36, 32, 0.88);
    backdrop-filter: blur(16px) saturate(150%);
    -webkit-backdrop-filter: blur(16px) saturate(150%);
    border: 1px solid rgba(255, 255, 255, 0.28);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.24), 0 10px 24px rgba(22, 36, 30, 0.1);
    color: #F7F3EA;
}
div[data-testid="stButton"] > button[kind="primary"] * {
    color: #F7F3EA !important;
    background: transparent !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: rgba(34, 46, 42, 0.92);
    border-color: rgba(255, 255, 255, 0.4);
}

.hero { padding: 0.35rem 0 1.35rem; }
.eyebrow {
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: rgba(22, 48, 40, 0.62);
    margin: 0 0 0.55rem;
}
.hero h1 {
    font-size: 2.7rem;
    font-weight: 650;
    letter-spacing: -0.045em;
    line-height: 1.05;
    margin: 0 0 0.7rem;
    max-width: 16ch;
    color: #12261E;
}
.hero p {
    color: rgba(22, 48, 40, 0.78);
    font-size: 1.05rem;
    line-height: 1.55;
    margin: 0;
    max-width: 38rem;
}

.answer-card {
    background: rgba(255, 255, 255, 0.55);
    border: 1px solid rgba(255, 255, 255, 0.7);
    border-radius: 22px;
    padding: 1.25rem 1.35rem 1.3rem;
    margin: 0.7rem 0 1.3rem;
    font-size: 1.02rem;
    line-height: 1.65;
    color: #163028;
}
.answer-card.abstain { background: rgba(255, 214, 170, 0.45); border-color: rgba(196, 137, 58, 0.28); }
.answer-card .cite {
    color: #1C4A36;
    font-weight: 650;
    font-size: 0.86em;
    text-decoration: none;
    border-bottom: 1px solid rgba(28, 74, 54, 0.4);
}
a.cite:hover { border-bottom-color: #1C4A36; }
.type-cursor {
    display: inline-block; margin-left: 1px; color: #1C4A36;
    animation: blink 1s step-start infinite;
}
@keyframes blink { 50% { opacity: 0; } }

.ev {
    display: grid;
    grid-template-columns: 2.2rem 1fr;
    gap: 0.2rem 0.8rem;
    padding: 1.05rem 0.15rem;
    border-top: 1px solid rgba(22, 48, 40, 0.1);
}
.ev-num { font-size: 0.95rem; color: rgba(22, 48, 40, 0.42); padding-top: 0.2rem; }
.ev-title, .ev-title-link {
    font-weight: 600;
    color: #12261E;
    font-size: 1.08rem;
    line-height: 1.35;
    letter-spacing: -0.02em;
    text-decoration: none;
}
.ev-title-link:hover { color: #1C4A36; }
.ev-meta { color: rgba(22, 48, 40, 0.62); font-size: 0.84rem; margin: 0.25rem 0 0.5rem; }
.ev-meta a { color: #1C4A36; text-decoration: none; border-bottom: 1px solid rgba(28, 74, 54, 0.35); }
.ev-text { color: rgba(22, 48, 40, 0.84); font-size: 0.95rem; line-height: 1.6; margin: 0; }
.ev-scores { margin-top: 0.5rem; color: rgba(22, 48, 40, 0.5); font-size: 0.78rem; }

.section-label {
    font-weight: 600;
    color: #12261E;
    font-size: 1.45rem;
    letter-spacing: -0.03em;
    margin: 0.15rem 0 0.3rem;
}
.feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem; margin-top: 0.35rem; }
.feature-card {
    background: rgba(255, 255, 255, 0.48);
    border: 1px solid rgba(255, 255, 255, 0.7);
    border-radius: 20px;
    padding: 1rem 1rem 1.1rem;
}
.feature-card .kicker { font-size: 0.78rem; color: rgba(22, 48, 40, 0.5); }
.feature-card h4 { margin: 0.3rem 0 0.3rem; font-size: 1.05rem; color: #12261E; }
.feature-card p { margin: 0; color: rgba(22, 48, 40, 0.74); font-size: 0.9rem; line-height: 1.5; }
.welcome-hint { color: rgba(22, 48, 40, 0.68); font-size: 0.96rem; margin: 0 0 0.85rem; }

div[data-testid="stButton"] button[kind="secondary"] {
    text-align: left;
    justify-content: flex-start;
    border: 1px solid rgba(22, 48, 40, 0.1);
    background: rgba(255, 255, 255, 0.55);
    color: #163028;
    font-weight: 500;
    border-radius: 16px;
    min-height: 3.15rem;
    white-space: normal;
    line-height: 1.35;
    padding: 0.7rem 0.95rem;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: rgba(255, 255, 255, 0.82);
    border-color: rgba(31, 107, 74, 0.28);
    color: #12261E;
}

.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin: 0.35rem 0 1.1rem; }
.stat {
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.72);
    border-radius: 20px;
    padding: 0.85rem 1rem 0.95rem;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: rgba(22, 48, 40, 0.52);
}
.stat-value { font-size: 1.7rem; color: #12261E; margin-top: 0.25rem; letter-spacing: -0.04em; }
.stat-hint { font-size: 0.78rem; color: rgba(22, 48, 40, 0.52); margin-top: 0.15rem; }
.status {
    border-radius: 16px;
    padding: 0.7rem 0.9rem;
    margin: 0.1rem 0 0.85rem;
    font-size: 0.92rem;
    line-height: 1.45;
    border: 1px solid rgba(22, 48, 40, 0.1);
}
.status.ok { background: rgba(47, 122, 86, 0.12); color: #14543C; }
.status.warn { background: rgba(196, 137, 58, 0.16); color: #6A4A16; }
div[data-testid="stAlert"] {
    background: rgba(255, 255, 255, 0.62);
    color: #163028;
    border: 1px solid rgba(22, 48, 40, 0.1);
    border-radius: 16px;
}
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.72);
    border-radius: 18px;
    padding: 0.65rem 0.8rem;
}
[data-testid="stDataFrame"] { border: 1px solid rgba(22, 48, 40, 0.1); border-radius: 16px; overflow: hidden; }
[data-testid="stException"] { display: none !important; }
@media (max-width: 800px) {
    .feature-grid, .stat-grid { grid-template-columns: 1fr; }
    .hero h1 { font-size: 2.15rem; }
    .block-container { border-radius: 22px; padding: 1.2rem 1rem 1.6rem; }
}
</style>
"""

# st.markdown closes an HTML block on a blank line and then prints the rest
# of the stylesheet as text. st.html injects the style tag intact.
st.html("\n".join(line for line in CUSTOM_CSS.splitlines() if line.strip()))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@st.cache_resource
def load_retriever() -> HybridRetriever:
    """Load the hybrid retriever once and cache it across reruns.

    On a fresh deploy the prebuilt index won't exist (it's gitignored), so we
    bootstrap it from the corpus that ships with the repo. Call
    ``load_retriever.clear()`` after rebuilding the index.
    """
    if INDEX_PATH.exists():
        return HybridRetriever.load(INDEX_PATH)

    # Bootstrap: build the index from the shipped corpus (downloads the
    # embedding model on first run, then caches it).
    df = normalize_documents(read_jsonl(RAW_PATH))
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    retriever = HybridRetriever.build(df)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    retriever.save(INDEX_PATH)
    return retriever


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


def _answer_card_html(answer: str, hits=None, streaming: bool = False) -> str:
    is_abstain = (
        not streaming
        and answer.strip().upper().startswith(ABSTENTION_MARKER.upper())
    )
    cls = "answer-card abstain" if is_abstain else "answer-card"
    if streaming:
        heading = "Writing"
    else:
        heading = "No close match" if is_abstain else "Answer"
    body = _format_citations(for_visitors(answer), hits)
    cursor = '<span class="type-cursor">▌</span>' if streaming else ""
    return (
        f'<div class="{cls}"><div class="eyebrow" style="margin-bottom:0.45rem;">{heading}</div>'
        f"{body}{cursor}</div>"
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
    st.markdown(
        _stats_html(
            [
                ("Claims cited", f"{cov:.0%}", "Share of claims with a source"),
                ("On-topic citations", f"{sup:.0%}", "Cited passages that match the claim"),
                ("Invalid citations", str(invalid), "Numbers that point at nothing retrieved"),
            ]
        ),
        unsafe_allow_html=True,
    )

    cited = report.cited_claims
    if cited:
        with st.expander("Per-claim grounding detail"):
            for s in cited:
                mark = "Supported" if s.supported else "Weak"
                score = f"{s.best_support:.2f}" if s.best_support is not None else "—"
                cite_str = ", ".join(f"[{c}]" for c in s.valid_citations)
                st.markdown(
                    f"{mark} **sim {score}** {cite_str} — {strip_for_display(s.sentence)}"
                )


def strip_for_display(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


def render_evidence(hits) -> None:
    for i, h in enumerate(hits, start=1):
        text = h.text[:600] + ("…" if len(h.text) > 600 else "")
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        url = _source_url(h)
        title_safe = h.title.replace("<", "&lt;").replace(">", "&gt;")
        if url:
            title_html = (
                f'<a class="ev-title-link" href="{url}" target="_blank" '
                f'rel="noopener">{title_safe}</a>'
            )
            source_html = f'<a href="{url}" target="_blank" rel="noopener">{h.source}</a>'
        else:
            title_html = f'<span class="ev-title">{title_safe}</span>'
            source_html = h.source

        meta_bits = [bit for bit in (h.specialty, str(h.year) if h.year else "", source_html) if bit]
        scores = []
        if h.score_sparse is not None:
            scores.append(f"Keyword {h.score_sparse:.2f}")
        if h.score_dense is not None:
            scores.append(f"Meaning {h.score_dense:.2f}")
        if h.score_rerank is not None:
            scores.append(f"Closer read {h.score_rerank:.2f}")
        score_html = f'<div class="ev-scores">{" · ".join(scores)}</div>' if scores else ""

        st.markdown(
            f'<article class="ev">'
            f'<div class="ev-num">{i:02d}</div>'
            f"<div>{title_html}"
            f'<div class="ev-meta">{" · ".join(meta_bits)}</div>'
            f'<p class="ev-text">{text}</p>'
            f"{score_html}</div></article>",
            unsafe_allow_html=True,
        )


EXAMPLE_QUESTIONS = [
    "What is first-line treatment for stage 1 hypertension in adults with diabetes?",
    "How is stroke risk managed in atrial fibrillation?",
    "Which antibiotics are recommended for community-acquired pneumonia?",
    "What is the first-line treatment for major depressive disorder?",
]


def _use_example(question: str) -> None:
    """Populate the query box with an example and trigger a search on rerun."""
    st.session_state["question_input"] = question
    st.session_state["trigger_search"] = True


def render_welcome() -> None:
    """Inviting empty state shown before the first query: examples + how-it-works."""
    st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="welcome-hint">Choose a question to see cited evidence, '
        "or write your own above.</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        cols[i % 2].button(
            q, key=f"example_{i}", on_click=_use_example, args=(q,), use_container_width=True
        )

    st.markdown(
        '<div class="section-label" style="margin-top:1.5rem;">How an answer is made</div>'
        '<div class="feature-grid">'
        '<div class="feature-card"><div class="kicker">01</div>'
        "<h4>Hybrid retrieval</h4><p>Keyword and semantic search over PubMed "
        "abstracts, fused so exact terms and related meaning both count.</p></div>"
        '<div class="feature-card"><div class="kicker">02</div>'
        "<h4>Grounded answers</h4><p>The reply is built only from what was "
        "retrieved, and it declines when the evidence is too thin.</p></div>"
        '<div class="feature-card"><div class="kicker">03</div>'
        "<h4>Citation check</h4><p>Each cited claim is compared with the passage "
        "it points to, so support is visible rather than assumed.</p></div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero">'
    '<div class="eyebrow">Clinical evidence</div>'
    "<h1>Healthcare Evidence Assistant</h1>"
    "<p>Ask a clinical question. The reply stays with the abstracts it used, "
    "and it declines when this library has no close match.</p>"
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Live health (MLflow)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=20, show_spinner=False)
def _load_health(session_id: str, limit: int = 200):
    return recent_runs(limit, session_id=session_id)


def _record_health(**kwargs) -> None:
    """Log one query. Never raises; refreshes the Health tab cache."""
    kwargs.setdefault("llm_enabled", LLM_ENABLED)
    kwargs.setdefault("session_id", _health_session())
    try:
        log_query(QueryHealth(**kwargs))
    except Exception:
        return
    _load_health.clear()


def _pct(value) -> str:
    return "—" if value is None else f"{value:.0%}"


def _secs(value) -> str:
    return "—" if value is None else f"{value:.2f}s"


def _stats_html(items) -> str:
    cards = []
    for label, value, hint in items:
        cards.append(
            f'<div class="stat"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-hint">{hint}</div></div>'
        )
    return f'<div class="stat-grid">{"".join(cards)}</div>'


def _health_chart(df: pd.DataFrame, y: str, title: str, percent: bool = False, unit_domain: bool = False):
    plot = df.dropna(subset=["timestamp", y])
    if plot.empty:
        return None
    scale = alt.Scale(domain=[0, 1], clamp=True) if (percent or unit_domain) else alt.Scale(zero=True)
    axis_args = {"grid": True, "gridColor": "rgba(22,48,40,0.08)", "domainColor": "rgba(22,48,40,0.16)", "labelColor": "#3E564A"}
    if percent:
        axis_args["format"] = "%"
    axis = alt.Axis(**axis_args)
    y_enc = alt.Y(f"{y}:Q", title=None, scale=scale, axis=axis)
    tooltip_fmt = ".0%" if percent else ".2f"
    base = alt.Chart(plot).encode(
        x=alt.X(
            "timestamp:T",
            title=None,
            axis=alt.Axis(grid=False, tickCount=4, labelColor="#3E564A", domainColor="rgba(22,48,40,0.16)"),
        ),
        y=y_enc,
        tooltip=[
            alt.Tooltip("timestamp:T", title="When"),
            alt.Tooltip(f"{y}:Q", title=title, format=tooltip_fmt),
        ],
    )
    color = {
        "error_rate": "#C45B6A",
        "abstention_rate": "#C4893A",
        "citation_support": "#1F7A58",
        "latency_s": "#3D6B8A",
        "max_dense": "#5C6B3A",
    }.get(y, "#1F6B4A")
    chart = (base.mark_line(color=color, strokeWidth=2) + base.mark_circle(color=color, size=36))
    return (
        chart.properties(
            height=168,
            title=alt.TitleParams(text=title, anchor="start", fontSize=15, fontWeight=500, color="#163028"),
        )
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelFont="Figtree", labelColor="#3E564A", titleColor="#3E564A")
        .configure_title(font="Figtree", color="#163028")
    )


def _health_session() -> str:
    """One id per browser visit. Refreshing starts a new exhibit."""
    if "health_session" not in st.session_state:
        st.session_state["health_session"] = uuid.uuid4().hex
    return st.session_state["health_session"]


def render_health() -> None:
    """Charts for this visit only. Older MLflow runs stay out of the page."""
    st.markdown('<div class="section-label">This session</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="welcome-hint">These charts are a sample from this visit: '
        "latency, errors, abstention, retrieval confidence, and citation support. "
        "They start empty, reset when you refresh, and are not kept after the app "
        "restarts. Question text is not stored. Lines show an 8-query rolling mean.</div>",
        unsafe_allow_html=True,
    )

    if not monitoring_available():
        st.info(
            "MLflow is not installed in this environment, so live health is off. "
            "Install the app requirements and refresh."
        )
        return
    if not monitoring_enabled():
        st.info("Model monitoring is paused. Set `MLFLOW_MONITOR=1` to resume.")

    df = _load_health(_health_session())
    if df is None or df.empty:
        st.info("Nothing in this session yet. Ask a question and the sample will start here.")
        return

    stats = summarize(df)
    alerts = health_alerts(df)
    if alerts:
        st.markdown(
            '<div class="status warn">' + "<br>".join(alerts) + "</div>",
            unsafe_allow_html=True,
        )
    elif stats["n_queries"] >= 5:
        st.markdown(
            '<div class="status ok">Within bounds. No threshold breaches in the recent window.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("A few more queries are needed before the health checks are meaningful.")

    dense = "—" if stats["mean_max_dense"] is None else f"{stats['mean_max_dense']:.2f}"
    st.markdown(
        _stats_html(
            [
                ("Queries", str(stats["n_queries"]), "This session"),
                ("Error rate", _pct(stats["error_rate"]), "Queries that failed"),
                ("Abstention", _pct(stats["abstention_rate"]), "Refused for thin evidence"),
                ("Citation support", _pct(stats["mean_citation_support"]), "Claims matching their source"),
                ("Mean latency", _secs(stats["mean_latency_s"]), f"95th percentile {_secs(stats['p95_latency_s'])}"),
                ("Retrieval", dense, "Mean best semantic score"),
            ]
        ),
        unsafe_allow_html=True,
    )

    rolled = rolling_health(df)
    charts = [
        ("error_rate", "Error rate", True, False),
        ("abstention_rate", "Abstention rate", True, False),
        ("citation_support", "Citation support", True, False),
        ("max_dense", "Retrieval confidence", False, True),
        ("latency_s", "Latency, seconds", False, False),
    ]
    slots = st.columns(2)
    placed = 0
    for y, title, percent, unit in charts:
        chart = _health_chart(rolled, y, title, percent=percent, unit_domain=unit)
        if chart is None:
            continue
        with slots[placed % 2]:
            st.altair_chart(chart, use_container_width=True)
        placed += 1

    st.markdown('<div class="section-label" style="margin-top:0.6rem;">Recent queries</div>', unsafe_allow_html=True)
    recent = df.sort_values("timestamp", ascending=False).head(15)
    display = recent[
        ["timestamp", "outcome", "latency_s", "max_dense", "citation_support", "n_hits", "rerank", "generation"]
    ].copy()
    display["latency_s"] = display["latency_s"].map(lambda v: None if pd.isna(v) else round(float(v), 2))
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(
        "Session exhibit. Earlier queries remain in the local MLflow store "
        "(`mlflow ui --backend-store-uri ./mlruns`) and are not shown here. "
        "`MLFLOW_MONITOR=0` pauses logging."
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_ask, tab_health, tab_corpus = st.tabs(["Ask", "Model monitoring", "Corpus"])

# ===========================  ASK TAB  =====================================

with tab_ask:
    if not INDEX_PATH.exists() and not RAW_PATH.exists():
        st.caption(
            "No corpus or index found yet. Use the Corpus tab to fetch documents and build an index."
        )
        st.stop()

    try:
        with st.spinner("Loading retrieval index…"):
            retriever = load_retriever()
        meta = retriever.tfidf.metadata
    except Exception:
        st.caption("The library is still getting ready. Refresh in a moment.")
        st.stop()

    # ── Sidebar: retrieval settings ──────────────────────────────────────────
    with st.sidebar:
        st.header("Search")

        top_k = st.slider("Passages to show", 1, 10, 3)

        rerank = st.toggle(
            "Closer reading",
            value=True,
            help="On by default. A second model reads the question together with "
            "each candidate passage and keeps the closest matches. Turn it off "
            "for a faster, less precise search.",
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
        st.caption(f"{len(meta)} indexed passages · {len(specialties)} specialties")
        if monitoring_available() and monitoring_enabled():
            st.caption("Model monitoring: this session")
        elif not monitoring_available():
            st.caption("Model monitoring: MLflow is not installed")
        else:
            st.caption("Model monitoring: paused")

    # ── Main query area ──────────────────────────────────────────────────────
    question = st.text_input(
        "Ask a clinical question",
        placeholder="e.g. How is atrial fibrillation stroke risk managed?",
        key="question_input",
    )

    run = st.button("Search evidence", type="primary", use_container_width=True)

    # An example-question click sets this flag so we run without a second click.
    triggered = st.session_state.pop("trigger_search", False)
    do_search = run or triggered

    if not do_search:
        if not question.strip():
            render_welcome()
    else:
        if not question.strip():
            st.caption("Please enter a question first.")
            st.stop()

        t0 = time.perf_counter()
        retrieval_s = generation_s = grounding_s = 0.0
        logged = False
        specialty_label = specialty_sel if specialty_sel != "(all)" else "all"
        hits = []
        try:
            specialty_filter = None if specialty_sel == "(all)" else specialty_sel
            yr_filter = tuple(year_range) if year_range else None

            spinner_msg = (
                "Retrieving + re-ranking evidence…" if rerank else "Retrieving evidence…"
            )
            t_r = time.perf_counter()
            with st.spinner(spinner_msg):
                hits = retriever.search(
                    question,
                    k=top_k,
                    specialty=specialty_filter,
                    year_range=yr_filter,
                    rerank=rerank,
                )
            retrieval_s = time.perf_counter() - t_r
            max_dense, mean_rrf = retrieval_scores(hits)

            if not hits:
                _record_health(
                    outcome="no_hits",
                    n_hits=0,
                    max_dense=max_dense,
                    mean_rrf=mean_rrf,
                    latency_retrieval_s=retrieval_s,
                    latency_total_s=time.perf_counter() - t0,
                    top_k=top_k,
                    rerank=rerank,
                    specialty=specialty_label,
                )
                logged = True
                st.caption(
                    "No documents matched the selected filters. Try broadening your criteria."
                )
            else:
                t_g = time.perf_counter()
                answer_placeholder = st.empty()
                answer = stream_answer_into(answer_placeholder, question, hits)
                generation_s = time.perf_counter() - t_g

                is_abstain = answer.strip().upper().startswith(ABSTENTION_MARKER.upper())
                report = None
                if not is_abstain:
                    def _embed_fn(texts):
                        enc = retriever._encoder_model()
                        return [list(v) for v in enc.embed(list(texts))]

                    t_f = time.perf_counter()
                    with st.spinner("Checking citation grounding…"):
                        report = check_faithfulness(answer, hits, embed_fn=_embed_fn)
                    grounding_s = time.perf_counter() - t_f
                    render_grounding(report)

                _record_health(
                    outcome="abstain" if is_abstain else "answered",
                    n_hits=len(hits),
                    max_dense=max_dense,
                    mean_rrf=mean_rrf,
                    abstained=is_abstain,
                    citation_coverage=None if report is None else report.citation_coverage,
                    citation_support=None if report is None else report.citation_support,
                    invalid_citations=None if report is None else report.invalid_citations,
                    n_claims=None if report is None else report.n_claims,
                    latency_retrieval_s=retrieval_s,
                    latency_generation_s=generation_s,
                    latency_grounding_s=grounding_s,
                    latency_total_s=time.perf_counter() - t0,
                    top_k=top_k,
                    rerank=rerank,
                    specialty=specialty_label,
                )
                logged = True

                st.markdown('<div class="section-label" style="margin-top:0.4rem;">Retrieved evidence</div>', unsafe_allow_html=True)
                render_evidence(hits)
        except Exception:
            if not logged:
                _record_health(
                    outcome="error",
                    error=True,
                    latency_retrieval_s=retrieval_s,
                    latency_generation_s=generation_s,
                    latency_grounding_s=grounding_s,
                    latency_total_s=time.perf_counter() - t0,
                    top_k=top_k,
                    rerank=rerank,
                    specialty=specialty_label,
                )
            if hits:
                st.markdown(
                    '<div class="section-label" style="margin-top:0.4rem;">Retrieved evidence</div>',
                    unsafe_allow_html=True,
                )
                render_evidence(hits)


# The monitoring panel is built after a query is logged. Streamlit draws every
# tab in one run and does not rerun when the visitor switches tabs, so this
# panel has to be filled after _record_health or a new question looks missing.
with tab_health:
    render_health()


# =======================  MANAGE CORPUS TAB  ================================

with tab_corpus:
    st.subheader("Current corpus")

    if RAW_PATH.exists():
        existing_ids = fp._load_existing_ids(RAW_PATH)
        total_docs = len(existing_ids)
        raw_df = pd.read_json(RAW_PATH, lines=True)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Documents", total_docs)
        col_b.metric(
            "Specialties",
            raw_df["specialty"].nunique() if "specialty" in raw_df.columns else "—",
        )
        col_c.metric(
            "Year range",
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
            st.caption("Please enter a search query.")
        else:
            specialty_label = pm_specialty.strip() or "general"

            key = fp._api_key()
            if key:
                st.caption("Using the configured PubMed rate limit.")

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

                except Exception:
                    status.update(label="The request did not complete.", state="complete")

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
            st.caption("No corpus file found. Fetch documents first.")
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

                except Exception:
                    rebuild_status.update(label="The index was not rebuilt.", state="complete")
