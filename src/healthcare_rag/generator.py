from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List

import openai

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass

from healthcare_rag.retriever import SearchResult

# Marker that signals the model chose to abstain.
# eval.py imports this to detect abstention programmatically.
# The Streamlit UI strips it before display; visitors see VISITOR_ABSTAIN.
ABSTENTION_MARKER = "INSUFFICIENT EVIDENCE:"

VISITOR_ABSTAIN = (
    "I couldn't find a close match for this question in the library, "
    "so I won't guess. Try a broader question, or one of the examples."
)

# A passage is cited only when it sits close to the best match.
# Dense scores are cosine similarities. Cross-encoder scores are raw logits.
_DENSE_CITE_GAP = 0.08
_RERANK_CITE_GAP = 1.5

# Dense cosine threshold below which the offline answer abstains.
# Re-set after the starter notes were removed: real PubMed abstracts for the
# example questions land around 0.75–0.78, while off-topic matches stay near 0.70.
LOW_EVIDENCE_THRESHOLD = 0.72

_SYSTEM_PROMPT = (
    "You are a healthcare evidence assistant. "
    "Answer using ONLY the numbered evidence passages provided — do not use outside knowledge. "
    "If the passages do not contain enough information to answer the question, you MUST start "
    f"your response with '{ABSTENTION_MARKER}' and briefly explain what is missing. "
    "If evidence is present but weak or conflicting, state that explicitly. "
    "Do not provide diagnosis or emergency guidance. "
    "Keep your answer under 180 words and cite sources as [1], [2], etc."
)


def _build_context(hits: List[SearchResult]) -> str:
    max_dense = max(
        (h.score_dense for h in hits if h.score_dense is not None),
        default=None,
    )
    warning = ""
    if max_dense is not None and max_dense < LOW_EVIDENCE_THRESHOLD:
        warning = (
            f"[NOTE: Highest semantic similarity to this question is {max_dense:.2f}, "
            "which is low. If the passages below do not directly address the question, "
            f"respond with '{ABSTENTION_MARKER}'.]\n\n"
        )

    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] {h.title} ({h.year}, {h.source})\n"
            f"doc_id: {h.doc_id}\n"
            f"text: {h.text[:1300]}"
        )
    return warning + "\n\n".join(blocks)


def _build_messages(question: str, hits: List[SearchResult]) -> List[Dict[str, str]]:
    context = _build_context(hits)
    user_msg = f"Question:\n{question}\n\nEvidence:\n{context}"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _passages_to_cite(hits: List[SearchResult]) -> List[SearchResult]:
    """Drop passages that are clearly weaker than the best match."""
    if not hits:
        return []
    rerank_scores = [h.score_rerank for h in hits if h.score_rerank is not None]
    if rerank_scores:
        best = max(rerank_scores)
        kept = [
            h
            for h in hits
            if h.score_rerank is not None and h.score_rerank >= best - _RERANK_CITE_GAP
        ]
        return kept or hits[:1]
    dense_scores = [h.score_dense for h in hits if h.score_dense is not None]
    if not dense_scores:
        return list(hits)
    best = max(dense_scores)
    kept = [
        h
        for h in hits
        if h.score_dense is None or h.score_dense >= best - _DENSE_CITE_GAP
    ]
    return kept or hits[:1]


def _snippet(text: str, limit: int = 280) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def for_visitors(answer: str) -> str:
    """Remove the eval abstention token. A still-arriving token stays hidden."""
    text = answer.lstrip()
    marker = ABSTENTION_MARKER
    if text.upper().startswith(marker.upper()):
        rest = text[len(marker) :].lstrip()
        return rest or VISITOR_ABSTAIN
    if text and marker.upper().startswith(text.upper()):
        return ""
    return answer


def _offline_answer(_question: str, hits: List[SearchResult]) -> str:
    """Deterministic fallback when no API key is set: abstain or cite passages."""
    max_dense = max(
        (h.score_dense for h in hits if h.score_dense is not None),
        default=0.0,
    )
    if max_dense < LOW_EVIDENCE_THRESHOLD:
        return f"{ABSTENTION_MARKER} {VISITOR_ABSTAIN}"
    lines = [
        f"{h.title}. {_snippet(h.text)} [{i}]"
        for i, h in enumerate(_passages_to_cite(hits), start=1)
    ]
    return "Closest matches.\n\n" + "\n\n".join(lines)


def generate_answer(question: str, hits: List[SearchResult]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _offline_answer(question, hits)

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=_build_messages(question, hits),
            temperature=0.1,
        )
        content = response.choices[0].message.content
        if content and content.strip():
            return content.strip()
    except Exception:
        pass
    return _offline_answer(question, hits)


def stream_answer(question: str, hits: List[SearchResult]) -> Iterator[str]:
    """Yield the answer incrementally as text chunks (for live UI rendering).

    Falls back to yielding the full offline answer in one chunk when no API key
    is configured, so callers can use a single streaming code path.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield _offline_answer(question, hits)
        return

    produced = False
    try:
        client = openai.OpenAI(api_key=api_key)
        stream = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=_build_messages(question, hits),
            temperature=0.1,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                produced = True
                yield delta
        if not produced:
            yield _offline_answer(question, hits)
    except Exception:
        if not produced:
            yield _offline_answer(question, hits)

