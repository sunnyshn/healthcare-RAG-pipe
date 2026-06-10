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
ABSTENTION_MARKER = "INSUFFICIENT EVIDENCE:"

# Dense cosine threshold below which we warn the model that evidence is weak.
# Set by inspecting the score gap between answerable (~0.83) and off-topic (~0.70)
# questions against the toy corpus. Re-evaluate after major corpus expansions.
LOW_EVIDENCE_THRESHOLD = 0.78

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


def _offline_answer(question: str, hits: List[SearchResult]) -> str:
    """Deterministic fallback when no API key is set: abstain or extract."""
    max_dense = max(
        (h.score_dense for h in hits if h.score_dense is not None),
        default=0.0,
    )
    if max_dense < LOW_EVIDENCE_THRESHOLD:
        return (
            f"{ABSTENTION_MARKER} No closely matching evidence found in the "
            f"corpus for this question (max semantic similarity: {max_dense:.2f}, "
            f"threshold: {LOW_EVIDENCE_THRESHOLD}). Add relevant documents via "
            "fetch_pubmed.py and rebuild the index."
        )
    snippets = "\n".join([f"- {h.title}: {h.text[:240]}…" for h in hits])
    return (
        "Showing the most relevant evidence from the corpus below. "
        "Connect a language model to generate a single synthesized, cited answer.\n\n"
        f"Top evidence for: {question}\n\n"
        f"{snippets}"
    )


def generate_answer(question: str, hits: List[SearchResult]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _offline_answer(question, hits)

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=_build_messages(question, hits),
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


def stream_answer(question: str, hits: List[SearchResult]) -> Iterator[str]:
    """Yield the answer incrementally as text chunks (for live UI rendering).

    Falls back to yielding the full offline answer in one chunk when no API key
    is configured, so callers can use a single streaming code path.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield _offline_answer(question, hits)
        return

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
            yield delta

