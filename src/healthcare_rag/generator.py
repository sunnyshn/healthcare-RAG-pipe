from __future__ import annotations

import os
from pathlib import Path
from typing import List

import openai

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env")
except ImportError:
    pass

from healthcare_rag.retriever import SearchResult

_SYSTEM_PROMPT = (
    "You are a healthcare evidence assistant. "
    "Answer using only the numbered evidence passages provided. "
    "If the evidence is weak, conflicting, or absent, state that explicitly. "
    "Do not provide diagnosis or emergency guidance. "
    "Keep your answer under 180 words and cite sources as [1], [2], etc."
)


def _build_context(hits: List[SearchResult]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[{i}] {h.title} ({h.year}, {h.source})\n"
            f"doc_id: {h.doc_id}\n"
            f"text: {h.text[:1300]}"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, hits: List[SearchResult]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    context = _build_context(hits)

    if not api_key:
        # Offline fallback to keep the project usable without paid APIs.
        snippets = "\n".join([f"- {h.title}: {h.text[:240]}..." for h in hits])
        return (
            "No OPENAI_API_KEY detected, so this is an extractive baseline.\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Evidence snippets:\n"
            f"{snippets}\n\n"
            "Suggested next step: add an API key to enable grounded generation with citations."
        )

    client = openai.OpenAI(api_key=api_key)
    user_msg = f"Question:\n{question}\n\nEvidence:\n{context}"

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

