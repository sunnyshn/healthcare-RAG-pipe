from __future__ import annotations

import os
from typing import List

import openai

from healthcare_rag.retriever import SearchResult


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

    openai.api_key = api_key
    prompt = (
        "You are a healthcare evidence assistant. Use only the provided evidence. "
        "If evidence is weak or conflicting, say so explicitly. "
        "Do not provide diagnosis. Keep answer under 180 words and cite [1], [2], etc.\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence:\n{context}"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response["choices"][0]["message"]["content"].strip()

