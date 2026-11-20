from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterator, List

try:
    from dotenv import load_dotenv

    from healthcare_rag.config import PROJECT_ROOT

    # Only the project .env. Never a file in the parent folder.
    load_dotenv(PROJECT_ROOT / ".env")
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
    "Write one short recommendation that answers the question, using ONLY the numbered passages. "
    "Say it as a single paragraph in your own words. "
    "Do not describe the papers one by one, and do not use outside knowledge. "
    "If they disagree, say so in one sentence. "
    "Do not diagnose the reader or give emergency guidance. "
    "If the passages do not contain enough information to answer, your entire response "
    f"must be exactly '{ABSTENTION_MARKER}' and nothing else. "
    "Do not explain that refusal, and do not put it in brackets. "
    "Use at most 4 sentences and end the last one completely. "
    "Put citations at the end of the sentence they support, as [1] or [2]."
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


def is_abstention(answer: str) -> bool:
    """True when the reply refuses to answer.

    The marker counts wherever it appears. A small model often writes a
    sentence and then appends the marker, sometimes inside brackets.
    """
    return ABSTENTION_MARKER.upper() in (answer or "").upper()


def for_visitors(answer: str) -> str:
    """Remove the eval abstention token. A still-arriving token stays hidden."""
    text = (answer or "").lstrip()
    marker = ABSTENTION_MARKER
    if is_abstention(text):
        return VISITOR_ABSTAIN
    if text and marker.upper().startswith(text.upper()):
        return ""
    return answer


def _as_abstention(text: str) -> str:
    """Replace a mixed refusal with the marker plus the visitor sentence."""
    if is_abstention(text):
        return f"{ABSTENTION_MARKER} {VISITOR_ABSTAIN}"
    return text


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


def local_generator_ready() -> bool:
    """True when a local weights file is configured. Does not load the model."""
    path = os.getenv("LOCAL_LLM_PATH", "").strip()
    return bool(path) and os.path.isfile(path)


def _generator_threads() -> int:
    """Threads for reading the prompt and writing the reply.

    The hosted app sets LOCAL_LLM_THREADS to match its two CPUs.
    On a larger machine, leave one core free for the page itself.
    """
    chosen = _bounded_int("LOCAL_LLM_THREADS", 0, 0, 8)
    if chosen:
        return chosen
    count = os.cpu_count() or 2
    if count <= 2:
        return count
    return min(6, count - 1)


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(low, min(high, value))


# One process-wide reader. Weights stay read-only. Nothing from a question is written back.
_local_model = None
_local_unavailable = False


def warm_local_generator() -> None:
    """Map the weights into memory. No-op when no weights file is configured."""
    _local_llm()


def _local_llm():
    global _local_model, _local_unavailable
    if _local_unavailable:
        return None
    if _local_model is not None:
        return _local_model
    if not local_generator_ready():
        return None
    try:
        from llama_cpp import Llama
    except ImportError:
        _local_unavailable = True
        return None
    try:
        n_ctx = _bounded_int("LOCAL_LLM_CTX", 1024, 512, 2048)
        threads = _generator_threads()
        _local_model = Llama(
            model_path=os.environ["LOCAL_LLM_PATH"],
            n_ctx=n_ctx,
            n_threads=threads,
            n_threads_batch=threads,
            n_batch=min(512, n_ctx),
            verbose=False,
            use_mmap=True,
            use_mlock=False,
            chat_format="chatml",
        )
    except Exception:
        _local_unavailable = True
        return None
    return _local_model


def _local_messages(question: str, hits: List[SearchResult]) -> List[Dict[str, str]]:
    """A short prompt. Passages are capped so the context window cannot grow."""
    cited = (_passages_to_cite(hits) or list(hits))[:3]
    blocks = []
    for index, hit in enumerate(cited, start=1):
        text = " ".join(hit.text.split())[:320]
        blocks.append(f"[{index}] {hit.title}\n{text}")
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Write one recommendation that answers the question. "
                "Do not describe each paper separately. "
                "If the passages do not answer the question, reply with only "
                f"{ABSTENTION_MARKER}\n\n"
                f"Question:\n{question}\n\nEvidence:\n" + "\n\n".join(blocks)
            ),
        },
    ]


def _control_flow(exc: BaseException) -> bool:
    """Streamlit uses exceptions to stop a run. Those must not become a fallback answer."""
    return type(exc).__name__ in {"RerunException", "StopException", "ScriptControlException"}


def _closed_prefix(text: str) -> str:
    """Text through the last finished sentence, including a citation that follows it.

    A token limit can slice the reply mid-word. Anything after the last period,
    question mark, or exclamation point stays hidden.
    """
    last = -1
    for index, char in enumerate(text):
        if char not in ".!?":
            continue
        if (
            char == "."
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isdigit()
            and text[index + 1].isdigit()
        ):
            continue
        last = index
    if last < 0:
        return ""
    rest = text[last + 1 :]
    gap = 0
    while rest[gap:].startswith(" ") or rest[gap:].startswith("\n"):
        gap += 1
    extra = gap
    found_cite = False
    while True:
        cite = rest[extra:]
        if len(cite) < 3 or cite[0] != "[":
            break
        end = cite.find("]")
        if end < 2 or not cite[1:end].isdigit():
            break
        found_cite = True
        extra += end + 1
        while rest[extra:].startswith(" "):
            extra += 1
    if not found_cite:
        extra = 0
    return text[: last + 1 + extra]


def _stream_local(question: str, hits: List[SearchResult]) -> Iterator[str]:
    """Yield the recommendation as it is written. Weights are not updated."""
    max_dense = max(
        (hit.score_dense for hit in hits if hit.score_dense is not None),
        default=0.0,
    )
    if max_dense < LOW_EVIDENCE_THRESHOLD:
        yield f"{ABSTENTION_MARKER} {VISITOR_ABSTAIN}"
        return
    llm = _local_llm()
    if llm is None:
        yield _offline_answer(question, hits)
        return
    produced = False
    full = ""
    shown = 0
    pending = ""
    yielded = ""
    for chunk in llm.create_chat_completion(
        messages=_local_messages(question, hits),
        max_tokens=220,
        temperature=0.1,
        repeat_penalty=1.05,
        stream=True,
    ):
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content") or ""
        if not piece:
            continue
        full += piece
        closed = _closed_prefix(full)
        if len(closed) <= shown:
            continue
        pending += closed[shown:]
        shown = len(closed)
        if len(pending) >= 24 or pending.endswith((" ", "\n", ".", ",", "!", "?")):
            produced = True
            yielded += pending
            yield pending
            pending = ""
        if is_abstention(full):
            break
    if pending and not is_abstention(full):
        produced = True
        yielded += pending
        yield pending
    if is_abstention(full):
        if not is_abstention(yielded):
            produced = True
            yield f" {ABSTENTION_MARKER}"
        return
    if not produced:
        yield _offline_answer(question, hits)


def _local_answer(question: str, hits: List[SearchResult]) -> str:
    """One stateless reply. The weights are not updated and the prompt is not stored."""
    text = _as_abstention("".join(_stream_local(question, hits)).strip())
    return text or _offline_answer(question, hits)


def generate_answer(question: str, hits: List[SearchResult]) -> str:
    """Reply from the local model, or the extractive list when it is unavailable."""
    if local_generator_ready():
        try:
            return _local_answer(question, hits)
        except Exception:
            return _offline_answer(question, hits)
    return _offline_answer(question, hits)


def stream_answer(question: str, hits: List[SearchResult]) -> Iterator[str]:
    """Yield the reply as it is written.

    Without the local model, the extractive list is yielded in one chunk.
    """
    if local_generator_ready():
        produced = False
        try:
            for piece in _stream_local(question, hits):
                produced = True
                yield piece
        except Exception as exc:
            if _control_flow(exc):
                raise
            if not produced:
                yield _offline_answer(question, hits)
        return
    yield _offline_answer(question, hits)

