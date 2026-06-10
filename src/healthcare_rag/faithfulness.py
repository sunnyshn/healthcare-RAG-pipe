"""Citation faithfulness: do the model's [n] citations actually support its claims?

A grounded RAG answer should (a) attribute each factual claim to a retrieved
passage with an inline ``[n]`` marker, and (b) only cite passages that actually
support the claim. This module measures both:

- **citation coverage** — share of claim sentences that carry at least one
  valid citation (did the model attribute its claims at all?);
- **citation support** — share of cited claim sentences whose cited passage is
  semantically close to the claim (are the citations on-topic?);
- **invalid citation rate** — share of ``[n]`` markers pointing at a passage
  index that wasn't retrieved (did the model invent a source number?).

Support is estimated with the *same* dense embedding model used for retrieval,
so the check is fully offline and needs no extra dependency. Embeddings are a
proxy for entailment, not a substitute for it: a high score means the claim and
passage are about the same thing, not that the passage logically entails the
claim. The encoder is injectable so tests stay fast and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from healthcare_rag.config import EMBEDDING_MODEL_NAME
from healthcare_rag.generator import ABSTENTION_MARKER
from healthcare_rag.retriever import SearchResult

# Cosine threshold above which a cited passage is treated as supporting the
# claim. Calibrated for BGE-small query/passage similarity; tune per corpus.
DEFAULT_SUPPORT_THRESHOLD = 0.55

# Minimum word count for a sentence to count as a substantive "claim".
MIN_CLAIM_WORDS = 4

_CITATION_RE = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# An embedder maps a list of strings to a list of vectors.
EmbedFn = Callable[[Sequence[str]], List[np.ndarray]]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation (newlines too)."""
    parts: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(line) if s.strip())
    return parts


def extract_citations(sentence: str) -> List[int]:
    """Return all citation indices (as written, 1-indexed) found in a sentence."""
    return [int(m) for m in _CITATION_RE.findall(sentence)]


def strip_citations(sentence: str) -> str:
    """Remove ``[n]`` markers so the bare claim text can be embedded."""
    return _CITATION_RE.sub("", sentence).strip()


def _is_claim(sentence: str) -> bool:
    bare = strip_citations(sentence)
    return len(bare.split()) >= MIN_CLAIM_WORDS


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _default_embed_fn(texts: Sequence[str]) -> List[np.ndarray]:
    from fastembed import TextEmbedding

    encoder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
    return [np.asarray(v, dtype=np.float64) for v in encoder.embed(list(texts))]


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass
class SentenceCheck:
    sentence: str
    is_claim: bool
    citations: List[int]            # all indices as written
    valid_citations: List[int]      # within [1, n_passages]
    invalid_citations: List[int]    # out of range / hallucinated
    best_support: Optional[float]   # max cosine over valid cited passages
    supported: bool                 # best_support >= threshold


@dataclass
class FaithfulnessReport:
    threshold: float
    abstained: bool
    sentences: List[SentenceCheck]

    @property
    def claim_sentences(self) -> List[SentenceCheck]:
        return [s for s in self.sentences if s.is_claim]

    @property
    def cited_claims(self) -> List[SentenceCheck]:
        return [s for s in self.claim_sentences if s.valid_citations]

    @property
    def n_claims(self) -> int:
        return len(self.claim_sentences)

    @property
    def total_citations(self) -> int:
        return sum(len(s.citations) for s in self.sentences)

    @property
    def invalid_citations(self) -> int:
        return sum(len(s.invalid_citations) for s in self.sentences)

    @property
    def citation_coverage(self) -> float:
        """Share of claim sentences carrying at least one valid citation."""
        if not self.n_claims:
            return 0.0
        return len(self.cited_claims) / self.n_claims

    @property
    def citation_support(self) -> float:
        """Share of cited claim sentences whose citation is semantically supported."""
        cited = self.cited_claims
        if not cited:
            return 0.0
        return sum(1 for s in cited if s.supported) / len(cited)

    @property
    def invalid_citation_rate(self) -> float:
        if not self.total_citations:
            return 0.0
        return self.invalid_citations / self.total_citations

    @property
    def mean_support(self) -> float:
        scores = [s.best_support for s in self.cited_claims if s.best_support is not None]
        return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_faithfulness(
    answer: str,
    hits: List[SearchResult],
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    embed_fn: Optional[EmbedFn] = None,
) -> FaithfulnessReport:
    """Check whether ``answer``'s inline citations are supported by ``hits``.

    Embeds each cited claim and its cited passages with the dense model and
    flags a citation as supported when their cosine similarity meets
    ``threshold``. Abstention answers are returned with ``abstained=True`` and
    no claims to score.
    """
    abstained = answer.strip().upper().startswith(ABSTENTION_MARKER.upper())
    n_passages = len(hits)
    sentences = split_sentences(answer)

    # Pre-parse every sentence's citations and claim status.
    parsed = []
    for sent in sentences:
        cites = extract_citations(sent)
        valid = [c for c in cites if 1 <= c <= n_passages]
        invalid = [c for c in cites if c not in valid]
        parsed.append((sent, _is_claim(sent), cites, valid, invalid))

    checks: List[SentenceCheck] = []

    if abstained or not any(valid for _, _, _, valid, _ in parsed):
        # Nothing to score against passages — just record structure.
        for sent, is_claim, cites, valid, invalid in parsed:
            checks.append(
                SentenceCheck(
                    sentence=sent,
                    is_claim=is_claim,
                    citations=cites,
                    valid_citations=valid,
                    invalid_citations=invalid,
                    best_support=None,
                    supported=False,
                )
            )
        return FaithfulnessReport(threshold=threshold, abstained=abstained, sentences=checks)

    if embed_fn is None:
        embed_fn = _default_embed_fn

    # Embed all passages once and each cited claim once (single batched call).
    passage_inputs = [f"passage: {h.text}" for h in hits]
    cited_sentences = [
        (idx, sent, valid)
        for idx, (sent, is_claim, _cites, valid, _inv) in enumerate(parsed)
        if is_claim and valid
    ]
    claim_inputs = [f"query: {strip_citations(sent)}" for _, sent, _ in cited_sentences]

    vectors = embed_fn(passage_inputs + claim_inputs)
    passage_vecs = vectors[:n_passages]
    claim_vecs = vectors[n_passages:]

    support_by_sentence_idx = {}
    for (sent_idx, _sent, valid), claim_vec in zip(cited_sentences, claim_vecs):
        best = max(_cosine(claim_vec, passage_vecs[c - 1]) for c in valid)
        support_by_sentence_idx[sent_idx] = best

    for idx, (sent, is_claim, cites, valid, invalid) in enumerate(parsed):
        best = support_by_sentence_idx.get(idx)
        checks.append(
            SentenceCheck(
                sentence=sent,
                is_claim=is_claim,
                citations=cites,
                valid_citations=valid,
                invalid_citations=invalid,
                best_support=best,
                supported=best is not None and best >= threshold,
            )
        )

    return FaithfulnessReport(threshold=threshold, abstained=abstained, sentences=checks)
