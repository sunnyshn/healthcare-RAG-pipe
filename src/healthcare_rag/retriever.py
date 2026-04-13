from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Reciprocal rank fusion constant (common default; balances sparse vs dense ranks).
RRF_K = 60


@dataclass
class SearchResult:
    doc_id: str
    title: str
    text: str
    score: float
    source: str
    year: int
    score_sparse: Optional[float] = None
    score_dense: Optional[float] = None


class TfidfRetriever:
    def __init__(self, vectorizer: TfidfVectorizer, matrix, metadata: pd.DataFrame):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.metadata = metadata.reset_index(drop=True)

    @classmethod
    def build(cls, docs: pd.DataFrame) -> "TfidfRetriever":
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(docs["text"].tolist())
        return cls(vectorizer=vectorizer, matrix=matrix, metadata=docs)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "matrix": self.matrix, "metadata": self.metadata},
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "TfidfRetriever":
        with path.open("rb") as f:
            payload = pickle.load(f)
        return cls(payload["vectorizer"], payload["matrix"], payload["metadata"])

    def search(self, query: str, k: int = 3) -> List[SearchResult]:
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        best_idx = np.argsort(scores)[::-1][:k]

        results: List[SearchResult] = []
        for idx in best_idx:
            row = self.metadata.iloc[idx]
            results.append(
                SearchResult(
                    doc_id=str(row["doc_id"]),
                    title=str(row["title"]),
                    text=str(row["text"]),
                    score=float(scores[idx]),
                    source=str(row["source"]),
                    year=int(row["year"]),
                )
            )
        return results


def _rows_to_results(
    metadata: pd.DataFrame,
    indices: np.ndarray,
    fused_scores: np.ndarray,
    sparse_scores: np.ndarray,
    dense_scores: np.ndarray,
) -> List[SearchResult]:
    results: List[SearchResult] = []
    for idx in indices:
        row = metadata.iloc[int(idx)]
        results.append(
            SearchResult(
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                text=str(row["text"]),
                score=float(fused_scores[int(idx)]),
                source=str(row["source"]),
                year=int(row["year"]),
                score_sparse=float(sparse_scores[int(idx)]),
                score_dense=float(dense_scores[int(idx)]),
            )
        )
    return results


class HybridRetriever:
    """
    TF-IDF + dense embeddings (FastEmbed / ONNX), fused with reciprocal rank fusion (RRF).
    """

    def __init__(
        self,
        tfidf: TfidfRetriever,
        doc_embeddings: np.ndarray,
        embedding_model_name: str,
    ):
        self.tfidf = tfidf
        self.doc_embeddings = doc_embeddings.astype(np.float64, copy=False)
        self.embedding_model_name = embedding_model_name
        self._encoder = None

    def _encoder_model(self):
        if self._encoder is None:
            from fastembed import TextEmbedding

            self._encoder = TextEmbedding(model_name=self.embedding_model_name)
        return self._encoder

    @classmethod
    def build(
        cls,
        docs: pd.DataFrame,
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    ) -> "HybridRetriever":
        from fastembed import TextEmbedding

        tfidf = TfidfRetriever.build(docs)
        encoder = TextEmbedding(model_name=embedding_model_name)
        # BGE retrieval: passage vs query prefixes (see model card).
        passages = [f"passage: {t}" for t in docs["text"].tolist()]
        emb = list(encoder.embed(passages))
        stacked = np.stack([np.asarray(e, dtype=np.float64) for e in emb], axis=0)
        return cls(
            tfidf=tfidf,
            doc_embeddings=stacked,
            embedding_model_name=embedding_model_name,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "kind": "hybrid",
                    "tfidf": {
                        "vectorizer": self.tfidf.vectorizer,
                        "matrix": self.tfidf.matrix,
                        "metadata": self.tfidf.metadata,
                    },
                    "doc_embeddings": self.doc_embeddings,
                    "embedding_model_name": self.embedding_model_name,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "HybridRetriever":
        with path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("kind") != "hybrid":
            raise ValueError(
                "Index file is not a hybrid index. Re-run `python index.py` to rebuild."
            )
        tf = payload["tfidf"]
        tfidf = TfidfRetriever(
            vectorizer=tf["vectorizer"],
            matrix=tf["matrix"],
            metadata=tf["metadata"],
        )
        return cls(
            tfidf=tfidf,
            doc_embeddings=payload["doc_embeddings"],
            embedding_model_name=payload["embedding_model_name"],
        )

    def search(self, query: str, k: int = 3) -> List[SearchResult]:
        n = len(self.tfidf.metadata)
        q_vec = self.tfidf.vectorizer.transform([query])
        sparse_scores = cosine_similarity(q_vec, self.tfidf.matrix)[0]

        encoder = self._encoder_model()
        q_raw = list(encoder.embed([f"query: {query}"]))[0]
        q_emb = np.asarray(q_raw, dtype=np.float64).reshape(1, -1)
        dense_scores = cosine_similarity(q_emb, self.doc_embeddings)[0]

        sparse_rank = np.argsort(sparse_scores)[::-1]
        dense_rank = np.argsort(dense_scores)[::-1]
        rrf = np.zeros(n, dtype=np.float64)
        for rank, idx in enumerate(sparse_rank, start=1):
            rrf[idx] += 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(dense_rank, start=1):
            rrf[idx] += 1.0 / (RRF_K + rank)

        best_idx = np.argsort(rrf)[::-1][:k]
        return _rows_to_results(
            self.tfidf.metadata,
            best_idx,
            rrf,
            sparse_scores,
            dense_scores,
        )

