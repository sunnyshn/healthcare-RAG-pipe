"""Open the library as soon as the process starts.

The page is allowed to render while the models are still loading. A question
waits only for whatever is left.
"""

from __future__ import annotations

import threading

from healthcare_rag.config import INDEX_PATH, PROCESSED_PATH, RAW_PATH

_models_ready = threading.Event()
_index_ready = threading.Event()
_lock = threading.Lock()
_started = False
_retriever = None
_error: BaseException | None = None


def models_ready() -> bool:
    return _models_ready.is_set()


def wait_for_models() -> None:
    _models_ready.wait()


def replace_retriever(retriever) -> None:
    """Swap in a rebuilt index without waiting for the background load."""
    global _retriever, _error
    _retriever = retriever
    _error = None
    _index_ready.set()
    _models_ready.set()


def get_retriever():
    """Return the index. Blocks only until the file is open, not until the models are warm."""
    if not _started:
        start()
    _index_ready.wait()
    if _error is not None:
        raise _error
    return _retriever


def start() -> None:
    """Start one background load. Later calls do nothing."""
    global _started
    with _lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_run, name="library-boot", daemon=True).start()


def _load_index():
    from healthcare_rag.data_io import normalize_documents, read_jsonl
    from healthcare_rag.retriever import HybridRetriever

    if INDEX_PATH.exists():
        return HybridRetriever.load(INDEX_PATH)

    df = normalize_documents(read_jsonl(RAW_PATH))
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    retriever = HybridRetriever.build(df)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    retriever.save(INDEX_PATH)
    return retriever


def _warm(retriever) -> None:
    from healthcare_rag.generator import warm_local_generator
    from healthcare_rag.retriever import SearchResult

    encoder = retriever._encoder_model()
    list(encoder.embed(["query: warmup"], batch_size=1, parallel=1))
    dummy = SearchResult(
        doc_id="warmup",
        title="warmup",
        text="warmup passage",
        score=0.0,
        source="",
        year=0,
    )
    retriever._reranker_model().rerank("warmup", [dummy], top_k=1)
    warm_local_generator()


def _run() -> None:
    global _retriever, _error
    try:
        _retriever = _load_index()
    except Exception as exc:
        _error = exc
        _models_ready.set()
        return
    finally:
        _index_ready.set()

    try:
        _warm(_retriever)
    except Exception:
        return
    finally:
        _models_ready.set()
