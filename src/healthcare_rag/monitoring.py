"""Live RAG health logging via MLflow.

Each answered question becomes one run in the ``healthcare-rag-health``
experiment: latency, retrieval confidence, abstention, and citation
faithfulness. Question text is never stored.

Logging is a no-op when MLflow is not installed or ``MLFLOW_MONITOR`` is set
to ``0`` / ``false`` / ``off``. Failures are swallowed so a tracking problem
cannot break a user query; the last error is kept for the Health tab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from healthcare_rag.config import MLFLOW_HEALTH_EXPERIMENT, MLFLOW_TRACKING_URI

# Recent-window checks. Alerts wait until the window has enough queries so a
# single odd question does not look like an outage.
_ALERT_WINDOW = 50
_MIN_ALERT_N = 5
_ERROR_RATE_MAX = 0.10
_ABSTENTION_RATE_MAX = 0.50
_SUPPORT_MIN = 0.45
_DENSE_MIN = 0.35
_P95_LATENCY_MAX_S = 10.0

_METRIC_COLUMNS = {
    "latency_s": "metrics.latency_total_s",
    "retrieval_s": "metrics.latency_retrieval_s",
    "generation_s": "metrics.latency_generation_s",
    "grounding_s": "metrics.latency_grounding_s",
    "max_dense": "metrics.max_dense",
    "mean_rrf": "metrics.mean_rrf",
    "abstained": "metrics.abstained",
    "citation_support": "metrics.citation_support",
    "citation_coverage": "metrics.citation_coverage",
    "invalid_citations": "metrics.invalid_citations",
    "error": "metrics.error",
    "n_hits": "metrics.n_hits",
}

_last_error: Optional[str] = None


@dataclass
class QueryHealth:
    """One query's operational signals. No question text."""

    latency_retrieval_s: float = 0.0
    latency_generation_s: float = 0.0
    latency_grounding_s: float = 0.0
    latency_total_s: float = 0.0
    n_hits: int = 0
    max_dense: Optional[float] = None
    mean_rrf: Optional[float] = None
    abstained: Optional[bool] = None
    citation_coverage: Optional[float] = None
    citation_support: Optional[float] = None
    invalid_citations: Optional[int] = None
    n_claims: Optional[int] = None
    top_k: int = 3
    rerank: bool = False
    specialty: str = "all"
    session_id: str = ""
    llm_enabled: bool = False
    generator: str = ""
    error: bool = False
    outcome: str = "answered"


def monitoring_enabled() -> bool:
    flag = os.getenv("MLFLOW_MONITOR", "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def monitoring_available() -> bool:
    try:
        import mlflow  # noqa: F401
    except Exception:
        return False
    return True


def last_monitor_error() -> Optional[str]:
    return _last_error


def log_query(record: QueryHealth) -> bool:
    """Persist one query. Returns True when a run was written."""
    global _last_error
    if not monitoring_enabled() or not monitoring_available():
        return False
    try:
        client = _client()
        _write(client, record)
    except Exception as exc:
        _last_error = str(exc)
        return False
    _last_error = None
    return True


def recent_runs(limit: int = 200, session_id: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Recent health runs, oldest first. None when MLflow is not installed.

    ``session_id`` limits the frame to one visit. Older runs stay in the
    tracking store and are left out of the session exhibit.
    """
    if not monitoring_available():
        return None
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = _client()
    experiment = client.get_experiment_by_name(MLFLOW_HEALTH_EXPERIMENT)
    if experiment is None:
        return _empty_frame()
    filter_string = None
    if session_id:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in "-_")
        filter_string = f"params.session = '{safe}'"
    raw = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=limit,
        order_by=["start_time DESC"],
        filter_string=filter_string,
    )
    return _normalize(raw)


def rolling_health(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Rolling means used by the health charts. One row per logged query."""
    columns = [
        "timestamp",
        "error_rate",
        "abstention_rate",
        "citation_support",
        "latency_s",
        "max_dense",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({"timestamp": df["timestamp"] if "timestamp" in df.columns else pd.NaT})

    def _roll(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series([pd.NA] * len(df), index=df.index)
        return pd.to_numeric(df[column], errors="coerce").rolling(window, min_periods=1).mean()

    out["error_rate"] = _roll("error")
    out["abstention_rate"] = _roll("abstained")
    out["citation_support"] = _roll("citation_support")
    out["latency_s"] = _roll("latency_s")
    out["max_dense"] = _roll("max_dense")
    return out.reset_index(drop=True)


def summarize(df: pd.DataFrame) -> dict:
    """Aggregate a health frame into rates the dashboard can show."""
    lat = _series(df, "latency_s")
    return {
        "n_queries": int(len(df)),
        "error_rate": _mean(df, "error"),
        "abstention_rate": _mean(df, "abstained"),
        "mean_citation_support": _mean(df, "citation_support"),
        "mean_citation_coverage": _mean(df, "citation_coverage"),
        "mean_max_dense": _mean(df, "max_dense"),
        "mean_latency_s": float(lat.mean()) if len(lat) else None,
        "p95_latency_s": float(lat.quantile(0.95)) if len(lat) else None,
    }


def health_alerts(df: pd.DataFrame) -> List[str]:
    """Plain-language warnings for the recent query window."""
    if df is None or df.empty or len(df) < _MIN_ALERT_N:
        return []
    window = df.tail(_ALERT_WINDOW)
    alerts: List[str] = []

    error_rate = _mean(window, "error")
    if error_rate is not None and error_rate >= _ERROR_RATE_MAX:
        alerts.append(
            f"Answers are failing often ({error_rate:.0%} of recent queries errored)."
        )

    abstention = _mean(window, "abstained")
    abstain_n = _series(window, "abstained")
    if (
        abstention is not None
        and len(abstain_n) >= _MIN_ALERT_N
        and abstention >= _ABSTENTION_RATE_MAX
    ):
        alerts.append(
            f"The assistant is abstaining often ({abstention:.0%}). "
            "Recent questions may sit outside the corpus."
        )

    support = _mean(window, "citation_support")
    support_n = _series(window, "citation_support")
    if support is not None and len(support_n) >= _MIN_ALERT_N and support < _SUPPORT_MIN:
        alerts.append(
            f"Citation support is low (mean {support:.0%}). "
            "Claims are only weakly backed by the passages they cite."
        )

    dense = _mean(window, "max_dense")
    dense_n = _series(window, "max_dense")
    if dense is not None and len(dense_n) >= _MIN_ALERT_N and dense < _DENSE_MIN:
        alerts.append(
            f"Retrieval confidence is low (mean best semantic score {dense:.2f}). "
            "The corpus may not cover what people are asking."
        )

    lat = _series(window, "latency_s")
    if len(lat) >= _MIN_ALERT_N:
        p95 = float(lat.quantile(0.95))
        if p95 > _P95_LATENCY_MAX_S:
            alerts.append(
                f"Responses are slow (95th percentile {p95:.1f}s)."
            )
    return alerts


def _client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def _experiment_id(client) -> str:
    experiment = client.get_experiment_by_name(MLFLOW_HEALTH_EXPERIMENT)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return client.create_experiment(MLFLOW_HEALTH_EXPERIMENT)
    except Exception:
        experiment = client.get_experiment_by_name(MLFLOW_HEALTH_EXPERIMENT)
        if experiment is None:
            raise
        return experiment.experiment_id


def _write(client, record: QueryHealth) -> None:
    experiment_id = _experiment_id(client)
    run = client.create_run(
        experiment_id,
        tags={
            "mlflow.runName": "query",
            "source": "streamlit",
        },
    )
    run_id = run.info.run_id
    for key, value in _metrics(record).items():
        client.log_metric(run_id, key, float(value))
    for key, value in _params(record).items():
        client.log_param(run_id, key, value)
    client.set_terminated(run_id, status="FAILED" if record.error else "FINISHED")


def _metrics(record: QueryHealth) -> dict:
    metrics = {
        "latency_total_s": record.latency_total_s,
        "latency_retrieval_s": record.latency_retrieval_s,
        "latency_generation_s": record.latency_generation_s,
        "latency_grounding_s": record.latency_grounding_s,
        "n_hits": record.n_hits,
        "error": 1.0 if record.error else 0.0,
        "llm_enabled": 1.0 if record.llm_enabled else 0.0,
    }
    optional = {
        "max_dense": record.max_dense,
        "mean_rrf": record.mean_rrf,
        "abstained": None if record.abstained is None else (1.0 if record.abstained else 0.0),
        "citation_coverage": record.citation_coverage,
        "citation_support": record.citation_support,
        "invalid_citations": record.invalid_citations,
        "n_claims": record.n_claims,
    }
    for key, value in optional.items():
        if value is not None:
            metrics[key] = value
    return metrics


def _params(record: QueryHealth) -> dict:
    specialty = (record.specialty or "all").strip() or "all"
    params = {
        "top_k": str(int(record.top_k)),
        "rerank": "true" if record.rerank else "false",
        "specialty": specialty[:250],
        "outcome": record.outcome[:250],
        "generation": (record.generator or ("llm" if record.llm_enabled else "offline"))[:32],
    }
    session = (record.session_id or "").strip()
    if session:
        params["session"] = session[:250]
    return params


def _empty_frame() -> pd.DataFrame:
    columns = ["timestamp", "outcome", "rerank", "specialty", "generation", *_METRIC_COLUMNS]
    return pd.DataFrame(columns=columns)


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_frame()
    out = pd.DataFrame(index=raw.index)
    if "start_time" in raw.columns:
        out["timestamp"] = pd.to_datetime(raw["start_time"])
    else:
        out["timestamp"] = pd.NaT
    for dest, src in {
        "outcome": "params.outcome",
        "rerank": "params.rerank",
        "specialty": "params.specialty",
        "generation": "params.generation",
    }.items():
        out[dest] = raw[src] if src in raw.columns else pd.NA
    for dest, src in _METRIC_COLUMNS.items():
        if src in raw.columns:
            out[dest] = pd.to_numeric(raw[src], errors="coerce")
        else:
            out[dest] = pd.NA
    out = out.dropna(subset=["timestamp"]) if out["timestamp"].notna().any() else out
    return out.sort_values("timestamp").reset_index(drop=True)


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    if df is None or df.empty or column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _mean(df: pd.DataFrame, column: str) -> Optional[float]:
    series = _series(df, column)
    if series.empty:
        return None
    return float(series.mean())


def retrieval_scores(hits: Sequence) -> Tuple[Optional[float], Optional[float]]:
    """Best dense score and mean RRF score for a hit list."""
    if not hits:
        return None, None
    dense = [h.score_dense for h in hits if getattr(h, "score_dense", None) is not None]
    max_dense = max(dense) if dense else None
    mean_rrf = sum(float(h.score) for h in hits) / len(hits)
    return max_dense, mean_rrf
