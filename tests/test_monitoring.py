"""Tests for live RAG health logging. No MLflow server required."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthcare_rag import monitoring as mon


def _record(**overrides) -> mon.QueryHealth:
    base = dict(latency_total_s=1.2, n_hits=3, max_dense=0.8, outcome="answered")
    base.update(overrides)
    return mon.QueryHealth(**base)


def _client(monkeypatch) -> MagicMock:
    client = MagicMock()
    client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
    run = MagicMock()
    run.info.run_id = "run-1"
    client.create_run.return_value = run
    monkeypatch.setattr(mon, "monitoring_enabled", lambda: True)
    monkeypatch.setattr(mon, "monitoring_available", lambda: True)
    monkeypatch.setattr(mon, "_client", lambda: client)
    return client


def _keys(client: MagicMock, method: str, index: int) -> list:
    return [call.args[index] for call in getattr(client, method).call_args_list]


class TestLogQuery:
    def test_disabled_does_not_touch_mlflow(self, monkeypatch):
        monkeypatch.setattr(mon, "monitoring_enabled", lambda: False)

        def _boom():
            raise AssertionError("client should not be created")

        monkeypatch.setattr(mon, "_client", _boom)
        assert mon.log_query(_record()) is False

    def test_writes_metrics_without_question_text(self, monkeypatch):
        client = _client(monkeypatch)
        ok = mon.log_query(
            _record(abstained=False, citation_support=0.9, specialty="cardiology")
        )
        assert ok is True
        metric_keys = _keys(client, "log_metric", 1)
        param_values = " ".join(_keys(client, "log_param", 2))
        assert "latency_total_s" in metric_keys
        assert "max_dense" in metric_keys
        assert "abstained" in metric_keys
        assert "citation_support" in metric_keys
        assert "question" not in metric_keys
        assert "hypertension" not in param_values
        client.set_terminated.assert_called_once_with("run-1", status="FINISHED")

    def test_session_is_logged_without_question_text(self, monkeypatch):
        client = _client(monkeypatch)
        mon.log_query(_record(session_id="abc123", specialty="cardiology"))
        params = {call.args[1]: call.args[2] for call in client.log_param.call_args_list}
        assert params["session"] == "abc123"
        assert "question" not in params

    def test_skips_unset_optional_metrics(self, monkeypatch):
        client = _client(monkeypatch)
        mon.log_query(_record(abstained=None, citation_support=None))
        assert "abstained" not in _keys(client, "log_metric", 1)
        assert "citation_support" not in _keys(client, "log_metric", 1)

    def test_failed_query_is_marked_failed(self, monkeypatch):
        client = _client(monkeypatch)
        assert mon.log_query(_record(error=True, outcome="error")) is True
        client.set_terminated.assert_called_once_with("run-1", status="FAILED")

    def test_write_failure_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(mon, "monitoring_enabled", lambda: True)
        monkeypatch.setattr(mon, "monitoring_available", lambda: True)

        def _boom():
            raise RuntimeError("tracking store down")

        monkeypatch.setattr(mon, "_client", _boom)
        assert mon.log_query(_record()) is False
        assert "tracking store down" in (mon.last_monitor_error() or "")


class TestSummarizeAndAlerts:
    def _df(self, n, **columns):
        data = {"latency_s": [1.0] * n, "error": [0.0] * n}
        data.update(columns)
        frame = pd.DataFrame(data)
        frame["timestamp"] = pd.date_range("2026-01-01", periods=len(frame), freq="min")
        return frame

    def test_summarize_rates(self):
        df = self._df(4, error=[1.0, 0.0, 0.0, 0.0], abstained=[1.0, 0.0, None, 0.0])
        stats = mon.summarize(df)
        assert stats["n_queries"] == 4
        assert stats["error_rate"] == 0.25
        assert abs(stats["abstention_rate"] - (1 / 3)) < 1e-9
        assert stats["p95_latency_s"] == 1.0

    def test_rolling_rates_follow_recent_queries(self):
        df = self._df(4, error=[0.0, 0.0, 1.0, 1.0], abstained=[0.0, 1.0, 1.0, 1.0])
        rolled = mon.rolling_health(df, window=2)
        assert list(rolled["error_rate"]) == [0.0, 0.0, 0.5, 1.0]
        assert abs(rolled["abstention_rate"].iloc[-1] - 1.0) < 1e-9

    def test_no_alert_below_minimum_sample(self):
        df = self._df(4, error=[1.0, 1.0, 1.0, 1.0])
        assert mon.health_alerts(df) == []

    def test_alerts_on_failures_abstention_support_and_latency(self):
        df = self._df(
            5,
            error=[1.0, 0.0, 0.0, 0.0, 0.0],
            abstained=[1.0, 1.0, 1.0, 0.0, 0.0],
            citation_support=[0.2, 0.2, 0.2, 0.2, 0.2],
            max_dense=[0.1, 0.1, 0.1, 0.1, 0.1],
            latency_s=[12.0, 12.0, 12.0, 12.0, 12.0],
        )
        messages = " ".join(mon.health_alerts(df))
        assert "failing" in messages
        assert "abstaining" in messages
        assert "Citation support" in messages
        assert "Retrieval confidence" in messages
        assert "slow" in messages


class TestRecentRuns:
    def test_none_when_mlflow_missing(self, monkeypatch):
        monkeypatch.setattr(mon, "monitoring_available", lambda: False)
        assert mon.recent_runs() is None


class TestRetrievalScores:
    def test_empty(self):
        assert mon.retrieval_scores([]) == (None, None)

    def test_max_dense_and_mean_rrf(self):
        hits = [
            MagicMock(score=0.2, score_dense=0.4),
            MagicMock(score=0.4, score_dense=None),
            MagicMock(score=0.6, score_dense=0.9),
        ]
        max_dense, mean_rrf = mon.retrieval_scores(hits)
        assert max_dense == 0.9
        assert abs(mean_rrf - 0.4) < 1e-9
