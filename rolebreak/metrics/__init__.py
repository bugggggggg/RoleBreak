"""RoleVoiceBench metrics: score a speech role-play rollout on several axes."""

from __future__ import annotations

from rolebreak.metrics import dimensions  # noqa: F401  -- triggers self-registration of built-in metrics
from rolebreak.metrics.base import Judge, JudgeVerdict, Metric
from rolebreak.metrics.judges import JudgeCallError
from rolebreak.metrics.registry import (
    DEFAULT_METRICS,
    available_metrics,
    evaluate_all,
    evaluate_many,
    get_metric,
    load_metric,
    register_metric,
)
from rolebreak.metrics.store import ScoreStore
from rolebreak.types import (
    BenchmarkReport,
    Dimension,
    Exchange,
    MetricScore,
    Rubric,
    Transcript,
)

__all__ = [
    # contract
    "Metric",
    "Judge",
    "JudgeVerdict",
    "JudgeCallError",
    # registry
    "register_metric",
    "load_metric",
    "get_metric",
    "available_metrics",
    "evaluate_all",
    "evaluate_many",
    "DEFAULT_METRICS",
    # score persistence / resume
    "ScoreStore",
    # types
    "Dimension",
    "Transcript",
    "Exchange",
    "MetricScore",
    "Rubric",
    "BenchmarkReport",
]
