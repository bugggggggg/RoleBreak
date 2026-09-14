"""Resolve metrics by name, and run a suite over a transcript.

    from rolebreak.metrics import load_metric, evaluate_all

    persona = load_metric("persona_consistency")   # metric builds its own judge
    [score] = persona.evaluate(transcript)          # evaluate always returns a list

    report = evaluate_all(transcript)               # the default suite, one transcript
    reports = evaluate_many(transcripts)            # same suite, one judge load for all

Register a custom metric with the :func:`register_metric` decorator. Built-in
dimension metrics in :mod:`rolebreak.metrics.dimensions` self-register on import.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from rolebreak.metrics.base import Metric
from rolebreak.metrics.store import PARTIAL_SUFFIX, ScoreStore
from rolebreak.types import BenchmarkReport, MetricScore, Transcript

logger = logging.getLogger(__name__)

#: Strips the "<run> turn <n>: " context a metric prefixes onto a failure, so
#: the same underlying fault from different runs counts as one reason.
_RUN_CONTEXT = re.compile(r"^\S+ turn \d+: ")

# name -> Metric subclass.
_REGISTRY: dict[str, type[Metric]] = {}

DEFAULT_METRICS: list[str] = [
    "voice_consistency",
    # "text_quality",
    # "emotion",
    "naturalness",
]


def register_metric(cls: type[Metric]) -> type[Metric]:
    """Class decorator registering a metric under its ``name`` attribute."""
    if not getattr(cls, "name", None) or cls.name == "metric":
        raise ValueError(f"{cls.__name__} must set a unique `name` to be registered.")
    _REGISTRY[cls.name] = cls
    return cls


def get_metric(name: str) -> type[Metric]:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown metric {name!r}. Registered: {sorted(_REGISTRY)}.")
    return _REGISTRY[name]


def load_metric(name: str, **config: Any) -> Metric:
    """Instantiate the metric registered under ``name``.

    Each metric hardcodes its own judge in ``__init__``; ``config`` carries only
    metric-specific knobs, not the rater.
    """
    return get_metric(name)(**config)


def available_metrics() -> list[str]:
    """Names of all registered metrics."""
    return sorted(_REGISTRY)


def evaluate_all(
    transcript: Transcript,
    *,
    store: ScoreStore,
    metrics: list[str] | None = None,
    skip_missing_audio: bool = True,
    **config: Any,
) -> BenchmarkReport:
    """Run a set of metrics over one transcript and collect the scores.

    Defaults to the hand-picked :data:`DEFAULT_METRICS` suite; pass ``metrics=``
    to restrict or extend it (this also opts in probes, which are excluded from
    the default suite so they don't double-count the metrics they subsume).
    Metrics that need audio are skipped (not failed) when the transcript has
    none, so a text-only run still produces a partial report.
    """
    # `store` is named explicitly rather than left to **config, which is metric
    # config and would be handed to load_metric.
    [report] = evaluate_many([transcript], store=store, metrics=metrics, **config)
    return report


def evaluate_many(
    transcripts: Sequence[Transcript],
    *,
    store: ScoreStore,
    metrics: list[str] | None = None,
    parallel: int = 1,
    **config: Any,
) -> list[BenchmarkReport]:
    """Run a set of metrics over several transcripts, one report each."""
    if parallel < 1:
        raise ValueError(f"parallel must be >= 1, got {parallel}.")
    names = metrics if metrics is not None else list(DEFAULT_METRICS)
    reports = [BenchmarkReport(transcript=t.name) for t in transcripts]
    for name in names:
        done = store.finished(name)
        if done is not None and all(t.name in done for t in transcripts):
            for report, transcript in zip(reports, transcripts, strict=True):
                report.scores.extend(done[transcript.name])
            continue

        with store.open(name):
            hits: list[list[MetricScore] | None] = [store.cached(t) for t in transcripts]
            if all(hit is not None for hit in hits):
                for report, hit in zip(reports, hits, strict=True):
                    report.scores.extend(hit or [])
                continue

            metric = load_metric(name, **config)
            todo = [i for i, hit in enumerate(hits) if hit is None]
            failures: list[BaseException] = []
            if parallel > 1 and len(todo) > 1:
                produced = _score_concurrently(metric, transcripts, todo, parallel=parallel, failures=failures)
            else:
                produced = ((i, metric.evaluate(transcripts[i])) for i in tqdm(todo, desc=name))
            scored: dict[int, list[MetricScore]] = {}
            for i, verdict in produced:
                store.record(transcripts[i], verdict)
                scored[i] = verdict
            if failures:
                # Skipped runs have no verdict, so the sweep isn't finished: the
                # rows stay in the partial and a re-run retries only these.
                store.incomplete = True
                _report_failures(name, failures, len(todo))
            for i, (report, hit) in enumerate(zip(reports, hits, strict=True)):
                report.scores.extend(hit if hit is not None else scored.get(i, []))
            del metric  # drop the judge model before the next metric loads its own
    return reports


def _score_concurrently(
    metric: Metric,
    transcripts: Sequence[Transcript],
    todo: Sequence[int],
    *,
    parallel: int,
    failures: list[BaseException],
) -> Iterator[tuple[int, list[MetricScore]]]:
    """Score the ``todo`` transcripts on a thread pool, yielding as they land.

    ``(index, scores)`` in completion order. Only ``evaluate`` runs on the pool;
    a verdict is a handful of numbers, so it's handed back to the caller's thread
    to be recorded there rather than having every worker write to the store.

    The metric instance (and its judge) is shared, so its ``evaluate`` must be
    re-entrant — the built-in ones only read the transcript they're handed. The
    judge is built here, before the pool starts, so the workers can't race to
    load the same model; metrics without a judge simply have nothing to warm.

    A worker that raises is collected into ``failures`` and its run skipped, so
    one bad run can't throw away the verdicts of the ones still in flight — every
    future is drained, and every success reaches the caller to be recorded.
    """
    try:
        _ = metric.judge
    except NotImplementedError:
        pass  # a judge-free metric (a formula, a lookup) — nothing to pre-build

    with ThreadPoolExecutor(max_workers=min(parallel, len(todo))) as pool:
        futures = {pool.submit(metric.evaluate, transcripts[i]): i for i in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{metric.name} (x{parallel})"):
            index = futures[future]
            try:
                scores = future.result()
            except Exception as e:  # noqa: BLE001 — whatever a metric raises, the other runs still count
                failures.append(e)
                logger.debug("%s: %s failed", metric.name, transcripts[index].name, exc_info=True)
                continue
            yield index, scores


def _report_failures(metric: str, failures: Sequence[BaseException], attempted: int) -> None:
    """Warn how many runs the sweep skipped, and what most of them tripped on."""
    reasons = Counter(f"{type(e).__name__}: {_RUN_CONTEXT.sub('', str(e))}" for e in failures)
    reason, count = reasons.most_common(1)[0]
    logger.warning(
        "%s: skipped %d of %d run(s); most common reason (%d of %d): %s",
        metric,
        len(failures),
        attempted,
        count,
        len(failures),
        reason,
    )
    logger.warning(
        "%s: left unpublished in <run-dir>/%s.jsonl%s — re-run to retry the skipped runs",
        metric,
        metric,
        PARTIAL_SUFFIX,
    )
