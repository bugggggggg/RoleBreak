"""Persist metric scores next to a run, one file per metric, so eval can resume."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rolebreak.types import MetricScore, Transcript

#: Bumped when the row layout changes, so stale files are ignored, not misread.
#: 2: the run key moved from ``transcript`` to ``name``, matching the run log.
SCHEMA = 2

PARTIAL_SUFFIX = ".partial"  # a score file whose sweep hasn't finished


class ScoreStore:
    """Per-metric score files in ``directory``, read for resume and appended to."""

    def __init__(
        self,
        directory: str | Path,
        *,
        config: dict[str, Any] | None = None,
        resume: bool = True,
    ) -> None:
        self.directory = Path(directory)
        self.config = config or {}
        self.resume = resume
        #: The metric whose sweep is open, set by :meth:`open`.
        self.metric: str | None = None
        #: Set by a caller that skipped runs it couldn't score. The sweep didn't
        #: cover every run, so its partial is held back rather than published as
        #: a finished file the next run would resume straight past.
        self.incomplete = False
        # That metric's rows: run name -> scores.
        self._scores: dict[str, list[MetricScore]] = {}

    def path_for(self, metric: str) -> Path:
        """``metric``'s published file — present only when its sweep finished."""
        return self.directory / f"{metric}.jsonl"

    @property
    def path(self) -> Path:
        """The published file for the open metric — only ever a finished sweep."""
        assert self.metric, "ScoreStore is used inside a `with store.open(metric):` block."
        return self.path_for(self.metric)

    @property
    def partial_path(self) -> Path:
        """Where the open metric's rows land until the sweep finishes."""
        return self.path.with_name(self.path.name + PARTIAL_SUFFIX)

    def open(self, metric: str) -> "ScoreStore":
        """Start ``metric``'s sweep: read its rows for resume, open its partial."""
        self.metric = metric
        self.incomplete = False
        self.directory.mkdir(parents=True, exist_ok=True)
        self._scores = self._load()
        self._start_partial()
        return self

    # ------------------------------------------------------------------ read --
    def finished(self, metric: str) -> dict[str, list[MetricScore]] | None:
        """``metric``'s verdicts if its sweep is already done, else ``None``."""
        path = self.path_for(metric)
        if not self.resume or not path.exists():
            return None
        found: dict[str, list[MetricScore]] = {}
        self._parse_into(found, path, metric)
        return found

    def _load(self) -> dict[str, list[MetricScore]]:
        """Every usable row on disk for the open metric: run name -> scores."""
        found: dict[str, list[MetricScore]] = {}
        if self.resume:
            # The published sweep first, then a partial over the top: a sweep that
            # died left its rows there, and they're the newer verdict.
            for path in (self.path, self.partial_path):
                if path.exists():
                    self._parse_into(found, path, self.metric)
        return found

    def _parse_into(self, found: dict[str, list[MetricScore]], path: Path, metric: str | None) -> None:
        """Read one score file's usable rows into ``found``."""
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a hard kill; drop it and rescore that run
            if row.get("schema") != SCHEMA or row.get("metric") != metric:
                continue
            if row.get("config", {}) != self.config:
                continue  # scored under different knobs; not a hit
            key = row.get("name")
            if not isinstance(key, str):
                continue
            # A later row for the same run wins, so re-scoring appends rather
            # than needing a rewrite.
            found[key] = [MetricScore.from_dict(s) for s in row.get("scores", [])]

    def cached(self, transcript: Transcript) -> list[MetricScore] | None:
        """Scores already on disk for this run, or ``None`` to score it.

        An empty list is a hit, not a miss: it records a metric that ran and
        emitted nothing (e.g. an audio metric on a text-only run).
        """
        return self._scores.get(transcript.name)

    # ----------------------------------------------------------------- write --
    def _start_partial(self) -> None:
        """Open the sweep's partial, holding everything :meth:`_load` just read.

        The partial is what :meth:`close` publishes, so it must start out holding
        every row :meth:`_load` answers ``cached`` from — otherwise a run counted
        as cached, and so never rescored, is dropped from the file at close.
        That means seeding it with the published sweep even when a partial from a
        dead sweep is already there, with those rows kept after the seed so the
        newer verdict still wins on read.
        """
        if not self.resume:
            self.partial_path.write_text("", encoding="utf-8")
            return
        text = self._rows(self.path) + self._rows(self.partial_path)
        # Via a temp file: rewriting the partial in place would lose the dead
        # sweep's rows if this process died mid-write.
        tmp = self.partial_path.with_name(self.partial_path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.partial_path)

    @staticmethod
    def _rows(path: Path) -> str:
        """``path``'s text, newline-terminated so rows can be concatenated."""
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            text += "\n"  # a torn last line stays one line, and is skipped on read
        return text

    def record(self, transcript: Transcript, scores: list[MetricScore]) -> None:
        """Append this run's verdict for the open metric, flushed to disk."""
        row = {
            "schema": SCHEMA,
            "name": transcript.name,
            "metric": self.metric,
            "config": self.config,
            "scores": [s.to_dict() for s in scores],
        }
        with self.partial_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._scores[transcript.name] = scores

    def close(self) -> None:
        """Publish the partial at the real path — the sweep is complete.

        A sweep that scored nothing new still republishes the rows it was seeded
        with, which costs a rename and keeps the rule simple: a finished sweep
        ends with its file at the real path.
        """
        if self.partial_path.exists():
            self.partial_path.replace(self.path)

    def __enter__(self) -> "ScoreStore":
        return self

    def __exit__(self, exc_type: object, *exc: object) -> None:
        if exc_type is None and not self.incomplete:
            self.close()
