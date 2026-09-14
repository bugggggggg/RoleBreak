"""The metric contract and the judge abstraction it leans on.

Every scorer implements :class:`Metric`. Most RoleVoiceBench dimensions can't be
measured with a closed-form formula — "did it stay in character?", "is this the
right emotion?", "is this a jailbreak?" are judgment calls. We funnel those
through a :class:`Judge`: a pluggable rater (an LLM, an audio-LM, a human, or a
specialized model like a speaker-verification or emotion-recognition net). A
metric declares *what* to assess; the judge decides *how*. This keeps scoring
logic independent of which model does the rating, mirroring how
:class:`~rolebreak.models.SpeechToSpeechModel` is independent of its backend.
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from rolebreak.types import Dimension, MetricScore, Transcript


@runtime_checkable
class Judge(Protocol):
    """A rater a metric delegates a judgment to.

    Implementations might wrap a Claude/LLM text judge, an audio language model
    that hears the reply, a wav2vec emotion classifier, an ECAPA speaker-
    verification model, or a human-in-the-loop queue. The metric supplies a
    rubric and the material; the judge returns a number and its reasoning.

    A judge rates on whatever scale is natural to it (``scale`` below, or a raw
    MOS / cosine similarity); the metric that owns it puts the result on the
    benchmark's 0-100.
    """

    def rate(
        self,
        instruction: str,
        *,
        text: str | None = None,
        audio: bytes | None = None,
        reference_audio: bytes | None = None,
        scale: tuple[float, float] = (0.0, 1.0),
        context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        """Score ``text``/``audio`` against ``instruction`` on ``scale``."""
        ...


@dataclass
class JudgeVerdict:
    """What a :class:`Judge` returns: a score plus a free-form ``meta`` dict.

    ``meta`` carries whatever extra the judge wants to surface — rationale,
    predicted label, raw model output (e.g. ``{"cosine": 0.83}``).
    """

    score: float | None
    meta: dict[str, Any] = field(default_factory=dict)


class Metric(abc.ABC):
    """Scores a :class:`Transcript`, emitting one or more :class:`MetricScore`.

    Each returned score names its own :class:`Dimension`, so a single metric
    (e.g. a combined safety+emotion probe) may report across several axes.

    Construction stays cheap (like the model backends): keep config here, defer
    loading any judge model until :meth:`evaluate` actually runs. A metric that
    delegates to a rater declares *how* to build it in :meth:`_build_judge` and
    reaches for it through the :attr:`judge` property; the object is built once,
    lazily, on first access and cached — so merely instantiating a metric (e.g.
    via :func:`~rolebreak.metrics.load_metric`) never eagerly constructs a judge
    or loads the model behind it. A metric that needs to hear the audio sets
    ``requires_audio = True`` so a runner can skip it (or warn) when only
    transcripts are available.

    A metric emits ``score`` and ``per_turn`` on the benchmark's ``[0, 100]``,
    higher-is-better scale, converting from whatever its judge returns.
    """

    #: Stable identifier, e.g. ``"persona_consistency"``.
    name: str = "metric"
    #: True if the metric inspects the waveform, not just the transcript text.
    requires_audio: bool = False

    def __init__(self, **config: Any) -> None:
        self.config = config
        self._judge: Any = None  # lazily built on first `judge` access; see below
        # A metric instance is shared across worker threads when a run is scored
        # with ``parallel > 1``, so the lazy build below must happen exactly once.
        self._judge_lock = threading.Lock()

    @property
    def judge(self) -> Any:
        """The rater this metric delegates to, built lazily and cached.

        The concrete object comes from :meth:`_build_judge`, called the first
        time this is accessed (inside :meth:`evaluate`), not in ``__init__`` — so
        constructing a metric never eagerly builds a judge or loads its model.

        Usually a :class:`Judge`, but intentionally untyped: some dimensions hold
        a plain LLM client (:class:`~rolebreak.metrics.judges.OpenAICompatibleLLM`)
        they drive via ``.generate`` rather than ``.rate``. Tests may inject a
        stand-in by assigning ``metric._judge`` before ``evaluate`` runs.

        The build is guarded by a lock: when several transcripts are scored
        concurrently they share one metric instance, and two threads racing here
        would otherwise load the judge model twice.
        """
        if self._judge is None:
            with self._judge_lock:
                if self._judge is None:
                    self._judge = self._build_judge()
        return self._judge

    def _build_judge(self) -> Any:
        """Construct this metric's rater. Overridden by metrics that use one."""
        raise NotImplementedError(f"{type(self).__name__} does not define a judge; override _build_judge().")

    @abc.abstractmethod
    def evaluate(self, transcript: Transcript) -> list[MetricScore]:
        """Score ``transcript``.

        Always returns a list of :class:`MetricScore` — usually one, but a probe
        that scores several dimensions from one underlying model call returns
        several. A single-dimension metric returns ``[self._score(...)]``.
        """

    # Small helper so subclasses build results without repeating the boilerplate.
    # Each score names its own axis: a metric may report several dimensions.
    def _score(self, score: float, *, dimension: Dimension, **kw: Any) -> MetricScore:
        return MetricScore(metric=self.name, dimension=dimension, score=score, **kw)
