"""Naturalness: does the generated speech *sound* like good speech?

Orthogonal to every other axis here. Persona, emotion and rubric ask what was
said and how it was delivered; voice asks *who* is speaking. This one ignores all
of that and asks only whether the waveform sounds like clean, natural, human
speech — the vocoder buzz, muffling, robotic monotone and glitchy artifacts a TTS
or speech-to-speech stack degrades into, especially late in a long rollout.

The judge is a no-reference MOS predictor: it hears one turn and returns the
naturalness MOS a human listener panel would give it (1 = bad, 5 = excellent).
Two are available, from different model families, so they cross-check each other:

* ``judge="utmosv2"`` (default) — :class:`~rolebreak.metrics.judges.UTMOSv2NaturalnessJudge`,
  the T05 system that won the VoiceMOS Challenge 2024 Track 1.
* ``judge="scoreq"`` — :class:`~rolebreak.metrics.judges.ScoreQQualityJudge` in
  its no-reference mode (wav2vec 2.0 + contrastive regression).

MOS is not this benchmark's unit: scores here are ``[0, 100]``, higher better,
so a predicted MOS is mapped linearly off :data:`MOS_RANGE` (``1 -> 0``,
``5 -> 100``). The raw MOS survives in ``meta["mos_per_turn"]`` /
``meta["mean_mos"]`` for anyone who wants to compare against published MOS
numbers. Because ``per_turn`` is populated, :attr:`~rolebreak.types.MetricScore.drift`
answers the long-horizon question directly: did the audio get worse as the
conversation ran on?

Prediction is reference-free, so nothing needs to be authored for this axis — but
it does need audio. Turns without any score ``None``; a rollout with no spoken
turns at all yields **no** score rather than a spurious zero::

    from rolebreak.metrics import load_metric

    metric = load_metric("naturalness")                 # UTMOSv2
    metric = load_metric("naturalness", judge="scoreq")  # or SCOREQ, no-reference
    [score] = metric.evaluate(transcript)                # evaluate always returns a list

Both judges pull weights on first use and need extra packages — see
:mod:`rolebreak.metrics.judges.utmos` and :mod:`rolebreak.metrics.judges.scoreq`
for setup. Pass judge constructor knobs through ``judge_kwargs``, e.g.
``load_metric("naturalness", judge_kwargs={"num_repetitions": 5})`` to average out
UTMOSv2's frame-sampling noise.
"""

from __future__ import annotations

from typing import Any

from rolebreak.metrics.base import Judge, Metric
from rolebreak.metrics.registry import register_metric
from rolebreak.types import Dimension, MetricScore, Transcript

#: The MOS scale both judges predict on (absolute category rating: 1 = bad,
#: 5 = excellent). Used to map a predicted MOS onto this benchmark's [0, 100].
MOS_RANGE = (1.0, 5.0)

_JUDGES = ("utmosv2", "scoreq")
_DEFAULT_JUDGE = "utmosv2"


def _normalize(mos: float) -> float:
    """Map a predicted MOS onto ``[0, 100]``, clamping outside :data:`MOS_RANGE`."""
    lo, hi = MOS_RANGE
    return max(0.0, min(100.0, 100.0 * (mos - lo) / (hi - lo)))


@register_metric
class NaturalnessMetric(Metric):
    name = "naturalness"
    requires_audio = True  # the waveform *is* the thing being judged

    def __init__(
        self,
        *,
        judge: str = _DEFAULT_JUDGE,
        judge_kwargs: dict[str, Any] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(judge=judge, judge_kwargs=judge_kwargs, **config)
        self.judge_name = judge.lower()
        self.judge_kwargs = dict(judge_kwargs or {})
        if self.judge_name not in _JUDGES:
            raise ValueError(f"Unknown naturalness judge {judge!r}. Available: {list(_JUDGES)}.")
        # SCOREQ's ref mode returns an embedding *distance* (lower is better)
        # against a clean copy of the same utterance — not a MOS, and not
        # something this axis can normalize. Reject it up front.
        if self.judge_name == "scoreq" and self.judge_kwargs.get("mode", "nr") != "nr":
            raise ValueError("naturalness needs SCOREQ's no-reference MOS; mode='ref' predicts a distance instead.")

    def _build_judge(self) -> Judge:
        if self.judge_name == "scoreq":
            from rolebreak.metrics.judges import ScoreQQualityJudge

            return ScoreQQualityJudge(**self.judge_kwargs)
        from rolebreak.metrics.judges import UTMOSv2NaturalnessJudge

        return UTMOSv2NaturalnessJudge(**self.judge_kwargs)

    def evaluate(self, transcript: Transcript) -> list[MetricScore]:
        from rolebreak.models.audio import decode_b64

        judge = self.judge

        per_turn: list[float | None] = []
        mos_per_turn: list[float | None] = []
        for ex in transcript.exchanges:
            audio = ex.assistant_audio
            if audio is None:
                per_turn.append(None)  # nothing spoken to listen to this turn
                mos_per_turn.append(None)
                continue
            verdict = judge.rate(
                audio=decode_b64(audio.data),
            )
            mos = verdict.score
            if mos is None:
                per_turn.append(None)  # judge declined / failed on this turn
                mos_per_turn.append(None)
                continue
            mos_per_turn.append(float(mos))
            per_turn.append(_normalize(float(mos)))

        scored = [s for s in per_turn if s is not None]
        if not scored:
            return []  # no audio anywhere — this axis doesn't apply

        rated = [m for m in mos_per_turn if m is not None]
        mean_mos = sum(rated) / len(rated)
        return [
            self._score(
                sum(scored) / len(scored),
                dimension=Dimension.NATURALNESS,
                per_turn=per_turn,
                meta={"judge": self.judge_name, "mean_mos": mean_mos, "mos_per_turn": mos_per_turn},
            )
        ]
