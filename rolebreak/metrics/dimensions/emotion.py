"""Emotion appropriateness: does the spoken reply *sound* the way it should?

This judges emotional delivery from the assistant's **audio**, not its text. A
reply can say the right words in a mismatched tone — cheerful through tragedy,
bored through danger — and that is the failure this metric catches.

The judge does one job: from the audio, name the emotion it *perceives* (one of
the :class:`~rolebreak.types.Emotion` labels). The *score* is then computed here
by checking that label against ``accepted_emotions``:

* **Targeted** — when the authored :class:`~rolebreak.examples.Turn` sets one or
  more accepted :class:`~rolebreak.types.Emotion` values (carried onto the
  :class:`~rolebreak.types.Exchange` as ``accepted_emotions``): the reply
  scores ``100`` when the perceived emotion belongs to that list, else ``0``.
* **No target** — with no authored ``accepted_emotions`` there is nothing to
  judge the delivery against, so the turn is skipped (scores ``None``).
* **No expressive target** — when ``neutral`` or ``calm`` is among the accepted
  values, the turn is skipped too (see :data:`_UNEXPRESSIVE` below).

The rater is a dedicated speech-emotion classifier,
:class:`~rolebreak.metrics.judges.Emotion2vecEmotionJudge` (emotion2vec+ large),
rather than a prompted audio LM: it hears only the waveform, so a transcript full
of emotional words can't talk it into a label, and it is deterministic, offline,
and free to re-run — which matters for a benchmark number others must reproduce.

This axis is deliberately narrow: it asks only whether the model *acted* an
emotion it was told to act. A turn that accepts ``neutral`` or ``calm`` — even
alongside an expressive alternative, as ``angry+calm`` does — can be passed
without expressing anything, because ``neutral`` is what the classifier answers
for any undramatic delivery (and it has no ``calm`` class at all, folding
low-arousal speech into ``neutral``). Those turns are the overwhelming majority
of the authored targets, so counting them would let flat delivery carry the
score. Every one of them is left unscored, and what remains is the turns where
an expressive emotion was the only way to pass. Accepted labels are matched
exactly; a target the classifier cannot name never reaches the comparison.

Turns without audio score ``None``, as do turns whose audio holds no samples; a
rollout with no audio at all is skipped by the runner (``requires_audio = True``).
A rollout where *no* turn is scoreable yields no score at all — this axis simply
doesn't apply to it, and reporting the ``0.0`` an empty mean falls back to would
read as a failure the model never had.

    from rolebreak.metrics import load_metric

    metric = load_metric("emotion")
    [score] = metric.evaluate(transcript)   # evaluate always returns a list
"""

from __future__ import annotations

from rolebreak.metrics.base import Metric
from rolebreak.metrics.judges import Emotion2vecEmotionJudge
from rolebreak.metrics.registry import register_metric
from rolebreak.types import Dimension, Emotion, MetricScore, Transcript


#: Authored emotions that don't ask the model to *act* anything. `neutral` is the
#: judge's default answer for undramatic delivery, and `calm` is a category it
#: cannot name at all; a turn that accepts either can be satisfied without
#: expressing anything, so any turn accepting one is left unscored.
_UNEXPRESSIVE = frozenset({Emotion.NEUTRAL, Emotion.CALM})


@register_metric
class EmotionMetric(Metric):
    name = "emotion"
    requires_audio = True  # delivery can only be judged from the waveform

    def _build_judge(self) -> Emotion2vecEmotionJudge:
        return Emotion2vecEmotionJudge()

    def evaluate(self, transcript: Transcript) -> list[MetricScore]:
        from rolebreak.models.audio import decode_b64

        judge = self.judge
        per_turn: list[float | None] = []
        # Per-turn diagnostics, aligned with `per_turn`: what the classifier heard,
        # how sure it was, and how much probability mass the accepted set drew.
        # The last one shows whether a 0 was a near miss or an outright mismatch —
        # the hard 100/0 above hides that, and it is what an error analysis needs.
        perceived: list[str | None] = []
        confidence: list[float | None] = []
        accepted_mass: list[float | None] = []
        scores: list[dict[str, float] | None] = []  # the classifier's own distribution

        for ex in transcript.exchanges:
            accepted = ex.accepted_emotions
            if not accepted or set(accepted) & _UNEXPRESSIVE or ex.assistant_audio is None:
                # No authored target, or none that asks for an expressive
                # delivery: nothing to judge against. No audio: nothing spoken to
                # judge. Either way the turn goes unscored.
                per_turn.append(None)
                perceived.append(None)
                confidence.append(None)
                accepted_mass.append(None)
                scores.append(None)
                continue

            verdict = judge.rate(audio=decode_b64(ex.assistant_audio.data))
            label = verdict.meta["emotion"]
            perceived.append(label)
            confidence.append(verdict.score)
            if label is None:  # empty waveform; the judge heard nothing to label
                per_turn.append(None)
                accepted_mass.append(None)
                scores.append(None)
                continue

            accepted_labels = {e.value for e in accepted}
            probabilities = verdict.meta["emotion_scores"]
            scores.append(verdict.meta["scores"])
            accepted_mass.append(sum(probabilities.get(v, 0.0) for v in accepted_labels))
            per_turn.append(100.0 if label in accepted_labels else 0.0)

        scored = [s for s in per_turn if s is not None]
        if not scored:
            # Nothing expressive was ever asked for (or nothing was spoken), so
            # this axis doesn't apply to this rollout. Emitting the 0.0 an empty
            # mean falls back to would publish a failure that never happened.
            return []
        overall = sum(scored) / len(scored)
        return [
            self._score(
                overall,
                dimension=Dimension.EMOTION,
                per_turn=per_turn,
                meta={
                    "judge_model": judge.model_id,
                    "perceived": perceived,
                    "confidence": confidence,
                    "scores": scores,
                    "accepted_mass": accepted_mass,
                },
            )
        ]
