"""Voice consistency: does the agent keep the *same voice* the whole time?

Speech role-players drift in timbre, pitch, accent, or speaking rate across a
long conversation — or snap back to a generic default voice under stress. We
measure the speaker identity of each turn against the model's **own first spoken
reply** (the first one long enough to carry a voice — see
:data:`~rolebreak.metrics.judges.wavlm_sv.MIN_CLIP_SECONDS`): whatever voice it
picked up front is the voice it has to keep. That makes the axis
self-referential, so it applies to every rollout regardless of
whether a seed/clone clip exists — and rollouts are expected to carry no
``reference_audio`` at all (asserted below). A rollout with no turn long enough
to anchor on is skipped, and a turn too short to embed is left unscored rather
than counted as drift.

The judge is a speaker-verification model: ``rate`` returns a cosine similarity
between the turn's audio and the anchor, on ``[0, 1]``, which this metric puts
on the benchmark's ``[0, 100]``. This metric hardcodes
:class:`~rolebreak.metrics.judges.WavLMSpeakerVerificationJudge` — the same
WavLM-large + ECAPA-TDNN model seed-tts-eval uses for its SIM score. Point it at
a checkpoint via ``$ROLEBREAK_WAVLM_CKPT``::

    from rolebreak.metrics import load_metric

    metric = load_metric("voice_consistency")
    [score] = metric.evaluate(transcript)   # evaluate always returns a list
"""

from __future__ import annotations

from rolebreak.metrics.base import Judge, Metric
from rolebreak.metrics.judges import WavLMSpeakerVerificationJudge
from rolebreak.metrics.judges.wavlm_sv import MIN_CLIP_SECONDS, is_long_enough
from rolebreak.metrics.registry import register_metric
from rolebreak.types import Dimension, MetricScore, Transcript


@register_metric
class VoiceConsistencyMetric(Metric):
    name = "voice_consistency"
    requires_audio = True

    def _build_judge(self) -> Judge:
        return WavLMSpeakerVerificationJudge()

    def evaluate(self, transcript: Transcript) -> list[MetricScore]:
        from rolebreak.models.audio import decode_b64

        # This metric is self-referential: rollouts carry no persona target clip,
        # and silently preferring one would change what the number means.
        assert transcript.reference_audio is None, (
            "voice_consistency scores against the model's own first spoken reply; "
            "transcript.reference_audio must be None."
        )

        # Anchor = the first turn that produced enough speech to hold a voice. The
        # "enough" matters: a backend that opens with a 0.04s blip would otherwise
        # have every later turn compared against noise, and the whole rollout would
        # score as drift. Without such a turn there's nothing to verify against, so
        # this axis doesn't apply and we skip.
        anchor: tuple[int, bytes] | None = None
        for i, ex in enumerate(transcript.exchanges):
            if ex.assistant_audio is None:
                continue
            data = decode_b64(ex.assistant_audio.data)
            if is_long_enough(data):
                anchor = (i, data)
                break
        if anchor is None:
            return []
        anchor_index, anchor_bytes = anchor

        judge = self.judge

        per_turn: list[float | None] = []
        for i, ex in enumerate(transcript.exchanges):
            audio = ex.assistant_audio
            # The anchor is trivially identical to itself; leave it unscored so a
            # free 100 doesn't inflate the mean (and the early half of `drift`).
            # Turns under the judge's floor come back as None from `rate` below.
            if audio is None or i == anchor_index:
                per_turn.append(None)
                continue
            verdict = judge.rate(
                audio=decode_b64(audio.data),
                reference_audio=anchor_bytes,
            )
            # The judge returns similarity on [0, 1]; report it on the benchmark's 0-100.
            per_turn.append(None if verdict.score is None else 100.0 * verdict.score)

        scored = [s for s in per_turn if s is not None]
        overall = sum(scored) / len(scored) if scored else 0.0
        return [
            self._score(
                overall,
                dimension=Dimension.VOICE,
                per_turn=per_turn,
                meta={
                    "reference": "first_output_audio",
                    "anchor_turn": anchor_index,
                    "min_clip_seconds": MIN_CLIP_SECONDS,
                },
            )
        ]
