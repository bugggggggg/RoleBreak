"""Concrete :class:`~rolebreak.metrics.base.Judge` implementations.

Judges are the raters metrics delegate to. They are kept separate from the
metrics (and import heavy deps lazily) so the metrics package stays importable
without torch / s3prl / any model weights installed.
"""

from __future__ import annotations

from rolebreak.metrics.judges.emotion2vec import Emotion2vecEmotionJudge
from rolebreak.metrics.judges.llm import JudgeCallError, OpenAICompatibleLLM
from rolebreak.metrics.judges.scoreq import ScoreQQualityJudge
from rolebreak.metrics.judges.utmos import UTMOSv2NaturalnessJudge
from rolebreak.metrics.judges.wavlm_sv import WavLMSpeakerVerificationJudge

__all__ = [
    "Emotion2vecEmotionJudge",
    "JudgeCallError",
    "OpenAICompatibleLLM",
    "ScoreQQualityJudge",
    "UTMOSv2NaturalnessJudge",
    "WavLMSpeakerVerificationJudge",
]
