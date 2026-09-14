"""SCOREQ speech-quality judge (no-reference MOS, or distance to a reference).

This wraps the ``scoreq`` package — "SCOREQ: Speech Quality Assessment with
Contrastive Regression" (NeurIPS 2024) — a wav2vec 2.0 model trained with
contrastive regression to predict perceived audio quality. It has two modes,
both exposed here through :meth:`rate`:

* ``mode="nr"`` (default) — *no-reference*: ``rate(audio=...)`` returns a
  predicted MOS (roughly 1-5, **higher is better**), like
  :class:`~rolebreak.metrics.judges.UTMOSv2NaturalnessJudge` but from a different
  model family, so the two are useful as a cross-check.
* ``mode="ref"`` — *full-reference*: ``rate(audio=..., reference_audio=...)``
  returns the distance between the two embeddings (**lower is better**, 0 =
  indistinguishable). This scores degradation against a clean reference of the
  *same utterance*, not speaker similarity — for "is it the same voice?" use
  :class:`~rolebreak.metrics.judges.WavLMSpeakerVerificationJudge`.

``data_domain="synthetic"`` (the default here) selects the model adapted to
TTS/vocoded speech; ``"natural"`` selects the telephone-speech model for
recorded audio::

    from rolebreak.metrics.judges import ScoreQQualityJudge

    judge = ScoreQQualityJudge()                      # synthetic, no-reference
    verdict = judge.rate(audio=wav_bytes)             # score is the MOS itself
    verdict.meta["mos"]

    ref_judge = ScoreQQualityJudge(mode="ref")
    ref_judge.rate(audio=wav_bytes, reference_audio=clean_bytes).meta["distance"]

Setup
-----
* ``scoreq`` is a plain project dependency -- ``uv sync`` installs it.
* Weights are the upstream ONNX exports, auto-downloaded from Zenodo into
  ``~/.cache/scoreq/onnx-models/`` on first use (a truncated download there
  surfaces as an ``INVALID_PROTOBUF`` error — delete the file and retry).
  ``use_onnx=False`` falls back to the original PyTorch path, which additionally
  needs ``fairseq``.
* Heads-up: ``import scoreq`` builds a default *natural / nr* model at module
  scope, so the first import downloads that checkpoint and opens an
  onnxruntime session even when you asked for a different variant. That import
  is deferred to the first :meth:`rate` call here, not paid at construction.

The model is loaded once, lazily, on the first :meth:`rate` call and cached.
SCOREQ reads audio from disk, so in-memory bytes are staged through a temporary
file.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

from rolebreak.metrics.base import JudgeVerdict

DataDomain = Literal["natural", "synthetic"]
Mode = Literal["nr", "ref"]


class ScoreQQualityJudge:
    """Speech-quality judge backed by SCOREQ.

    Implements the :class:`~rolebreak.metrics.base.Judge` protocol. ``mode="nr"``
    uses only ``audio``; ``mode="ref"`` also needs ``reference_audio``.
    ``instruction`` / ``text`` / ``scale`` are ignored — the score is SCOREQ's own
    number (a MOS in ``nr`` mode, an embedding distance in ``ref`` mode), left in
    its native units rather than rescaled.
    """

    def __init__(
        self,
        *,
        data_domain: DataDomain = "synthetic",
        mode: Mode = "nr",
        use_onnx: bool = True,
    ) -> None:
        self.data_domain = data_domain
        self.mode = mode
        self.use_onnx = use_onnx
        self._model: Any = None

    # -- model loading ------------------------------------------------------ #
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from scoreq import Scoreq  # noqa: PLC0415  (heavy; also downloads weights on import)
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError(
                "ScoreQQualityJudge needs the `scoreq` package (`pip install scoreq`); "
                "it is preinstalled in the rolebreak:cuda128 image."
            ) from e

        # SCOREQ picks its own device (CUDA provider when torch sees a GPU).
        self._model = Scoreq(data_domain=self.data_domain, mode=self.mode, use_onnx=self.use_onnx)
        return self._model

    # -- audio -> temp file ------------------------------------------------- #
    def score(self, audio: bytes, reference_audio: bytes | None = None) -> float:
        """SCOREQ's raw score: predicted MOS (``nr``) or distance (``ref``)."""
        model = self._load()
        # `Scoreq.predict` takes paths and loads them with torchaudio, so stage the
        # in-memory bytes on disk. The suffix is only a hint for the decoder.
        with tempfile.TemporaryDirectory(prefix="rolebreak-scoreq-") as tmp:
            test_path = Path(tmp) / "test.wav"
            test_path.write_bytes(audio)
            ref_path: Path | None = None
            if reference_audio is not None:
                ref_path = Path(tmp) / "ref.wav"
                ref_path.write_bytes(reference_audio)
            return float(model.predict(test_path=str(test_path), ref_path=str(ref_path) if ref_path else None))

    # -- Judge protocol ----------------------------------------------------- #
    def rate(
        self,
        instruction: str = "",
        *,
        text: str | None = None,
        audio: bytes | None = None,
        reference_audio: bytes | None = None,
        scale: tuple[float, float] = (0.0, 1.0),
        context: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        if audio is None:
            raise ValueError("ScoreQQualityJudge.rate needs `audio`.")
        if self.mode == "ref" and reference_audio is None:
            raise ValueError("ScoreQQualityJudge.rate in mode='ref' also needs `reference_audio`.")

        value = self.score(audio, reference_audio if self.mode == "ref" else None)
        if self.mode == "nr":
            meta = {"rationale": f"SCOREQ ({self.data_domain}, nr) predicted MOS={value:.3f}", "mos": value}
        else:
            meta = {
                "rationale": f"SCOREQ ({self.data_domain}, ref) distance={value:.4f} (lower is better)",
                "distance": value,
            }
        return JudgeVerdict(score=value, meta=meta)
