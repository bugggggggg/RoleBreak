"""UTMOSv2 naturalness (MOS) judge for speech quality.

This wraps the pretrained UTMOSv2 MOS predictor — the T05 system that won the
VoiceMOS Challenge 2024 Track 1 — vendored at ``external/UTMOSv2``.
``rate(audio=...)`` returns a predicted naturalness MOS: how a human listener
would rate the *quality* of the waveform (1 = bad, 5 = excellent), independent of
what was said. It complements
:class:`~rolebreak.metrics.judges.WavLMSpeakerVerificationJudge`, which scores
*who* is speaking rather than how good the speech sounds::

    from rolebreak.metrics.judges import UTMOSv2NaturalnessJudge

    judge = UTMOSv2NaturalnessJudge()
    verdict = judge.rate(audio=wav_bytes)  # score is the MOS itself (~1-5)

Setup
-----
* The ``utmosv2`` package is imported off ``sys.path`` from the vendored
  ``external/UTMOSv2`` checkout (override with ``repo_dir=``), the same way the
  model backends pick up their vendored repos. It needs ``timm`` on top of the
  usual torch/torchaudio/librosa/transformers stack — all of them plain
  project dependencies, installed by ``uv sync``.

* Weights are the upstream pretrained ``fusion_stage3`` checkpoint (~820 MB). On
  first use UTMOSv2 downloads it (via ``wget``) into ``~/.cache/utmosv2/models/``
  — point at a local ``.pth`` instead with ``checkpoint=`` or
  ``$ROLEBREAK_UTMOS_CKPT``.

Prediction runs on randomly sampled frames of the waveform, so repeated calls
differ by a few hundredths of a MOS. Pass ``num_repetitions=5`` (and/or a
different ``fold``) to average that away, as the paper does.

The model is loaded once, lazily, on the first :meth:`rate` call and cached.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any

from rolebreak.metrics.base import JudgeVerdict

_REPO_DIR = Path(__file__).resolve().parents[3] / "external/UTMOSv2"
_DEFAULT_CONFIG = "fusion_stage3"


class UTMOSv2NaturalnessJudge:
    """Speech-naturalness judge backed by UTMOSv2 (predicted MOS).

    Implements the :class:`~rolebreak.metrics.base.Judge` protocol; only the
    ``audio`` argument of :meth:`rate` is used — MOS prediction is reference-free
    and text-independent, so ``text`` / ``reference_audio`` / ``instruction`` are
    ignored, and the returned score is the MOS itself rather than something
    rescaled onto ``scale``.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        repo_dir: str | Path | None = None,
        config: str = _DEFAULT_CONFIG,
        fold: int = 0,
        device: str | None = None,
        predict_dataset: str = "sarulab",
        num_repetitions: int = 1,
    ) -> None:
        ckpt = checkpoint or os.environ.get("ROLEBREAK_UTMOS_CKPT")
        self.checkpoint = Path(ckpt) if ckpt else None
        self.repo_dir = Path(repo_dir or _REPO_DIR)
        self.config = config
        self.fold = fold
        self.predict_dataset = predict_dataset
        self.num_repetitions = num_repetitions
        self._device = device
        self._model: Any = None

    # -- model loading ------------------------------------------------------ #
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch

        # `utmosv2` is vendored, not installed — put its checkout on sys.path.
        if not (self.repo_dir / "utmosv2").is_dir():
            raise FileNotFoundError(
                f"UTMOSv2 checkout not found at {self.repo_dir}. Pass repo_dir= to point at a clone."
            )
        repo = str(self.repo_dir)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import utmosv2  # noqa: PLC0415  (vendored; imported lazily off sys.path)

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.checkpoint is not None and not self.checkpoint.exists():
            raise FileNotFoundError(
                f"UTMOSv2 checkpoint not found at {self.checkpoint}. Set checkpoint= or $ROLEBREAK_UTMOS_CKPT."
            )

        # pretrained=True downloads the upstream fusion_stage3 weights into
        # ~/.cache/utmosv2 when no explicit checkpoint is given.
        self._model = utmosv2.create_model(
            pretrained=True,
            config=self.config,
            fold=self.fold,
            checkpoint_path=self.checkpoint,
            device=self._device,
        )
        return self._model

    # -- audio -> mono float32 array --------------------------------------- #
    def _to_array(self, audio: bytes) -> tuple[Any, int]:
        import numpy as np
        import soundfile as sf

        wav, sr = sf.read(io.BytesIO(audio), dtype="float32")
        if wav.ndim == 2:  # (T, channels) -> first channel
            wav = wav[:, 0]
        # UTMOSv2 resamples to its own cfg.sr (16 kHz) internally, so pass `sr`.
        return np.ascontiguousarray(wav), int(sr)

    def mos(self, audio: bytes) -> float | None:
        """Predicted naturalness MOS (roughly 1-5) for one waveform.

        ``None`` when the clip holds no samples: MOS is undefined for silence,
        and UTMOSv2 tiles the waveform up to its window length by dividing by the
        sample count, so an empty one raises ZeroDivisionError deep inside the
        vendored dataset code. Checked before the model loads, so an empty clip
        never pulls the 820 MB checkpoint.
        """
        wav, sr = self._to_array(audio)
        if wav.shape[0] == 0:
            return None
        model = self._load()
        pred = model.predict(
            data=wav,
            sr=sr,
            device=self._device,
            predict_dataset=self.predict_dataset,
            num_repetitions=self.num_repetitions,
            verbose=False,
        )
        return float(pred[0])  # (1,) array for a single 1-D waveform

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
            raise ValueError("UTMOSv2NaturalnessJudge.rate needs `audio`.")
        # The score *is* the MOS — a 5-point absolute category rating is already
        # the natural unit here, so `scale` is ignored rather than remapped.
        mos = self.mos(audio)
        if mos is None:  # nothing audible to rate; the metric leaves the turn unscored
            return JudgeVerdict(score=None, meta={"rationale": "empty waveform (0 samples); no MOS", "mos": None})
        return JudgeVerdict(score=mos, meta={"rationale": f"UTMOSv2 predicted MOS={mos:.3f}", "mos": mos})
