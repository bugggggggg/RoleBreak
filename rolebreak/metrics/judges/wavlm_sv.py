"""WavLM-large speaker-verification judge for the voice-consistency metric.

This wraps the exact model seed-tts-eval uses for its SIM score: an
ECAPA-TDNN head over WavLM-large features
(``external/seed-tts-eval/thirdparty/UniSpeech/downstreams/speaker_verification``).
``rate(audio=, reference_audio=)`` returns the cosine similarity of the two
speaker embeddings — the same number ``verification.py`` computes, just driven
from in-memory audio bytes instead of file paths.

Setup
-----
* Checkpoint: the fine-tuned head, ``~/models/seed-tts-eval-wavlm_large/model.pth``
  (override with ``checkpoint=`` or ``$ROLEBREAK_WAVLM_CKPT``).
* No ``s3prl``/``fairseq`` required. The checkpoint already ships the full
  WavLM-large weights, so the upstream is rebuilt from the vendored
  ``UniSpeech/WavLM`` implementation (torch/numpy only) — see
  :mod:`rolebreak.metrics.judges._wavlm_local`.

The model is loaded once, lazily, on the first :meth:`rate` call and cached.
A pair where either clip is under :data:`MIN_CLIP_SECONDS` is declined
(``score=None``) instead of embedded — see that constant.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from rolebreak.metrics.base import JudgeVerdict

"""
download the model from https://github.com/BytedanceSpeech/seed-tts-eval

wget -O ~/models/seed-tts-eval-wavlm_large/model.pth \
  "https://drive.usercontent.google.com/download?id=1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP&export=download&authuser=0&confirm=t"
"""
_DEFAULT_CKPT = Path.home() / "models/seed-tts-eval-wavlm_large/model.pth"
_TARGET_SR = 16000

#: Shortest clip this judge will embed, in seconds.
#:
#: The hard floor is the model's: WavLM strides 20ms with a 25ms window, so a
#: clip under 720 samples (45ms) collapses to a single frame and the ECAPA head's
#: ``InstanceNorm1d`` raises on it. The floor here is far above that on purpose —
#: an ECAPA embedding taken from a fraction of a second is dominated by whatever
#: phoneme happened to be in the window rather than by the speaker, so the cosine
#: it produces is not a speaker judgment and must not be averaged in as one.
#: Clips below this are declined (``score=None``), not scored.
MIN_CLIP_SECONDS = 0.5


def clip_seconds(audio: bytes) -> float | None:
    """Duration of ``audio`` in seconds, or ``None`` if the header can't be read.

    Reads the header only (no decode), so it's cheap enough to call on every clip.
    """
    import soundfile as sf

    try:
        info = sf.info(io.BytesIO(audio))
    except Exception:
        return None
    return info.frames / info.samplerate if info.samplerate else None


def is_long_enough(audio: bytes) -> bool:
    """Whether ``audio`` carries enough speech for a speaker embedding.

    A clip whose duration can't be read is taken at its word and reported as long
    enough, matching :func:`rolebreak.models.audio.has_audio_frames`.
    """
    seconds = clip_seconds(audio)
    return seconds is None or seconds >= MIN_CLIP_SECONDS


def _describe(audio: bytes) -> str:
    """``0.04s``-style tag for a clip, for the declined-verdict rationale."""
    seconds = clip_seconds(audio)
    return "unreadable" if seconds is None else f"{seconds:.2f}s"


class WavLMSpeakerVerificationJudge:
    """Speaker-similarity judge backed by WavLM-large + ECAPA-TDNN.

    Implements the :class:`~rolebreak.metrics.base.Judge` protocol; only the
    ``audio`` / ``reference_audio`` arguments of :meth:`rate` are used.
    """

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint or os.environ.get("ROLEBREAK_WAVLM_CKPT") or _DEFAULT_CKPT)
        self._device = device
        self._model: Any = None
        self._resamplers: dict[int, Any] = {}

    # -- model loading ------------------------------------------------------ #
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch

        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"WavLM SV checkpoint not found at {self.checkpoint}. Set checkpoint= or $ROLEBREAK_WAVLM_CKPT."
            )
        from rolebreak.metrics.judges._wavlm_local import build_wavlm_sv_model

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = build_wavlm_sv_model(self.checkpoint, device=self._device)
        return self._model

    # -- audio -> 16k mono tensor ------------------------------------------ #
    def _to_tensor(self, audio: bytes) -> Any:
        import numpy as np
        import soundfile as sf
        import torch
        from torchaudio.transforms import Resample

        wav, sr = sf.read(io.BytesIO(audio), dtype="float32")
        if wav.ndim == 2:  # (T, channels) -> first channel
            wav = wav[:, 0]
        t = torch.from_numpy(np.ascontiguousarray(wav)).unsqueeze(0).float()
        if sr != _TARGET_SR:
            if sr not in self._resamplers:
                self._resamplers[sr] = Resample(orig_freq=sr, new_freq=_TARGET_SR)
            t = self._resamplers[sr](t)
        return t.to(self._device)

    def embed(self, audio: bytes) -> Any:
        import torch

        model = self._load()
        with torch.no_grad():
            return model(self._to_tensor(audio))

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
        if audio is None or reference_audio is None:
            raise ValueError("WavLMSpeakerVerificationJudge.rate needs both `audio` and `reference_audio`.")
        # Too little audio to hold a speaker: an empty waveform (which WavLM's
        # convolutional front end can't consume at all), or a clip below
        # MIN_CLIP_SECONDS (whose embedding would be phoneme noise, not identity).
        # Decline the pair rather than crash or report a similarity computed from
        # nothing.
        short = [
            f"{label}={_describe(clip)}"
            for label, clip in (("turn", audio), ("reference", reference_audio))
            if not is_long_enough(clip)
        ]
        if short:
            return JudgeVerdict(
                score=None,
                meta={"rationale": f"clip shorter than {MIN_CLIP_SECONDS}s ({', '.join(short)}); no similarity"},
            )
        import torch.nn.functional as F

        sim = F.cosine_similarity(self.embed(audio), self.embed(reference_audio)).item()
        # Cosine is in [-1, 1]; affine-map to [0, 1] (monotonic, no clipping) for
        # the headline score, but keep the raw cosine for SIM-style reporting.
        return JudgeVerdict(
            score=(sim + 1.0) / 2.0, meta={"rationale": f"WavLM-large SV cosine={sim:.4f}", "cosine": sim}
        )
