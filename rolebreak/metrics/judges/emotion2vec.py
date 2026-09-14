"""emotion2vec speech-emotion-recognition judge.

This wraps the fine-tuned `emotion2vec+ large
<https://huggingface.co/emotion2vec/emotion2vec_plus_large>`_ classifier — a
self-supervised speech-emotion model, run through ``funasr``'s ``AutoModel``.
``rate(audio=...)`` hears one clip and returns the emotion it *perceives*, with
the model's confidence as the score::

    from rolebreak.metrics.judges import Emotion2vecEmotionJudge

    judge = Emotion2vecEmotionJudge()
    verdict = judge.rate(audio=wav_bytes)
    verdict.score              # confidence of the winning label, in [0, 1]
    verdict.meta["emotion"]    # "happy" — a `rolebreak.types.Emotion` value, or None
    verdict.meta["scores"]     # {"angry": 0.01, "happy": 0.87, ...} full distribution

This is the rater behind :mod:`~rolebreak.metrics.dimensions.emotion`, which
turns the perceived label into a score against the turn's authored
``accepted_emotions``. A dedicated classifier rather than a prompted audio LM: it
hears only the waveform, and it is deterministic and offline, so the benchmark
number it feeds is reproducible without an endpoint.

Labels
------
emotion2vec+ predicts nine classes: ``angry``, ``disgusted``, ``fearful``,
``happy``, ``neutral``, ``other``, ``sad``, ``surprised`` and ``<unk>`` (the
checkpoint's own spelling of "unknown"). Seven map straight onto
:class:`~rolebreak.types.Emotion`; ``other`` and ``<unk>`` — the model finding no
emotion class that fits — fold into :attr:`~rolebreak.types.Emotion.NEUTRAL`,
which is also where the classifier puts undramatic delivery.
:attr:`~rolebreak.types.Emotion.CALM` has no counterpart in this taxonomy, so the
judge never predicts it.

Folding the two abstentions into ``neutral`` does turn a non-answer into a
positive claim, which a metric scoring against an authored ``neutral`` target
will count as a hit. The raw label and the whole distribution stay in ``meta``,
so a metric that wants to treat an abstention as unscored can still tell the two
apart.

Setup
-----
* ``funasr``, ``torchaudio`` and ``soundfile`` are plain project dependencies
  -- ``uv sync`` installs them.
* Weights (~1.2 GB) are downloaded on first use. ``hub="hf"`` (the default here)
  pulls from HuggingFace; pass ``hub="ms"`` for ModelScope, which is the faster
  mirror from China mainland. The model id stays ``iic/emotion2vec_plus_large``
  either way — funasr maps it to the ``emotion2vec/…`` HF repo.

The model is loaded once, lazily, on the first :meth:`rate` call and cached.
"""

from __future__ import annotations

import io
import threading
from typing import Any

from rolebreak.metrics.base import JudgeVerdict
from rolebreak.types import Emotion

_DEFAULT_MODEL = "iic/emotion2vec_plus_large"
_TARGET_SR = 16000

# emotion2vec+ class -> our taxonomy. `other` and `<unk>` are the model failing to
# place the clip in any of its eight emotion classes; we read that as "no emotion
# came through" and fold it into `neutral`, which is where the classifier already
# puts undramatic delivery. Both spellings of the abstention class are listed
# because the checkpoints disagree on it (`<unk>` in emotion2vec+, `unknown` in
# the model card's documented taxonomy). `meta["label"]` keeps the raw class, so
# an abstention stays distinguishable from a confident `neutral` downstream.
_LABEL_TO_EMOTION: dict[str, Emotion] = {
    "angry": Emotion.ANGRY,
    "disgusted": Emotion.DISGUST,
    "fearful": Emotion.FEARFUL,
    "happy": Emotion.HAPPY,
    "neutral": Emotion.NEUTRAL,
    "sad": Emotion.SAD,
    "surprised": Emotion.SURPRISED,
    "other": Emotion.NEUTRAL,
    "unknown": Emotion.NEUTRAL,
    "<unk>": Emotion.NEUTRAL,
}


def _english(label: str) -> str:
    """The English half of a funasr label token (``"开心/happy"`` -> ``"happy"``)."""
    return label.split("/")[-1].strip().lower()


class Emotion2vecEmotionJudge:
    """Speech-emotion judge backed by emotion2vec+.

    Implements the :class:`~rolebreak.metrics.base.Judge` protocol; only the
    ``audio`` argument of :meth:`rate` is used. Emotion recognition here is
    reference-free and text-independent, so ``instruction`` / ``text`` /
    ``reference_audio`` are ignored, and ``scale`` is too: the returned score is
    the classifier's own posterior for the winning label, already in ``[0, 1]``.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        *,
        hub: str = "hf",
        device: str | None = None,
    ) -> None:
        self.model_id = model
        self.hub = hub
        self._device = device
        self._model: Any = None
        self._resamplers: dict[int, Any] = {}
        # One metric instance (and so one judge) is shared across workers when a
        # run is scored with `parallel > 1`. funasr's AutoModel mutates its own
        # kwargs during `generate`, so serialize both the load and inference.
        self._lock = threading.Lock()

    # -- model loading ------------------------------------------------------ #
    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        from funasr import AutoModel  # noqa: PLC0415  (heavy; pulls in the funasr stack)

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel(
            model=self.model_id,
            hub=self.hub,
            device=self._device,
            disable_update=True,
            disable_pbar=True,
        )
        return self._model

    # -- audio -> 16k mono float32 array ------------------------------------ #
    def _to_array(self, audio: bytes) -> Any:
        import numpy as np
        import soundfile as sf
        import torch
        from torchaudio.transforms import Resample

        wav, sr = sf.read(io.BytesIO(audio), dtype="float32")
        if wav.ndim == 2:  # (T, channels) -> first channel
            wav = wav[:, 0]
        # emotion2vec is a 16 kHz model and funasr trusts the `fs` we declare, so
        # resample here rather than letting it mislabel the sample rate.
        if sr != _TARGET_SR:
            if sr not in self._resamplers:
                self._resamplers[sr] = Resample(orig_freq=sr, new_freq=_TARGET_SR)
            wav = self._resamplers[sr](torch.from_numpy(np.ascontiguousarray(wav))).numpy()
        return np.ascontiguousarray(wav, dtype=np.float32)

    # -- inference ---------------------------------------------------------- #
    def classify(self, audio: bytes) -> dict[str, float]:
        """Posterior over emotion2vec+'s classes for one clip, keyed by English label.

        Empty when the clip holds no samples: there is nothing to hear, and the
        model's convolutional front end can't consume a zero-length waveform.
        Checked before the model loads, so an empty clip never pulls the weights.
        """
        wav = self._to_array(audio)
        if wav.shape[0] == 0:
            return {}
        with self._lock:
            model = self._load()
            # `extract_embedding=False` keeps the 1024-d features out of the
            # result; we only need the head's scores. No `output_dir`, so nothing
            # is written to disk.
            [result] = model.generate(
                wav,
                fs=_TARGET_SR,
                granularity="utterance",
                extract_embedding=False,
            )
        return {_english(label): float(score) for label, score in zip(result["labels"], result["scores"], strict=True)}

    def emotion_probabilities(self, scores: dict[str, float]) -> dict[str, float]:
        """Fold a :meth:`classify` distribution onto :class:`~rolebreak.types.Emotion`.

        Keyed by ``Emotion`` value, with the classes that share a target summed
        (``other``/``<unk>`` into ``neutral``). Classes outside the mapping are
        dropped, so this needn't sum to 1 — but with the current checkpoints it
        does. Lets a caller ask "how much mass landed on the emotions I accept?"
        rather than only reading off the winner.
        """
        folded: dict[str, float] = {}
        for label, p in scores.items():
            emotion = _LABEL_TO_EMOTION.get(label)
            if emotion is not None:
                folded[emotion.value] = folded.get(emotion.value, 0.0) + p
        return folded

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
            raise ValueError("Emotion2vecEmotionJudge.rate needs `audio`.")
        scores = self.classify(audio)
        if not scores:  # nothing audible to label; the metric leaves the turn unscored
            return JudgeVerdict(
                score=None,
                meta={"rationale": "empty waveform (0 samples); no emotion", "label": None, "emotion": None},
            )

        label = max(scores, key=lambda k: scores[k])
        emotion = _LABEL_TO_EMOTION.get(label)
        return JudgeVerdict(
            score=scores[label],
            meta={
                "rationale": f"emotion2vec+ predicted {label} (p={scores[label]:.3f})",
                "label": label,  # the model's own class, before other/<unk> fold into neutral
                "emotion": emotion.value if emotion is not None else None,
                "scores": scores,  # the model's own taxonomy
                "emotion_scores": self.emotion_probabilities(scores),  # folded onto ours
            },
        )
