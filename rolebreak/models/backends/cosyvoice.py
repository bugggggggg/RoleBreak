"""CosyVoice 3 text-to-speech backend (zero-shot voice cloning, multilingual)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from rolebreak.models.tts import TextToSpeechModel, register_tts_backend

# The CosyVoice checkout vendored under the repo root; used as the default
# ``repo_dir`` so the upstream ``cosyvoice`` package imports without the caller
# locating the clone.
_VENDORED_REPO = Path(__file__).resolve().parents[3] / "external" / "CosyVoice"

# Default checkpoint dir (relative paths are resolved against ``repo_dir``); a
# bare HF/ModelScope id also works — AutoModel will ``snapshot_download`` it.
DEFAULT_MODEL = "pretrained_models/Fun-CosyVoice3-0.5B"

# CosyVoice 3 prompt texts are prefixed with a system instruction terminated by
# this sentinel; we prepend it when the caller's prompt_text omits it.
_ENDOFPROMPT = "<|endofprompt|>"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

# The reference clip shipped in the repo (and the transcript that goes with it),
# used as the default cloning voice when the caller names none.
_DEFAULT_PROMPT_WAV = "asset/zero_shot_prompt.wav"
_DEFAULT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"

# Normalised language token -> CosyVoice cross-lingual control token. CosyVoice
# selects the target language from a leading ``<|xx|>`` token in cross-lingual
# mode; these are the languages it documents.
_LANG_TOKENS = {
    "zh": "<|zh|>",
    "cmn": "<|zh|>",
    "mandarin": "<|zh|>",
    "chinese": "<|zh|>",
    "en": "<|en|>",
    "english": "<|en|>",
    "ja": "<|ja|>",
    "jp": "<|ja|>",
    "jpn": "<|ja|>",
    "japanese": "<|ja|>",
    "yue": "<|yue|>",
    "cantonese": "<|yue|>",
    "ko": "<|ko|>",
    "kr": "<|ko|>",
    "korean": "<|ko|>",
}


@register_tts_backend("cosyvoice3")
class CosyVoiceTTSModel(TextToSpeechModel):
    """Adapter for ``FunAudioLLM/Fun-CosyVoice3-0.5B`` (and CosyVoice 2)."""

    backend = "cosyvoice3"
    default_language = "zh"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str = "auto",
        repo_dir: str | None = None,
        prompt_wav: str | None = None,
        prompt_text: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        mode: str | None = None,
        fp16: bool = False,
        load_jit: bool = False,
        **config: Any,
    ) -> None:
        """``model`` is the checkpoint dir (relative paths resolve against
        ``repo_dir``) or an HF/ModelScope id. ``repo_dir`` is the cloned CosyVoice
        checkout we import the ``cosyvoice`` package + ``third_party/Matcha-TTS``
        from (defaults to the vendored ``external/CosyVoice``). ``prompt_wav`` /
        ``prompt_text`` are the default cloning reference and its transcript;
        ``system_prompt`` is prepended (CosyVoice 3 convention) to any prompt_text
        that lacks a ``<|endofprompt|>`` marker. ``mode`` fixes the synthesis mode
        (``zero_shot`` / ``cross_lingual`` / ``instruct``) when set."""
        super().__init__(model, device=device, **config)
        self.device = device
        self.repo_dir = str(Path(repo_dir).resolve()) if repo_dir else str(_VENDORED_REPO)
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.system_prompt = system_prompt
        self.mode = mode
        self.fp16 = fp16
        self.load_jit = load_jit
        self._cosyvoice = None

    # -- lifecycle ---------------------------------------------------------- #
    def _load(self) -> None:
        repo = Path(self.repo_dir)
        if not repo.is_dir():
            raise FileNotFoundError(
                f"CosyVoice repo not found at {repo}. Clone it (with submodules) or "
                "pass repo_dir=... pointing at your checkout."
            )
        # Both the repo root and its bundled Matcha-TTS must be importable; upstream
        # does exactly this `sys.path.append('third_party/Matcha-TTS')` in example.py.
        for p in (str(repo), str(repo / "third_party" / "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)

        from cosyvoice.cli.cosyvoice import AutoModel  # upstream package, off sys.path

        use_cuda = self.device != "cpu"
        if self.device == "auto":
            import torch

            use_cuda = torch.cuda.is_available()

        # AutoModel forwards kwargs verbatim to the version it picks from the
        # checkpoint yaml, and those signatures differ: only CosyVoice / 2 accept
        # ``load_jit`` (CosyVoice 3 has ``load_trt`` / ``load_vllm`` instead). JIT
        # and fp16 are CUDA-only upstream, so only request them on a GPU — and pass
        # ``load_jit`` only when actually enabled, so the common (default) path
        # loads on every version.
        kwargs: dict[str, Any] = {"model_dir": self._resolve_model_dir(), "fp16": self.fp16 and use_cuda}
        if self.load_jit and use_cuda:
            kwargs["load_jit"] = True
        self._cosyvoice = AutoModel(**kwargs)

        # CosyVoice's `cosyvoice/utils/file_utils.py` calls logging.basicConfig(
        # level=DEBUG) at import, which flips the *root* logger to DEBUG and makes
        # every library in the process (transformers, gradio, httpx, ...) spew debug
        # logs. Restore a sane level now that CosyVoice has finished importing.
        import logging

        logging.getLogger().setLevel(logging.WARNING)

    def _resolve_model_dir(self) -> str:
        """Local checkpoint dir, or a bare HF/ModelScope id for AutoModel to fetch.
        A relative path is resolved against ``repo_dir`` (where upstream keeps
        ``pretrained_models/``); an existing/absolute path is used verbatim."""
        model = self.model or DEFAULT_MODEL
        p = Path(model)
        if p.is_absolute() or p.exists():
            return str(p)
        candidate = Path(self.repo_dir) / model
        if candidate.exists():
            return str(candidate)
        # Not on disk: hand the id to AutoModel, which snapshot_downloads it.
        return model

    def close(self) -> None:
        self._cosyvoice = None

    # -- synthesis ---------------------------------------------------------- #
    # CosyVoice has two *independent* streaming axes, kept separate throughout:
    #   * text input  — is ``text`` a whole string, or an iterable of chunks fed
    #     to the LLM incrementally (upstream ``inference_bistream``)?
    #   * audio output — does upstream emit the waveform in chunks as it decodes
    #     (``stream=True``), or as one block?
    # ``_infer`` takes one flag for the audio axis (``stream_audio``) and reads the
    # text axis from the *type* of ``text``; every 2x2 combination is valid.
    def _synthesize(
        self,
        text: str | Iterable[str],
        *,
        voice: str | None,
        language: str | None,
        speed: float,
        prompt_text: str | None = None,
        instruct: str | None = None,
        mode: str | None = None,
        stream_audio: bool = False,
        text_frontend: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """Collect the whole waveform. ``text`` may still stream *in* (a chunk
        iterable); ``stream_audio`` only controls whether upstream decodes the
        audio in chunks internally — either way it is concatenated here."""
        chunks = self._infer(
            text,
            voice=voice,
            language=language,
            speed=speed,
            prompt_text=prompt_text,
            instruct=instruct,
            mode=mode,
            stream_audio=stream_audio,
            text_frontend=text_frontend,
            **kwargs,
        )
        return self._collect(chunks), int(self._cosyvoice.sample_rate)

    def _stream(
        self,
        text: str | Iterable[str],
        *,
        voice: str | None,
        language: str | None,
        speed: float,
        prompt_text: str | None = None,
        instruct: str | None = None,
        mode: str | None = None,
        text_frontend: bool = True,
        **kwargs: Any,
    ) -> Iterator[tuple[Any, int]]:
        """Stream the audio *out*: yield each inference chunk's waveform as it lands.

        This is the audio-output axis. It composes with the (independent) text-input
        axis: ``text`` may be a plain string, or an *iterable of text chunks* — e.g.
        tokens streaming out of an upstream LLM — which is fed to CosyVoice as a
        generator and tokenized incrementally (``inference_bistream``) so the LLM can
        start speaking before the full text exists. Streaming *text in* needs a
        CosyVoice 2/3 checkpoint (upstream ``inference_bistream`` is not implemented
        for CosyVoice 1, nor with the vLLM backend); streaming *audio out* works on
        any version, with a whole-string or a chunked ``text``.
        """
        kwargs.pop("stream_audio", None)  # this path always streams audio out
        chunks = self._infer(
            text,
            voice=voice,
            language=language,
            speed=speed,
            prompt_text=prompt_text,
            instruct=instruct,
            mode=mode,
            stream_audio=True,
            text_frontend=text_frontend,
            **kwargs,
        )
        sample_rate = int(self._cosyvoice.sample_rate)
        for out in chunks:
            yield self._one(out), sample_rate

    def _infer(
        self,
        text: str | Iterable[str],
        *,
        voice: str | None,
        language: str | None,
        speed: float,
        prompt_text: str | None,
        instruct: str | None,
        mode: str | None,
        stream_audio: bool,
        text_frontend: bool,
        **kwargs: Any,
    ) -> Any:
        """Dispatch to the upstream ``inference_*`` method for the resolved mode and
        return its chunk generator.

        The two streaming axes are handled separately here: the *text-input* axis is
        read from the type of ``text`` (a non-string is wrapped in a real generator
        so upstream takes its ``inference_bistream`` path), and the *audio-output*
        axis is the explicit ``stream_audio`` flag passed straight to upstream as
        ``stream=``."""
        if self._cosyvoice is None:
            self.load()

        streaming_text = not isinstance(text, str)
        if streaming_text and self._is_cosyvoice1():
            raise ValueError(
                "Streaming text input (a chunk iterable) requires a CosyVoice 2/3 "
                "checkpoint; the loaded model is CosyVoice 1, whose upstream LLM has "
                "no inference_bistream. Pass the text as a single string instead."
            )

        prompt_wav = self._resolve_prompt_wav(voice)
        ptext = prompt_text if prompt_text is not None else self.prompt_text
        resolved_mode = self._resolve_mode(mode, instruct, ptext)

        # CosyVoice 3's streaming LLM (inference_bistream) asserts <|endofprompt|>
        # is present in prompt_text — but cross_lingual mode *deletes* prompt_text
        # upstream (frontend_cross_lingual), so streaming text in cross_lingual
        # crashes on v3. zero_shot keeps prompt_text, and _with_system injects the
        # marker even with no reference transcript, so route streaming v3 through
        # zero_shot instead; v3 infers the target language from the text itself, so
        # cross-lingual synthesis still works. (v1/v2 have no such assertion and use
        # a <|xx|> token in the text, so they keep the cross_lingual path.)
        if streaming_text and resolved_mode == "cross_lingual" and self._is_cosyvoice3():
            resolved_mode = "zero_shot"

        if resolved_mode == "instruct":
            if not instruct:
                raise ValueError("mode='instruct' requires an `instruct` argument.")
            tts_text = self._as_stream(text) if streaming_text else text
            return self._cosyvoice.inference_instruct2(
                tts_text,
                self._format_instruct(instruct),
                prompt_wav,
                stream=stream_audio,
                speed=speed,
                text_frontend=text_frontend,
                **kwargs,
            )
        if resolved_mode == "cross_lingual":
            # The language / system marker `_with_lang` would prepend to a string
            # is emitted as the generator's leading chunk when streaming text.
            tts_text = (
                self._as_stream(text, prefix=self._lang_prefix(language))
                if streaming_text
                else self._with_lang(text, language)
            )
            return self._cosyvoice.inference_cross_lingual(
                tts_text,
                prompt_wav,
                stream=stream_audio,
                speed=speed,
                text_frontend=text_frontend,
                **kwargs,
            )
        # zero_shot: the streamed text passes straight through; the system marker
        # rides on prompt_text (the reference transcript), not the spoken text.
        tts_text = self._as_stream(text) if streaming_text else text
        return self._cosyvoice.inference_zero_shot(
            tts_text,
            self._with_system(ptext or ""),
            prompt_wav,
            stream=stream_audio,
            speed=speed,
            text_frontend=text_frontend,
            **kwargs,
        )

    # -- helpers ----------------------------------------------------------- #
    def _resolve_prompt_wav(self, voice: str | None) -> str:
        """Resolve the cloning reference clip. ``voice`` (a path) wins, then the
        constructor ``prompt_wav``, then the repo's bundled sample clip."""
        ref = voice or self.prompt_wav
        if ref:
            if not Path(ref).exists():
                raise FileNotFoundError(f"CosyVoice reference clip not found: {ref!r}")
            return ref
        default = Path(self.repo_dir) / _DEFAULT_PROMPT_WAV
        if not default.exists():
            raise FileNotFoundError(
                f"No reference clip given and the bundled default is missing ({default}). "
                "Pass voice=<path to a reference .wav> (CosyVoice clones zero-shot)."
            )
        return str(default)

    def _resolve_mode(self, mode: str | None, instruct: str | None, prompt_text: str | None) -> str:
        chosen = mode or self.mode
        if chosen:
            if chosen not in {"zero_shot", "cross_lingual", "instruct"}:
                raise ValueError(
                    f"Unknown CosyVoice mode {chosen!r}; expected 'zero_shot', 'cross_lingual' or 'instruct'."
                )
            return chosen
        if instruct:
            return "instruct"
        # zero_shot needs the reference transcript; without one, cross_lingual
        # (which takes only the reference audio) is the right zero-prompt path.
        if prompt_text:
            return "zero_shot"
        if self.prompt_wav is None and self.prompt_text is None:
            # Using the bundled default clip: its transcript is known, so prefer
            # the higher-fidelity zero_shot path.
            return "zero_shot"
        return "cross_lingual"

    def _format_instruct(self, instruct: str) -> str:
        """Shape the style instruction the way upstream expects it.

        In instruct mode the instruction takes the ``prompt_text`` slot and must be
        *terminated* by ``<|endofprompt|>`` — the marker separates instruction from
        spoken text, so an instruction placed after it would be read as reference
        transcript and silently ignored.

        The ``system_prompt`` prefix is added for CosyVoice 3 only: its training
        recipe fills the instruct slot with ``"You are a helpful assistant.
        <|endofprompt|>"`` on every utterance (upstream
        ``examples/libritts/cosyvoice3/run.sh``) and its documented control set is
        uniformly ``"You are a helpful assistant. <指令>。<|endofprompt|>"``
        (``cosyvoice/utils/common.py``), so the prefix keeps v3 in distribution.
        CosyVoice 1/2 were not trained that way — upstream passes their instructions
        bare (``"用四川话说这句话<|endofprompt|>"``) — so they get no prefix."""
        instruct = instruct.strip()
        if _ENDOFPROMPT in instruct:
            return instruct
        prefix = f"{self.system_prompt} " if self.system_prompt and self._is_cosyvoice3() else ""
        return f"{prefix}{instruct}{_ENDOFPROMPT}"

    def _with_system(self, prompt_text: str) -> str:
        """Prepend the CosyVoice 3 system instruction unless the caller already
        included a ``<|endofprompt|>`` marker. For the bundled default clip, fill
        in its known transcript when none was supplied."""
        if not prompt_text and self.prompt_wav is None and self.prompt_text is None:
            prompt_text = _DEFAULT_PROMPT_TEXT
        if _ENDOFPROMPT in prompt_text or not self.system_prompt:
            return prompt_text
        return f"{self.system_prompt}{_ENDOFPROMPT}{prompt_text}"

    def _is_cosyvoice3(self) -> bool:
        """Whether the loaded checkpoint is CosyVoice 3 (its LLM requires the
        ``<|endofprompt|>`` marker; v1/v2 use ``<|xx|>`` language tokens)."""
        return type(self._cosyvoice).__name__ == "CosyVoice3"

    def _is_cosyvoice1(self) -> bool:
        """Whether the loaded checkpoint is CosyVoice 1 — its LLM has no
        ``inference_bistream``, so it cannot take streaming (generator) text."""
        return type(self._cosyvoice).__name__ == "CosyVoice"

    def _lang_prefix(self, language: str | None) -> str:
        """The marker :meth:`_with_lang` prepends to cross-lingual text, as a bare
        string (``""`` when none): the CosyVoice 3 system instruction terminated by
        ``<|endofprompt|>``, or the v1/v2 ``<|xx|>`` language control token."""
        if self._is_cosyvoice3():
            return f"{self.system_prompt}{_ENDOFPROMPT}" if self.system_prompt else ""
        lang = language or self.default_language
        return (_LANG_TOKENS.get(lang.strip().lower()) or "") if lang else ""

    def _with_lang(self, text: str, language: str | None) -> str:
        """Prepare the cross-lingual ``text``.

        CosyVoice 3 carries no ``<|xx|>`` language token; instead its LLM requires
        the system instruction terminated by ``<|endofprompt|>`` in the text — and
        cross-lingual mode drops ``prompt_text``, so without this the upstream
        ``<|endofprompt|> not detected`` assertion fires. CosyVoice 1/2 select the
        target language from a leading ``<|xx|>`` control token instead."""
        prefix = self._lang_prefix(language)
        if self._is_cosyvoice3():
            if _ENDOFPROMPT in text or not prefix:
                return text
            return f"{prefix}{text}"
        if not prefix or text.lstrip().startswith("<|"):
            return text
        return f"{prefix}{text}"

    @staticmethod
    def _as_stream(text: Iterable[str], prefix: str = "") -> Any:
        """Wrap an iterable of text chunks in a real generator so upstream detects
        streaming input (it dispatches on ``isinstance(text, Generator)``, which a
        list or other iterable would not satisfy). ``prefix``, when given, is
        emitted as the leading chunk — where the string path prepends a control
        marker."""

        def gen() -> Iterator[str]:
            if prefix:
                yield prefix
            for chunk in text:
                yield chunk

        return gen()

    @staticmethod
    def _one(out: Any) -> Any:
        """Extract one inference chunk's ``tts_speech`` (torch tensor, shape
        ``[1, T]``) as a 1-D float32 array."""
        import numpy as np

        speech = out["tts_speech"]
        if hasattr(speech, "detach"):
            speech = speech.detach().cpu().numpy()
        return np.asarray(speech, dtype="float32").reshape(-1)

    @classmethod
    def _collect(cls, chunks: Any) -> Any:
        """Concatenate the per-sentence ``tts_speech`` tensors the inference
        generator yields into one mono float32 waveform."""
        import numpy as np

        pieces = [cls._one(out) for out in chunks]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype="float32")

    @property
    def languages(self) -> list[str]:
        return sorted({t.strip("<>|") for t in _LANG_TOKENS.values()})
