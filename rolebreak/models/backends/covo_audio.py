"""Covo-Audio-Chat speech-to-speech backend, served over an OpenAI-compatible API.

``tencent/Covo-Audio-Chat`` is a 7B end-to-end audio LLM: it perceives audio and
emits interleaved text and speech tokens in one autoregressive stage, which a
second stage turns into a waveform. As with the Qwen and MiniCPM omni models we
drive it out-of-process: run the checkpoint behind ``vllm serve --omni`` and
point this backend at that endpoint. See ``docs/models.md`` for the launch
command — it needs the derived ``rolebreak:vllm-omni`` image (for ``torchdiffeq``)
and an explicit ``CUDA_VISIBLE_DEVICES``.

A sibling of :mod:`rolebreak.models.backends.minicpm_o_4p5` rather than a
subclass. The wire shape is the same and the resilience machinery is duplicated
from it deliberately: what differs is the *prompt contract* (see
:class:`CovoAudioAPIModel`), which is the whole substance of this backend, so a
shared base class would be one override away from being a copy anyway.
"""

from __future__ import annotations

import http.client
import io
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from dataclasses import replace
from typing import Any

from rolebreak.models.audio import decode_b64, encode_bytes, wav_frame_count
from rolebreak.models.base import SpeechToSpeechModel
from rolebreak.types import (
    ChatRequest,
    ChatResponse,
    Message,
    OutputAudio,
)

#: The voice the checkpoint's code2wav stage speaks with. Recorded for
#: provenance only: the stage loads one bundled speaker prompt
#: (``speaker_prompt/{prompt_token,prompt_latent,speaker_embed}.npy``, with
#: ``n_speakers = 1``), so the ``发音人`` named in the system prompt is nominal
#: and there is no per-request voice to set. See :class:`CovoAudioAPIModel`.
DEFAULT_VOICE_REF = "default_female"

# A locally deployed `vllm serve --omni` OpenAI-compatible endpoint for
# Covo-Audio-Chat. Distinct from the Qwen and MiniCPM omni ports so they can all
# run side by side.
DEFAULT_API_BASE_URL = "http://localhost:8093/v1"

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The prompt contract
# --------------------------------------------------------------------------- #
#: The instruction that makes Covo speak at all.
#:
#: The model emits speech only when its *system* prompt tells it to interleave
#: modalities; the checkpoint's chat template inserts nothing of the sort. This
#: is the operative tail of the official prompt, kept verbatim in Chinese
#: because that is what the checkpoint was trained on — an English rendering of
#: the same instruction is measurably less reliable (see ``docs/models.md``).
#:
#: Translated: "converse in text and audio, alternately generating 5 text tokens
#: and 15 audio tokens; the audio uses speaker: default_female".
COVO_INTERLEAVE_DIRECTIVE = (
    "请用文本和音频进行对话，交替生成5个文本token和15个音频token，音频部分使用发音人：default_female"
)

#: The official prompt's style and safety rules, without its identity line.
#:
#: The directive alone is *not* enough once a long English persona precedes it —
#: the reply comes back text-only. These rules are what carry it (measured on a
#: six-question battery: persona + rules + directive spoke on 5/6 turns, persona
#: + directive on 1/3). Translated: (1) chat in concise, colloquial language,
#: positive and patient, like a trustworthy friend; (2) no lists or numbering,
#: avoid URLs, emoji and complex formulas; (3) do not rate competitors or give
#: political opinions, and handle sexual, political, violent or discriminatory
#: questions safely, with humour and reassurance.
#:
#: Rule 3 is a safety instruction, so a run that scores safety is scoring the
#: persona *plus* Tencent's own guardrail wording. That is a milder confound
#: than the identity line below, but it is one -- ``scaffold="none"`` is the
#: control condition if it needs isolating.
COVO_STYLE_RULES = (
    "1、请使用简洁、口语化的语言和用户聊天，你的态度积极、耐心，像一位值得信赖的朋友。\n"
    "2、不要使用列表或编号，避免输出网址、表情符号和复杂的公式。\n"
    "3、不评价竞争对手，不发表主观政治观点，"
    "针对色情类、政治类、恐怖类、歧视类、暴力类的用户问题，"
    "你要妥善应对潜在的安全风险，并给出幽默，情绪安抚以及安全的劝导。"
)

#: The identity line of the official prompt: "You are 小腾, English name Covo,
#: an AI assistant developed by Tencent."
#:
#: Included only by the ``"canonical"`` scaffold. It buys the most reliable
#: speech (6/6 on the same battery) at the cost of telling a role-play model it
#: is somebody else — asked its name under it, the model has answered "I'm Covo"
#: and "I'm Xiao Teng" rather than in persona. See :class:`CovoAudioAPIModel`.
COVO_IDENTITY = '你是"小腾"，英文名是"Covo"，由腾讯开发的AI助手。'

#: The checkpoint's own system prompt, reproduced verbatim
#: (``modeling_covo_audio.py``; also ``vllm_omni``'s
#: ``covo_audio/prompt_utils.py::COVO_AUDIO_SYSTEM_PROMPT``).
COVO_CANONICAL_SYSTEM_PROMPT = "\n".join([COVO_IDENTITY, COVO_STYLE_RULES, COVO_INTERLEAVE_DIRECTIVE])

#: What each ``scaffold`` appends to the persona's system prompt.
SCAFFOLDS: dict[str, str] = {
    # Identity-free: the persona keeps its name. The default.
    "rules": "\n".join([COVO_STYLE_RULES, COVO_INTERLEAVE_DIRECTIVE]),
    # The whole official prompt, identity included.
    "canonical": COVO_CANONICAL_SYSTEM_PROMPT,
    # The bare instruction. Speaks reliably only when the persona is short.
    "directive": COVO_INTERLEAVE_DIRECTIVE,
    # Send the persona untouched — for measuring the model without the scaffold.
    "none": "",
}
DEFAULT_SCAFFOLD = "rules"

#: Longest clip still treated as "the model said nothing".
#:
#: A turn that emits no audio codes is not answered with a missing ``audio``
#: field: vllm-omni's input processor substitutes a single placeholder code
#: (``covo_audio.py::_filter_audio_codes`` returns ``[-1]``), and the code2wav
#: stage renders it as a well-formed 2,400-frame / 0.1 s clip of noise. Handing
#: that to the voice and naturalness judges would score silence as speech, so it
#: is dropped here. One code is 0.1 s, so anything at or under a quarter of a
#: second is the placeholder and not an utterance.
SILENT_MAX_SECONDS = 0.25

DEFAULT_MAX_RETRIES = 3
# A cold two-stage Omni server needs minutes to come back. Measured cold start
# here is ~80 seconds; the budget stays generous.
DEFAULT_HEALTH_TIMEOUT = 600.0
HEALTH_POLL_SECONDS = 10.0
# Let the freshly-restarted engine settle before the next request: /health answers
# as soon as the API server is up, which is a little before the stages are warm.
RETRY_SETTLE_SECONDS = 5.0

# Transport failures that mean "the server went away mid-request", not "this
# request is bad" — the stream is cut when the engine takes the process down.
_TRANSPORT_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    ConnectionError,
    socket.timeout,
    TimeoutError,
)

# An in-band error is worth another attempt only when the *server* broke.
_ENGINE_DEATH_MARKERS = (
    "enginecore",
    "engine core",
    "engine dead",
    "enginedead",
    "illegal memory access",
    "cuda error",
)
# A request the server refuses on its merits fails identically every time, so
# retrying only burns the budget and hides the real cause.
_FATAL_MARKERS = (
    "context",
    "maximum",
    "too long",
    "too large",
    "invalid",
)


def apply_scaffold(messages: list[Message], scaffold: str) -> list[Message]:
    """Return ``messages`` with the interleave scaffold on the system prompt.

    The directive is only honoured in the **system** message: appended to the
    user turn instead — as text, or as a second content part — it never produced
    speech in testing (0/6 turns), so this deliberately has no user-side path.

    A conversation with no system message gets one, since a persona-less run
    would otherwise be silent. The persona comes first and the scaffold last:
    the directive works best adjacent to the generation point, and it keeps the
    persona's own words at the top where they read as the prompt's subject.
    """
    suffix = SCAFFOLDS[scaffold]
    if not suffix:
        return messages
    out = list(messages)
    for i, message in enumerate(out):
        if message.role != "system":
            continue
        persona = message.content if isinstance(message.content, str) else None
        merged = f"{persona}\n{suffix}" if persona else suffix
        out[i] = Message(role="system", content=merged, audio=message.audio)
        return out
    return [Message(role="system", content=suffix), *out]


def is_placeholder_audio(audio: OutputAudio) -> bool:
    """Whether ``audio`` is the code2wav stage's "nothing was said" clip.

    See :data:`SILENT_MAX_SECONDS`. A payload that isn't parseable WAV is taken
    at its word and reported as real, the same way
    :func:`~rolebreak.models.audio.has_audio_frames` treats mp3 and opus.
    """
    raw = decode_b64(audio.data)
    frames = wav_frame_count(raw)
    if frames is None:
        return False
    if frames == 0:
        return True
    try:
        with wave.open(io.BytesIO(raw), "rb") as w:
            rate = w.getframerate()
    except (wave.Error, EOFError):
        return False
    return bool(rate) and frames / rate <= SILENT_MAX_SECONDS


class CovoAudioAPIModel(SpeechToSpeechModel):
    """Covo-Audio-Chat via a locally deployed vLLM OpenAI-compatible endpoint.

    This backend keeps no weights: it forwards our OpenAI-shaped
    :class:`ChatRequest` to a running ``vllm serve --omni`` at ``base_url`` and
    parses the chat-completion back.

    Three properties of this server shape the backend:

    * **speech has to be asked for, in the system prompt.** Covo interleaves
      text and audio tokens only when instructed to; with an ordinary persona
      prompt it answers in text and the code2wav stage still returns a 0.1 s
      placeholder clip. :func:`apply_scaffold` appends the instruction, and
      :func:`is_placeholder_audio` drops the placeholder so a mute turn is
      recorded as mute rather than scored as speech. Which scaffold is a real
      trade-off, so it is a constructor argument:

      - ``"rules"`` (default) appends the official prompt's style/safety rules
        and the interleave directive, but *not* its identity line. Measured
        5/6 turns vocal on a six-question battery, with the persona intact.
      - ``"canonical"`` appends the whole official prompt. 6/6 vocal, but it
        tells the model it is "小腾 / Covo, an AI assistant developed by
        Tencent" — asked its name under it, the model has answered as Covo
        instead of as the persona. For a benchmark that scores persona
        adherence that is a confound, which is why it is not the default.
      - ``"directive"`` appends the bare instruction; reliable only behind a
        short system prompt.
      - ``"none"`` sends the persona untouched, for measuring the model without
        any scaffold at all.

      The scaffold is Chinese even for an English persona, because that is the
      form the checkpoint was trained on; an English rendering of the same
      instruction was measurably less reliable. Replies stay in the user's
      language regardless.
    * like the Qwen and MiniCPM omni servers, the **non-streaming** call returns
      speech but splits the reply across **two ``choices`` that both carry
      ``index: 0``** — one holds ``message.content`` (text, with ``audio:
      null``), the other ``message.audio`` (base64 WAV, mono 24 kHz, with an
      empty ``transcript``). Reading ``choices[0]`` the usual way would silently
      drop the speech, so :meth:`_parse_completion` folds every choice into one
      reply and backfills the transcript from the text choice.
    * **there is no voice to pick.** The code2wav stage loads a single bundled
      speaker prompt at import time, so the ``发音人`` named in the directive is
      nominal and an ``audio.voice`` in the request would be ignored. This
      backend does not pretend to send one.

    Unlike MiniCPM-o there is no ``<think>`` preamble to strip: Covo does not
    reason before answering.

    Non-streaming is the default path. Pass ``stream=True`` to use the SSE path
    instead: it carries the same content — the waveform still arrives as a
    single segment at the end — and additionally yields a time-to-first-token
    latency.

    Note for callers: audio tokens are counted against ``max_tokens`` alongside
    text (15 audio per 5 text, per the directive), so a text budget sized for a
    text-only model truncates the speech mid-word. A couple of spoken sentences
    costs 300-500 tokens; see the eval's ``--max-tokens`` default.
    """

    backend = "covo-audio-api"

    def __init__(
        self,
        model: str = "tencent/Covo-Audio-Chat",
        *,
        base_url: str = DEFAULT_API_BASE_URL,
        api_key: str = "EMPTY",
        timeout: float = 300.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        health_timeout: float = DEFAULT_HEALTH_TIMEOUT,
        stream: bool = False,
        scaffold: str = DEFAULT_SCAFFOLD,
        **config: Any,
    ) -> None:
        super().__init__(model, base_url=base_url, **config)
        if scaffold not in SCAFFOLDS:
            raise ValueError(f"unknown scaffold {scaffold!r}; expected one of {sorted(SCAFFOLDS)}")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.health_timeout = health_timeout
        self.stream = stream
        self.scaffold = scaffold
        # Held by whichever thread is nursing a dead engine back up. An eval that
        # replays several examples at once has every in-flight turn fail at the
        # same instant when a stage takes the orchestrator down, and a cold
        # server does not need one warmup turn per thread on top of the restart
        # it is already paying for -- see :meth:`_recover`.
        self._recovery = threading.Lock()

    def generate(self, request: ChatRequest) -> ChatResponse:
        speaks = "audio" in request.modalities
        if speaks:
            request = replace(request, messages=apply_scaffold(request.messages, self.scaffold))
        payload = request.to_dict()
        payload["model"] = self.model
        # No voice field: the speaker prompt is fixed server-side, so anything we
        # sent here would be silently ignored (see the class docstring).
        if self.stream:
            payload["stream"] = True
        return self._generate_with_retry(payload, expect_audio=speaks)

    # -- resilience --------------------------------------------------------- #
    def _dispatch(self, payload: dict[str, Any]) -> ChatResponse:
        """One attempt at the turn, over whichever path the payload asks for."""
        if payload.get("stream"):
            return self._generate_stream(payload)
        return self._parse_completion(self._post(payload))

    def _generate_with_retry(self, payload: dict[str, Any], *, expect_audio: bool) -> ChatResponse:
        """Re-issue the turn when the *server* dies under it.

        A stage of vllm-omni can crash the whole engine on a CUDA fault under
        load, which would otherwise end a long eval partway through. Re-issuing
        is sound here because this backend keeps no server-side session: every
        turn already resends the full windowed history as audio, so an attempt
        against a restarted server is indistinguishable from a first attempt. A
        partial reply from a killed stream is discarded rather than stitched.

        A turn that came back mute is *not* retried. Covo is seeded server-side
        (``seed: 42`` in the deploy YAML), so the same prompt yields the same
        reply: re-issuing it returns the same silence, and varying the seed or
        the temperature did not recover speech in testing either. Silence is a
        property of the prompt, so it is reported rather than fought — the count
        rides back on :attr:`ChatResponse.raw` as ``no_speech``.

        The retry count rides back the same way so a run can log that a turn
        needed one — a recovered turn should be visible in the transcript, not
        silently identical to a clean one.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = self._dispatch(payload)
            except Exception as exc:
                if attempt >= self.max_retries or not self._is_retryable(exc):
                    raise
                logger.warning(
                    "%s: attempt %d/%d failed (%s); waiting for the server to come back",
                    self.backend,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                self._recover()
                continue
            extra: dict[str, Any] = {}
            if attempt:
                extra["retries"] = attempt
            if expect_audio and response.message.audio is None:
                extra["no_speech"] = True
            if extra:
                response.raw = {**(response.raw or {}), **extra}
            return response
        raise AssertionError("unreachable: the loop either returns or raises")

    def _recover(self) -> None:
        """Wait for the engine to come back, warming it once for all callers.

        One thread does the polling and the throwaway warmup turn; any other
        thread that failed on the same crash blocks here until that finishes and
        then retries against an already-warm server. Without this, N concurrent
        workers would each poll ``/health`` and each fire a warmup turn at a
        server that has just restarted -- which is the load that killed it.
        """
        if self._recovery.acquire(blocking=False):
            try:
                if self._wait_for_health():
                    self._warmup()
            finally:
                self._recovery.release()
            return
        logger.info("%s: another thread is recovering the server; waiting for it", self.backend)
        with self._recovery:
            pass

    def _is_retryable(self, exc: BaseException) -> bool:
        """Whether ``exc`` looks like a dead server rather than a bad request."""
        if isinstance(exc, urllib.error.HTTPError):
            return exc.code >= 500  # a 4xx is our fault and will not improve
        if isinstance(exc, _TRANSPORT_ERRORS):
            return True
        if isinstance(exc, RuntimeError):
            # An in-band failure: vLLM answers 200 before generation runs, so a
            # mid-generation crash arrives as a payload, not an HTTP status.
            message = str(exc).lower()
            if any(marker in message for marker in _FATAL_MARKERS):
                return False
            return any(marker in message for marker in _ENGINE_DEATH_MARKERS)
        return False

    def _warmup(self) -> bool:
        """Run one throwaway turn so the retried request does not pay cold-start costs.

        ``/health`` answers as soon as the API server binds its port, which is
        before the stages have JIT-compiled their kernels — a restarted server
        stalls on that during its first real request. Spending it on a turn whose
        answer we discard keeps it out of the eval's latency numbers, and keeps a
        slow first reply from tripping the request timeout.

        Best-effort by design: a failed warmup says nothing the next real attempt
        would not say better, so it logs and returns instead of raising.
        """
        # The canonical prompt, so both stages are exercised: this turn is
        # discarded, and it is the one place where reliable speech matters more
        # than keeping an identity out of the context.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": COVO_CANONICAL_SYSTEM_PROMPT},
                {"role": "user", "content": "Introduce yourself in one short sentence."},
            ],
            "modalities": ["text", "audio"],
            "max_tokens": 256,
        }
        started = time.monotonic()
        try:
            self._post(payload)
        except Exception as exc:  # noqa: BLE001 - warmup is advisory, never fatal
            logger.warning("%s: warmup turn failed (%s); trying the real request anyway", self.backend, exc)
            return False
        logger.info("%s: warmup turn done in %.1fs", self.backend, time.monotonic() - started)
        return True

    def _health_url(self) -> str:
        """``/health`` on the endpoint's origin — it sits beside ``/v1``, not under it."""
        parts = urllib.parse.urlsplit(self.base_url)
        return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))

    def _wait_for_health(self) -> bool:
        """Poll ``/health`` until the restarted server answers, or give up.

        Giving up does not raise: the next attempt's own failure is a better
        error to surface than one from this helper.
        """
        url = self._health_url()
        deadline = time.monotonic() + self.health_timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=10.0) as resp:
                    if resp.status < 400:
                        logger.info("%s: server healthy again at %s", self.backend, url)
                        time.sleep(RETRY_SETTLE_SECONDS)
                        return True
            except Exception:  # noqa: BLE001 - still down; keep polling until the deadline
                pass
            time.sleep(HEALTH_POLL_SECONDS)
        logger.error(
            "%s: server did not answer %s within %.0fs",
            self.backend,
            url,
            self.health_timeout,
        )
        return False

    # -- transport --------------------------------------------------------- #
    def _request(self, payload: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _parse_completion(self, data: dict[str, Any]) -> ChatResponse:
        """Fold the reply's several ``choices`` into one message.

        This server answers a spoken turn with two choices that *both* say
        ``index: 0`` — one text-only, one audio-only — so we take the first
        non-empty of each rather than indexing. A text-only turn comes back as
        the single ordinary choice, which this reads unchanged.
        """
        if "choices" not in data:
            raise RuntimeError(self._response_error(data))
        text: str | None = None
        audio: OutputAudio | None = None
        for choice in data["choices"]:
            message = choice.get("message") or {}
            if text is None and message.get("content"):
                text = (message["content"] or "").strip() or None
            if audio is None and message.get("audio"):
                audio = OutputAudio.from_dict(message["audio"])
        audio = self._drop_placeholder(audio)
        if audio is not None and not audio.transcript:
            # The audio choice ships an empty transcript; the text choice is it.
            audio.transcript = text
        return ChatResponse(
            message=Message(role="assistant", content=text, audio=audio),
            model=data.get("model", self.model),
            raw=data,
        )

    def _drop_placeholder(self, audio: OutputAudio | None) -> OutputAudio | None:
        """Discard the code2wav stage's "nothing was said" clip. See :data:`SILENT_MAX_SECONDS`."""
        if audio is None or not is_placeholder_audio(audio):
            return audio
        logger.warning(
            "%s: reply carried no audio codes; dropping the placeholder clip (the turn is text-only)",
            self.backend,
        )
        return None

    def _generate_stream(self, payload: dict[str, Any]) -> ChatResponse:
        """Consume the SSE stream, splitting text deltas from base64 WAV segments.

        Streaming is also the only place this backend can time the reply: the
        turn-based contract hands the model a finished clip, so the wait starts
        when the request goes out and ends at the first text token — or, for a
        turn that produced only speech, the code2wav stage's first WAV segment.

        The deploy config runs this pipeline with ``async_chunk: false``, so the
        waveform arrives as a single segment once generation finishes rather
        than incrementally; the merge below still handles either.
        """
        text_parts: list[str] = []
        wav_segments: list[bytes] = []
        started = time.monotonic()
        first_text_s: float | None = None
        first_audio_s: float | None = None
        with urllib.request.urlopen(self._request(payload), timeout=self.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[len("data:") :].strip()
                if chunk == "[DONE]":
                    break
                data = json.loads(chunk)
                if "choices" not in data:
                    raise RuntimeError(self._response_error(data))
                if not data["choices"]:
                    continue  # usage-only final chunk carries no delta
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content")
                if not content:
                    continue
                # Base64 WAV segments start with "UklGR" ("RIFF"); else it's text.
                if content.startswith("UklGR"):
                    wav_segments.append(decode_b64(content))
                    if first_audio_s is None:
                        first_audio_s = time.monotonic() - started
                else:
                    text_parts.append(content)
                    if first_text_s is None:
                        first_text_s = time.monotonic() - started

        transcript = "".join(text_parts).strip() or None
        audio = None
        if wav_segments:
            audio = OutputAudio(
                data=encode_bytes(_merge_wav_segments(wav_segments)),
                format="wav",
                transcript=transcript,
            )
        audio = self._drop_placeholder(audio)
        return ChatResponse(
            message=Message(role="assistant", content=transcript, audio=audio),
            model=self.model,
            raw={"text": transcript, "audio_segments": len(wav_segments)},
            latency=first_text_s if first_text_s is not None else first_audio_s,
        )

    def _response_error(self, data: dict[str, Any]) -> str:
        """Describe a payload that carries a failure instead of a completion.

        vLLM answers 200 and opens the stream before generation runs, so a
        failure raised partway through (context overflow, a stage crash) can
        only be reported in-band: it yields one payload shaped
        ``{"object": "error", "message": ...}`` with no ``choices``.
        """
        err = data.get("error", data)
        detail = err.get("message") if isinstance(err, dict) else err
        return f"{self.backend} request failed at {self.base_url}: {detail or json.dumps(data)[:500]}"


def _merge_wav_segments(segments: list[bytes]) -> bytes:
    """Concatenate the PCM frames of several WAV byte-strings into one WAV.

    Each streamed segment is a complete standalone WAV; joining them at the raw
    bytes level would leave stray headers mid-stream, so we read each with
    :mod:`wave` and rewrite the merged frames under a single header.
    """
    params = None
    frames = bytearray()
    for seg in segments:
        with wave.open(io.BytesIO(seg), "rb") as w:
            if params is None:
                params = w.getparams()
            frames += w.readframes(w.getnframes())
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(frames)
    return buf.getvalue()
