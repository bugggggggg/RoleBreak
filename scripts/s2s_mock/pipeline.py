"""STT -> LLM -> TTS for one whole turn, with nothing streamed in between.

The real pipeline is built for live conversation: its VAD decides where the
user's turn ended, and TTS audio leaves the server as soon as the first block
exists. Both are wrong for a benchmark replaying fixed clips. The VAD can
endpoint on a pause inside an utterance, so the reply answers a prefix; and a
reply that trickles out gives the client no way to tell "the model is still
synthesizing" from "the reply is over" except by timing out on silence, which
misfiles a slow turn's audio onto the next turn.

This module keeps the three models and throws the streaming away: one clip in,
one finished reply out. The handlers are the pipeline's own, built by
``get_stt_handler`` / ``get_llm_handler`` / ``get_tts_handler`` from the same
parsed arguments, so the models behave exactly as they do in production. Only
the plumbing differs:

* no VAD -- the turn boundary is the client's ``end_of_turn``, full stop;
* no queues or handler threads -- three direct ``process()`` calls;
* one TTS call for the whole reply instead of one per sentence batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock
from time import perf_counter
from typing import Any

import numpy as np
from openai.types.realtime import RealtimeSessionCreateRequest

from speech_to_speech.api.openai_realtime.runtime_config import RuntimeConfig
from speech_to_speech.LLM.chat import Chat, make_user_message
from speech_to_speech.pipeline.messages import (
    AUDIO_RESPONSE_DONE,
    PIPELINE_END,
    EndOfResponse,
    GenerateResponseRequest,
    LLMResponseChunk,
    TokenUsage,
    Transcription,
    TTSInput,
    VADAudio,
)
from speech_to_speech.s2s_pipeline import (
    ParsedArguments,
    get_llm_handler,
    get_stt_handler,
    get_tts_handler,
)
from speech_to_speech.utils.utils import int2float

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass
class TurnResult:
    """Everything one committed turn produced, once it is completely finished."""

    transcript: str
    reply_text: str
    audio: bytes  # int16 little-endian mono @ SAMPLE_RATE, the whole reply
    stt_s: float = 0.0
    llm_s: float = 0.0
    tts_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def audio_s(self) -> float:
        return len(self.audio) / 2 / SAMPLE_RATE


def trim_padding(pcm: np.ndarray) -> np.ndarray:
    """Drop the exactly-zero samples at both ends of a buffered turn.

    A client streams its clip and then keeps the socket warm with silence until
    the reply arrives, so the buffer this turn hands over is
    ``[silence from the previous turn] + clip + [zero padding of the last
    frame]``. Those pads are *exact* zeros -- real recorded or synthesized audio
    never is -- so cutting at the first and last non-zero sample removes the
    padding without touching a single sample of speech, and without needing a
    threshold to guess with. The clip's own quiet lead-in survives, which is
    what the STT model expects to see anyway.
    """
    nonzero = np.flatnonzero(pcm)
    if nonzero.size == 0:
        return pcm[:0]
    return pcm[nonzero[0] : nonzero[-1] + 1]


class TurnPipeline:
    """The three models, loaded once, shared by every session on this port.

    Concurrency here is deliberately narrow. The two GPU stages hold a lock, so
    only one session is ever inside STT or TTS: the handlers keep mutable state
    (Parakeet's revision keys, Qwen3-TTS's voice override and its captured CUDA
    graphs), and replaying a CUDA graph from two threads at once corrupts both.
    The LLM call runs *outside* that lock -- it is a request to a separate vLLM
    server, so a session waiting on it should not be occupying the GPU. That
    overlap is the whole gain: one session synthesizes while another thinks.

    Nothing bigger is on the table without batching inside the TTS handler.
    Extra *processes* on one GPU do not help either -- without CUDA MPS their
    kernels time-slice rather than run concurrently, so they only fill the idle
    gaps that this overlap already fills, at N times the model memory.
    """

    def __init__(self, args: ParsedArguments) -> None:
        # The handlers only touch these when driven by ``BaseHandler.run()``,
        # which the mock never calls -- except the TTS handler, which peeks at
        # its input queue to coalesce whatever text is already pending. Leaving
        # that queue empty is what makes it synthesize exactly the text passed
        # to ``process()`` and nothing else.
        stop_event = Event()
        should_listen = Event()
        should_listen.set()

        lm_vars = vars(args.responses_api_language_model_handler_kwargs)
        self._chat_size = int(lm_vars.get("chat_size", 30))
        self._default_instructions = lm_vars.get("init_chat_prompt")

        # Live (progressive) transcription exists to show text while the user is
        # still speaking. Here the clip is already over before STT is called, so
        # the progressive path could only ever add contention.
        args.module_kwargs.enable_live_transcription = False

        logger.info("Loading STT (%s)...", args.module_kwargs.stt)
        self.stt = get_stt_handler(
            args.module_kwargs,
            stop_event,
            Queue(),
            Queue(),
            None,
            whisper_stt_handler_kwargs=args.whisper_stt_handler_kwargs,
            faster_whisper_stt_handler_kwargs=args.faster_whisper_stt_handler_kwargs,
            paraformer_stt_handler_kwargs=args.paraformer_stt_handler_kwargs,
            mlx_audio_whisper_stt_handler_kwargs=args.mlx_audio_whisper_stt_handler_kwargs,
            parakeet_tdt_stt_handler_kwargs=args.parakeet_tdt_stt_handler_kwargs,
        )
        logger.info("Loading LLM (%s)...", args.module_kwargs.llm_backend)
        self.llm = get_llm_handler(
            args.module_kwargs,
            stop_event,
            Queue(),
            Queue(),
            args.language_model_handler_kwargs,
            args.responses_api_language_model_handler_kwargs,
        )
        logger.info("Loading TTS (%s)...", args.module_kwargs.tts)
        self.tts = get_tts_handler(
            args.module_kwargs,
            stop_event,
            Queue(),
            Queue(),
            should_listen,
            chat_tts_handler_kwargs=args.chat_tts_handler_kwargs,
            facebook_mms_tts_handler_kwargs=args.facebook_mms_tts_handler_kwargs,
            pocket_tts_handler_kwargs=args.pocket_tts_handler_kwargs,
            kokoro_tts_handler_kwargs=args.kokoro_tts_handler_kwargs,
            qwen3_tts_handler_kwargs=args.qwen3_tts_handler_kwargs,
        )
        # Guards the GPU stages only; see the class docstring.
        self._gpu = Lock()
        logger.info("All three models loaded; ready for turns")

    # -- session ----------------------------------------------------------- #
    def new_session(self) -> RuntimeConfig:
        """A conversation's own history and persona.

        The real server keeps one process-global ``RuntimeConfig`` and resets it
        per connection; a fresh one per connection is the same thing with no way
        for a previous conversation to leak into the next.
        """
        return RuntimeConfig(
            chat=Chat(self._chat_size),
            session=RealtimeSessionCreateRequest(type="realtime", instructions=self._default_instructions),
        )

    @property
    def default_instructions(self) -> str | None:
        return self._default_instructions

    def end_session(self) -> None:
        """Let the handlers drop their per-session state (STT revisions, TTS voice)."""
        for handler in (self.stt, self.llm, self.tts):
            try:
                handler.on_session_end()
            except Exception:
                logger.exception("%s.on_session_end() failed", type(handler).__name__)

    # -- one turn ---------------------------------------------------------- #
    def run_turn(
        self,
        runtime_config: RuntimeConfig,
        pcm_int16: np.ndarray,
        *,
        turn_id: str,
        committed_at: float,
    ) -> TurnResult:
        """Transcribe, answer, and speak one turn. Blocking; returns when done.

        ``committed_at`` is the ``perf_counter()`` of the client's ``end_of_turn``
        -- the moment the user stopped talking -- which is what the handlers'
        own latency logs are measured from.

        Safe to call from several session threads at once: the GPU stages
        serialize internally and everything else is either per-session state
        (``runtime_config``) or thread-safe (the OpenAI client).
        """
        return self._run_turn(runtime_config, pcm_int16, turn_id=turn_id, committed_at=committed_at)

    def _run_turn(
        self,
        runtime_config: RuntimeConfig,
        pcm_int16: np.ndarray,
        *,
        turn_id: str,
        committed_at: float,
    ) -> TurnResult:
        # The VAD hands STT float32 in [-1, 1]; use its own converter so the
        # model sees bit-identical input to a live run.
        audio = int2float(pcm_int16)

        t0 = perf_counter()
        with self._gpu:
            transcript, language_code = self._transcribe(audio, turn_id)
        stt_s = perf_counter() - t0
        logger.info(
            'STT %.2fs (%.2fs of audio) turn=%s: "%s"',
            stt_s,
            audio.shape[-1] / SAMPLE_RATE,
            turn_id,
            transcript,
        )
        if not transcript:
            return TurnResult("", "", b"", stt_s=stt_s, error="empty transcript")

        # Unlocked on purpose: this is the vLLM round trip, and it is the only
        # part of a turn that can overlap another session's GPU work.
        t0 = perf_counter()
        reply_text, usage, error = self._answer(
            runtime_config, transcript, language_code, turn_id=turn_id, committed_at=committed_at
        )
        llm_s = perf_counter() - t0
        logger.info('LLM %.2fs turn=%s: "%s"', llm_s, turn_id, reply_text)
        if error is not None:
            return TurnResult(transcript, reply_text, b"", stt_s=stt_s, llm_s=llm_s, error=error)
        if not reply_text:
            return TurnResult(transcript, "", b"", stt_s=stt_s, llm_s=llm_s, error="empty reply")

        t0 = perf_counter()
        with self._gpu:
            audio_out = self._speak(
                runtime_config, reply_text, language_code, turn_id=turn_id, committed_at=committed_at
            )
        tts_s = perf_counter() - t0
        result = TurnResult(
            transcript=transcript,
            reply_text=reply_text,
            audio=audio_out,
            stt_s=stt_s,
            llm_s=llm_s,
            tts_s=tts_s,
            input_tokens=usage[0],
            output_tokens=usage[1],
            error=None if audio_out else "tts produced no audio",
        )
        logger.info(
            "TTS %.2fs -> %.2fs of audio (RTF %.2f) turn=%s",
            tts_s,
            result.audio_s,
            (result.audio_s / tts_s) if tts_s > 0 else 0.0,
            turn_id,
        )
        return result

    def _transcribe(self, audio: np.ndarray, turn_id: str) -> tuple[str, str | None]:
        vad_audio = VADAudio(audio=audio, mode="final", turn_id=turn_id, turn_revision=0)
        transcript, language_code = "", None
        for out in self.stt.process(vad_audio):
            if isinstance(out, Transcription):
                transcript = str(out.text).strip()
                language_code = out.language_code
        return transcript, language_code

    def _answer(
        self,
        runtime_config: RuntimeConfig,
        transcript: str,
        language_code: str | None,
        *,
        turn_id: str,
        committed_at: float,
    ) -> tuple[str, tuple[int, int], str | None]:
        """Run the LLM and reassemble its chunks into the reply it meant to send.

        The handler streams the reply as sentence batches joined by a single
        space (``stream_batch_sentences`` of them per chunk), so joining the
        chunks the same way reproduces the reply exactly -- including the
        space at a chunk boundary, which a client concatenating the raw chunks
        loses. It also writes the assistant turn back into ``runtime_config``'s
        history on its own, which is what carries the conversation forward.
        """
        runtime_config.chat.add_item(make_user_message(transcript))
        request = GenerateResponseRequest(
            runtime_config=runtime_config,
            language_code=language_code,
            turn_id=turn_id,
            turn_revision=0,
            speech_stopped_at_s=committed_at,
        )
        parts: list[str] = []
        usage = (0, 0)
        error: str | None = None
        for out in self.llm.process(request):
            if isinstance(out, LLMResponseChunk):
                if out.text:
                    parts.append(out.text)
            elif isinstance(out, TokenUsage):
                usage = (out.input_tokens, out.output_tokens)
            elif isinstance(out, EndOfResponse) and out.error:
                error = out.error
        return " ".join(parts).strip(), usage, error

    def _speak(
        self,
        runtime_config: RuntimeConfig,
        text: str,
        language_code: str | None,
        *,
        turn_id: str,
        committed_at: float,
    ) -> bytes:
        """Synthesize the whole reply in one call and return it as int16 bytes."""
        tts_input = TTSInput(
            text=text,
            language_code=language_code,
            runtime_config=runtime_config,
            response=None,
            turn_id=turn_id,
            turn_revision=0,
            speech_stopped_at_s=committed_at,
        )
        blocks: list[np.ndarray] = []
        for chunk in self.tts.process(tts_input):
            block = _as_int16(chunk)
            if block is not None and block.size:
                blocks.append(block)
        if not blocks:
            return b""
        return np.concatenate(blocks).tobytes()


def _as_int16(chunk: Any) -> np.ndarray | None:
    """Normalize one TTS output block to int16 samples, or ``None`` if it is a sentinel."""
    if isinstance(chunk, bytes):
        # The queue's sentinels travel as bytes alongside real audio.
        if chunk in (AUDIO_RESPONSE_DONE, PIPELINE_END) or len(chunk) % 2:
            return None
        return np.frombuffer(chunk, dtype=np.int16)
    audio = getattr(chunk, "audio", chunk)
    if not isinstance(audio, np.ndarray):
        return None
    return audio.astype(np.int16, copy=False).reshape(-1)
