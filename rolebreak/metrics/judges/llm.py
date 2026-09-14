"""OpenAI-compatible LLM text client.

:class:`OpenAICompatibleLLM` is a thin text client whose only job is to call an
OpenAI Chat Completions-compatible endpoint. Its input is a
:class:`~rolebreak.types.ChatRequest` and its output a
:class:`~rolebreak.types.ChatResponse`, exactly like the model backends in
:mod:`rolebreak.models` — so it stays a plain "send messages, get generated
text back" component with no scoring knowledge. Dimensions call it directly.

    from rolebreak.metrics.judges import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM(                                    # e.g. a local server
        model="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )
    llm = OpenAICompatibleLLM()   # or from $OPENAI_MODEL / $OPENAI_BASE_URL

There is no built-in default model or endpoint: both must come from the
arguments or the environment, or ``__init__`` raises — an unnoticed fallback to
some other judge would quietly make scores incomparable across runs.

The call goes through the official ``openai`` SDK, so connection handling,
retries with exponential backoff, and ``Retry-After`` are the library's rather
than ours.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any

import openai

from rolebreak.types import ChatRequest, ChatResponse, Message

logger = logging.getLogger(__name__)

_EMPTY_REPLY_ATTEMPTS = 2
_TRANSIENT_ATTEMPTS = 2
_TRANSIENT_BACKOFF = 2.0
# Retried rather than raised: the server is telling us to come back later, not
# that the request is wrong. 5xx is covered separately by a range check.
_TRANSIENT_STATUS = frozenset({408, 409, 429})


def _without_thinking(request: ChatRequest) -> ChatRequest:
    """Return ``request`` with thinking turned off."""
    extra = dict(request.extra)
    if "thinking" in extra:  # Anthropic-style {"type": "enabled"}
        extra["thinking"] = {"type": "disabled"}
    if "enable_thinking" in extra:
        extra["enable_thinking"] = False
    template_kwargs = extra.get("chat_template_kwargs")  # vLLM (Qwen3, …)
    if isinstance(template_kwargs, dict) and "enable_thinking" in template_kwargs:
        extra["chat_template_kwargs"] = {**template_kwargs, "enable_thinking": False}
    # "none", not a dropped field: an absent key leaves a thinking-on default on.
    return dataclasses.replace(request, reasoning_effort="none", extra=extra)


class JudgeCallError(RuntimeError):
    """The judge endpoint never returned a usable answer.

    Raised once the SDK's own retries and both retry loops in
    :meth:`OpenAICompatibleLLM.generate` are spent, so it always means
    *infrastructure*, not judgment: a dead endpoint, a bad key, an exhausted
    balance, a request the server rejects outright, a decode that keeps coming
    back with nothing in it. It is emphatically
    **not** "the judge marked this reply down" — a metric that turns one of these
    into a low score would publish an outage as a model's benchmark result.

    Metrics must therefore let it propagate. Scoring a suite aborts, the runs
    already judged stay in ``<metric>.jsonl.partial``, and the sweep resumes
    there once the endpoint is healthy again.

    Subclasses :class:`RuntimeError` so callers that only want "the call didn't
    work" (the rollout authoring scripts) keep catching it as they do today.
    """


class _TransientJudgeError(JudgeCallError):
    """A failure that says *later*, not *no*: rate limit, 5xx, dropped connection.

    Retried by :meth:`OpenAICompatibleLLM.generate` with the request left
    exactly as it was — unlike an empty reply, nothing about the request needs
    to change for the next attempt to work. Internal only: it subclasses
    :class:`JudgeCallError` so a spent retry budget propagates to callers as the
    one error they already handle.
    """


# --------------------------------------------------------------------------- #
# LLM client: ChatRequest in, ChatResponse out. Nothing benchmark-specific.
# --------------------------------------------------------------------------- #
class OpenAICompatibleLLM:
    """Minimal text client for an OpenAI-compatible ``/chat/completions`` endpoint.

    Its sole responsibility is to call the model and return the generated text
    as a :class:`~rolebreak.types.ChatResponse`. Works against OpenAI, a running
    ``vllm serve``, Together, Groq, a local llama.cpp server, and so on.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        # No implicit fallback to a hosted default: a judge model silently
        # differing from the one the run was scored with would invalidate the
        # numbers, so an unset model or endpoint is an error, not a default.
        resolved_model = model or os.environ.get("OPENAI_MODEL")
        if not resolved_model:
            raise ValueError("No judge model: pass model= or set $OPENAI_MODEL.")
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not resolved_base_url:
            raise ValueError("No judge endpoint: pass base_url= or set $OPENAI_BASE_URL.")
        self.model = resolved_model
        self.base_url = resolved_base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        # A judged run makes one request per sample, so a single dropped
        # connection or a rate limit from a busy endpoint would otherwise throw
        # away the whole evaluation. The SDK retries those (and 408/409/429/5xx)
        # with exponential backoff; anything the server would reject the same
        # way twice — bad key, bad payload, context too long — is raised at once.
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def generate(self, request: ChatRequest) -> ChatResponse:
        """Send ``request`` to the endpoint and return the assistant's reply.

        Two failures are worth another attempt, and they want opposite things
        from it. A transient one (rate limit, 5xx, dropped connection) is about
        the endpoint, so the request is resent unchanged after a backoff. An
        empty reply is about *this* request, so the resend drops thinking — the
        one change that makes a spent-on-reasoning decode come back with an
        answer. Sending a modified request at a rate-limited endpoint would
        silently score the sample with a different judge configuration than the
        rest of the suite, which is why the two paths stay separate.
        """
        detail = ""
        attempt_request = request
        empty_attempts = 0
        transient_attempts = 0
        while True:
            try:
                data = self._create(attempt_request)
            except _TransientJudgeError as e:
                transient_attempts += 1
                if transient_attempts >= _TRANSIENT_ATTEMPTS:
                    raise
                delay = _TRANSIENT_BACKOFF * 2 ** (transient_attempts - 1)
                logger.warning("%s: %s; retrying in %.0fs", self.model, e, delay)
                time.sleep(delay)
                continue
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return ChatResponse(
                    message=Message(role="assistant", content=content),
                    model=data.get("model", request.model or self.model),
                    raw=data,
                )
            detail = self._empty_reply_detail(choice, data)
            empty_attempts += 1
            if empty_attempts >= _EMPTY_REPLY_ATTEMPTS:
                raise JudgeCallError(f"{self.model} returned an empty reply {empty_attempts} times ({detail})")
            # An empty reply is most often a decode that spent its budget
            # reasoning, so the retry drops thinking rather than losing the
            # sample the same way twice.
            logger.warning("%s returned an empty reply (%s); retrying without thinking", self.model, detail)
            attempt_request = _without_thinking(attempt_request)

    @staticmethod
    def _empty_reply_detail(choice: dict[str, Any], data: dict[str, Any]) -> str:
        """Say *how* a reply came back empty, which is what names the cause.

        ``finish_reason='length'`` with the completion tokens pinned at the cap is
        a runaway decode; a long ``reasoning_content`` alongside it is a thinking
        model that never reached its answer; anything else is the endpoint itself.
        """
        bits = [f"finish_reason={choice.get('finish_reason')!r}"]
        completion_tokens = (data.get("usage") or {}).get("completion_tokens")
        if completion_tokens is not None:
            bits.append(f"completion_tokens={completion_tokens}")
        message = choice.get("message") or {}
        # A dumped response, so the vendor extension is a plain key here, not
        # the SDK model's ``model_extra``: reading it off the object raised.
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            bits.append(f"reasoning_content={len(str(reasoning))} chars")
        return ", ".join(bits)

    # -- request / http ----------------------------------------------------- #
    def _create(self, request: ChatRequest) -> dict[str, Any]:
        """Run one chat completion and return it as the raw response dict.

        The typed request is unpacked by hand rather than through
        ``ChatRequest.to_dict()``: that emits the speech-to-speech
        ``modalities``/``audio`` fields, which a plain text endpoint rejects.
        ``request.extra`` rides along as ``extra_body`` so backend-specific
        knobs (``response_format``, ``repetition_penalty``, …) reach the server
        whether or not the SDK knows them as named parameters.

        Failures the endpoint may recover from become
        :class:`_TransientJudgeError` for :meth:`generate` to retry; a request
        the server would reject identically next time raises
        :class:`JudgeCallError` straight away.
        """
        messages: Any = [m.to_dict() for m in request.messages]
        # NOT_GIVEN, not None: an explicit `"max_tokens": null` in the body is
        # rejected by some OpenAI-compatible servers, where an absent key is not.
        max_tokens = request.max_tokens if request.max_tokens is not None else openai.NOT_GIVEN
        reasoning_effort = request.reasoning_effort or openai.NOT_GIVEN
        try:
            completion = self.client.chat.completions.create(
                model=request.model or self.model,
                messages=messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                extra_body=request.extra or None,
            )
        except openai.APIConnectionError as e:  # connection reset, DNS, timeout
            raise _TransientJudgeError(f"LLM request failed: {e}") from e
        except openai.APIStatusError as e:  # surface the server's error body
            failed = f"LLM request failed ({e.status_code}): {e.response.text}"
            if e.status_code in _TRANSIENT_STATUS or e.status_code >= 500:
                raise _TransientJudgeError(failed) from e
            raise JudgeCallError(failed) from e
        except openai.OpenAIError as e:  # bad config, unusable client
            raise JudgeCallError(f"LLM request failed: {e}") from e
        # Dumped rather than kept as SDK models so `raw` stays JSON-serialisable
        # for the run logs, and so vendor extras (`reasoning_content`) survive.
        return completion.model_dump()
