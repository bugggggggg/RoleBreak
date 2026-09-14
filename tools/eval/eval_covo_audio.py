"""Replay the example conversations against a served Covo-Audio-Chat.

Usage
-----
    # one example; text turns are spoken from data/audio by default
    python -m tools.eval.eval_covo_audio --example ip-conan

    # synthesize the spoken user turns first if they aren't in data/audio yet
    python -m tools.synthesize_user_turns ip-conan --out-dir data/audio

    # every example instead of one; either way the out-dir holds one <example>.tar
    # run shard per example, indexed in runs.jsonl, with a shared chat.log
    python -m tools.eval.eval_covo_audio --all

    # an interrupted run is picked back up by default — the examples runs.jsonl
    # already lists are skipped; replay every example instead
    python -m tools.eval.eval_covo_audio --all --no-resume

    # score what came out
    python -m tools.score_run --run-dir var/generation/tencent_Covo-Audio-Chat/happy

    # replay an older batch of synthesized clips instead of the newest one
    python -m tools.eval.eval_covo_audio --all --audio-version neutral

    # a remote endpoint, and time-to-first-token latency
    python -m tools.eval.eval_covo_audio --all \\
        --base-url http://10.0.0.5:8093/v1 --stream

    # trade persona fidelity for speech: the checkpoint's own system prompt
    # speaks on more turns, but tells the model it is Tencent's assistant
    python -m tools.eval.eval_covo_audio --all --scaffold canonical

A sibling of ``eval_minicpm_o_4p5`` rather than a flag on it. The two servers are
deployed on different ports, and Covo has its own prompt contract: it emits
speech only when the *system* prompt tells it to interleave text and audio
tokens, so this eval appends a scaffold to every persona (``--scaffold``) and
counts the turns that came back mute anyway. There is no ``--voice``: the
code2wav stage loads one bundled speaker prompt, so the voice is fixed. See
:class:`~rolebreak.models.backends.covo_audio.CovoAudioAPIModel`. Launch the
server with the command in ``docs/models.md`` (it needs the derived
``rolebreak:vllm-omni`` image and an explicit ``CUDA_VISIBLE_DEVICES``).
"""

from __future__ import annotations

import contextlib
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
from tqdm import tqdm

from rolebreak.examples import EXAMPLES
from rolebreak.models import load_model, user_audio_message
from rolebreak.models.audio import decode_b64
from rolebreak.models.backends.covo_audio import DEFAULT_SCAFFOLD, DEFAULT_VOICE_REF, SCAFFOLDS
from rolebreak.runlog import RUNS_FILENAME, append_run
from rolebreak.runshard import RunShardWriter, wav_member
from rolebreak.types import Message
from tools.audio_version import generation_dir, label, read_dir
from tools.eval.resume import recorded_examples
from tools.eval.user_turns import has_shard, load_synth_audio, report_drift, resolve_turn_audio


DEFAULT_HISTORY_TURNS = 40

#: How many examples to replay at once by default.
#:
#: Turns *within* an example are strictly sequential -- turn N is answered against
#: the reply to turn N-1 -- so the only parallelism a replay has is across
#: examples, which are independent.
#:
#: The shipped ``covo_audio.yaml`` pins **both** stages at ``max_num_seqs: 1``,
#: so against it extra workers buy latency, not throughput (1.5x however many
#: arrive) -- pass ``--workers 1`` there. The launch command in
#: ``docs/models.md`` widens the thinker to 16 and code2wav to 4, and eight is
#: the sweet spot on that server: 2.8x the sequential throughput, and 3x on a
#: real replay (six examples, 65 turns: 3m33s against 10m33s).
#:
#: Note that eight is *not* capped by code2wav's ``max_num_seqs: 4``. Unlike
#: MiniCPM-o, where the narrow stages set a hard ceiling, sequences queued at
#: code2wav here overlap with the thinker's work on the other card, so
#: throughput keeps climbing past it. What sets the default is the latency
#: curve: the mean turn is 3.8s alone, 8.9s at eight, 11.4s at sixteen, and past
#: twelve the gain is mostly queueing. Code2Wav is held at 4 for memory, not
#: throughput -- see ``docs/models.md``.
DEFAULT_WORKERS = 8

#: Text budget per turn.
#:
#: Larger than the other evals' 256 on purpose: Covo's audio codes are counted
#: against ``max_tokens`` alongside its text (the directive asks for 15 audio
#: tokens per 5 text tokens), so a text-sized budget truncates the speech
#: mid-word. A couple of spoken sentences measured 300-500 tokens here; the
#: server's own per-stage cap is 2048.
DEFAULT_MAX_TOKENS = 1024


#: Appended to every persona's system prompt before it is sent.
#:
#: Carried over from the Qwen2.5-Omni and MiniCPM-o evals, and it earns its place
#: here too: without it Covo answers an audio turn with Markdown headings and
#: numbered sections, which is not something a persona can say out loud.
#:
#: The backend then appends its own interleave scaffold *after* this, so the
#: instruction the model sees last is the one that makes it speak.
SPOKEN_STYLE_SUFFIX = "\n\nKeep answer to a couple of sentences. Never use Markdown or any other markup. Write only words that would be said."


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def windowed(history: list[Message], max_turns: int) -> list[Message]:
    """The system prompt plus the most recent ``max_turns`` turns of ``history``.

    Every user turn is sent as a full audio clip, so an untrimmed history grows
    monotonically in audio tokens and eventually overruns the server's context.
    Covo is served with the same 32k window as the Qwen and MiniCPM omni models,
    so it gets the same sliding window to bound what each request carries.

    The system prompt is never dropped: it is the persona under test *and* the
    only place Covo honours the instruction to speak at all, so a turn answered
    without it is neither the run we mean to score nor an audible one.
    ``max_turns <= 0`` disables the window and sends everything.
    """
    if max_turns <= 0:
        return history
    system, rest = history[:1], history[1:]
    # ``rest`` is [user, assistant, ..., user]: the turn in flight has no reply
    # yet, so keep its user message plus the max_turns - 1 exchanges before it.
    return system + rest[-(2 * max_turns - 1) :]


def run_example(
    example_name: str,
    *,
    model,
    model_id: str,
    voice_ref: str,
    root: Path,
    audio_dir: str,
    modalities: list[str],
    temperature: float,
    max_tokens: int,
    history_turns: int,
    emit,
) -> dict:
    """Replay one example over a single chat session into its run shard.

    Returns the run's row for the consolidated ``runs.jsonl`` — only reached once
    the shard is published, so what the index lists and what is on disk agree.
    The row carries a ``mute_turns`` count: see :func:`main`.
    """
    example = EXAMPLES[example_name]
    synth_audio = load_synth_audio(audio_dir, example_name)

    system_prompt = example.system_prompt + SPOKEN_STYLE_SUFFIX
    history: list[Message] = [Message(role="system", content=system_prompt)]
    emit(f"\n{'#' * 72}\n# example: {example.name}\n{'#' * 72}")
    emit(f"system> {system_prompt}")
    emit(f"scaff > {model.scaffold} (interleave scaffold appended by the backend)")
    emit(f"voice > {voice_ref} (server-side speaker prompt)")

    # Resolve every turn's audio up front — a turn that authored its own clip keeps
    # it, the rest come from the shard — so a missing clip fails before the first
    # request rather than partway through the conversation.
    clips = [resolve_turn_audio(turn, synth_audio.get(i)) for i, turn in enumerate(example.turns)]
    report_drift(example.turns, clips, emit)

    mute = 0
    shard = RunShardWriter(
        root,
        example=example.name,
        persona=example.system_prompt,
        model=model_id,
        voice=voice_ref,
    )
    with shard:
        for i, turn in enumerate(example.turns, start=1):
            emit(f"\n{'=' * 72}\n[{i:02d}/{len(example.turns)}]")
            emit(f"  user> {turn.label}")

            clip = clips[i - 1]
            emit(f"  in  > {clip.ref}")
            history.append(user_audio_message(clip.source))
            sent = windowed(history, history_turns)
            if len(sent) < len(history):
                emit(f"  ctx > last {history_turns} turns ({(len(history) - len(sent)) // 2} earlier dropped)")
            resp = model.chat(
                sent,
                modalities=modalities,
                audio={"format": "wav"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = (resp.transcript or "").strip()

            # Feed our own reply back as text so the reply's audio doesn't bloat
            # context on long horizons.
            history.append(Message(role="assistant", content=reply))
            emit(f"  bot > {reply or '(no assistant text received)'}")
            if resp.latency is not None:
                emit(f"  lat > {resp.latency:.2f}s")
            # A turn the server had to be restarted for is still a valid turn, but
            # it should be visible as one rather than reading like a clean reply.
            raw = resp.raw if isinstance(resp.raw, dict) else {}
            if raw.get("retries"):
                emit(f"  rtry> recovered after {raw['retries']} engine restart(s)")
            if raw.get("no_speech"):
                # Covo answered in text only. The backend has already discarded
                # the code2wav stage's 0.1s placeholder, so the turn is stored
                # with no WAV rather than with a clip of nothing -- which is what
                # keeps the voice and naturalness judges off it.
                mute += 1
                emit("  mute> text only: the model emitted no audio codes for this turn")

            wav_bytes = decode_b64(resp.audio.data) if resp.audio is not None else None
            if wav_bytes is not None:
                emit(f"  wav > {shard.path}#{wav_member(example.name, i - 1)}")

            shard.add_turn(
                i - 1,
                user_text=clip.text,
                assistant_text=reply,
                user_audio=clip.ref,
                wav=wav_bytes,
                accepted_emotions=turn.accepted_emotions,
                expected_rubric=turn.rubric,
                latency=resp.latency,
            )
    if mute:
        emit(f"\nmute turns: {mute}/{len(example.turns)} answered in text only")
    return {**shard.row(), "mute_turns": mute}


@click.command()
@click.option(
    "--example",
    "example_name",
    default="ip-conan",
    show_default=True,
    type=click.Choice(sorted(EXAMPLES)),
    help="Which example conversation to run (ignored when --all is set).",
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    default=False,
    help="Replay every example rather than the one --example names; the out-dir is laid out the same either way.",
)
@click.option(
    "--model",
    "model_id",
    default="tencent/Covo-Audio-Chat",
    show_default=True,
    help="Model id or local checkpoint path.",
)
@click.option(
    "--voice-ref",
    default=DEFAULT_VOICE_REF,
    show_default=True,
    help="Name of the speaker prompt the server's code2wav stage speaks with, recorded in the run "
    "shard for provenance. Covo has no per-request voice: the stage loads one bundled speaker "
    "embedding, so the name in the interleave directive is nominal.",
)
@click.option(
    "--base-url",
    default="http://localhost:8093/v1",
    show_default=True,
    help="vLLM OpenAI-compatible endpoint serving Covo-Audio-Chat.",
)
@click.option(
    "--scaffold",
    default=DEFAULT_SCAFFOLD,
    show_default=True,
    type=click.Choice(sorted(SCAFFOLDS)),
    help="What the backend appends to each persona's system prompt to make Covo speak. "
    "'rules' keeps the persona's identity and spoke on 5/6 probe turns; 'canonical' sends the "
    "checkpoint's whole prompt and spoke on 6/6 but tells the model it is Tencent's assistant, "
    "which confounds the persona axis; 'directive' is the bare instruction; 'none' is the control "
    "condition and is mostly mute. See the backend's docstring.",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(),
    help="Root for run shards/logs; the run lands in <out-dir>/<audio-version>[.<revision>]/. "
    "Defaults to var/{model} (path-safe).",
)
@click.option("--temperature", default=0.0, show_default=True, type=float, help="Sampling temperature (0 = greedy).")
@click.option(
    "--max-tokens",
    default=DEFAULT_MAX_TOKENS,
    show_default=True,
    type=int,
    help="Token cap per turn. Covo spends this on audio codes as well as text (15 audio tokens per "
    "5 text tokens), so a text-sized budget cuts the speech off mid-word.",
)
@click.option(
    "--history-turns",
    default=DEFAULT_HISTORY_TURNS,
    show_default=True,
    type=int,
    help="Sliding window: send only the last N turns (plus the system prompt) with each request. "
    "0 sends the whole conversation, which overruns the context on long examples.",
)
@click.option(
    "--workers",
    default=DEFAULT_WORKERS,
    show_default=True,
    type=int,
    help="Replay this many examples concurrently. Turns within an example are sequential, so this "
    "is the only parallelism a replay has. The default suits the widened stage overrides in "
    "docs/models.md; the shipped deploy config pins both Covo stages at max_num_seqs=1, so pass "
    "--workers 1 against a server launched without them.",
)
@click.option("--no-speech", is_flag=True, default=False, help="Text-only run (no interleave scaffold, no WAVs).")
@click.option(
    "--stream/--no-stream",
    default=False,
    show_default=True,
    help="Request the reply over SSE. Same content either way — the waveform still arrives as one "
    "segment at the end — but it changes what the logged latency means: streaming records "
    "time-to-first-token, the default records the whole turn.",
)
@click.option(
    "--audio-dir",
    default="data/audio",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Speak text turns from the tools/synthesize_user_turns.py shards under this dir (its --out-dir).",
)
@click.option(
    "--audio-version",
    default=None,
    help="Which audio version under --audio-dir to speak, e.g. 'angry' or 'neutral.1'; "
    "'none' reads clips straight from --audio-dir. Default: the newest neutral version present.",
)
@click.option(
    "--bump-version",
    "bump_version",
    is_flag=True,
    default=False,
    help="Generate into the next revision of this audio version -- neutral -> neutral.1 -- keeping the "
    "run already under it. Default writes the newest revision present, which is what --resume "
    "picks back up.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Pick an interrupted run back up: skip the examples already recorded in "
    "<out-dir>/runs.jsonl and append to chat.log. --no-resume replays every example, "
    "overwriting both.",
)
def main(
    example_name: str,
    run_all: bool,
    model_id: str,
    voice_ref: str,
    base_url: str,
    scaffold: str,
    out_dir: str | None,
    temperature: float,
    max_tokens: int,
    history_turns: int,
    workers: int,
    no_speech: bool,
    stream: bool,
    audio_dir: str,
    audio_version: str | None,
    bump_version: bool,
    resume: bool,
) -> None:
    backend = "covo-audio-api"
    if workers < 1:
        raise click.UsageError(f"--workers must be at least 1, got {workers}")
    if out_dir is None:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")
        out_dir = f"var/generation/{safe_model}"
    root = Path(out_dir)
    modalities = ["text"] if no_speech else ["text", "audio"]

    examples = sorted(EXAMPLES) if run_all else [example_name]

    # Synthesized clips live in versioned subdirs of --audio-dir; pin one version for
    # the whole run so every example is spoken from the same batch.
    resolved = read_dir(audio_dir, audio_version)
    audio_dir = str(resolved)
    # Which clips were spoken is part of what a run is, so the version names the
    # output dir too: a replay against a newer batch lands beside the old run
    # rather than on top of it, and a scored run traces back to what it heard.
    spoken_version = label(resolved)
    if spoken_version is not None:
        # A second generation against the same clips is a run of its own, so it gets
        # its own revision of that version rather than being mixed into the first.
        root = generation_dir(root, spoken_version, bump=bump_version)
    elif bump_version:
        raise click.UsageError(
            f"--bump-version numbers a generation against its audio version, and {audio_dir} "
            "has none -- name a versioned batch with --audio-version, or pass --out-dir yourself."
        )

    click.echo(f"Writing run shards and logs under: {root}/")

    root.mkdir(parents=True, exist_ok=True)

    # --example is a --all run with one example in it — same out-dir, same shared
    # chat.log, same runs.jsonl index — so the two differ only in what gets run.
    log_path = root / "chat.log"
    # A resumed run continues the log: what the interrupted run recorded for the
    # examples it did finish is the log of those examples, and they aren't re-run.
    log_file = log_path.open("a" if resume else "w")

    # chat.log and runs.jsonl are one file each shared by every worker, so both
    # are written under this lock. It also orders them against each other: a row
    # is indexed after its example's lines are on disk.
    output_lock = threading.Lock()

    def emit(line: str = "") -> None:
        # tqdm.write, not click.echo: plain writes to stdout tear the bar apart.
        log_file.write(line + "\n")
        log_file.flush()

    def example_log():
        """An ``emit`` scoped to one example, plus the ``flush`` that commits it.

        With one worker the lines go straight out, so watching a single
        ``--example`` run still shows the conversation as it happens. With
        several, the examples overlap in time, so each one's lines are held and
        written as a single block: a chat.log with two conversations spliced
        together line by line is not something a scored run can be read against.
        """
        if workers <= 1:
            return emit, lambda: None
        held: list[str] = []

        def emit_held(line: str = "") -> None:
            held.append(line)

        def flush() -> None:
            if held:
                log_file.write("\n".join(held) + "\n")
                log_file.flush()
                held.clear()

        return emit_held, flush

    # An audio version only holds the examples that were synthesized into it, and
    # a text turn has nothing to send without its clip: drop the examples this
    # batch never spoke rather than failing the whole run on the first one.
    unspoken = [name for name in examples if not has_shard(audio_dir, name)]
    if unspoken:
        note = f"Skipping {len(unspoken)} example(s) with no shard under {audio_dir}: {', '.join(unspoken)}"
        emit(note)
        click.echo(note)
        examples = [name for name in examples if name not in set(unspoken)]
    if not examples:
        log_file.close()
        click.echo(f"Nothing to run — synthesize with tools/synthesize_user_turns.py --out-dir {audio_dir}")
        return

    # A text-only run has no speech to ask for, so the scaffold would only add an
    # instruction the model cannot follow.
    effective_scaffold = "none" if no_speech else scaffold
    model = load_model(backend, model_id, base_url=base_url, stream=stream, scaffold=effective_scaffold)
    emit(f"Using {model_id} (backend={backend!r}) at {base_url!r} ...")
    emit(f"Transport: {'SSE stream' if stream else 'non-streaming'}")
    emit(f"Interleave scaffold: {effective_scaffold}")
    emit(f"User-turn audio: {audio_dir}")
    emit(f"Run dir: {root}")
    emit(f"History sent per turn: {f'last {history_turns} turns' if history_turns > 0 else 'whole conversation'}")
    emit(f"Concurrent examples: {workers}")
    emit(f"Running {len(examples)} example(s).")

    # Every finished example is appended to runs.jsonl as it lands, so the file is
    # the record of what got through — of this run and of any earlier one against
    # this out-dir, including the runs written before the shards. That makes it the
    # one thing a resumed run has to read to know what not to replay.
    combined_path = root / RUNS_FILENAME
    if not resume:
        # Truncate once, then each example appends its own row again.
        combined_path.write_text("", encoding="utf-8")
    done = recorded_examples(combined_path) if resume else set()
    already = [name for name in examples if name in done]
    if already:
        note = f"Skipping {len(already)} example(s) already in {combined_path.name}"
        emit(note)
        click.echo(note)
    pending = [name for name in examples if name not in done]

    # How many turns came back in text only, across the whole run. Covo does not
    # always take the instruction to speak, so this is a property of the run
    # worth reporting rather than an error: a scored run should say how much of
    # it the voice metrics actually had audio for.
    mute_total = 0

    def replay(name: str) -> dict:
        """One example, start to finish, on whichever worker picked it up."""
        emit_ex, flush = example_log()
        try:
            return run_example(
                name,
                model=model,
                model_id=model_id,
                voice_ref=voice_ref,
                root=root,
                audio_dir=audio_dir,
                modalities=modalities,
                temperature=temperature,
                max_tokens=max_tokens,
                history_turns=history_turns,
                emit=emit_ex,
            )
        finally:
            # An example that raised still gets its lines written: its shard is
            # left unpublished so the next run replays it, and the log is where
            # the turn it died on is recorded.
            with output_lock:
                flush()

    with contextlib.ExitStack() as stack:
        if pending:  # nothing to say to the server if every example is already recorded
            stack.enter_context(model)
        # Examples are independent, so they can be in flight together; the turns
        # inside one cannot. The server admits what arrives up to its per-stage
        # max_num_seqs and queues the rest, so --workers past that buys nothing.
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers, thread_name_prefix="replay"))
        futures = [pool.submit(replay, name) for name in pending]
        progress = tqdm(
            as_completed(futures), total=len(futures), desc="examples", unit="ex", disable=len(pending) <= 1
        )
        try:
            for future in progress:
                row = future.result()
                mute_total += row.get("mute_turns", 0)
                # Indexed only once its shard is on disk, so an interrupted run
                # leaves runs.jsonl listing exactly the shards that are there to
                # be scored.
                with output_lock:
                    append_run(combined_path, row)
        except BaseException:
            # Fail the run the way the sequential loop did: drop what has not
            # started rather than replaying the whole batch behind a dead server.
            # Examples already in flight are still joined by the pool's exit.
            for future in futures:
                future.cancel()
            raise
        finally:
            progress.close()

    if mute_total:
        emit(f"\nMute turns across the run: {mute_total}")
    log_file.close()
    skipped = f" ({len(already)} skipped)" if already else ""
    click.echo(f"\nDone. {len(examples)} example(s){skipped}. Outputs under: {root}/")
    if mute_total:
        click.echo(
            f"{mute_total} turn(s) came back in text only and were stored without a WAV. "
            "Try --scaffold canonical if the voice metrics need more coverage."
        )
    click.echo(f"Chat log: {log_path}")
    click.echo(f"Run index: {combined_path}")
    click.echo(f"Score with: python -m tools.score_run --run-dir {root}")


if __name__ == "__main__":
    main()
