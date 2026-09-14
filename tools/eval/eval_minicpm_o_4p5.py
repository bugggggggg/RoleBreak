"""Replay the example conversations against a served MiniCPM-o 4.5.

Usage
-----
    # one example; text turns are spoken from data/audio by default
    python -m tools.eval.eval_minicpm_o_4p5 --example ip-conan

    # synthesize the spoken user turns first if they aren't in data/audio yet
    python -m tools.synthesize_user_turns ip-conan --out-dir data/audio

    # every example instead of one; either way the out-dir holds one <example>.tar
    # run shard per example, indexed in runs.jsonl, with a shared chat.log
    python -m tools.eval.eval_minicpm_o_4p5 --all

    # an interrupted run is picked back up by default — the examples runs.jsonl
    # already lists are skipped; replay every example instead
    python -m tools.eval.eval_minicpm_o_4p5 --all --no-resume

    # score what came out
    python -m tools.score_run --run-dir var/generation/openbmb_MiniCPM-o-4_5/happy

    # replay an older batch of synthesized clips instead of the newest one
    python -m tools.eval.eval_minicpm_o_4p5 --all --audio-version neutral

    # a remote endpoint, and time-to-first-token latency
    python -m tools.eval.eval_minicpm_o_4p5 --all \\
        --base-url http://10.0.0.5:8092/v1 --stream

    # replay several examples at once -- match the server's per-stage
    # max_num_seqs (docs/models.md), since that is what caps the batch
    python -m tools.eval.eval_minicpm_o_4p5 --all --workers 8

A sibling of ``eval_qwen2p5_omni`` rather than a flag on it. The two servers are
deployed on different ports, and MiniCPM-o has no voice to pick: its Code2Wav
stage clones a reference clip fixed at deploy time, so there is no ``--voice``
here — see
:class:`~rolebreak.models.backends.minicpm_o_4p5.MiniCPMO4p5APIModel`. Launch the
server with the command in ``docs/models.md`` (it needs the derived
``rolebreak:vllm-omni`` image and an explicit ``--deploy-config``).
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
from rolebreak.models.backends.minicpm_o_4p5 import DEFAULT_VOICE_REF
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
#: The useful ceiling is the *narrowest* stage's ``max_num_seqs``, since that is
#: what caps the batch and everything past it only queues. Talker and Code2Wav
#: are 60% of a turn and are the stages held at 4 in ``docs/models.md`` -- they
#: share a card and the talker OOMs if widened -- so four is the ceiling worth
#: having by default. Raise it only alongside the server: see the
#: ``max_num_seqs`` notes under the MiniCPM-o entry there.
DEFAULT_WORKERS = 4


#: Appended to every persona's system prompt before it is sent.
#:
#: Carried over from the Qwen2.5-Omni eval, and it earns its place here too:
#: without it MiniCPM-o answers an audio turn with Markdown headings and
#: numbered sections, which is not something a persona can say out loud.
SPOKEN_STYLE_SUFFIX = "\n\nKeep answer to a couple of sentences. Never use Markdown or any other markup. Write only words that would be said."


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def windowed(history: list[Message], max_turns: int) -> list[Message]:
    """The system prompt plus the most recent ``max_turns`` turns of ``history``.

    Every user turn is sent as a full audio clip, so an untrimmed history grows
    monotonically in audio tokens and eventually overruns the server's context.
    MiniCPM-o is served with the same 32k window as the Qwen omni models, so it
    gets the same sliding window to bound what each request carries.

    The system prompt is never dropped: it is the persona under test, and a turn
    answered without it is not the run we mean to score. ``max_turns <= 0``
    disables the window and sends everything.
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
    """
    example = EXAMPLES[example_name]
    synth_audio = load_synth_audio(audio_dir, example_name)

    system_prompt = example.system_prompt + SPOKEN_STYLE_SUFFIX
    history: list[Message] = [Message(role="system", content=system_prompt)]
    emit(f"\n{'#' * 72}\n# example: {example.name}\n{'#' * 72}")
    emit(f"system> {system_prompt}")
    emit(f"voice > {voice_ref} (server-side reference clip)")

    # Resolve every turn's audio up front — a turn that authored its own clip keeps
    # it, the rest come from the shard — so a missing clip fails before the first
    # request rather than partway through the conversation.
    clips = [resolve_turn_audio(turn, synth_audio.get(i)) for i, turn in enumerate(example.turns)]
    report_drift(example.turns, clips, emit)

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

            # Feed our own reply back as text so the talker's audio doesn't
            # bloat context on long horizons.
            history.append(Message(role="assistant", content=reply))
            emit(f"  bot > {reply or '(no assistant text received)'}")
            if resp.latency is not None:
                emit(f"  lat > {resp.latency:.2f}s")
            # A turn the server had to be restarted for is still a valid turn, but
            # it should be visible as one rather than reading like a clean reply.
            retries = resp.raw.get("retries") if isinstance(resp.raw, dict) else None
            if retries:
                emit(f"  rtry> recovered after {retries} engine restart(s)")

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
    return shard.row()


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
    default="openbmb/MiniCPM-o-4_5",
    show_default=True,
    help="Model id or local checkpoint path.",
)
@click.option(
    "--voice-ref",
    default=DEFAULT_VOICE_REF,
    show_default=True,
    help="Name of the reference clip the server's Code2Wav stage speaks with, recorded in the run "
    "shard for provenance. MiniCPM-o has no per-request voice: change the served voice with the "
    "stage connector's extra.prompt_wav, then name it here so the run says what it heard.",
)
@click.option(
    "--base-url",
    default="http://localhost:8092/v1",
    show_default=True,
    help="vLLM OpenAI-compatible endpoint serving MiniCPM-o 4.5.",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(),
    help="Root for run shards/logs; the run lands in <out-dir>/<audio-version>[.<revision>]/. "
    "Defaults to var/{model} (path-safe).",
)
@click.option("--temperature", default=0.0, show_default=True, type=float, help="Sampling temperature (0 = greedy).")
@click.option("--max-tokens", default=256, show_default=True, type=int, help="Thinker (text) token cap per turn.")
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
    "is the only parallelism a replay has; set it to the server's per-stage max_num_seqs, since "
    "that is what caps the batch. 1 replays one at a time and streams chat.log live.",
)
@click.option("--no-speech", is_flag=True, default=False, help="Text-only run (skip the talker / no WAVs).")
@click.option(
    "--stream/--no-stream",
    default=False,
    show_default=True,
    help="Request the reply over SSE. Same content either way — this server returns speech on the "
    "plain call too — but it changes what the logged latency means: streaming records "
    "time-to-first-token, the default records the whole turn.",
)
@click.option(
    "--thinking/--no-thinking",
    "enable_thinking",
    default=False,
    show_default=True,
    help="Let the model reason before answering. The <think> block is stripped from the reply "
    "either way (it lands in message.content, not a separate field), so this only changes "
    "whether the tokens are spent.",
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
    out_dir: str | None,
    temperature: float,
    max_tokens: int,
    history_turns: int,
    workers: int,
    no_speech: bool,
    stream: bool,
    enable_thinking: bool,
    audio_dir: str,
    audio_version: str | None,
    bump_version: bool,
    resume: bool,
) -> None:
    backend = "minicpm-o-4.5-api"
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

    model = load_model(backend, model_id, base_url=base_url, stream=stream, enable_thinking=enable_thinking)
    emit(f"Using {model_id} (backend={backend!r}) at {base_url!r} ...")
    emit(f"Transport: {'SSE stream' if stream else 'non-streaming'}")
    emit(f"Thinking: {'on' if enable_thinking else 'off'} (<think> stripped from the reply either way)")
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
        # inside one cannot. The server batches what arrives up to its per-stage
        # max_num_seqs and queues the rest, so --workers past that buys nothing.
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=workers, thread_name_prefix="replay"))
        futures = [pool.submit(replay, name) for name in pending]
        progress = tqdm(
            as_completed(futures), total=len(futures), desc="examples", unit="ex", disable=len(pending) <= 1
        )
        try:
            for future in progress:
                row = future.result()
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

    log_file.close()
    skipped = f" ({len(already)} skipped)" if already else ""
    click.echo(f"\nDone. {len(examples)} example(s){skipped}. Outputs under: {root}/")
    click.echo(f"Chat log: {log_path}")
    click.echo(f"Run index: {combined_path}")
    click.echo(f"Score with: python -m tools.score_run --run-dir {root}")


if __name__ == "__main__":
    main()
