from __future__ import annotations

import contextlib
import re
from pathlib import Path

import click
from tqdm import tqdm

from rolebreak.examples import EXAMPLES
from rolebreak.models import load_model, user_audio_message
from rolebreak.models.audio import decode_b64
from rolebreak.runlog import RUNS_FILENAME, append_run
from rolebreak.runshard import RunShardWriter, wav_member
from rolebreak.types import Message
from tools.audio_version import generation_dir, label, read_dir
from tools.eval.resume import recorded_examples
from tools.eval.user_turns import has_shard, load_synth_audio, report_drift, resolve_turn_audio


DEFAULT_HISTORY_TURNS = 40


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def windowed(history: list[Message], max_turns: int) -> list[Message]:
    """The system prompt plus the most recent ``max_turns`` turns of ``history``.

    Every user turn is sent as a full audio clip, so an untrimmed history grows
    monotonically in audio tokens and eventually overruns the server's context —
    a 20-turn example died partway through against a Qwen3-Omni endpoint. A
    sliding window bounds what each request carries.

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
    voice: str,
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

    history: list[Message] = [Message(role="system", content=example.system_prompt)]
    emit(f"\n{'#' * 72}\n# example: {example.name}\n{'#' * 72}")
    emit(f"system> {example.system_prompt}")
    emit(f"voice > {voice}")

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
        voice=voice,
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
                audio={"voice": voice, "format": "wav"},
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = (resp.transcript or "").strip()

            # Feed our own reply back as text so the talker's audio doesn't
            # bloat context or trip Qwen's audio-turn limit on long horizons.
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
    "--backend",
    default="qwen3-omni-api",
    show_default=True,
    type=click.Choice(["qwen3-omni-api", "qwen2-audio-api"]),
    help="Which backend to run. qwen2-audio-api is text-out only.",
)
@click.option(
    "--model",
    "model_id",
    required=True,
    help="Model id or local checkpoint path.",
)
@click.option("--voice", default="Ethan", show_default=True, help="Speaker voice (Chelsie / Ethan / Aiden).")
@click.option(
    "--base-url",
    default="http://localhost:8901/v1",
    show_default=True,
    help="vLLM OpenAI-compatible endpoint serving Qwen3-Omni.",
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
@click.option("--no-speech", is_flag=True, default=False, help="Text-only run (skip the talker / no WAVs).")
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
    backend: str,
    model_id: str,
    voice: str,
    base_url: str,
    out_dir: str | None,
    temperature: float,
    max_tokens: int,
    history_turns: int,
    no_speech: bool,
    audio_dir: str,
    audio_version: str | None,
    bump_version: bool,
    resume: bool,
) -> None:
    if out_dir is None:
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")
        out_dir = f"var/generation/{safe_model}"
    root = Path(out_dir)
    # Qwen2-Audio has no talker head — force text-out regardless of --no-speech.
    text_only = no_speech or backend == "qwen2-audio-api"
    modalities = ["text"] if text_only else ["text", "audio"]

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

    def emit(line: str = "") -> None:
        # tqdm.write, not click.echo: plain writes to stdout tear the bar apart.
        # tqdm.write(line)
        log_file.write(line + "\n")
        log_file.flush()

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

    model = load_model(backend, model_id, base_url=base_url)
    emit(f"Using {model_id} (backend={backend!r}) at {base_url!r} ...")
    emit(f"User-turn audio: {audio_dir}")
    emit(f"Run dir: {root}")
    emit(f"History sent per turn: {f'last {history_turns} turns' if history_turns > 0 else 'whole conversation'}")
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

    with contextlib.ExitStack() as stack:
        if pending:  # nothing to say to the server if every example is already recorded
            stack.enter_context(model)
        for name in tqdm(pending, desc="examples", unit="ex", disable=len(pending) <= 1):
            row = run_example(
                name,
                model=model,
                model_id=model_id,
                voice=voice,
                root=root,
                audio_dir=audio_dir,
                modalities=modalities,
                temperature=temperature,
                max_tokens=max_tokens,
                history_turns=history_turns,
                emit=emit,
            )
            # Indexed only once its shard is on disk, so an interrupted run leaves
            # runs.jsonl listing exactly the shards that are there to be scored.
            append_run(combined_path, row)

    log_file.close()
    skipped = f" ({len(already)} skipped)" if already else ""
    click.echo(f"\nDone. {len(examples)} example(s){skipped}. Outputs under: {root}/")
    click.echo(f"Chat log: {log_path}")
    click.echo(f"Run index: {combined_path}")
    click.echo(f"Score with: python -m tools.score_run --run-dir {root}")


if __name__ == "__main__":
    main()
