"""Score a finished role-play run with the RoleVoiceBench metrics.

Usage
-----
    # score the captain run produced under var/qwen3_omni_examples/captain/
    CUDA_VISIBLE_DEVICES=1 \
    python -m tools.score_run --run-dir var/qwen3_omni_examples/captain

    # score every example in an --all out-dir root (consolidated runs.jsonl)
    python -m tools.score_run --run-dir var/Qwen_Qwen2-Audio-7B-Instruct

    # pick the metrics instead of the default suite (repeat --metric; order is kept)
    python -m tools.score_run --run-dir var/qwen3_omni_examples/captain \
        --metric naturalness --metric text_quality

    # just parse + show the rebuilt transcript, no model load (sanity check)
    python -m tools.score_run --run-dir var/qwen3_omni_examples/captain --dry-run

    # write the full report(s) as JSON
    python -m tools.score_run --run-dir var/qwen3_omni_examples/captain --out report.json

    # throw away the cached per-metric scores and judge everything again
    python -m tools.score_run --run-dir var/Qwen_Qwen2-Audio-7B-Instruct --rescore

    # score 8 runs at a time (worth it for LLM judges, which just wait on the API)
    python -m tools.score_run --run-dir var/Qwen_Qwen2-Audio-7B-Instruct \
        --metric text_quality --parallel 8

    # only score the first 5 examples (quick smoke test of a big consolidated run)
    python -m tools.score_run --run-dir var/Qwen_Qwen2-Audio-7B-Instruct --max-examples 5
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from rolebreak.metrics import DEFAULT_METRICS, JudgeCallError, ScoreStore, available_metrics, evaluate_many
from rolebreak.metrics.store import PARTIAL_SUFFIX
from rolebreak.runlog import FILENAME, RUNS_FILENAME, load_runs
from rolebreak.runshard import SUFFIX as SHARD_SUFFIX


@click.command()
@click.option(
    "--run-dir",
    "run_dir",
    required=True,
    type=click.Path(exists=True),
    help="A runner output dir — one holding run.jsonl + turn_NN.wav, or one holding per-example "
    "<example>.tar run shards (e.g. var/qwen3_omni_examples/captain). A single .tar may be named directly.",
)
@click.option(
    "--metric",
    "metric_names",
    multiple=True,
    type=click.Choice(available_metrics()),
    help=(f"Metric to run; repeat to run several, in the order given. Default suite: {', '.join(DEFAULT_METRICS)}."),
)
@click.option("--out", "out_path", default=None, type=click.Path(), help="Write the full report as JSON here.")
@click.option("--dry-run", is_flag=True, default=False, help="Parse + rebuild the transcript only; don't load judges.")
@click.option(
    "--rescore",
    is_flag=True,
    default=False,
    help="Ignore any <metric>.jsonl already in the run dir and score every run again (replaces each once its pass finishes).",
)
@click.option(
    "--parallel",
    default=1,
    type=click.IntRange(min=1),
    show_default=True,
    help=(
        "Score this many runs at once within each metric. Use it for LLM-judge metrics (the time goes on the "
        "API call); leave it at 1 for local GPU judges like the voice one."
    ),
)
@click.option(
    "--max-examples",
    "max_examples",
    default=None,
    type=click.IntRange(min=1),
    help="Score only the first N examples of the run dir (in load order). Default: every example.",
)
def main(
    run_dir: str,
    metric_names: tuple[str, ...],
    out_path: str | None,
    dry_run: bool,
    rescore: bool,
    parallel: int,
    max_examples: int | None,
) -> None:
    target = Path(run_dir)
    if (target / RUNS_FILENAME).exists():
        # A consolidated index with no shards beside it — a run of the older
        # directory format, one row per example.
        run_path_dir = target
        transcripts = load_runs(target, filename=RUNS_FILENAME)
    else:
        raise click.UsageError(
            f"No {FILENAME}, {RUNS_FILENAME} or *{SHARD_SUFFIX} run shard in {target}. Point --run-dir at a runner's "
            f"per-example output folder, an --all out-dir root, or a run shard."
        )
    if max_examples is not None and len(transcripts) > max_examples:
        click.echo(f"--max-examples {max_examples}: scoring {max_examples} of {len(transcripts)} run(s).")
        transcripts = transcripts[:max_examples]
    if dry_run:
        for transcript in transcripts:
            click.echo(f"--dry-run: skipping judges. First exchange of {transcript.name}:")
            if transcript.exchanges:
                ex = transcript.exchanges[0]
                click.echo(f"  user> {ex.user.content}")
                click.echo(f"  bot > {ex.assistant_text}")
                click.echo(f"  audio> {'yes' if ex.assistant_audio else 'no'}")
                click.echo(f"  lat  > {f'{ex.latency:.2f}s' if ex.latency is not None else 'not recorded'}")
        return

    # Metric-outer: every transcript is scored by one metric before the next
    # metric (and its judge model) loads, so the suite is built once, not per run.
    metrics = list(metric_names) or None
    suite = ", ".join(metrics) if metrics else f"default suite ({', '.join(DEFAULT_METRICS)})"
    click.echo(f"\nRunning {suite} over {len(transcripts)} run(s)...")

    # Each metric's verdicts land in <run-dir>/<metric>.jsonl.partial as they're
    # produced and are published at <metric>.jsonl once that metric has been over
    # every run, so an interrupted pass resumes instead of re-earning them — and a
    # metric whose runs are all on disk never loads its judge model at all.
    store = ScoreStore(run_path_dir, resume=not rescore)
    names = {t.name for t in transcripts}
    if len(names) != len(transcripts):
        raise ValueError(f"{len(transcripts) - len(names)} run(s) share a name; their cached scores will collide.")

    try:
        reports = evaluate_many(transcripts, store=store, metrics=metrics, parallel=parallel)
    except JudgeCallError as e:
        # The judge endpoint, not the model under test. Nothing is published, so
        # the half-finished sweep stays in <metric>.jsonl.partial and resumes
        # once the endpoint is healthy — no zeros are written for runs that were
        # never actually judged.
        raise click.ClickException(
            f"The judge endpoint failed, so scoring stopped and nothing was published: {e}\n"
            f"Runs judged before the failure are kept in {run_path_dir}/<metric>.jsonl{PARTIAL_SUFFIX}; "
            "fix the endpoint ($OPENAI_BASE_URL / $OPENAI_MODEL / $OPENAI_API_KEY) and re-run to resume."
        ) from e
    click.echo(f"Per-metric scores -> {run_path_dir}/<metric>.jsonl")
    # Only the aggregate is printed; per-example detail lives in --out.
    overalls = [r.overall for r in reports if r.overall is not None]
    if not overalls:
        click.echo("\nNo metrics ran. (Needs assistant audio for the voice judge — was this a --no-speech run?)")
    else:
        mean_overall = sum(overalls) / len(overalls)
        click.echo(f"\n=== mean final score over {len(overalls)} run(s): {mean_overall:.1f}/100 ===")
        metric_buckets: dict[str, list[float]] = {}
        for r in reports:
            for metric, val in r.by_metric().items():
                metric_buckets.setdefault(metric, []).append(val)
        click.echo("mean by metric:")
        for metric, vals in metric_buckets.items():
            click.echo(f"  {metric:<12} {sum(vals) / len(vals):6.1f}")

    # Latency is measured by the runner, not a judge, so report it whether or
    # not any metric ran. Turns from a backend that can't time itself are None
    # and simply don't count toward the mean.
    latencies = [ex.latency for t in transcripts for ex in t.exchanges if ex.latency is not None]
    if latencies:
        n_turns = sum(len(t.exchanges) for t in transcripts)
        suffix = f" ({len(latencies)}/{n_turns} turns recorded)" if len(latencies) < n_turns else ""
        click.echo(f"mean latency: {sum(latencies) / len(latencies):.2f}s{suffix}")
    else:
        click.echo("mean latency: not recorded")

    if out_path and reports:
        # Single run -> one object; consolidated -> a list, one report per example.
        payload = reports[0].to_dict() if len(reports) == 1 else [r.to_dict() for r in reports]
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"\nWrote {len(reports)} report(s) -> {out_path}")


if __name__ == "__main__":
    main()
