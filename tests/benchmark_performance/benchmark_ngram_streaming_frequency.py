#!/usr/bin/env python3
"""Benchmark eager versus streaming n-gram frequency counting.

Run this script from the repository root on macOS or Linux. Memory samples are
collected in fresh processes so that ``ru_maxrss`` is comparable across
variants. The workload uses a single repeated n-gram to isolate the removed
occurrence-list overhead in highly repetitive documents.
"""

import argparse
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from data_juicer.ops.common import SPECIAL_CHARACTERS, get_words_from_document, words_refinement
from data_juicer.ops.filter.character_repetition_filter import CharacterRepetitionFilter
from data_juicer.ops.filter.word_repetition_filter import WordRepetitionFilter
from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = Path(__file__).resolve()
_OPERATORS = ("character", "word")
_VARIANTS = ("baseline", "streaming")
_WORKER_RESULT_PREFIX = "NGRAM_BENCHMARK_RESULT="


def _compute_character_eager(op, samples):
    samples_list = samples[op.text_key]
    samples_stats = samples[Fields.stats]

    for idx, stat in enumerate(samples_stats):
        if StatsKeys.char_rep_ratio in stat:
            continue

        cur_text = samples_list[idx]
        char_ngrams = [cur_text[i : i + op.n] for i in range(len(cur_text) - op.n + 1)]
        freq_char_ngrams = {}
        for char_ngram in char_ngrams:
            freq_char_ngrams[char_ngram] = freq_char_ngrams.get(char_ngram, 0) + 1

        if len(freq_char_ngrams) == 0:
            samples_stats[idx][StatsKeys.char_rep_ratio] = 0.0
            continue

        freq_char_ngrams = sorted(freq_char_ngrams.values(), reverse=True)
        num_no_rep_char_ngrams = freq_char_ngrams.count(1)
        num_rep_char_ngrams = min(
            int(np.sqrt(len(freq_char_ngrams))),
            len(freq_char_ngrams) - num_no_rep_char_ngrams,
        )
        total = sum(freq_char_ngrams)
        samples_stats[idx][StatsKeys.char_rep_ratio] = (
            (sum(freq_char_ngrams[:num_rep_char_ngrams]) / total) if total != 0 else 0.0
        )

    return samples


def _compute_word_eager(op, samples, context=True):
    samples_list = samples[op.text_key]
    samples_stats = samples[Fields.stats]
    words_key = f"{InterVars.words}-{op.model_key}"

    for idx, stat in enumerate(samples_stats):
        if StatsKeys.word_rep_ratio in stat:
            continue
        if context and words_key in samples[Fields.context][idx]:
            words = samples[Fields.context][idx][words_key]
        else:
            tokenizer = get_model(op.model_key)
            words = get_words_from_document(
                samples_list[idx], token_func=tokenizer.encode_as_pieces if tokenizer else None
            )
            if context:
                samples[Fields.context][idx][words_key] = words

        refined_words_key = f"{InterVars.refined_words}-True-SPECIAL_CHARS-False-[2]-"
        if context and refined_words_key in samples[Fields.context][idx]:
            words = samples[Fields.context][idx][refined_words_key]
        else:
            words = words_refinement(words, lower_case=True, strip_chars=SPECIAL_CHARACTERS)
            if context:
                samples[Fields.context][idx][refined_words_key] = words
        word_ngrams = [" ".join(words[i : i + op.n]) for i in range(len(words) - op.n + 1)]
        freq_word_ngrams = {}
        for word_ngram in word_ngrams:
            freq_word_ngrams[word_ngram] = freq_word_ngrams.get(word_ngram, 0) + 1

        if len(freq_word_ngrams) == 0:
            samples_stats[idx][StatsKeys.word_rep_ratio] = 0.0
            continue

        freq_word_ngrams = list(freq_word_ngrams.values())
        rep_more_than_one = [freq for freq in freq_word_ngrams if freq > 1]
        total = sum(freq_word_ngrams)
        samples_stats[idx][StatsKeys.word_rep_ratio] = (sum(rep_more_than_one) / total) if total != 0 else 0.0

    return samples


def _build_workload(operator, occurrences, rep_len):
    if operator == "character":
        op = CharacterRepetitionFilter(rep_len=rep_len, auto_op_parallelism=False, num_proc=1)
        samples = {
            op.text_key: ["a" * (occurrences + rep_len - 1)],
            Fields.stats: [{}],
        }
        stats_key = StatsKeys.char_rep_ratio
    else:
        op = WordRepetitionFilter(rep_len=rep_len, auto_op_parallelism=False, num_proc=1)
        words = ["word"] * (occurrences + rep_len - 1)
        words_key = f"{InterVars.words}-{op.model_key}"
        refined_words_key = f"{InterVars.refined_words}-True-SPECIAL_CHARS-False-[2]-"
        samples = {
            op.text_key: [""],
            Fields.stats: [{}],
            Fields.context: [{words_key: words, refined_words_key: words}],
        }
        stats_key = StatsKeys.word_rep_ratio
    return op, samples, stats_key


def _compute(operator, variant, op, samples):
    if operator == "character":
        return _compute_character_eager(op, samples) if variant == "baseline" else op.compute_stats_batched(samples)
    return (
        _compute_word_eager(op, samples) if variant == "baseline" else op.compute_stats_batched(samples, context=True)
    )


def _max_rss_bytes():
    if resource is None:
        raise RuntimeError("peak RSS measurement requires the Unix resource module")
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(max_rss if sys.platform == "darwin" else max_rss * 1024)


def _memory_worker(args):
    op, samples, stats_key = _build_workload(args.operator, args.occurrences, args.rep_len)
    _compute(args.operator, args.variant, op, samples)
    ratio = samples[Fields.stats][0][stats_key]
    if ratio != 1.0:
        raise AssertionError(f"unexpected repetition ratio: {ratio}")
    print(f"{_WORKER_RESULT_PREFIX}{json.dumps({'max_rss_bytes': _max_rss_bytes(), 'ratio': ratio})}")


def _latency_worker(args):
    op, samples, stats_key = _build_workload(args.operator, args.occurrences, args.rep_len)

    def run_once():
        samples[Fields.stats][0].clear()
        _compute(args.operator, args.variant, op, samples)
        if samples[Fields.stats][0][stats_key] != 1.0:
            raise AssertionError("benchmark variants produced different semantics")

    for _ in range(args.warmup_calls):
        run_once()

    observations = []
    for _ in range(args.latency_observations):
        started = time.perf_counter()
        for _ in range(args.latency_calls_per_observation):
            run_once()
        observations.append((time.perf_counter() - started) / args.latency_calls_per_observation)

    print(f"{_WORKER_RESULT_PREFIX}{json.dumps({'seconds_per_call': observations})}")


def _run_worker(arguments):
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_REPO_ROOT) if not current_pythonpath else f"{_REPO_ROOT}{os.pathsep}{current_pythonpath}"
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), *arguments],
        cwd=_REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker failed ({' '.join(arguments)}):\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    results = [
        line.removeprefix(_WORKER_RESULT_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(_WORKER_RESULT_PREFIX)
    ]
    if len(results) != 1:
        raise RuntimeError(f"worker emitted {len(results)} result records; expected exactly one")
    return json.loads(results[0])


def _percentile(values, percentile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _linear_slope(xs, ys):
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _summarize_memory(samples, occurrences_per_scale, scales):
    summary = {}
    occurrence_counts = [occurrences_per_scale * scale for scale in scales]
    for operator in _OPERATORS:
        summary[operator] = {"by_scale": {}}
        medians = {variant: [] for variant in _VARIANTS}
        for scale in scales:
            scale_key = str(scale)
            baseline = statistics.median(samples[operator]["baseline"][scale_key])
            streaming = statistics.median(samples[operator]["streaming"][scale_key])
            medians["baseline"].append(baseline)
            medians["streaming"].append(streaming)
            summary[operator]["by_scale"][scale_key] = {
                "occurrences": occurrences_per_scale * scale,
                "baseline_median_rss_bytes": baseline,
                "streaming_median_rss_bytes": streaming,
                "rss_reduction": 1 - streaming / baseline,
            }
        baseline_slope = _linear_slope(occurrence_counts, medians["baseline"])
        streaming_slope = _linear_slope(occurrence_counts, medians["streaming"])
        slope_reduction = None if baseline_slope <= 0 or streaming_slope < 0 else 1 - streaming_slope / baseline_slope
        summary[operator]["rss_growth"] = {
            "baseline_bytes_per_occurrence": baseline_slope,
            "streaming_bytes_per_occurrence": streaming_slope,
            "slope_reduction": slope_reduction,
        }
    return summary


def _summarize_latency(samples):
    summary = {}
    for operator in _OPERATORS:
        baseline = samples[operator]["baseline"]
        streaming = samples[operator]["streaming"]
        baseline_median = statistics.median(baseline)
        streaming_median = statistics.median(streaming)
        baseline_p95 = _percentile(baseline, 0.95)
        streaming_p95 = _percentile(streaming, 0.95)
        summary[operator] = {
            "baseline_median_seconds": baseline_median,
            "streaming_median_seconds": streaming_median,
            "throughput_ratio": baseline_median / streaming_median,
            "baseline_p95_seconds": baseline_p95,
            "streaming_p95_seconds": streaming_p95,
            "p95_latency_ratio": streaming_p95 / baseline_p95,
        }
    return summary


def _format_mib(value):
    return f"{value / (1024 * 1024):.1f}"


def _format_percentage(value):
    return "n/a" if value is None else f"{value:.1%}"


def _print_markdown(result):
    print(
        "> Workload: one repeated n-gram. Peak RSS includes interpreter and import overhead; "
        "the RSS growth slope is the primary scaling signal."
    )
    print()
    print("## Peak RSS")
    print()
    print("| Operator | Scale | Occurrences | Baseline MiB | Streaming MiB | Reduction |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for operator in _OPERATORS:
        for scale, values in result["memory"]["summary"][operator]["by_scale"].items():
            print(
                f"| {operator} | {scale}x | {values['occurrences']:,} | "
                f"{_format_mib(values['baseline_median_rss_bytes'])} | "
                f"{_format_mib(values['streaming_median_rss_bytes'])} | "
                f"{values['rss_reduction']:.1%} |"
            )
    print()
    print("| Operator | Baseline RSS slope (B/ngram) | Streaming RSS slope (B/ngram) | Slope reduction |")
    print("| --- | ---: | ---: | ---: |")
    for operator in _OPERATORS:
        values = result["memory"]["summary"][operator]["rss_growth"]
        print(
            f"| {operator} | {values['baseline_bytes_per_occurrence']:.2f} | "
            f"{values['streaming_bytes_per_occurrence']:.2f} | "
            f"{_format_percentage(values['slope_reduction'])} |"
        )
    print()
    print("## Latency")
    print()
    print(
        "| Operator | Baseline median ms | Streaming median ms | "
        "Throughput ratio | Streaming/baseline P95 observation mean |"
    )
    print("| --- | ---: | ---: | ---: | ---: |")
    for operator in _OPERATORS:
        values = result["latency"]["summary"][operator]
        print(
            f"| {operator} | {values['baseline_median_seconds'] * 1000:.3f} | "
            f"{values['streaming_median_seconds'] * 1000:.3f} | {values['throughput_ratio']:.3f}x | "
            f"{values['p95_latency_ratio']:.3f}x |"
        )


def _benchmark(args):
    memory_samples = {
        operator: {variant: {str(scale): [] for scale in args.scales} for variant in _VARIANTS}
        for operator in _OPERATORS
    }
    memory_tasks = [
        (operator, variant, scale, repetition)
        for operator in _OPERATORS
        for variant in _VARIANTS
        for scale in args.scales
        for repetition in range(args.memory_repetitions)
    ]
    random.Random(args.seed).shuffle(memory_tasks)
    for index, (operator, variant, scale, repetition) in enumerate(memory_tasks, start=1):
        occurrences = args.occurrences_per_scale * scale
        print(
            f"memory {index}/{len(memory_tasks)}: {operator} {variant} {scale}x rep {repetition + 1}",
            file=sys.stderr,
            flush=True,
        )
        worker_result = _run_worker(
            [
                "--worker",
                "memory",
                "--operator",
                operator,
                "--variant",
                variant,
                "--occurrences",
                str(occurrences),
                "--rep-len",
                str(args.rep_len),
            ]
        )
        memory_samples[operator][variant][str(scale)].append(worker_result["max_rss_bytes"])

    latency_samples = {operator: {variant: [] for variant in _VARIANTS} for operator in _OPERATORS}
    latency_tasks = [
        (operator, variant, repetition)
        for operator in _OPERATORS
        for variant in _VARIANTS
        for repetition in range(args.latency_repetitions)
    ]
    random.Random(args.seed + 1).shuffle(latency_tasks)
    for index, (operator, variant, repetition) in enumerate(latency_tasks, start=1):
        print(
            f"latency {index}/{len(latency_tasks)}: {operator} {variant} rep {repetition + 1}",
            file=sys.stderr,
            flush=True,
        )
        worker_result = _run_worker(
            [
                "--worker",
                "latency",
                "--operator",
                operator,
                "--variant",
                variant,
                "--occurrences",
                str(args.latency_occurrences),
                "--rep-len",
                str(args.rep_len),
                "--warmup-calls",
                str(args.warmup_calls),
                "--latency-observations",
                str(args.latency_observations),
                "--latency-calls-per-observation",
                str(args.latency_calls_per_observation),
            ]
        )
        latency_samples[operator][variant].extend(worker_result["seconds_per_call"])

    result = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "parameters": {
            "occurrences_per_scale": args.occurrences_per_scale,
            "scales": args.scales,
            "memory_repetitions": args.memory_repetitions,
            "rep_len": args.rep_len,
            "workload": "single_repeated_ngram",
            "latency_occurrences": args.latency_occurrences,
            "latency_repetitions": args.latency_repetitions,
            "latency_observations": args.latency_observations,
            "latency_calls_per_observation": args.latency_calls_per_observation,
            "warmup_calls": args.warmup_calls,
            "seed": args.seed,
        },
        "memory": {
            "samples": memory_samples,
            "summary": _summarize_memory(memory_samples, args.occurrences_per_scale, args.scales),
        },
        "latency": {
            "samples": latency_samples,
            "summary": _summarize_latency(latency_samples),
        },
    }
    _print_markdown(result)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrences-per-scale", type=int, default=600_000)
    parser.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--memory-repetitions", type=int, default=5)
    parser.add_argument("--rep-len", type=int, default=10)
    parser.add_argument("--latency-occurrences", type=int, default=1_000)
    parser.add_argument("--latency-repetitions", type=int, default=3)
    parser.add_argument("--latency-observations", type=int, default=40)
    parser.add_argument("--latency-calls-per-observation", type=int, default=500)
    parser.add_argument("--warmup-calls", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--json-output", type=Path)

    parser.add_argument("--worker", choices=("memory", "latency"), help=argparse.SUPPRESS)
    parser.add_argument("--operator", choices=_OPERATORS, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=_VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--occurrences", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.worker:
        if args.operator is None or args.variant is None or args.occurrences is None:
            raise ValueError("worker mode requires operator, variant, and occurrences")
        if args.occurrences < 2 or args.rep_len <= 0:
            raise ValueError("worker occurrences must be at least 2 and rep_len must be positive")
        if args.worker == "latency" and (
            args.warmup_calls <= 0 or args.latency_observations <= 0 or args.latency_calls_per_observation <= 0
        ):
            raise ValueError("latency worker counts must be positive")
        if args.worker == "memory":
            _memory_worker(args)
        else:
            _latency_worker(args)
    else:
        positive_values = {
            "occurrences_per_scale": args.occurrences_per_scale,
            "memory_repetitions": args.memory_repetitions,
            "rep_len": args.rep_len,
            "latency_occurrences": args.latency_occurrences,
            "latency_repetitions": args.latency_repetitions,
            "latency_observations": args.latency_observations,
            "latency_calls_per_observation": args.latency_calls_per_observation,
            "warmup_calls": args.warmup_calls,
        }
        if any(value <= 0 for value in positive_values.values()):
            raise ValueError("benchmark sizes and repetition counts must be positive")
        if (
            len(args.scales) < 2
            or len(set(args.scales)) != len(args.scales)
            or any(scale <= 0 for scale in args.scales)
        ):
            raise ValueError("at least two unique positive scales are required")
        args.scales.sort()
        if min(args.occurrences_per_scale * scale for scale in args.scales) < 2 or args.latency_occurrences < 2:
            raise ValueError("each memory and latency workload must contain at least 2 n-gram occurrences")
        if resource is None:
            raise RuntimeError("peak RSS benchmark requires macOS or Linux")
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
        _benchmark(args)


if __name__ == "__main__":
    main()
