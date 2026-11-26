#!/usr/bin/env python3
import argparse
import gzip
import importlib.util
import json
import statistics
import time
from pathlib import Path

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.clean_html_mapper import CleanHtmlMapper
from data_juicer.ops.mapper.extract_html_content_mapper import ExtractHtmlContentMapper


def load_benchmark(benchmark_dir, limit):
    ground_truth = json.loads((benchmark_dir / "ground-truth.json").read_text(encoding="utf-8"))
    paths = sorted((benchmark_dir / "html").glob("*.html.gz"))
    if limit:
        paths = paths[:limit]
    pages = []
    for path in paths:
        item_id = path.name.removesuffix(".html.gz")
        with gzip.open(path, "rt", encoding="utf-8") as file:
            pages.append((item_id, file.read()))
    return pages, {item_id: ground_truth[item_id] for item_id, _ in pages}


def load_evaluator(benchmark_dir):
    spec = importlib.util.spec_from_file_location("article_extraction_evaluate", benchmark_dir / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_operator(op, pages):
    dataset = Dataset.from_list([{"text": html} for _, html in pages])
    return op.run(dataset)[op.text_key]


def run_clean_html(pages):
    return run_operator(CleanHtmlMapper(num_proc=1), pages)


def run_html_content(pages):
    return run_operator(ExtractHtmlContentMapper(num_proc=1), pages)


def benchmark(name, runner, pages, repeats):
    durations = []
    predictions = None
    for _ in range(repeats):
        start = time.perf_counter()
        predictions = runner(pages)
        durations.append(time.perf_counter() - start)
    return name, predictions, statistics.median(durations)


def main():
    parser = argparse.ArgumentParser(
        description="Compare HTML extraction operators on " "scrapinghub/article-extraction-benchmark."
    )
    parser.add_argument(
        "benchmark_dir",
        type=Path,
        help="Path to scrapinghub/article-extraction-benchmark",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pages")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed runs; reports the median",
    )
    args = parser.parse_args()

    pages, ground_truth = load_benchmark(args.benchmark_dir, args.limit)
    evaluator = load_evaluator(args.benchmark_dir)
    print(f"pages={len(pages)} repeats={args.repeats}")
    print(f"{'operator':<42} {'F1':>7} {'precision':>10} " f"{'recall':>8} {'pages/s':>9}")

    for name, runner in (
        ("clean_html_mapper", run_clean_html),
        ("extract_html_content_mapper", run_html_content),
    ):
        name, extracted, duration = benchmark(name, runner, pages, args.repeats)
        prediction = {item_id: {"articleBody": text or ""} for (item_id, _), text in zip(pages, extracted)}
        metrics = evaluator.evaluate(ground_truth, prediction, n_bootstrap=0)
        print(
            f"{name:<42} {metrics['f1']:7.3f} {metrics['precision']:10.3f} "
            f"{metrics['recall']:8.3f} {len(pages) / duration:9.1f}"
        )


if __name__ == "__main__":
    main()
