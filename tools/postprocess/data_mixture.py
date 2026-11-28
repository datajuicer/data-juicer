import argparse
import shlex

from data_juicer.core.data.dataset_builder import DatasetBuilder, rewrite_cli_datapath
from data_juicer.core.exporter import Exporter
from data_juicer.utils.data_mixture import (
    build_token_counter,
    quality_weighted_rows,
    shuffle_rows,
    token_budget_for_rows,
    token_budget_shards,
    write_jsonl_shards,
)


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="Mix multiple datasets Arguments")
    parser.add_argument(
        "--data_path",
        nargs="+",
        required=True,
        help="Path to datasets. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-weight dataset1-path dataset2-weight "
        "dataset2-path ...",
    )

    parser.add_argument(
        "--export_path",
        default="mixed.jsonl",
        help="Path to save the mixed dataset. " "Supported suffixes include " '["jsonl", "json", "parquet"]',
    )

    parser.add_argument(
        "--export_shard_size",
        type=int,
        default=0,
        help="Shard size of exported dataset in Byte. In "
        "default, it's 0, which means export the whole "
        "dataset into only one file. If it's set a "
        "positive number, the exported dataset will be "
        "split into several dataset shards, and the max "
        "size of each shard won't larger than the "
        "export_shard_size",
    )

    parser.add_argument("--max_samples", type=int, default=None, help="Number of samples of mixed dataset.")

    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes to process dataset.")

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Globally shuffle the mixed rows before exporting (seeded by --seed).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle/weighted sampling.")
    parser.add_argument(
        "--shard_by_tokens",
        action="store_true",
        help="Write shards bounded by --token_budget instead of --export_shard_size bytes.",
    )
    parser.add_argument(
        "--token_budget",
        type=int,
        default=None,
        help="Maximum tokens per shard when --shard_by_tokens is set. "
        "A single row larger than the budget is allowed to exceed it.",
    )
    parser.add_argument(
        "--tokenizer",
        default="whitespace",
        help="Token counter: 'whitespace' (default, no model), 'tiktoken', " "or a Hugging Face tokenizer name.",
    )
    parser.add_argument(
        "--text_field",
        default="text",
        help="Field used for token counting (default: text).",
    )
    parser.add_argument(
        "--quality_field",
        default=None,
        help="Optional numeric field; rows are weighted by quality ** --quality_weight.",
    )
    parser.add_argument(
        "--quality_weight",
        type=float,
        default=0.0,
        help="Exponent for quality-aware sampling. 0 disables the quality weighting.",
    )

    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max_samples must be a positive integer")
    if args.shard_by_tokens and (args.token_budget is None or args.token_budget <= 0):
        parser.error("--shard_by_tokens requires a positive --token_budget")
    if args.quality_weight < 0:
        parser.error("--quality_weight must be >= 0")

    return args


def _export_rows(rows, args):
    counter = build_token_counter(args.tokenizer)
    counts = token_budget_for_rows(rows, counter, args.text_field)
    shards = token_budget_shards(rows, counts, args.token_budget)
    written = write_jsonl_shards(shards, args.export_path)
    print(
        f"wrote {len(written)} shard(s), {len(rows)} rows, " f"{sum(counts)} tokens (budget {args.token_budget}/shard)"
    )
    return written


def run_mixture():
    """
    Mix multiple datasets into one dataset.

    `data_path` with optional weight(1.0 as default), e.g.
        1) a single data path
        2) multiple datasets in the format: <w1> dataset1-path
           <w2> dataset1-file  <w3> dataset3-path ...

    When `--shard_by_tokens` is set the tool writes JSONL shards bounded by
    `--token_budget`; otherwise it keeps the legacy byte-sharded Exporter path.
    """
    args = parse_args()
    data_path = shlex.join(args.data_path)
    args.dataset = rewrite_cli_datapath(data_path, max_sample_num=args.max_samples)
    dataset_builder = DatasetBuilder(args)
    dataset = dataset_builder.load_dataset(num_proc=args.num_proc)

    needs_rows = args.shuffle or args.shard_by_tokens or args.quality_field
    if not needs_rows:
        exporter = Exporter(
            export_path=args.export_path,
            export_shard_size=args.export_shard_size,
            num_proc=args.num_proc,
            export_stats=False,
        )
        exporter.export(dataset)
        return

    rows = dataset.to_list()
    if args.quality_field:
        rows = quality_weighted_rows(rows, args.quality_field, args.quality_weight, args.seed)
    if args.shuffle:
        rows = shuffle_rows(rows, args.seed)

    if args.shard_by_tokens:
        _export_rows(rows, args)
        return

    # Shuffle/quality-only mode: one JSONL file with the reordered rows.
    written = write_jsonl_shards([rows], args.export_path)
    print(f"wrote {len(written)} file(s), {len(rows)} rows")


if __name__ == "__main__":
    run_mixture()
