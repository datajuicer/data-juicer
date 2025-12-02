import argparse

from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.exporter import Exporter


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="Mix multiple datasets Arguments")
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=None,
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

    args = parser.parse_args()

    return args


def run_mixture():
    """
    Mix multiple datasets into one dataset.
    Randomly select samples from every dataset and mix these
    samples, then export to a new mixed dataset

    `data_path` with optional weight(1.0 as default),
        e.g.
        1) a single data path
        2) multiple datasets in the format: <w1> dataset1-path
            <w2> dataset1-file  <w3> dataset3-path ...'

    """
    args = parse_args()
    data_path = " ".join(args.data_path)
    args.dataset_path = data_path
    dataset_builder = DatasetBuilder(args)
    dataset = dataset_builder.load_dataset(num_proc=args.num_proc)
    exporter = Exporter(
        export_path=args.export_path,
        export_shard_size=args.export_shard_size,
        num_proc=args.num_proc,
        export_stats=False,
    )
    exporter.export(dataset)


if __name__ == "__main__":
    run_mixture()
