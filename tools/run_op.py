#!/usr/bin/env python3
"""
Quick one-shot run of a Data-Juicer operator on a single sample.

Usage:
  dj-op <op_name> [sample_json]
  echo '{"arxiv_id":"2501.14755"}' | dj-op arxiv_to_markdown_mapper
  dj-op whitespace_normalization_mapper '{"text":"  hello  "}'
  dj-op arxiv_to_markdown_mapper '{"arxiv_id":"2501.14755"}' --backend mineru

Sample can be a JSON object from positional arg, stdin, or minimal default.
Output: JSON object of the processed sample to stdout.
"""
import argparse
import base64
import json
import sys

from loguru import logger

from data_juicer.ops import OPERATORS


def main():
    parser = argparse.ArgumentParser(
        description="Run a Data-Juicer operator on one sample (minimal/quick test).",
        epilog="Example: dj-op arxiv_to_markdown_mapper '{\"arxiv_id\":\"2501.14755\"}' (default: mineru)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available operator names and exit",
    )
    parser.add_argument(
        "op_name",
        type=str,
        nargs="?",
        default=None,
        help="Operator name (e.g. arxiv_to_markdown_mapper, whitespace_normalization_mapper)",
    )
    parser.add_argument(
        "sample_json",
        type=str,
        nargs="?",
        default=None,
        help="JSON object for one sample; if omitted, read from stdin or use minimal default",
    )
    # Forward unknown args as op kwargs (e.g. --backend mineru)
    args, unknown = parser.parse_known_args()

    if args.list:
        names = sorted(OPERATORS.modules.keys())
        for n in names:
            print(n)
        return 0

    if not args.op_name or args.op_name not in OPERATORS.modules:
        logger.error(f"Unknown operator: {args.op_name!r}. Use --list to list names.")
        return 1

    # Parse op kwargs from --key val
    op_kwargs = {}
    i = 0
    while i < len(unknown):
        key = unknown[i]
        if key.startswith("--"):
            key = key[2:].replace("-", "_")
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val = unknown[i + 1]
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    pass
                op_kwargs[key] = val
                i += 2
            else:
                op_kwargs[key] = True
                i += 1
        else:
            i += 1

    # Sample: from arg, stdin, or minimal default
    if args.sample_json:
        try:
            sample = json.loads(args.sample_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid sample JSON: {e}")
            return 1
    elif not sys.stdin.isatty():
        try:
            sample = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from stdin: {e}")
            return 1
    else:
        sample = {}
        if "text" not in sample and "arxiv_id" not in sample:
            sample["text"] = ""

    op_class = OPERATORS.modules[args.op_name]
    op = op_class(**op_kwargs)
    out = op.process(sample)
    if isinstance(out, dict) and not any(isinstance(v, list) for v in out.values()):
        result = out
    else:
        # Batched return: dict of lists -> take first
        result = {k: (v[0] if isinstance(v, list) else v) for k, v in out.items()}

    def _json_serial(obj):
        if isinstance(obj, bytes):
            return "<bytes base64:" + base64.b64encode(obj).decode("ascii")[:80] + ">"
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    try:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_serial))
    except TypeError:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
