"""Helpers for token-budget data mixing.

These helpers are intentionally dependency-light so they can be unit-tested
without loading a full Data-Juicer pipeline. The CLI in
``tools/postprocess/data_mixture.py`` wires them to a loaded dataset.
"""

import json
import math
import random
from pathlib import Path
from typing import Callable, List, Sequence


def whitespace_token_count(text: str) -> int:
    """Cheap, dependency-free token proxy (whitespace words)."""
    return len(text.split())


def build_token_counter(tokenizer: str = "whitespace") -> Callable[[str], int]:
    """Build a token counter.

    ``whitespace`` is the default and needs no model. ``tiktoken`` uses the
    ``cl100k_base`` encoding when the optional package is installed. Any other
    value is treated as a Hugging Face tokenizer name.
    """
    if tokenizer in ("whitespace", "space", "words", ""):
        return whitespace_token_count
    if tokenizer == "tiktoken":
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("--tokenizer tiktoken requires `pip install tiktoken`") from exc
        encoding = tiktoken.get_encoding("cl100k_base")

        def count_tiktoken(text: str) -> int:
            return len(encoding.encode(text))

        return count_tiktoken
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("a Hugging Face --tokenizer requires `pip install transformers`") from exc
    tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer)

    def count_hf(text: str) -> int:
        return len(tokenizer_obj.encode(text, add_special_tokens=False))

    return count_hf


def shuffle_rows(rows: Sequence[dict], seed: int = 42) -> List[dict]:
    """Return a deterministically shuffled copy of ``rows``."""
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def quality_weighted_rows(
    rows: Sequence[dict],
    quality_field: str,
    alpha: float = 1.0,
    seed: int = 42,
) -> List[dict]:
    """Weighted sampling without replacement by ``(1 + quality) ** alpha``.

    Uses the Efraimidis-Spirakis key (``u ** (1 / w)``) and sorts descending:
    every row is kept exactly once, but high-quality rows tend to appear first.
    ``alpha == 0`` degrades to a uniform shuffle.
    """
    if alpha < 0:
        raise ValueError("quality_weight must be >= 0")
    rng = random.Random(seed)
    keyed = []
    for row in rows:
        value = row.get(quality_field)
        try:
            quality = float(value)
        except (TypeError, ValueError):
            quality = 0.0
        if not math.isfinite(quality) or quality <= 0:
            quality = 0.0
        weight = (1.0 + quality) ** alpha
        key = rng.random() ** (1.0 / weight)
        keyed.append((key, row))
    keyed.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in keyed]


def token_budget_shards(
    rows: Sequence[dict],
    counts: Sequence[int],
    budget: int,
    tail_merge_ratio: float = 0.5,
) -> List[List[dict]]:
    """Greedily pack rows into shards with at most ``budget`` tokens.

    A single row larger than ``budget`` is allowed to exceed it (it cannot be
    split); the shard containing it is still emitted. If the final shard is
    smaller than ``tail_merge_ratio * budget`` it is merged into the previous
    one to avoid a tiny tail shard.
    """
    if budget <= 0:
        raise ValueError("token_budget must be positive")
    if len(rows) != len(counts):
        raise ValueError("rows and counts must have the same length")
    if not rows:
        return []

    shards: List[List[dict]] = []
    shard_counts: List[int] = []
    current: List[dict] = []
    current_tokens = 0

    for row, count in zip(rows, counts):
        if count < 0:
            raise ValueError("token counts must be non-negative")
        if current and current_tokens + count > budget:
            shards.append(current)
            shard_counts.append(current_tokens)
            current = []
            current_tokens = 0
        current.append(row)
        current_tokens += count

    if current:
        shards.append(current)
        shard_counts.append(current_tokens)

    if len(shards) > 1 and shard_counts[-1] < budget * tail_merge_ratio:
        shards[-2].extend(shards[-1])
        shard_counts[-2] += shard_counts[-1]
        shards.pop()
        shard_counts.pop()

    return shards


def _shard_path(export_path: str, index: int, total: int) -> Path:
    path = Path(export_path)
    if total == 1:
        return path
    return path.with_name(f"{path.stem}-{index:05d}{path.suffix}")


def write_jsonl_shards(
    shards: Sequence[Sequence[dict]],
    export_path: str,
    ensure_ascii: bool = False,
) -> List[str]:
    """Write each shard as JSONL and return the written paths."""
    written = []
    total = len(shards)
    for index, shard in enumerate(shards):
        path = _shard_path(export_path, index, total)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in shard:
                handle.write(json.dumps(row, ensure_ascii=ensure_ascii) + "\n")
        written.append(str(path))
    return written


def token_budget_for_rows(
    rows: Sequence[dict],
    token_counter: Callable[[str], int],
    text_field: str = "text",
) -> List[int]:
    """Count tokens for each row; missing/non-string text counts as 0."""
    counts = []
    for row in rows:
        text = row.get(text_field, "")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        counts.append(token_counter(text))
    return counts
