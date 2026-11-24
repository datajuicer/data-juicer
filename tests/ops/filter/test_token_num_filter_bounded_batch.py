import random
from unittest import mock

import pytest

from data_juicer.ops.filter import token_num_filter as token_num_module
from data_juicer.ops.filter.token_num_filter import TokenNumFilter
from data_juicer.utils.constant import Fields, StatsKeys


def _token_ids(text):
    return list(text.encode("utf-8")) + [0] * len(text.split())


class RecordingTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        input_ids = [_token_ids(text) for text in texts]
        return {
            "input_ids": input_ids,
            "attention_mask": [[1] * len(ids) for ids in input_ids],
        }


def test_tokenizes_uncached_rows_in_bounded_batches():
    tokenizer = RecordingTokenizer()
    op = TokenNumFilter(hf_tokenizer="unused")
    cached_indices = {3, 200}
    payload = object()
    samples = {
        "text": [f"row-{index}" for index in range(259)],
        Fields.stats: [
            ({StatsKeys.num_token: 99, "marker": index} if index in cached_indices else {"marker": index})
            for index in range(259)
        ],
        "payload": payload,
    }

    with mock.patch.object(token_num_module, "get_model", return_value=tokenizer) as get_model_mock:
        returned = op.compute_stats_batched(samples)

    assert returned is samples
    assert samples["payload"] is payload
    get_model_mock.assert_called_once_with(op.model_key)
    assert [len(texts) for texts, _kwargs in tokenizer.calls] == [128, 128, 1]
    assert all(kwargs == {"add_special_tokens": False} for _texts, kwargs in tokenizer.calls)
    assert [text for texts, _kwargs in tokenizer.calls for text in texts] == [
        text for index, text in enumerate(samples["text"]) if index not in cached_indices
    ]
    assert [stat["marker"] for stat in samples[Fields.stats]] == list(range(259))
    assert [stat[StatsKeys.num_token] for stat in samples[Fields.stats]] == [
        99 if index in cached_indices else len(_token_ids(text)) for index, text in enumerate(samples["text"])
    ]


@pytest.mark.parametrize(
    ("uncached_rows", "expected_batch_sizes"),
    [
        (0, []),
        (1, [1]),
        (127, [127]),
        (128, [128]),
        (129, [128, 1]),
        (255, [128, 127]),
        (256, [128, 128]),
        (257, [128, 128, 1]),
    ],
)
def test_tokenization_batch_boundaries(uncached_rows, expected_batch_sizes):
    tokenizer = RecordingTokenizer()
    op = TokenNumFilter(hf_tokenizer="unused")
    samples = {
        "text": [f"boundary {index}" for index in range(uncached_rows)],
        Fields.stats: [{} for _index in range(uncached_rows)],
    }

    with mock.patch.object(token_num_module, "get_model", return_value=tokenizer):
        op.compute_stats_batched(samples)

    assert [len(texts) for texts, _kwargs in tokenizer.calls] == expected_batch_sizes
    assert [stat[StatsKeys.num_token] for stat in samples[Fields.stats]] == [
        len(_token_ids(text)) for text in samples["text"]
    ]


def test_fully_cached_batch_bypasses_tokenizer():
    op = TokenNumFilter(hf_tokenizer="unused")
    samples = {
        "text": ["cached", "缓存", "🙂"],
        Fields.stats: [
            {StatsKeys.num_token: 7, "marker": 0},
            {StatsKeys.num_token: 8, "marker": 1},
            {StatsKeys.num_token: 9, "marker": 2},
        ],
    }

    with mock.patch.object(token_num_module, "get_model") as get_model_mock:
        returned = op.compute_stats_batched(samples)

    assert returned is samples
    get_model_mock.assert_not_called()
    assert samples[Fields.stats] == [
        {StatsKeys.num_token: 7, "marker": 0},
        {StatsKeys.num_token: 8, "marker": 1},
        {StatsKeys.num_token: 9, "marker": 2},
    ]


def test_later_tokenizer_failure_does_not_partially_update_stats():
    class FailingTokenizer:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, texts, **_kwargs):
            self.batch_sizes.append(len(texts))
            if "explode" in texts:
                raise RuntimeError("tokenizer failed on explode")
            return {"input_ids": [[1] for _text in texts]}

    tokenizer = FailingTokenizer()
    op = TokenNumFilter(hf_tokenizer="unused")
    samples = {
        "text": ["ok"] * 128 + ["explode"],
        Fields.stats: [{"marker": index} for index in range(129)],
    }
    expected_stats = [dict(stat) for stat in samples[Fields.stats]]

    with mock.patch.object(token_num_module, "get_model", return_value=tokenizer):
        with pytest.raises(RuntimeError, match="tokenizer failed on explode"):
            op.compute_stats_batched(samples)

    assert tokenizer.batch_sizes in ([129], [128, 1])
    assert samples[Fields.stats] == expected_stats


def test_releases_each_encoded_batch_before_tokenizing_the_next():
    released_batches = []

    class TrackedInputIds(list):
        def __init__(self, batch_index, values):
            super().__init__(values)
            self.batch_index = batch_index

        def __del__(self):
            released_batches.append(self.batch_index)

    class LifetimeCheckingTokenizer:
        def __init__(self):
            self.batch_sizes = []

        def __call__(self, texts, **_kwargs):
            if self.batch_sizes:
                assert released_batches == [len(self.batch_sizes) - 1]
            batch_index = len(self.batch_sizes)
            self.batch_sizes.append(len(texts))
            return {"input_ids": TrackedInputIds(batch_index, [[1] for _text in texts])}

    tokenizer = LifetimeCheckingTokenizer()
    op = TokenNumFilter(hf_tokenizer="unused")
    samples = {
        "text": [f"row-{index}" for index in range(129)],
        Fields.stats: [{} for _index in range(129)],
    }

    with mock.patch.object(token_num_module, "get_model", return_value=tokenizer):
        op.compute_stats_batched(samples)

    assert tokenizer.batch_sizes == [128, 1]
    assert released_batches == [0, 1]


def test_randomized_counts_match_eager_reference():
    rng = random.Random(20260806)
    tokenizer = RecordingTokenizer()
    op = TokenNumFilter(hf_tokenizer="unused")
    alphabet = ["alpha", "two words", "中文", "🙂emoji", "", "tabs\tand\nlines"]

    with mock.patch.object(token_num_module, "get_model", return_value=tokenizer):
        for _case in range(200):
            row_count = rng.randint(0, 350)
            texts = [f"{rng.choice(alphabet)}-{index}-{rng.randint(0, 9999)}" for index in range(row_count)]
            cached_indices = {index for index in range(row_count) if rng.random() < 0.25}
            stats = [
                (
                    {StatsKeys.num_token: 10_000 + index, "marker": index}
                    if index in cached_indices
                    else {"marker": index}
                )
                for index in range(row_count)
            ]
            samples = {"text": texts, Fields.stats: stats}
            calls_before = len(tokenizer.calls)

            returned = op.compute_stats_batched(samples)

            assert returned is samples
            expected_counts = [
                10_000 + index if index in cached_indices else len(_token_ids(text)) for index, text in enumerate(texts)
            ]
            assert [stat[StatsKeys.num_token] for stat in stats] == expected_counts
            assert [stat["marker"] for stat in stats] == list(range(row_count))
            new_calls = tokenizer.calls[calls_before:]
            assert all(1 <= len(batch_texts) <= 128 for batch_texts, _kwargs in new_calls)
            assert [text for batch_texts, _kwargs in new_calls for text in batch_texts] == [
                text for index, text in enumerate(texts) if index not in cached_indices
            ]
