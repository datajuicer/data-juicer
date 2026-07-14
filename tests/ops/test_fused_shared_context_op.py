import unittest

from data_juicer.ops.base_op import (
    NON_STATS_FILTERS,
    OPERATORS,
    TAGGING_OPS,
    Filter,
    Mapper,
)
from data_juicer.ops.filter.average_line_length_filter import AverageLineLengthFilter
from data_juicer.ops.filter.maximum_line_length_filter import MaximumLineLengthFilter
from data_juicer.ops.filter.suffix_filter import SuffixFilter
from data_juicer.ops.fused_shared_context_op import FusedSharedContextOp
from data_juicer.ops.load import load_ops
from data_juicer.utils.constant import Fields, InterVars, StatsKeys


class _SharedLinesContextFilter(Filter):
    _batched_op = True
    compute_count = 0

    def __init__(
        self,
        stat_key="line_count",
        min_value=0,
        context_key=InterVars.lines,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stat_key = stat_key
        self.min_value = min_value
        self.context_key = context_key
        self.auto_op_parallelism = False
        self.num_proc = 1

    @classmethod
    def reset_count(cls):
        cls.compute_count = 0

    def compute_stats_batched(self, samples, context=False):
        for idx, stat in enumerate(samples[Fields.stats]):
            if context and self.context_key in samples[Fields.context][idx]:
                lines = samples[Fields.context][idx][self.context_key]
            else:
                type(self).compute_count += 1
                lines = samples["text"][idx].splitlines()
                if context:
                    samples[Fields.context][idx][self.context_key] = lines
            stat[self.stat_key] = len(lines)
        return samples

    def process_batched(self, samples):
        return [stat[self.stat_key] >= self.min_value for stat in samples[Fields.stats]]


class _ContextAwareMapper(Mapper):
    _batched_op = True
    _requires_meta = True

    def __init__(self, context_key=InterVars.lines, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_key = context_key
        self.context_flags = []
        self.auto_op_parallelism = False
        self.num_proc = 1

    def process_batched(self, samples, context=False):
        self.context_flags.append(context)
        for idx, meta in enumerate(samples[Fields.meta]):
            meta["context_enabled"] = context
            meta["context_available"] = self.context_key in samples[Fields.context][idx]
        samples["text"] = [f"{text}|mapped" for text in samples["text"]]
        return samples


class _CountingText(str):
    splitlines_count = 0

    @classmethod
    def reset_count(cls):
        cls.splitlines_count = 0

    def splitlines(self, *args, **kwargs):
        type(self).splitlines_count += 1
        return super().splitlines(*args, **kwargs)


class _TaggingNonStatsFilter(Filter):
    def compute_stats_single(self, sample, context=False):
        sample[Fields.meta]["tagged"] = context
        return sample

    def process_single(self, sample):
        return sample[Fields.meta]["tagged"]


class _RegisteredTaggingNonStatsFilter(_TaggingNonStatsFilter):
    pass


class _ContainerContextMapper(Mapper):
    _batched_op = True

    def __init__(self, containers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.containers = containers

    def process_batched(self, samples, context=False):
        if context:
            for sample_context, container in zip(samples[Fields.context], self.containers):
                sample_context[InterVars.loaded_videos] = {"nested": [container]}
        return samples


class _ReturnWithoutContextMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples, context=False):
        return {"text": [f"{text}|copied" for text in samples["text"]]}


class _RaisingMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples, context=False):
        raise RuntimeError("expected test error")


class _FakeVideoStream:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class _FakeStreams:
    def __init__(self, video_stream):
        self.video = [video_stream]


def _fake_container_init(self):
    self.video_stream = _FakeVideoStream()
    self.streams = _FakeStreams(self.video_stream)
    self.close_count = 0


def _fake_container_close(self):
    self.close_count += 1


FakeInputContainer = type(
    "InputContainer",
    (),
    {
        "__module__": "av.container.core",
        "__init__": _fake_container_init,
        "close": _fake_container_close,
    },
)


class _RegisteredSharedLinesContextFilter(_SharedLinesContextFilter):
    pass


class _RegisteredContextAwareMapper(_ContextAwareMapper):
    pass


OPERATORS.register_module(
    "test_shared_lines_context_filter",
    _RegisteredSharedLinesContextFilter,
    force=True,
)
OPERATORS.register_module(
    "test_context_aware_mapper",
    _RegisteredContextAwareMapper,
    force=True,
)
OPERATORS.register_module(
    "test_tagging_non_stats_filter",
    _RegisteredTaggingNonStatsFilter,
    force=True,
)
TAGGING_OPS.register_module(
    "test_tagging_non_stats_filter",
    _RegisteredTaggingNonStatsFilter,
    force=True,
)
NON_STATS_FILTERS.register_module(
    "test_tagging_non_stats_filter",
    _RegisteredTaggingNonStatsFilter,
    force=True,
)


class TestFusedSharedContextOp(unittest.TestCase):

    def test_real_line_filters_reuse_context(self):
        _CountingText.reset_count()
        fused = FusedSharedContextOp(
            fused_ops=[
                AverageLineLengthFilter(min_len=0),
                MaximumLineLengthFilter(min_len=0),
            ],
            accelerator="cpu",
        )

        result = fused.process_batched(
            {
                "text": [
                    _CountingText("one\ntwo"),
                    _CountingText("one\ntwo\nthree"),
                ]
            }
        )

        self.assertEqual(_CountingText.splitlines_count, 2)
        self.assertEqual(
            [stat[StatsKeys.max_line_length] for stat in result[Fields.stats]],
            [3, 5],
        )
        self.assertNotIn(Fields.context, result)

    def test_shared_context_computed_once_across_filters(self):
        _SharedLinesContextFilter.reset_count()
        fused = FusedSharedContextOp(
            fused_ops=[
                _SharedLinesContextFilter(stat_key="lines_a"),
                _SharedLinesContextFilter(stat_key="lines_b"),
            ],
            accelerator="cpu",
        )

        result = fused.process_batched({"text": ["one\ntwo", "one\ntwo\nthree"]})

        self.assertEqual(_SharedLinesContextFilter.compute_count, 2)
        self.assertEqual([stat["lines_a"] for stat in result[Fields.stats]], [2, 3])
        self.assertEqual([stat["lines_b"] for stat in result[Fields.stats]], [2, 3])
        self.assertNotIn(Fields.context, result)

    def test_filter_drop_rows_and_downstream_mapper(self):
        _SharedLinesContextFilter.reset_count()
        mapper = _ContextAwareMapper()
        fused = FusedSharedContextOp(
            fused_ops=[
                _SharedLinesContextFilter(stat_key="line_count", min_value=2),
                mapper,
            ],
            accelerator="cpu",
        )

        result = fused.process_batched({"text": ["one", "one\ntwo", "one\ntwo\nthree"]})

        self.assertEqual(result["text"], ["one\ntwo|mapped", "one\ntwo\nthree|mapped"])
        self.assertEqual([stat["line_count"] for stat in result[Fields.stats]], [2, 3])
        self.assertEqual([meta["context_enabled"] for meta in result[Fields.meta]], [True, True])
        self.assertEqual([meta["context_available"] for meta in result[Fields.meta]], [True, True])
        self.assertEqual(mapper.context_flags, [True])
        self.assertNotIn(Fields.context, result)

    def test_non_batched_non_stats_filter(self):
        fused = FusedSharedContextOp(
            fused_ops=[SuffixFilter(suffixes=[".txt"])],
            accelerator="cpu",
        )

        result = fused.process_batched(
            {
                "text": ["keep", "drop"],
                Fields.suffix: [".txt", ".json"],
            }
        )

        self.assertEqual(result["text"], ["keep"])
        self.assertEqual(result[Fields.suffix], [".txt"])
        self.assertNotIn(Fields.stats, result)
        self.assertNotIn(Fields.context, result)

    def test_tagging_filter_initializes_meta(self):
        tagging_filter = _RegisteredTaggingNonStatsFilter()
        fused = FusedSharedContextOp(
            fused_ops=[tagging_filter],
            accelerator="cpu",
        )

        result = fused.process_batched({"text": ["a", "b"]})

        self.assertEqual(result[Fields.meta], [{"tagged": True}, {"tagged": True}])
        self.assertNotIn(Fields.stats, result)

    def test_mapper_result_without_context_keeps_owned_context(self):
        mapper = _ContextAwareMapper()
        fused = FusedSharedContextOp(
            fused_ops=[
                _SharedLinesContextFilter(stat_key="line_count"),
                _ReturnWithoutContextMapper(),
                mapper,
            ],
            accelerator="cpu",
        )

        result = fused.process_batched({"text": ["one\ntwo"]})

        self.assertEqual(result["text"], ["one\ntwo|copied|mapped"])
        self.assertEqual(result[Fields.meta][0]["context_available"], True)
        self.assertNotIn(Fields.context, result)

    def test_context_resources_closed_once_after_row_drop(self):
        shared_container = FakeInputContainer()
        fused = FusedSharedContextOp(
            fused_ops=[
                _ContainerContextMapper([shared_container, shared_container]),
                _SharedLinesContextFilter(stat_key="line_count", min_value=2),
            ],
            accelerator="cpu",
        )

        result = fused.process_batched({"text": ["one", "one\ntwo"]})

        self.assertEqual(result["text"], ["one\ntwo"])
        self.assertEqual(shared_container.video_stream.close_count, 1)
        self.assertEqual(shared_container.close_count, 1)

    def test_context_resources_closed_on_error(self):
        containers = [FakeInputContainer(), FakeInputContainer()]
        fused = FusedSharedContextOp(
            fused_ops=[
                _ContainerContextMapper(containers),
                _RaisingMapper(),
            ],
            accelerator="cpu",
        )
        samples = {"text": ["a", "b"]}

        with self.assertRaisesRegex(RuntimeError, "expected test error"):
            fused.process_batched(samples)

        self.assertNotIn(Fields.context, samples)
        for container in containers:
            self.assertEqual(container.video_stream.close_count, 1)
            self.assertEqual(container.close_count, 1)

    def test_rejects_misaligned_batch_columns(self):
        fused = FusedSharedContextOp(
            fused_ops=[_ContextAwareMapper()],
            accelerator="cpu",
        )

        with self.assertRaisesRegex(ValueError, "does not match batch size"):
            fused.process_batched({"text": ["a", "b"], "other": [1]})

    def test_rejects_preexisting_context_column(self):
        fused = FusedSharedContextOp(
            fused_ops=[_ContextAwareMapper()],
            accelerator="cpu",
        )

        with self.assertRaisesRegex(ValueError, "owns Fields.context"):
            fused.process_batched({"text": ["a"], Fields.context: [{}]})

    def test_batch_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            FusedSharedContextOp(batch_size=0, fused_ops=[], accelerator="cpu")

    def test_load_ops_constructs_fused_shared_context_op(self):
        ops = load_ops(
            [
                {
                    "fused_shared_context_op": {
                        "batch_size": 2,
                        "fused_op_list": [
                            {"test_shared_lines_context_filter": {"stat_key": "lines_a"}},
                            {"test_context_aware_mapper": {}},
                        ],
                    }
                }
            ]
        )

        self.assertEqual(len(ops), 1)
        self.assertIsInstance(ops[0], FusedSharedContextOp)

        result = ops[0].process_batched({"text": ["abc"]})
        self.assertEqual(result["text"], ["abc|mapped"])
        self.assertEqual(result[Fields.stats][0]["lines_a"], 1)
        self.assertTrue(result[Fields.meta][0]["context_available"])

    def test_empty_fused_op_returns_samples_unchanged(self):
        fused = FusedSharedContextOp(fused_ops=[], accelerator="cpu")
        samples = {"text": ["abc"]}

        result = fused.process_batched(samples)

        self.assertIs(result, samples)
        self.assertNotIn(Fields.context, result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
