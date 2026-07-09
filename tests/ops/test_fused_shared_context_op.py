import unittest

from data_juicer.ops.base_op import OPERATORS, Filter, Mapper
from data_juicer.ops.fused_shared_context_op import FusedSharedContextOp
from data_juicer.ops.load import load_ops
from data_juicer.utils.constant import Fields, InterVars
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


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


class TestFusedSharedContextOp(DataJuicerTestCaseBase):

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
