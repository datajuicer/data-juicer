"""
Tests for FusedParallelMapper and mapper fusion logic.

Covers:
- _is_gpu_mapper / _are_ops_independent / fuse_mapper_group / fuse_consecutive_mappers
- fuse_operators() mapper_fusion parameter
- FusedParallelMapper: single-op fast path, multi-op parallel, profiling, cleanup
"""

import unittest

from data_juicer.ops.base_op import Mapper
from data_juicer.ops.fused_parallel_mapper import FusedParallelMapper
from data_juicer.ops.op_fusion import (
    _are_ops_independent,
    _is_fusible_gpu_mapper,
    _is_gpu_mapper,
    fuse_consecutive_mappers,
    fuse_mapper_group,
    fuse_operators,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


# ---------------------------------------------------------------------------
# Mock mappers (CPU-only, no model loading)
# ---------------------------------------------------------------------------
class _MockMapper(Mapper):
    """Base mock mapper that writes a value to Fields.meta."""

    _batched_op = True

    def __init__(self, name="mock_mapper", value="v", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name
        self._mock_value = value

    def process_batched(self, samples, rank=None, **kwargs):
        meta = samples.get(Fields.meta, [])
        for i in range(len(meta)):
            meta[i][self._name] = self._mock_value
        return samples


class _MockGPUMapper(_MockMapper):
    """Mock mapper that declares num_gpus > 0."""

    _accelerator = "cuda"
    _fused_parallel_mapper_safe = True

    def __init__(self, num_gpus=0.3, **kwargs):
        super().__init__(**kwargs)
        self.num_gpus = num_gpus
        self._input_columns = ["text"]
        self._output_columns = [f"{Fields.meta}.{self._name}"]
        # Disable auto parallelism so runtime_np() returns num_proc
        # directly, avoiding calculate_np() which requires a real GPU.
        self.auto_op_parallelism = False
        self.num_proc = kwargs.get("num_proc", 1)


class _MockCPUFilter:
    """Minimal non-Mapper stub to break GPU mapper groups."""

    _name = "mock_cpu_filter"
    num_gpus = 0


# ===========================================================================
# Tests for op_fusion.py helper functions
# ===========================================================================
class TestIsGpuMapper(DataJuicerTestCaseBase):

    def test_gpu_mapper(self):
        op = _MockGPUMapper(num_gpus=1.0, name="gpu_op")
        self.assertTrue(_is_gpu_mapper(op))

    def test_cpu_mapper(self):
        op = _MockMapper(name="cpu_op")
        self.assertFalse(_is_gpu_mapper(op))

    def test_non_mapper(self):
        self.assertFalse(_is_gpu_mapper(_MockCPUFilter()))

    def test_zero_gpus(self):
        op = _MockGPUMapper(num_gpus=0, name="zero_gpu")
        self.assertFalse(_is_gpu_mapper(op))


class TestIsFusibleGpuMapper(DataJuicerTestCaseBase):

    def test_gpu_mapper_with_opt_in(self):
        op = _MockGPUMapper(num_gpus=1.0, name="gpu_op")
        self.assertTrue(_is_fusible_gpu_mapper(op))

    def test_gpu_mapper_without_opt_in(self):
        op = _MockGPUMapper(num_gpus=1.0, name="gpu_op")
        op._fused_parallel_mapper_safe = False
        self.assertFalse(_is_fusible_gpu_mapper(op))


class TestAreOpsIndependent(DataJuicerTestCaseBase):

    def test_independent_no_attrs(self):
        """Ops without declared output columns are not assumed independent."""
        ops = [_MockMapper(name="a"), _MockMapper(name="b")]
        self.assertFalse(_are_ops_independent(ops))

    def test_dependent_via_column_overlap(self):
        """Op B reads a column produced by Op A → dependent."""
        a = _MockMapper(name="a")
        a._output_columns = ["col_x"]
        b = _MockMapper(name="b")
        b._input_columns = ["col_x"]
        self.assertFalse(_are_ops_independent([a, b]))

    def test_independent_disjoint_columns(self):
        """Ops with disjoint _input/_output_columns are independent."""
        a = _MockMapper(name="a")
        a._output_columns = ["col_a"]
        b = _MockMapper(name="b")
        b._input_columns = ["col_b"]
        b._output_columns = ["col_b_out"]
        self.assertTrue(_are_ops_independent([a, b]))

    def test_dependent_via_write_overlap(self):
        """Two ops writing the same column are dependent."""
        a = _MockMapper(name="a")
        a._output_columns = ["col_x"]
        b = _MockMapper(name="b")
        b._output_columns = ["col_x"]
        self.assertFalse(_are_ops_independent([a, b]))


class TestFuseMapperGroup(DataJuicerTestCaseBase):

    def test_empty_group(self):
        self.assertEqual(fuse_mapper_group([]), [])

    def test_single_op_still_fuses(self):
        """fuse_mapper_group fuses any group size; single-op filtering
        is done by fuse_consecutive_mappers before calling this function."""
        op = _MockGPUMapper(name="single")
        result = fuse_mapper_group([op])
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], FusedParallelMapper)

    def test_vram_exceeds_limit(self):
        ops = [_MockGPUMapper(num_gpus=0.6, name=f"op{i}") for i in range(2)]
        result = fuse_mapper_group(ops, vram_limit=0.9)
        # 0.6 + 0.6 = 1.2 > 0.9 → not fused
        self.assertEqual(len(result), 2)

    def test_mapper_without_opt_in_is_not_fused(self):
        ops = [_MockGPUMapper(num_gpus=0.2, name=f"op{i}") for i in range(2)]
        ops[0]._fused_parallel_mapper_safe = False
        result = fuse_mapper_group(ops, vram_limit=0.9)
        self.assertEqual(result, ops)

    def test_successful_fusion(self):
        ops = [_MockGPUMapper(num_gpus=0.3, name=f"op{i}") for i in range(3)]
        for op in ops:
            op._op_cfg = {op._name: {"num_gpus": op.num_gpus}}
        result = fuse_mapper_group(ops, vram_limit=0.9)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], FusedParallelMapper)
        self.assertEqual(result[0].num_gpus, 1.0)
        self.assertEqual(
            result[0]._op_cfg,
            {"fused:op0,op1,op2": [{op._name: {"num_gpus": op.num_gpus}} for op in ops]},
        )

    def test_batch_size_uses_min(self):
        """Verify the fix: batch_size should use min, not max."""
        ops = [
            _MockGPUMapper(num_gpus=0.2, name="heavy"),
            _MockGPUMapper(num_gpus=0.2, name="light"),
        ]
        ops[0].batch_size = 4  # heavy mapper, small batch to avoid OOM
        ops[1].batch_size = 128  # light mapper
        result = fuse_mapper_group(ops, vram_limit=0.9)
        self.assertEqual(len(result), 1)
        # min(4, 128) = 4, not max = 128
        self.assertEqual(result[0].batch_size, 4)

    def test_num_proc_uses_runtime_np(self):
        """Verify the fix: num_proc uses runtime_np(), not raw num_proc."""
        ops = [
            _MockGPUMapper(num_gpus=0.2, name="op_a"),
            _MockGPUMapper(num_gpus=0.2, name="op_b"),
        ]
        ops[0].num_proc = 2
        ops[1].num_proc = 8
        result = fuse_mapper_group(ops, vram_limit=0.9)
        self.assertEqual(len(result), 1)
        # runtime_np() returns num_proc when auto_op_parallelism is False
        self.assertEqual(result[0].num_proc, min(ops[0].runtime_np(), ops[1].runtime_np()))


class TestFuseConsecutiveMappers(DataJuicerTestCaseBase):

    def test_all_gpu_mappers_fused(self):
        ops = [_MockGPUMapper(num_gpus=0.2, name=f"g{i}") for i in range(3)]
        result = fuse_consecutive_mappers(ops, vram_limit=0.9)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], FusedParallelMapper)

    def test_non_mapper_breaks_group(self):
        ops = [
            _MockGPUMapper(num_gpus=0.2, name="g0"),
            _MockGPUMapper(num_gpus=0.2, name="g1"),
            _MockCPUFilter(),
            _MockGPUMapper(num_gpus=0.2, name="g2"),
            _MockGPUMapper(num_gpus=0.2, name="g3"),
        ]
        result = fuse_consecutive_mappers(ops, vram_limit=0.9)
        # [FusedParallelMapper(g0,g1), MockCPUFilter, FusedParallelMapper(g2,g3)]
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], FusedParallelMapper)
        self.assertIsInstance(result[1], _MockCPUFilter)
        self.assertIsInstance(result[2], FusedParallelMapper)

    def test_single_gpu_mapper_passthrough(self):
        ops = [_MockCPUFilter(), _MockGPUMapper(num_gpus=0.2, name="lone")]
        result = fuse_consecutive_mappers(ops, vram_limit=0.9)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[1], _MockMapper)

    def test_unsafe_gpu_mapper_breaks_group(self):
        ops = [
            _MockGPUMapper(num_gpus=0.2, name="g0"),
            _MockGPUMapper(num_gpus=0.2, name="unsafe"),
            _MockGPUMapper(num_gpus=0.2, name="g1"),
            _MockGPUMapper(num_gpus=0.2, name="g2"),
        ]
        ops[1]._fused_parallel_mapper_safe = False
        result = fuse_consecutive_mappers(ops, vram_limit=0.9)
        self.assertEqual(len(result), 3)
        self.assertIs(result[0], ops[0])
        self.assertIs(result[1], ops[1])
        self.assertIsInstance(result[2], FusedParallelMapper)

    def test_cpu_mappers_not_fused(self):
        ops = [_MockMapper(name=f"c{i}") for i in range(3)]
        result = fuse_consecutive_mappers(ops, vram_limit=0.9)
        self.assertEqual(len(result), 3)


class TestFuseOperatorsWithMapperFusion(DataJuicerTestCaseBase):

    def test_mapper_fusion_disabled(self):
        """mapper_fusion=False should skip mapper fusion phase."""
        ops = [_MockGPUMapper(num_gpus=0.2, name=f"m{i}") for i in range(2)]
        result = fuse_operators(ops, mapper_fusion=False)
        # No fusion → ops pass through as-is
        self.assertEqual(len(result), 2)

    def test_mapper_fusion_enabled(self):
        """mapper_fusion=True should fuse consecutive GPU mappers."""
        ops = [_MockGPUMapper(num_gpus=0.2, name=f"m{i}") for i in range(2)]
        result = fuse_operators(ops, mapper_fusion=True, mapper_fusion_vram_limit=0.9)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], FusedParallelMapper)


# ===========================================================================
# Tests for FusedParallelMapper execution
# ===========================================================================
class TestFusedParallelMapperExecution(DataJuicerTestCaseBase):

    def _make_samples(self, n=4):
        return {
            "text": [f"sample_{i}" for i in range(n)],
            Fields.meta: [{} for _ in range(n)],
        }

    def _make_fused(self, ops, **kwargs):
        """Create a FusedParallelMapper with pre-built ops (CPU mode)."""
        fused = FusedParallelMapper(fused_ops=ops, **kwargs)
        # Override accelerator to cpu so tests run without GPU
        fused.accelerator = "cpu"
        return fused

    def test_single_op_fast_path(self):
        """Single op should execute without thread overhead."""
        op = _MockMapper(name="solo", value="done")
        fused = self._make_fused([op])

        samples = self._make_samples()
        result = fused.process_batched(samples)

        for meta in result[Fields.meta]:
            self.assertEqual(meta["solo"], "done")

    def test_multi_op_parallel_execution(self):
        """Multiple ops should all execute and write disjoint meta keys."""
        ops = [_MockMapper(name=f"op{i}", value=f"v{i}") for i in range(3)]
        fused = self._make_fused(ops)

        samples = self._make_samples()
        result = fused.process_batched(samples)

        for meta in result[Fields.meta]:
            self.assertEqual(meta["op0"], "v0")
            self.assertEqual(meta["op1"], "v1")
            self.assertEqual(meta["op2"], "v2")

    def test_profiling_stats_collected(self):
        """Profiling state should be updated after process_batched."""
        ops = [_MockMapper(name=f"p{i}", value="x") for i in range(2)]
        fused = self._make_fused(ops)
        # Force init
        fused._ensure_ops()

        samples = self._make_samples()
        fused.process_batched(samples)

        self.assertEqual(fused._prof_batch_count, 1)
        self.assertEqual(fused._prof_total_rows, 4)
        # Both ops should have timing entries
        self.assertIn("p0", fused._prof_op_wall_ms)
        self.assertIn("p1", fused._prof_op_wall_ms)

    def test_cleanup_columns(self):
        """cleanup_columns should remove specified keys from output."""
        ops = [
            _MockMapper(name="cleaner", value="ok"),
            _MockMapper(name="cleaner2", value="ok2"),
        ]
        fused = self._make_fused(ops, cleanup_columns=["temp_col"])

        samples = self._make_samples()
        samples["temp_col"] = [1, 2, 3, 4]
        result = fused.process_batched(samples)

        self.assertNotIn("temp_col", result)

    def test_empty_batch(self):
        """Empty batch should return immediately."""
        op = _MockMapper(name="empty", value="x")
        fused = self._make_fused([op])

        samples = {"text": [], Fields.meta: []}
        result = fused.process_batched(samples)
        self.assertEqual(result["text"], [])

    def test_no_ops(self):
        """No ops should return samples unchanged."""
        fused = self._make_fused([])
        samples = self._make_samples()
        result = fused.process_batched(samples)
        self.assertEqual(result["text"], ["sample_0", "sample_1", "sample_2", "sample_3"])

    def test_dual_init_modes_error(self):
        """Providing both op_specs and fused_ops should raise ValueError."""
        with self.assertRaises(ValueError):
            FusedParallelMapper(
                op_specs=[{"class_name": "fix_unicode_mapper"}],
                fused_ops=[_MockMapper(name="m")],
            )

    def test_sub_op_returning_new_batch_raises(self):
        """Sub-ops must mutate the shared batch in place."""

        class _ReturnNewBatchMapper(_MockMapper):
            def process_batched(self, samples, rank=None, **kwargs):
                return {key: list(value) for key, value in samples.items()}

        fused = self._make_fused([_ReturnNewBatchMapper(name="copying")])
        with self.assertRaisesRegex(ValueError, "returned a different batch object"):
            fused.process_batched(self._make_samples())


if __name__ == "__main__":
    unittest.main(verbosity=2)
