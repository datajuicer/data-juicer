from contextlib import contextmanager

import pytest

from data_juicer.core.elasticjuicer.adaptive_mapper import OOMSafeAdaptiveMapper
from data_juicer.core.elasticjuicer.batch_controller import AdaptiveBatchController
from data_juicer.core.elasticjuicer.oom import is_oom_error


class FakeCudaOutOfMemoryError(RuntimeError):
    pass


class ThresholdListMapper:
    def __init__(self, safe_batch_size=8):
        self.safe_batch_size = safe_batch_size
        self.calls = []
        self.successful_inputs = []

    def __call__(self, batch):
        self.calls.append(list(batch))
        if len(batch) > self.safe_batch_size:
            raise FakeCudaOutOfMemoryError(f"batch {len(batch)} is too large")
        self.successful_inputs.extend(batch)
        return [value * 2 for value in batch]


class RecordingSampler:
    def __init__(self):
        self.batch_sizes = []

    @contextmanager
    def measure(self, batch_size):
        self.batch_sizes.append(batch_size)
        yield


def _fast_probe_controller(initial_batch_size=32):
    return AdaptiveBatchController(
        initial_batch_size=initial_batch_size,
        min_batch_size=1,
        max_batch_size=64,
        successes_before_growth=1,
        cooldown_successes=0,
    )


def test_adaptive_mapper_matches_fixed_safe_batch_without_loss_or_duplicates():
    values = list(range(100))
    mapper = ThresholdListMapper(safe_batch_size=8)
    sampler = RecordingSampler()
    cleanup_calls = []
    wrapper = OOMSafeAdaptiveMapper(
        mapper,
        controller=_fast_probe_controller(),
        sampler=sampler,
        oom_cleanup=lambda: cleanup_calls.append("cleanup"),
    )

    result = wrapper(values)
    baseline = [value * 2 for value in values]

    assert result == baseline
    assert mapper.successful_inputs == values
    assert len(mapper.successful_inputs) == len(set(mapper.successful_inputs)) == 100
    failed_calls = [call for call in mapper.calls if len(call) > 8]
    assert [len(call) for call in failed_calls] == [32, 16, 12, 10, 9]
    assert len({len(call) for call in failed_calls}) == len(failed_calls)
    assert [call[0] for call in mapper.calls[:3]] == [0, 0, 0]
    assert sampler.batch_sizes == [len(call) for call in mapper.calls]
    assert len(cleanup_calls) == len(failed_calls)
    assert wrapper.oom_retries == len(failed_calls)


def test_non_oom_exception_is_not_retried_or_reported_to_controller():
    calls = []
    controller = _fast_probe_controller(initial_batch_size=8)

    def broken_mapper(batch):
        calls.append(list(batch))
        raise ValueError("invalid schema")

    wrapper = OOMSafeAdaptiveMapper(broken_mapper, controller=controller)

    with pytest.raises(ValueError, match="invalid schema"):
        wrapper(list(range(20)))

    assert len(calls) == 1
    assert wrapper.oom_retries == 0
    assert controller.oom_events == 0
    assert controller.current_batch_size == 8


def test_retry_count_is_bounded_per_slice():
    calls = []

    def always_oom(batch):
        calls.append(list(batch))
        raise FakeCudaOutOfMemoryError("still too large")

    wrapper = OOMSafeAdaptiveMapper(
        always_oom,
        controller=_fast_probe_controller(),
        max_retries_per_slice=2,
    )

    with pytest.raises(FakeCudaOutOfMemoryError):
        wrapper(list(range(100)))

    assert [len(call) for call in calls] == [32, 16, 8]
    assert wrapper.oom_retries == 3


def test_oom_at_minimum_re_raises_original_error():
    error = FakeCudaOutOfMemoryError("minimum still fails")

    def always_oom(_batch):
        raise error

    wrapper = OOMSafeAdaptiveMapper(
        always_oom,
        controller=AdaptiveBatchController(initial_batch_size=1, min_batch_size=1, max_batch_size=8),
    )

    with pytest.raises(FakeCudaOutOfMemoryError) as raised:
        wrapper([1, 2])

    assert raised.value is error


def test_mapping_batches_preserve_order_and_schema():
    batch = {
        "id": list(range(25)),
        "text": [f"row-{index}" for index in range(25)],
        "constant": "metadata",
    }

    def mapping_mapper(microbatch):
        if len(microbatch["id"]) > 8:
            raise RuntimeError("CUDA out of memory")
        return {
            "id": microbatch["id"],
            "text": microbatch["text"],
            "double": [value * 2 for value in microbatch["id"]],
        }

    wrapper = OOMSafeAdaptiveMapper(mapping_mapper, controller=_fast_probe_controller())

    result = wrapper(batch)

    assert list(result) == ["id", "text", "double"]
    assert result["id"] == batch["id"]
    assert result["text"] == batch["text"]
    assert result["double"] == [value * 2 for value in batch["id"]]


def test_mapper_output_row_count_mismatch_fails_instead_of_losing_rows():
    def drops_last_row(batch):
        return batch[:-1]

    wrapper = OOMSafeAdaptiveMapper(drops_last_row, controller=_fast_probe_controller(initial_batch_size=8))

    with pytest.raises(ValueError, match="returned 7 rows for an 8-row input"):
        wrapper(list(range(10)))


def test_empty_batch_is_forwarded_once():
    calls = []

    def mapper(batch):
        calls.append(batch)
        return batch

    wrapper = OOMSafeAdaptiveMapper(mapper, controller=_fast_probe_controller())

    assert wrapper([]) == []
    assert calls == [[]]


def test_before_slice_hook_can_lower_cap_during_one_outer_batch():
    controller = AdaptiveBatchController(
        initial_batch_size=8,
        min_batch_size=1,
        max_batch_size=8,
        successes_before_growth=1,
        cooldown_successes=0,
    )
    calls = []
    boundaries = []

    def before_slice():
        boundaries.append(len(calls))
        if len(calls) == 1:
            controller.set_hard_limit(3)

    def mapper(batch):
        calls.append(list(batch))
        return batch

    wrapper = OOMSafeAdaptiveMapper(mapper, controller=controller, before_slice=before_slice)

    assert wrapper(list(range(20))) == list(range(20))
    assert [len(batch) for batch in calls] == [8, 3, 3, 3, 3]
    assert boundaries == [0, 1, 2, 3, 4]


def test_snapshot_callback_runs_after_oom_and_success_controller_transitions():
    class Measurement:
        def __init__(self, batch_size):
            self.snapshot = None
            self.batch_size = batch_size

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.snapshot = (self.batch_size, exc_type is None)
            return False

    class SnapshotSampler:
        def measure(self, batch_size):
            return Measurement(batch_size)

    controller = _fast_probe_controller(initial_batch_size=8)
    observed = []

    def mapper(batch):
        if len(batch) > 4:
            raise FakeCudaOutOfMemoryError("too large")
        return batch

    wrapper = OOMSafeAdaptiveMapper(
        mapper,
        controller=controller,
        sampler=SnapshotSampler(),
        snapshot_callback=lambda snapshot: observed.append((snapshot, controller.state)),
    )

    assert wrapper(list(range(8))) == list(range(8))
    assert observed[0][0] == (8, False)
    assert observed[0][1].oom_upper_bound == 8
    assert observed[0][1].current_batch_size == 4
    assert observed[1][0] == (4, True)
    assert observed[1][1].success_lower_bound == 4


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MemoryError("host allocation failed"), True),
        (FakeCudaOutOfMemoryError("allocator failed"), True),
        (RuntimeError("CUDA out of memory"), True),
        (RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"), True),
        (RuntimeError("CUDA illegal memory access"), False),
        (ValueError("out of memory"), False),
    ],
)
def test_oom_classification_is_narrow(error, expected):
    assert is_oom_error(error) is expected


def test_optional_pyarrow_table_preserves_schema_and_order():
    pyarrow = pytest.importorskip("pyarrow")
    table = pyarrow.table({"id": list(range(20)), "text": [f"row-{index}" for index in range(20)]})

    def arrow_mapper(microbatch):
        if microbatch.num_rows > 8:
            raise FakeCudaOutOfMemoryError("arrow batch too large")
        doubled = pyarrow.array([value.as_py() * 2 for value in microbatch["id"]])
        return microbatch.append_column("double", doubled)

    wrapper = OOMSafeAdaptiveMapper(arrow_mapper, controller=_fast_probe_controller())

    result = wrapper(table)

    assert result.num_rows == 20
    assert result.column_names == ["id", "text", "double"]
    assert result["id"].to_pylist() == list(range(20))
    assert result["double"].to_pylist() == [value * 2 for value in range(20)]
