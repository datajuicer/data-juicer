from contextlib import contextmanager

import pytest

from data_juicer.core.elasticjuicer.adaptive_mapper import OOMSafeAdaptiveMapper
from data_juicer.core.elasticjuicer.batch_controller import AdaptiveBatchController
from data_juicer.core.elasticjuicer.oom import is_oom_error


class FakeCudaOutOfMemoryError(RuntimeError):
    pass


def test_failed_mutation_is_not_replayed_into_a_retry():
    batch = {"id": [0, 1, 2, 3], "stats": [{"count": 0} for _ in range(4)]}

    def mapper(part):
        for stats in part["stats"]:
            stats["count"] += 1
        if len(part["id"]) > 2:
            raise FakeCudaOutOfMemoryError("CUDA out of memory")
        return part

    wrapper = OOMSafeAdaptiveMapper(mapper, AdaptiveBatchController(4, max_batch_size=4))
    result = wrapper(batch)
    assert result["id"] == list(range(4))
    assert result["stats"] == [{"count": 1}] * 4
    assert batch["stats"] == [{"count": 0}] * 4


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
        max_floor_retries=0,
    )

    with pytest.raises(FakeCudaOutOfMemoryError) as raised:
        wrapper([1, 2])

    assert raised.value is error


def test_floor_oom_retries_transient_starvation_before_failing():
    # A co-located sibling holding a huge slice can starve this actor's
    # minimum-size batch for a few seconds (the W2' P4 entry transient:
    # one actor's 91.7 GiB slice vs a 5.6 GiB bs=1 request).  The floor
    # must wait it out with bounded same-slice retries instead of
    # declaring the slice fatal on the first try.

    class TransientFloorMapper:
        def __init__(self, failures_left):
            self.failures_left = failures_left
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            if self.failures_left > 0:
                self.failures_left -= 1
                raise FakeCudaOutOfMemoryError("sibling still holds the device")
            return list(batch)

    mapper = TransientFloorMapper(failures_left=2)
    wrapper = OOMSafeAdaptiveMapper(
        mapper,
        controller=AdaptiveBatchController(initial_batch_size=1, min_batch_size=1, max_batch_size=8),
        max_floor_retries=3,
        floor_retry_backoff_sec=0.0,
    )

    result = wrapper([1, 2, 3])

    assert result == [1, 2, 3]
    # slice [1]: attempt + 2 bounded wait-retries that survive; slices [2] and
    # [3] then succeed on their first attempt
    assert mapper.calls == 5


def test_floor_oom_still_propagates_after_bounded_retries():
    # The wait budget is bounded: permanent starvation still propagates
    # the original error explicitly (no silent skip, no infinite loop).

    class AlwaysFloorMapper:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            raise FakeCudaOutOfMemoryError("permanent starvation")

    wrapper = OOMSafeAdaptiveMapper(
        AlwaysFloorMapper(),
        controller=AdaptiveBatchController(initial_batch_size=1, min_batch_size=1, max_batch_size=8),
        max_floor_retries=2,
        floor_retry_backoff_sec=0.0,
    )

    with pytest.raises(FakeCudaOutOfMemoryError):
        wrapper([1, 2])

    # one first attempt plus the two bounded retries, then fatal
    assert wrapper.mapper.calls == 3


def test_recovered_floor_reprobes_after_stable_successes_without_captain():
    calls = []
    controller = AdaptiveBatchController(
        initial_batch_size=1,
        max_batch_size=8,
        recovery_requires_hint=True,
        oom_reprobe_successes=4,
        max_oom_reprobes=1,
        successes_before_growth=1,
    )

    def transient_mapper(batch):
        calls.append(list(batch))
        if len(calls) <= 2:
            raise FakeCudaOutOfMemoryError("transient contention")
        return list(batch)

    wrapper = OOMSafeAdaptiveMapper(transient_mapper, controller, floor_retry_backoff_sec=0)
    values = list(range(100))
    assert wrapper(values) == values
    assert calls[:3] == [[0], [0], [0]]
    assert all(len(batch) == 1 for batch in calls[:6])
    assert any(len(batch) > 1 for batch in calls[6:])
    assert controller.state.floor_recovery_events == 1


def test_floor_recovery_does_not_keep_reprobing_a_fresh_larger_oom():
    calls = []
    controller = AdaptiveBatchController(
        initial_batch_size=1,
        max_batch_size=8,
        recovery_requires_hint=True,
        oom_reprobe_successes=4,
        max_oom_reprobes=1,
        successes_before_growth=1,
    )

    def one_row_only(batch):
        calls.append(list(batch))
        if len(calls) == 1 or len(batch) > 1:
            raise FakeCudaOutOfMemoryError("only one row fits")
        return list(batch)

    wrapper = OOMSafeAdaptiveMapper(one_row_only, controller, floor_retry_backoff_sec=0)
    values = list(range(128))
    assert wrapper(values) == values
    assert sum(len(batch) > 1 for batch in calls) == 1
    assert controller.oom_upper_bound == 2
    assert controller.state.oom_reprobe_events == 1
    assert not controller.state.floor_recovery_pending


def test_invalid_floor_retry_output_cannot_authorize_recovery():
    calls = []
    controller = AdaptiveBatchController(initial_batch_size=1, max_batch_size=8, max_oom_reprobes=1)

    def invalid_retry(batch):
        calls.append(list(batch))
        if len(calls) == 1:
            raise FakeCudaOutOfMemoryError("transient contention")
        return []

    wrapper = OOMSafeAdaptiveMapper(invalid_retry, controller, floor_retry_backoff_sec=0)
    with pytest.raises(ValueError, match="returned 0 rows"):
        wrapper([1, 2])
    assert controller.oom_upper_bound == 1
    assert controller.state.floor_recovery_events == 0


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
