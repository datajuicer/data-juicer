import pytest

from data_juicer.core.elasticjuicer.batch_controller import (
    AdaptiveBatchController,
    MinimumBatchSizeOOM,
)


def test_controller_validates_configuration():
    with pytest.raises(ValueError, match="min_batch_size"):
        AdaptiveBatchController(initial_batch_size=1, min_batch_size=0)
    with pytest.raises(ValueError, match="max_batch_size"):
        AdaptiveBatchController(initial_batch_size=4, min_batch_size=8, max_batch_size=4)
    with pytest.raises(ValueError, match="initial_batch_size"):
        AdaptiveBatchController(initial_batch_size=16, min_batch_size=1, max_batch_size=8)
    with pytest.raises(ValueError, match="successes_before_growth"):
        AdaptiveBatchController(initial_batch_size=4, successes_before_growth=0)
    with pytest.raises(ValueError, match="growth_step"):
        AdaptiveBatchController(initial_batch_size=4, growth_step=0)
    with pytest.raises(ValueError, match="max_probe_ooms"):
        AdaptiveBatchController(initial_batch_size=4, max_probe_ooms=0)
    with pytest.raises(ValueError, match="minimum_growth_fraction"):
        AdaptiveBatchController(initial_batch_size=4, minimum_growth_fraction=0)


def test_next_batch_size_respects_remaining_samples_and_hard_limit():
    controller = AdaptiveBatchController(initial_batch_size=16, max_batch_size=64)

    assert controller.next_batch_size(remaining_samples=100) == 16
    assert controller.next_batch_size(remaining_samples=7) == 7
    assert controller.next_batch_size(remaining_samples=0) == 0

    controller.set_hard_limit(6)
    assert controller.current_batch_size == 6
    assert controller.next_batch_size(remaining_samples=100) == 6


def test_hard_limit_must_stay_inside_static_bounds():
    controller = AdaptiveBatchController(initial_batch_size=4, min_batch_size=2, max_batch_size=16)

    with pytest.raises(ValueError, match="hard_limit"):
        controller.set_hard_limit(1)
    with pytest.raises(ValueError, match="hard_limit"):
        controller.set_hard_limit(17)


def test_success_requires_streak_before_small_step_growth():
    controller = AdaptiveBatchController(
        initial_batch_size=4,
        max_batch_size=16,
        successes_before_growth=3,
        growth_step=2,
    )

    controller.observe_success(4)
    controller.observe_success(4)
    assert controller.current_batch_size == 4

    controller.observe_success(4)
    assert controller.current_batch_size == 6
    assert controller.success_lower_bound == 4


def test_final_partial_batch_does_not_count_toward_growth():
    controller = AdaptiveBatchController(
        initial_batch_size=16,
        max_batch_size=32,
        successes_before_growth=2,
        growth_step=4,
    )

    controller.observe_success(7)
    controller.observe_success(7)

    assert controller.current_batch_size == 16
    assert controller.success_lower_bound is None
    assert controller.consecutive_successes == 0


def test_oom_creates_exclusive_upper_bound_and_enters_cooldown():
    controller = AdaptiveBatchController(
        initial_batch_size=32,
        min_batch_size=1,
        max_batch_size=64,
        successes_before_growth=1,
        cooldown_successes=2,
    )

    controller.observe_oom(32)

    assert controller.oom_upper_bound == 32
    assert controller.current_batch_size == 16
    assert controller.next_batch_size(100) < 32

    controller.observe_success(16)
    controller.observe_success(16)
    assert controller.current_batch_size == 16
    controller.observe_success(16)
    assert 16 < controller.current_batch_size < 32


def test_known_success_is_not_discarded_by_later_probe_oom():
    controller = AdaptiveBatchController(
        initial_batch_size=16,
        min_batch_size=1,
        max_batch_size=32,
        successes_before_growth=1,
        cooldown_successes=0,
    )

    controller.observe_oom(16)
    assert controller.current_batch_size == 8
    controller.observe_success(8)
    assert controller.current_batch_size == 12

    controller.observe_oom(12)

    assert controller.success_lower_bound == 8
    assert controller.oom_upper_bound == 12
    assert controller.current_batch_size == 8


def test_synthetic_trace_converges_without_retrying_failed_batch_sizes():
    controller = AdaptiveBatchController(
        initial_batch_size=32,
        min_batch_size=1,
        max_batch_size=64,
        successes_before_growth=1,
        cooldown_successes=0,
    )
    failed_batch_sizes = set()

    for _ in range(20):
        batch_size = controller.next_batch_size(remaining_samples=100)
        if batch_size > 8:
            assert batch_size not in failed_batch_sizes
            failed_batch_sizes.add(batch_size)
            controller.observe_oom(batch_size)
        else:
            controller.observe_success(batch_size)

    assert failed_batch_sizes == {9, 10, 12, 16, 32}
    assert controller.success_lower_bound == 8
    assert controller.oom_upper_bound == 9
    assert controller.current_batch_size == 8


def test_oom_at_minimum_batch_size_is_terminal():
    controller = AdaptiveBatchController(initial_batch_size=1, min_batch_size=1, max_batch_size=8)

    with pytest.raises(MinimumBatchSizeOOM, match="minimum batch size 1"):
        controller.observe_oom(1)


def test_smaller_oom_invalidates_a_stale_success_lower_bound():
    controller = AdaptiveBatchController(
        initial_batch_size=8,
        min_batch_size=1,
        max_batch_size=16,
        successes_before_growth=1,
        cooldown_successes=0,
    )
    controller.observe_success(8)

    controller.observe_oom(8)

    assert controller.success_lower_bound is None
    assert controller.oom_upper_bound == 8
    assert controller.current_batch_size == 4


def test_equal_event_traces_produce_equal_state():
    left = AdaptiveBatchController(initial_batch_size=32, max_batch_size=64)
    right = AdaptiveBatchController(initial_batch_size=32, max_batch_size=64)

    for event, batch_size in [("oom", 32), ("success", 16), ("success", 16), ("success", 16)]:
        getattr(left, f"observe_{event}")(batch_size)
        getattr(right, f"observe_{event}")(batch_size)

    assert left.state == right.state


def test_probe_oom_budget_freezes_at_best_proven_size():
    controller = AdaptiveBatchController(
        initial_batch_size=8,
        max_batch_size=64,
        successes_before_growth=1,
        growth_step=16,
        cooldown_successes=0,
        max_probe_ooms=2,
        max_oom_reprobes=0,
    )

    for _ in range(20):
        requested = controller.next_batch_size(100)
        if requested <= 10:
            controller.observe_success(requested)
        else:
            controller.observe_oom(requested)

    assert controller.probe_oom_events == 2
    assert controller.current_batch_size == controller.success_lower_bound
    assert controller.oom_upper_bound == controller.success_lower_bound + 1


def test_capacity_recovery_reopens_bound_after_stable_successes():
    controller = AdaptiveBatchController(
        initial_batch_size=16,
        max_batch_size=32,
        successes_before_growth=1,
        growth_step=4,
        cooldown_successes=0,
        oom_reprobe_successes=4,
        max_oom_reprobes=3,
    )

    for _ in range(8):
        requested = controller.next_batch_size(100)
        if requested <= 8:
            controller.observe_success(requested)
        else:
            controller.observe_oom(requested)
    for _ in range(20):
        controller.observe_success(controller.next_batch_size(100))

    assert controller.current_batch_size == 32
    assert controller.oom_upper_bound is None


def test_explicit_oom_reset_reopens_capacity_without_changing_current_size():
    controller = AdaptiveBatchController(initial_batch_size=16, max_batch_size=32)
    controller.observe_oom(16)
    current = controller.current_batch_size

    assert controller.reset_oom_bound() == current
    assert controller.oom_upper_bound is None
