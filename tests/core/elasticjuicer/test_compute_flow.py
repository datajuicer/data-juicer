from collections import deque

import pytest

from data_juicer.core.elasticjuicer.compute_flow import (
    ComputeFlowConfig,
    ComputeFlowController,
    ComputeFlowError,
    ReadyWork,
)
from data_juicer.core.elasticjuicer.partition_compute_flow import _compiled


def controller(workers=8, **kwargs):
    return ComputeFlowController(
        _compiled(("screen", "features"), workers), ComputeFlowConfig(pools=("gpu",), **kwargs)
    )


def queue(count, since=0, rows=32):
    return deque(ReadyWork(i, rows, since) for i in range(count))


def test_default_pool_selection():
    c = ComputeFlowController(_compiled(("screen", "features"), 2), ComputeFlowConfig())
    assert c.pools == ("gpu",)
    assert c.targets == {"screen": 1, "features": 1}


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "bogus"},
        {"ewma_alpha": 0},
        {"interval_seconds": float("nan")},
        {"starvation_seconds": -1},
        {"hysteresis_workers": True},
        {"initial_slots": (("screen", 7),)},
        {"initial_slots": (("unknown", 8),)},
        {"initial_slots": (("screen", 8), ("features", 0)), "mode": "static"},
    ],
)
def test_reject_invalid_controls(options):
    with pytest.raises(ComputeFlowError):
        controller(**options)


def test_reject_unknown_pool_and_static_deadlock():
    with pytest.raises(ComputeFlowError):
        ComputeFlowController(_compiled(("screen", "features"), 2), ComputeFlowConfig(pools=("missing",)))
    with pytest.raises(ComputeFlowError, match="at least one"):
        controller(workers=1, mode="static")


def test_rate_weighting_cold_exclusion_and_reversal():
    c = controller(ewma_alpha=1)
    key = ("gpu", 0)
    c.observe("screen", key, 0, 1000)
    c.observe("screen", key, 32, 1000)  # Cold initialization is not a throughput prior.
    c.observe("features", key, 32, 1000)
    assert c.cost == {} and c.cold_samples == 2
    c.observe("screen", key, 32, 1)
    c.observe("features", key, 32, 8)
    ready = {"screen": queue(100), "features": queue(100)}
    c.replan(ready, {}, 0)
    assert c.targets == {"screen": 1, "features": 7}
    # An actual workload change, including smaller partial batches, reverses
    # the allocation using seconds/row rather than raw task latency.
    c.observe("screen", key, 4, 1)
    c.observe("features", key, 32, 1)
    c.replan(ready, {}, 1)
    assert c.targets == {"screen": 7, "features": 1}
    assert c.rate_samples == 4


def test_epoch_interval_and_hysteresis():
    c = controller(hysteresis_workers=3)
    ready = {"screen": queue(100), "features": queue(100)}
    c.replan(ready, {}, 0)
    c.cost = {"screen": 1, "features": 2}
    assert c.replan(ready, {}, 0.1) == []
    c.replan(ready, {}, 1)
    assert c.targets == {"screen": 4, "features": 4}
    # A stage becoming empty bypasses hysteresis, so all capacity drains the tail.
    c.replan({"features": queue(100)}, {}, 2)
    assert c.targets == {"screen": 0, "features": 8}


def test_busy_leases_are_not_preempted_and_idle_capacity_is_borrowed():
    c = controller()
    ready = {"features": queue(20)}
    running = {"screen": [32] * 7, "features": [32]}
    c.replan(ready, running, 0)
    assert c.choose("gpu", deque(), ready, {"screen": 7, "features": 1}, 0) is None
    assert running == {"screen": [32] * 7, "features": [32]}
    c.targets = {"screen": 8, "features": 0}  # Old plan during cooldown.
    key, group, reason = c.choose("gpu", deque([("gpu", 6)]), ready, {}, 0)
    assert group == "features" and reason == "borrow"
    c.dispatched(key, "screen", "target")
    lease = c.dispatched(key, group, reason)
    assert lease["reassigned"] and c.borrowed == 1 and c.reassignments == 1


def test_fairness_when_there_are_fewer_workers_than_groups():
    c = controller(workers=1, starvation_seconds=1)
    ready = {"screen": queue(1, since=0), "features": queue(100, since=0.5)}
    c.targets = {"screen": 0, "features": 1}
    assert c.choose("gpu", deque([("gpu", 0)]), ready, {}, 1.1)[1:] == ("screen", "age")


def test_static_ownership_does_not_lend_idle_slots():
    c = controller(mode="static")
    ready = {"features": queue(100)}
    assert c.choose("gpu", deque([("gpu", 0)]), ready, {}, 10) is None
    assert c.choose("gpu", deque([("gpu", 0), ("gpu", 6)]), ready, {}, 10)[0] == ("gpu", 6)


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_targets_stay_within_capacity(workers):
    c = controller(workers=workers)
    for i in range(100):
        ready = {"screen": queue(i % 19), "features": queue((i * 7) % 23)}
        running = {"screen": [32] * (i % (workers + 1))}
        c.replan(ready, running, i)
        assert all(0 <= n <= workers for n in c.targets.values())
        assert sum(c.targets.values()) <= workers
        assert all(c.targets[g] <= len(ready[g]) + len(running.get(g, ())) for g in ready)
