from collections import deque
from dataclasses import replace

import pytest

from data_juicer.core.elasticjuicer.compute_flow import (
    ComputeFlowConfig,
    ComputeFlowController,
    ReadyWork,
)
from data_juicer.core.elasticjuicer.dag_plan import DAGContractError
from data_juicer.core.executor.resident_dag_executor import ResidentDAGExecutor
from tests.core.elasticjuicer import test_resident_dag as dag_fixture
from tests.core.elasticjuicer.test_resident_dag import batches, plan, read_events

cluster = dag_fixture.cluster


def controller(workers=8, **kwargs):
    p = plan()
    p = replace(p, pools=(p.pools[0], replace(p.pools[1], workers=workers)))
    return ComputeFlowController(p.compile(), ComputeFlowConfig(pools=("models",), **kwargs))


def queue(count, since=0, rows=32):
    return deque(ReadyWork(i, rows, since) for i in range(count))


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
    with pytest.raises(DAGContractError):
        controller(**options)


def test_reject_unknown_pool_and_static_deadlock():
    with pytest.raises(DAGContractError):
        ComputeFlowController(plan().compile(), ComputeFlowConfig(pools=("missing",)))
    with pytest.raises(DAGContractError, match="at least one"):
        controller(workers=1, mode="static")


def test_rate_weighting_cold_exclusion_and_reversal():
    c = controller(ewma_alpha=1)
    key = ("models", 0)
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
    assert c.choose("models", deque(), ready, {"screen": 7, "features": 1}, 0) is None
    assert running == {"screen": [32] * 7, "features": [32]}
    c.targets = {"screen": 8, "features": 0}  # Old plan during cooldown.
    key, group, reason = c.choose("models", deque([("models", 6)]), ready, {}, 0)
    assert group == "features" and reason == "borrow"
    c.dispatched(key, "screen", "target")
    lease = c.dispatched(key, group, reason)
    assert lease["reassigned"] and c.borrowed == 1 and c.reassignments == 1


def test_fairness_when_there_are_fewer_workers_than_groups():
    c = controller(workers=1, starvation_seconds=1)
    ready = {"screen": queue(1, since=0), "features": queue(100, since=0.5)}
    c.targets = {"screen": 0, "features": 1}
    assert c.choose("models", deque([("models", 0)]), ready, {}, 1.1)[1:] == ("screen", "age")


def test_static_ownership_does_not_lend_idle_slots():
    c = controller(mode="static")
    ready = {"features": queue(100)}
    assert c.choose("models", deque([("models", 0)]), ready, {}, 10) is None
    assert c.choose("models", deque([("models", 0), ("models", 6)]), ready, {}, 10)[0] == ("models", 6)


def test_multiple_pools_use_their_own_plan_epoch():
    c = ComputeFlowController(plan().compile(), ComputeFlowConfig(pools=("cpu", "models")))
    snapshots = c.replan({"prepare": queue(5), "screen": queue(5)}, {}, 0)
    epochs = {s["pool"]: s["epoch"] for s in snapshots}
    assert c.dispatched(("cpu", 0), "prepare", "target")["epoch"] == epochs["cpu"]
    assert c.dispatched(("models", 0), "screen", "target")["epoch"] == epochs["models"]


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


@pytest.mark.parametrize("mode,adaptive", [("static", False), ("static", True), ("dynamic", False), ("dynamic", True)])
def test_real_ray_two_layers_preserve_join_and_instances(cluster, tmp_path, mode, adaptive):
    p = plan()
    p = replace(p, max_inflight_units=8, nodes=tuple(replace(n, initial_batch_size=1) for n in p.nodes))
    root = tmp_path / mode
    with ResidentDAGExecutor(
        p,
        adaptive=adaptive,
        compute_flow=ComputeFlowConfig(pools=("models",), mode=mode, interval_seconds=0.01),
        artifact_dir=root,
    ) as executor:
        identities = {w["worker_id"] for w in executor.worker_identities}
        for _ in range(2):
            values = {}
            for result in executor.iter_batches(batches(65)):
                output = result.columns()
                values.update(zip(output["id"], output["result"]))
            assert values == {i: 20 * i + 25 for i in range(0, 65, 2)}
            assert executor.last_summary["compute_flow_lifetime"]["rate_samples"] > 0
        events = read_events(root)
        assert {e["worker_id"] for e in events} <= identities
        groups_by_actor, instances = {}, {}
        for e in events:
            instances.setdefault((e["worker_id"], e["node"]), set()).add(e["operator_instance"])
            if e["group"] in ("screen", "features"):
                groups_by_actor.setdefault(e["worker_id"], set()).add(e["group"])
        assert all(len(ids) == 1 for ids in instances.values())
        if mode == "static":
            assert all(len(groups) == 1 for groups in groups_by_actor.values())
        else:
            assert any(len(groups) == 2 for groups in groups_by_actor.values())
            assert executor.flow.reassignments > 0
        sizes = {a["rows"] for e in events for a in e["attempts"]}
        assert (1 in sizes and 2 in sizes) if adaptive else 4 in sizes


def test_flow_keeps_oom_learning_local_and_fences_errors(cluster, tmp_path):
    p = plan()
    left = p.nodes[1]
    options = dict(left.operator["ej_test_dag_mapper"], oom_above=2)
    p = replace(p, nodes=(p.nodes[0], replace(left, operator={"ej_test_dag_mapper": options}), *p.nodes[2:]))
    root = tmp_path / "oom"
    with ResidentDAGExecutor(p, compute_flow=ComputeFlowConfig(pools=("models",)), artifact_dir=root) as executor:
        assert sum(len(b.accepted_ids) for b in executor.iter_batches(batches(35))) == 18
        events = read_events(root)
        assert any(e["oom_retries"] for e in events if e["node"] == "left")
        assert all(e["oom_retries"] == 0 for e in events if e["node"] != "left")
        iterator = executor.iter_batches(batches(35))
        next(iterator)
        iterator.close()
        assert executor.closed and executor.last_summary["status"] == "failed"


def test_initial_batch_validation():
    p = plan()
    for initial in (0, 5, True):
        with pytest.raises(DAGContractError, match="initial_batch_size"):
            replace(p, nodes=(replace(p.nodes[0], initial_batch_size=initial), *p.nodes[1:])).compile()
