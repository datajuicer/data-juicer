import json
from dataclasses import replace

import pytest
import ray

from data_juicer.core.elasticjuicer.dag_plan import (
    DAGContractError,
    DAGNode,
    DAGPayload,
    ExecutionGroup,
    ResidentDAGPlan,
    WorkerPool,
    merge_parents,
)
from data_juicer.core.elasticjuicer.dag_worker import materialize_payload
from data_juicer.core.executor.resident_dag_executor import ResidentDAGExecutor


def node(name, parents, inputs, output, **options):
    row_policy = "subset" if options.get("keep_even") else "preserve"
    return DAGNode(
        name,
        {"ej_test_dag_mapper": dict(inputs=list(inputs), output=output, **options)},
        tuple(parents),
        tuple(inputs),
        (output,),
        batch_size=4,
        row_policy=row_policy,
    )


def plan(parallel=True, **options):
    nodes = (
        node("prepare", (), ("value",), "x", scale=2),
        node("left", ("prepare",), ("x",), "a", offset=2, sleep_seconds=0.03, reverse=True),
        node("right", ("prepare",), ("x",), "b", offset=3, sleep_seconds=0.03),
        node("crop", ("left", "right"), ("a", "b"), "y", keep_even=True),
        node("embedding", ("crop",), ("y",), "c", scale=2, sleep_seconds=0.03),
        node("aesthetic", ("crop",), ("y",), "d", scale=3, sleep_seconds=0.03),
        node("pack", ("embedding", "aesthetic"), ("c", "d"), "result"),
    )
    groups = [ExecutionGroup(n, (n,), "cpu") for n in ("prepare", "crop", "pack")]
    if parallel:
        groups += [
            ExecutionGroup("screen", ("left", "right"), "models", True),
            ExecutionGroup("features", ("embedding", "aesthetic"), "models", True),
        ]
    else:
        groups += [ExecutionGroup(n, (n,), "models") for n in ("left", "right", "embedding", "aesthetic")]
    return ResidentDAGPlan(
        nodes,
        tuple(groups),
        (WorkerPool("cpu", 2), WorkerPool("models", 2)),
        "pack",
        operator_modules=("tests.core.elasticjuicer.dag_test_ops",),
        max_inflight_units=2,
        max_batch_bytes=1024**2,
        object_budget_bytes=32 * 1024**2,
        **options,
    )


def batches(count=17, size=5):
    return (
        {"id": list(range(i, min(count, i + size))), "value": list(range(i, min(count, i + size)))}
        for i in range(0, count, size)
    )


def test_plan_rejects_unsafe_groups_and_underbudget():
    p = plan()
    p.compile()
    with pytest.raises(DAGContractError, match="dependent"):
        replace(
            p,
            groups=(ExecutionGroup("all", tuple(n.name for n in p.nodes), "cpu", True),),
            pools=(WorkerPool("cpu", 1),),
        ).compile()
    # A non-convex sequential group creates a cycle after contraction.
    gs = tuple(g for g in p.groups if g.name not in ("prepare", "crop"))
    with pytest.raises(DAGContractError, match="cycle"):
        replace(p, groups=gs + (ExecutionGroup("bad", ("prepare", "crop"), "cpu"),)).compile()
    with pytest.raises(DAGContractError, match="budget"):
        replace(p, object_budget_bytes=1).compile()
    with pytest.raises(DAGContractError, match="ID is immutable"):
        replace(p, nodes=(replace(p.nodes[0], writes=("id",)),) + p.nodes[1:]).compile()


def test_exact_join_reorders_and_checks_versions_and_conflicts():
    p = plan()
    compiled = p.compile()
    join = replace(p.nodes[3], join_keys=("version",))
    a = DAGPayload({"id": [1, 2], "version": ["v1", "v2"], "a": [10, 20]}, {"a": "left"})
    b = DAGPayload({"id": [2, 1], "version": ["v2", "v1"], "b": [200, 100]}, {"b": "right"})
    result = merge_parents([a, b], join, compiled, "id")
    assert result.columns == {"id": [1, 2], "version": ["v1", "v2"], "a": [10, 20], "b": [100, 200]}
    b.columns["version"][0] = "stale"
    with pytest.raises(DAGContractError, match="version"):
        merge_parents([a, b], join, compiled, "id")
    with pytest.raises(DAGContractError, match="branch IDs"):
        merge_parents([a, DAGPayload({"id": [1]})], join, compiled, "id")
    with pytest.raises(DAGContractError, match="ambiguous"):
        merge_parents([a, DAGPayload(a.columns, {"a": "right"})], join, compiled, "id")


def test_delta_lineage_reconstructs_columns_and_rejects_forged_identity():
    compiled = plan().compile()
    source = DAGPayload({"id": [1, 2], "value": [10, 20]})
    prepared = DAGPayload({"id": [1, 2], "x": [20, 40]}, {"x": "prepare"}, "prepare", {"$input": source})
    delta = DAGPayload({"id": [1, 2], "a": [22, 42]}, {"a": "left"}, "left", {"prepare": prepared})
    result = materialize_payload(delta, compiled, "id")
    assert result.columns == {"id": [1, 2], "value": [10, 20], "x": [20, 40], "a": [22, 42]}
    assert "value" not in delta.columns and "x" not in delta.columns
    delta.writers["a"] = "right"
    with pytest.raises(DAGContractError, match="ownership"):
        materialize_payload(delta, compiled, "id")
    delta.writers["a"] = "left"
    delta.columns["id"] = [1, 999]
    with pytest.raises(DAGContractError, match="identity"):
        materialize_payload(delta, compiled, "id")


@pytest.fixture(scope="module")
def cluster():
    assert not ray.is_initialized(), "test owns an isolated Ray cluster"
    ray.init(
        address="local",
        num_cpus=6,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=256 * 1024**2,
        log_to_driver=False,
    )
    yield
    ray.shutdown()


def read_events(root):
    return [json.loads(line) for path in root.glob("*/events.jsonl") for line in path.read_text().splitlines()]


@pytest.mark.parametrize("adaptive,parallel", [(False, True), (True, True), (True, False)])
def test_real_ray_forks_filter_join_same_path_and_resident_reuse(cluster, tmp_path, adaptive, parallel):
    root = tmp_path / "run"
    with ResidentDAGExecutor(plan(parallel), adaptive=adaptive, artifact_dir=root) as executor:
        identities = {w["worker_id"] for w in executor.worker_identities}
        for _ in range(2):
            records, rejected = {}, []
            for result in executor.iter_batches(batches()):
                out = result.columns()
                records.update(zip(out["id"], out["result"]))
                rejected.extend(result.rejected_ids)
            assert records == {i: 20 * i + 25 for i in range(0, 17, 2)}
            assert sorted(rejected) == list(range(1, 17, 2))
            assert executor.last_summary["status"] == "complete"
            assert executor.last_summary["peak_inflight_units"] <= 2
        events = read_events(root)
        assert all(e["emitted_column_bytes"] < e["payload_bytes"] for e in events if e["node"] == "left")
        assert {e["worker_id"] for e in events} <= identities
        by_node_worker = {}
        for event in events:
            by_node_worker.setdefault((event["node"], event["worker_id"]), set()).add(event["operator_instance"])
        assert all(len(ids) == 1 for ids in by_node_worker.values())
        assert {e["worker_id"] for e in events if e["node"] == "left"} & {
            e["worker_id"] for e in events if e["node"] == "embedding"
        }
        paired = {}
        for e in events:
            paired.setdefault((e["run_id"], e["unit"]), {})[e["node"]] = e
        for pair in [("left", "right"), ("embedding", "aesthetic")]:
            assert any(
                min(v[pair[0]]["ended"], v[pair[1]]["ended"]) > max(v[pair[0]]["started"], v[pair[1]]["started"])
                for v in paired.values()
            )


def test_local_oom_recovery_preserves_rows_and_state(cluster, tmp_path):
    p = plan()
    left = replace(p.nodes[1], operator={"ej_test_dag_mapper": dict(inputs=["x"], output="a", offset=2, oom_above=2)})
    p = replace(p, nodes=(p.nodes[0], left) + p.nodes[2:])
    with ResidentDAGExecutor(p, artifact_dir=tmp_path / "oom") as executor:
        result = {
            i: value
            for batch in executor.iter_batches(batches(20))
            for i, value in zip(batch.columns()["id"], batch.columns()["result"])
        }
        assert result == {i: 20 * i + 25 for i in range(0, 20, 2)}
        left_events = [e for e in read_events(tmp_path / "oom") if e["node"] == "left"]
        assert any(a["oom"] for e in left_events for a in e["attempts"])
        assert all(e["controller"]["oom_upper_bound"] == 4 for e in left_events)
    with ResidentDAGExecutor(p, adaptive=False) as executor:
        with pytest.raises(Exception, match="out of memory"):
            list(executor.iter_batches(batches(5)))
        assert executor.closed


def test_sequential_group_can_use_local_intermediate_lineage(cluster):
    p = plan()
    p = replace(
        p,
        groups=(ExecutionGroup("all", tuple(n.name for n in reversed(p.nodes)), "cpu"),),
        pools=(WorkerPool("cpu", 1),),
        output_columns=("result",),
    )
    with ResidentDAGExecutor(p) as executor:
        results = [batch.columns() for batch in executor.iter_batches(batches(5))]
        assert results == [{"id": [0, 2, 4], "result": [25, 65, 105]}]


def test_bad_identity_fences_owned_pool(cluster):
    p = plan()
    left = replace(p.nodes[1], operator={"ej_test_dag_mapper": dict(inputs=["x"], output="a", corrupt_id=True)})
    with ResidentDAGExecutor(replace(p, nodes=(p.nodes[0], left) + p.nodes[2:])) as executor:
        with pytest.raises(Exception, match="identity"):
            list(executor.iter_batches(batches()))
        assert executor.closed and not executor.actors
        assert ray.is_initialized()


def test_pause_drains_without_consuming_remaining_input(cluster):
    source = batches(35)
    with ResidentDAGExecutor(plan()) as executor:
        emitted = []
        for result in executor.iter_batches(source):
            emitted.extend(result.input_ids)
            executor.request_pause()
        assert executor.last_summary["status"] == "paused"
        assert sorted(emitted) == list(range(10))
        rest = [i for result in executor.iter_batches(source) for i in result.input_ids]
        assert sorted(rest) == list(range(10, 35))
        assert executor.last_summary["status"] == "complete"


def test_resident_slots_are_fair_and_caller_recipe_is_frozen(cluster, tmp_path):
    p = ResidentDAGPlan(
        (node("only", (), ("value",), "result", offset=1),),
        (ExecutionGroup("only", ("only",), "models"),),
        (WorkerPool("models", 4),),
        "only",
        operator_modules=("tests.core.elasticjuicer.dag_test_ops",),
        max_inflight_units=1,
        output_columns=("result",),
    )
    root = tmp_path / "fair"
    with ResidentDAGExecutor(p, artifact_dir=root) as executor:
        p.nodes[0].operator["ej_test_dag_mapper"]["offset"] = 999
        outputs = [b.columns() for b in executor.iter_batches(batches(5, 1))]
        assert outputs == [{"id": [i], "result": [i + 1]} for i in range(5)]
        assert len({e["worker_id"] for e in read_events(root)}) == 4


@pytest.mark.parametrize("failure", ["duplicate", "oversize", "cancel"])
def test_input_errors_and_cancel_retire_pool(cluster, failure):
    p = plan()
    if failure == "oversize":
        p = replace(p, max_batch_bytes=100)
    with ResidentDAGExecutor(p) as executor:
        if failure == "duplicate":
            with pytest.raises(DAGContractError, match="duplicate"):
                list(executor.iter_batches([{"id": [1], "value": [1]}] * 2))
        elif failure == "oversize":
            with pytest.raises(DAGContractError, match="max_batch_bytes"):
                list(executor.iter_batches(batches()))
        else:
            iterator = executor.iter_batches(batches())
            next(iterator)
            iterator.close()
        assert executor.closed
        assert executor.last_summary["status"] == "failed"


def test_task_timeout_fences_pool(cluster):
    p = ResidentDAGPlan(
        (node("slow", (), ("value",), "result", sleep_seconds=0.1),),
        (ExecutionGroup("slow", ("slow",), "cpu"),),
        (WorkerPool("cpu", 1),),
        "slow",
        operator_modules=("tests.core.elasticjuicer.dag_test_ops",),
        task_timeout_seconds=0.01,
    )
    with ResidentDAGExecutor(p) as executor:
        with pytest.raises(TimeoutError):
            list(executor.iter_batches(batches(1)))
        assert executor.closed
