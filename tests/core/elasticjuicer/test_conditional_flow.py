"""Contract tests; logical GPU fixtures do not establish physical speedups."""

from types import SimpleNamespace

import pyarrow as pa
import pytest

from data_juicer.core.elasticjuicer.compute_flow import (
    ComputeFlowError,
    PartitionComputeFlowConfig,
)
from data_juicer.core.elasticjuicer.conditional_flow import (
    ROW,
    ConditionalPlan,
    invoke_selected,
)


def sample():
    return pa.table(
        {
            "id": [19, 2, 77],
            ROW: [0, 1, 2],
            "image_version": ["a", "b", "c"],
            "required_ops": [["quality", "ocr"], ["quality"], ["quality", "face"]],
        }
    )


def plan():
    return ConditionalPlan.compile(
        ("topic", "q", "o", "f"),
        [
            SimpleNamespace(conditional_topic=True),
            SimpleNamespace(conditional_stage="quality", conditional_outputs={"quality": "float32"}),
            SimpleNamespace(conditional_stage="ocr", conditional_outputs={"ocr": "string"}),
            SimpleNamespace(conditional_stage="face", conditional_outputs={"face": "string"}),
        ],
    )


def test_selected_captain_sees_only_required_rows():
    op = SimpleNamespace(conditional_stage="ocr", conditional_outputs={"ocr": "string"})
    seen = []

    def captain(table):
        seen.extend(table["id"].to_pylist())
        return table.append_column("ocr", pa.array(["real-result"]))

    out = invoke_selected(op, sample(), captain)
    assert seen == [19]
    assert out["id"].to_pylist() == [19, 2, 77]
    assert out["ocr"].to_pylist() == ["real-result", None, None]


def test_empty_branch_never_calls_model_and_keeps_typed_nulls():
    op = SimpleNamespace(conditional_stage="embedding", conditional_outputs={"embedding": "list_float32"})

    def forbidden(_):
        pytest.fail("Empty route called GPU mapper")

    out = invoke_selected(op, sample(), forbidden)
    assert out["embedding"].to_pylist() == [None, None, None]
    assert out["embedding"].type == pa.list_(pa.float32())


def test_heterogeneous_join_keeps_full_master_and_distinct_memberships():
    p, base = plan(), sample()
    deltas = {
        "q": base.append_column("quality", pa.array([0.1, 0.2, 0.3])),
        "o": base.slice(0, 1).append_column("ocr", pa.array(["text"])),
        "f": base.slice(2, 1).append_column("face", pa.array(["face-boxes"])),
    }
    out = p.join(base, deltas)
    assert out["id"].to_pylist() == [19, 2, 77]
    assert out["ocr"].to_pylist() == ["text", None, None]
    assert out["face"].to_pylist() == [None, None, "face-boxes"]
    deltas.pop("f")
    with pytest.raises(ComputeFlowError, match="exactly"):
        p.join(base, deltas)


@pytest.mark.parametrize("field", ["id", ROW, "image_version", "required_ops"])
def test_branch_cannot_modify_identity_version_or_routes(field):
    op = SimpleNamespace(conditional_stage="quality", conditional_outputs={"quality": "float32"})

    def corrupt(table):
        return table.take(pa.array([2, 1, 0])).append_column("quality", pa.array([1.0, 2.0, 3.0]))

    # Every immutable column is checked, including columns not used as join keys.
    def changed(table):
        out = corrupt(table)
        index = table.schema.get_field_index(field)
        return table.set_column(index, field, out[field]).append_column("quality", pa.array([1.0, 2.0, 3.0]))

    with pytest.raises(ComputeFlowError, match="preserved field"):
        invoke_selected(op, sample(), changed)


def test_unknown_or_duplicate_route_is_rejected():
    for routes in ([["unknown"], [], []], [["quality", "quality"], [], []]):
        table = sample().set_column(3, "required_ops", pa.array(routes))
        with pytest.raises(ComputeFlowError, match="required_ops"):
            plan().route(table)


def test_conditional_config_is_opt_in():
    assert PartitionComputeFlowConfig.parse(None).execution == "linear"
    assert PartitionComputeFlowConfig.parse({"execution": "conditional"}).execution == "conditional"
    assert PartitionComputeFlowConfig.parse({"allow_multi_node": True}).allow_multi_node
    with pytest.raises(ComputeFlowError, match="allow_multi_node"):
        PartitionComputeFlowConfig.parse({"allow_multi_node": "true"})
    with pytest.raises(ComputeFlowError, match="execution"):
        PartitionComputeFlowConfig.parse({"execution": "other"})


@pytest.mark.parametrize("slots", [[4, 3], [7, 0, 1], [True, 7], "8"])
def test_static_allocation_rejects_invalid_budget(slots):
    with pytest.raises(ComputeFlowError, match="stage_slots"):
        PartitionComputeFlowConfig.parse({"workers": 8, "stage_slots": slots})


def test_sibling_dependency_cannot_be_scheduled_as_independent_branch():
    with pytest.raises(ComputeFlowError, match="sibling"):
        ConditionalPlan.compile(
            ("topic", "q", "o"),
            [
                SimpleNamespace(conditional_topic=True),
                SimpleNamespace(conditional_stage="q", conditional_outputs={"quality": "float32"}),
                SimpleNamespace(
                    conditional_stage="o", conditional_outputs={"ocr": "string"}, _input_columns=("quality",)
                ),
            ],
        )


def test_join_replay_replaces_output_columns_without_duplicates():
    p, base = plan(), sample()
    base = base.append_column("ocr", pa.array(["stale", "stale", "stale"]))
    deltas = {
        "q": base.append_column("quality", pa.array([0.1, 0.2, 0.3])),
        "o": base.slice(0, 1).set_column(base.schema.get_field_index("ocr"), "ocr", pa.array(["fresh"])),
        "f": base.slice(2, 1).append_column("face", pa.array(["face-boxes"])),
    }
    out = p.join(base, deltas)
    assert len(out.column_names) == len(set(out.column_names))
    assert out["ocr"].to_pylist() == ["fresh", None, None]


def test_join_replay_clears_existing_outputs_of_empty_branches():
    p, base = plan(), sample()
    base = base.set_column(base.schema.get_field_index("required_ops"), "required_ops", pa.array([["quality"]] * 3))
    base = base.append_column("ocr", pa.array(["stale"] * 3))
    base = base.append_column("face", pa.array(["stale"] * 3))
    out = p.join(base, {"q": base.append_column("quality", pa.array([0.1, 0.2, 0.3]))})
    assert len(out.column_names) == len(set(out.column_names))
    assert out["id"].to_pylist() == base["id"].to_pylist()
    assert out["ocr"].to_pylist() == [None] * 3
    assert out["face"].to_pylist() == [None] * 3
    assert out["ocr"].type == out["face"].type == pa.string()


def test_real_ray_conditional_transport_and_order(tmp_path):
    import ray

    from data_juicer.core.elasticjuicer.conditional_session import (
        ConditionalFlowSession,
        ConditionalLinearSession,
    )
    from data_juicer.core.executor.ray_executor_partitioned import (
        PartitionedRayExecutor,
    )
    from data_juicer.ops.base_op import Mapper

    class Topic(Mapper):
        _accelerator, _batched_op, _supports_adaptive_batching = "cuda", True, True
        conditional_topic = True

        def process_batched(self, batch):
            batch["required_ops"] = [["q", "o"] if x % 2 else ["q"] for x in batch["id"]]
            return batch

    class Quality(Mapper):
        _accelerator, _batched_op, _supports_adaptive_batching = "cuda", True, True
        conditional_stage, conditional_outputs = "q", {"q": "float32"}

        def process_batched(self, batch):
            batch["q"] = [float(x) for x in batch["id"]]
            return batch

    class Other(Quality):
        conditional_stage, conditional_outputs = "o", {"o": "string"}

        def process_batched(self, batch):
            assert all(x % 2 for x in batch["id"])
            batch["o"] = [str(x) for x in batch["id"]]
            return batch

    class Source(Mapper):
        _batched_op = True
        _output_columns = ("source_tag",)

        def process_batched(self, batch):
            batch["source_tag"] = ["source"] * len(batch["id"])
            return batch

    class Pack(Mapper):
        _batched_op = True
        _output_columns = ("packed",)

        def process_batched(self, batch):
            assert all(v == "source" for v in batch["source_tag"])
            batch["packed"] = [True] * len(batch["id"])
            return batch

    ray.init(
        num_cpus=8,
        num_gpus=3,
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=256 * 1024**2,
    )
    try:
        ray.data.DataContext.get_current().execution_options.preserve_order = True
        ops = [
            cls(batch_size=4, num_proc=1, num_cpus=1, num_gpus=1, skip_op_error=False, ray_execution_mode="actor")
            for cls in (Topic, Quality, Other)
        ]
        for index, op in enumerate(ops):
            op._elastic_juicer_stage_identity = f"s{index}"
            op._gpu_memory_fraction = 0.01
        rows = [{"id": i, "image_version": str(i)} for i in range(17, -1, -1)]
        for mode, session_class in (
            ("dynamic", ConditionalFlowSession),
            ("static", ConditionalFlowSession),
            ("linear", ConditionalLinearSession),
        ):
            options = PartitionComputeFlowConfig(
                enabled=True, workers=3, execution="conditional", mode=mode, batch_size=4, max_inflight_batches=2
            )
            if mode == "linear":
                options = PartitionComputeFlowConfig(enabled=True, workers=3, batch_size=4, max_inflight_batches=2)
            session = session_class(options, str(tmp_path / mode))
            try:
                source = Source(batch_size=4, num_proc=2, num_cpus=1, ray_execution_mode="task", skip_op_error=False)
                pack = Pack(batch_size=4, num_proc=2, num_cpus=1, ray_execution_mode="task", skip_op_error=False)
                out = session.execute(ray.data.from_items(rows, override_num_blocks=3), [source, *ops, pack]).take_all()
                assert [r["id"] for r in out] == [r["id"] for r in rows]
                assert [r["o"] for r in out] == [str(r["id"]) if r["id"] % 2 else None for r in rows]
                assert all(r["q"] == float(r["id"]) for r in out)
                assert all(r["packed"] for r in out)
                assert all(ROW not in r for r in out)
            finally:
                session.finish()
        left = ray.data.from_items([dict(id=i) for i in range(4)], override_num_blocks=2)
        right = ray.data.from_items([dict(id=i) for i in range(4, 8)], override_num_blocks=2)
        concat = PartitionedRayExecutor._concatenate_partitions([left, right])
        assert [r["id"] for r in concat.take_all()] == list(range(8))
    finally:
        ray.shutdown()
