from copy import deepcopy
from unittest.mock import Mock

import pyarrow as pa
import pytest
from jsonargparse import Namespace

from data_juicer.config.config import build_base_parser, init_setup_from_cfg
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.elasticjuicer.adaptive_mapper import AdaptiveBatchContractError
from data_juicer.core.elasticjuicer.ray_adaptive_mapper import (
    RayAdaptiveMapperActor,
    adaptive_batching_enabled,
)
from data_juicer.core.elasticjuicer.stage_identity import assign_stage_identities
from data_juicer.core.executor.ray_executor_partitioned import _LOGICAL_PARTITION_COLUMN
from data_juicer.ops import Filter, Mapper


class ThresholdMapper(Mapper):
    _name = "ej_firstbatch_threshold_mapper"
    _batched_op = True
    _accelerator = "cuda"
    _supports_adaptive_batching = True

    def __init__(self, threshold=2, failure=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.failure = failure
        self.calls = []

    def process_batched(self, samples):
        self.calls.append(list(samples["id"]))
        for meta in samples["meta"]:
            meta["attempts"] += 1
        if self.failure == "ordinary" and samples["id"][0] >= 2:
            raise ValueError("bad image")
        if len(samples["id"]) > self.threshold:
            raise RuntimeError("CUDA out of memory")
        if self.failure == "rows":
            return {key: value[:-1] for key, value in samples.items()}
        if self.failure == "tags":
            samples.pop(_LOGICAL_PARTITION_COLUMN)
        samples["value"] = [value * 2 for value in samples["id"]]
        return samples


def make_op(**kwargs):
    args = dict(batch_size=8, num_proc=1, num_cpus=1, num_gpus=0.25, ray_execution_mode="actor")
    args.update(kwargs)
    return ThresholdMapper(**args)


def make_actor(**kwargs):
    op = make_op(**kwargs)
    identities = assign_stage_identities([op])
    return RayAdaptiveMapperActor(
        type(op), op._init_args, op._init_kwargs, op.batch_size, identities["stages"][0]["stage_id"]
    )


def make_batch(n=9):
    return {
        "id": list(range(n)),
        "meta": [{"attempts": 0} for _ in range(n)],
        _LOGICAL_PARTITION_COLUMN: [i // 3 for i in range(n)],
    }


@pytest.mark.parametrize("arrow", [False, True])
def test_actor_retries_without_losing_rows_tags_or_replaying_mutation(arrow):
    actor = make_actor(skip_op_error=True)
    original = make_batch()
    batch = pa.Table.from_pydict(original) if arrow else deepcopy(original)
    output = actor(batch)
    assert output["id"] == original["id"]
    assert output[_LOGICAL_PARTITION_COLUMN] == original[_LOGICAL_PARTITION_COLUMN]
    assert output["meta"] == [{"attempts": 1}] * 9
    assert output["value"] == [i * 2 for i in range(9)]
    assert actor.mapper.oom_retries >= 2
    assert [call[0] for call in actor.op.calls[:3]] == [0, 0, 0]
    assert actor.controller.state.current_batch_size <= 8


def test_actor_retains_learning_across_outer_batches():
    actor = make_actor()
    actor(make_batch())
    actor.op.calls.clear()
    actor(make_batch())
    assert len(actor.op.calls[0]) == 2
    # Growth may test 3, but previously failed sizes 4 and 8 stay excluded.
    assert max(map(len, actor.op.calls)) < 4


def test_terminal_oom_is_never_silently_skipped():
    actor = make_actor(threshold=0, skip_op_error=True)
    actor.mapper.max_floor_retries = 0
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        actor(make_batch())


@pytest.mark.parametrize("failure", ["rows", "tags"])
def test_contract_failure_is_never_silently_skipped(failure):
    actor = make_actor(failure=failure, skip_op_error=True)
    with pytest.raises(AdaptiveBatchContractError):
        actor(make_batch())


@pytest.mark.parametrize("skip", [False, True])
def test_ordinary_error_preserves_whole_outer_batch_skip_policy(skip):
    actor = make_actor(failure="ordinary", skip_op_error=skip)
    if skip:
        assert all(value == [] for value in actor(make_batch()).values())
    else:
        with pytest.raises(ValueError, match="bad image"):
            actor(make_batch())


def test_opt_in_requires_both_global_flag_and_operator_contract():
    assert not adaptive_batching_enabled(make_op())
    assert adaptive_batching_enabled(make_op(), True)
    assert not adaptive_batching_enabled(make_op(adaptive_batching=False), True)
    op = make_op()
    op._supports_adaptive_batching = False
    assert not adaptive_batching_enabled(op, True)
    op.adaptive_batching = True
    assert adaptive_batching_enabled(op, True)


@pytest.mark.parametrize(
    "kwargs", [{"accelerator": "cpu"}, {"ray_execution_mode": "task"}, {"num_gpus": 2}, {"batch_size": 0}]
)
def test_unsupported_mapper_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        adaptive_batching_enabled(make_op(**kwargs), True)


def test_filter_cannot_opt_in():
    with pytest.raises(ValueError, match="batched Mapper"):
        adaptive_batching_enabled(Filter(adaptive_batching=True), True)


def test_dispatch_rejects_unresolved_gpu_reservation():
    op = make_op(num_gpus=None)
    assert adaptive_batching_enabled(op, True)  # Preparation may precede preflight.
    dataset = RayDataset.__new__(RayDataset)
    dataset.data = Mock()
    dataset._adaptive_batching = True
    with pytest.raises(ValueError, match="resolved positive GPU reservation"):
        dataset._run_single_op(op, {"id"})
    dataset.data.map_batches.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_ray_dispatch_preserves_outer_batch_and_actor_resources(enabled):
    op = make_op()
    assign_stage_identities([op])
    dataset = RayDataset.__new__(RayDataset)
    dataset.data = Mock()
    dataset._adaptive_batching = enabled
    data = dataset.data
    dataset._run_single_op(op, {"id", "meta", _LOGICAL_PARTITION_COLUMN})
    args, kwargs = data.map_batches.call_args
    assert args[0] is (RayAdaptiveMapperActor if enabled else ThresholdMapper)
    assert kwargs["batch_size"] == 8
    assert kwargs["num_gpus"] == 0.25
    assert kwargs["num_cpus"] == 1
    assert kwargs["compute"].min_size == kwargs["compute"].max_size == 1
    if enabled:
        assert kwargs["fn_constructor_kwargs"]["stage_id"] == op._elastic_juicer_stage_identity


def test_config_defaults_off_and_accepts_explicit_enable():
    parser = build_base_parser()
    assert parser.get_defaults().elastic_juicer_adaptive_batching is False
    assert parser.parse_args(["--auto", "--elastic_juicer_adaptive_batching", "true"]).elastic_juicer_adaptive_batching


@pytest.mark.parametrize("config", [{"executor_type": "default"}, {"executor_type": "ray", "op_fusion": True}])
def test_unsupported_global_configuration_fails_before_setup(config):
    with pytest.raises(ValueError, match="elastic_juicer_adaptive_batching"):
        init_setup_from_cfg(Namespace(elastic_juicer_adaptive_batching=True, **config))
