from unittest.mock import MagicMock, patch

import pytest

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor
from data_juicer.ops.base_op import Mapper


class BatchedActorMapper(Mapper):
    _batched_op = True

    def __init__(self, marker="configured", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.marker = marker

    def process_batched(self, samples):
        return samples

    def use_ray_actor(self):
        return True


class SingleActorMapper(Mapper):
    def process_single(self, sample):
        return sample

    def use_ray_actor(self):
        return True


def _dataset(enabled):
    dataset = RayDataset.__new__(RayDataset)
    dataset.data = MagicMock()
    dataset._elastic_juicer_adaptive_batching = enabled
    dataset._elastic_juicer_job_id = "job-test"
    dataset._elastic_juicer_metrics_sink = MagicMock(name="metrics_sink")
    return dataset


def test_feature_flag_defaults_to_disabled():
    dataset = RayDataset.__new__(RayDataset)
    dataset.data = MagicMock()
    dataset._auto_proc = False

    dataset._configure_elastic_juicer({})

    assert dataset._elastic_juicer_adaptive_batching is False
    assert dataset._elastic_juicer_metrics_sink is None
    assert dataset._elastic_juicer_metrics_max_in_flight == 64
    assert dataset.elastic_juicer_metrics_sink is None


@patch("data_juicer.core.elasticjuicer.async_metrics_sink.create_ray_metrics_sink")
def test_enabled_datasets_reuse_one_job_scoped_sink_from_shared_config(create_sink):
    handle = MagicMock(name="metrics_sink")
    create_sink.return_value = handle
    cfg = {"elastic_juicer_adaptive_batching": True, "job_id": "job-a"}
    first = RayDataset.__new__(RayDataset)
    second = RayDataset.__new__(RayDataset)

    first._configure_elastic_juicer(cfg)
    second._configure_elastic_juicer(cfg)

    create_sink.assert_called_once_with("job-a")
    assert first._elastic_juicer_metrics_sink is handle
    assert second._elastic_juicer_metrics_sink is handle
    assert first.elastic_juicer_metrics_sink is handle


@patch("data_juicer.core.elasticjuicer.async_metrics_sink.create_ray_metrics_sink")
def test_metrics_in_flight_limit_is_configurable_and_validated(create_sink):
    dataset = RayDataset.__new__(RayDataset)
    dataset._configure_elastic_juicer(
        {
            "elastic_juicer_adaptive_batching": True,
            "elastic_juicer_metrics_max_in_flight": 7,
            "job_id": "job-a",
        }
    )
    assert dataset._elastic_juicer_metrics_max_in_flight == 7

    invalid = RayDataset.__new__(RayDataset)
    with pytest.raises(ValueError, match="metrics_max_in_flight"):
        invalid._configure_elastic_juicer(
            {
                "elastic_juicer_adaptive_batching": True,
                "elastic_juicer_metrics_max_in_flight": 0,
                "job_id": "job-b",
            }
        )
    create_sink.assert_called_once_with("job-a")


@patch("data_juicer.core.data.ray_dataset.ActorPoolStrategy")
def test_disabled_flag_preserves_existing_actor_map_batches_call(actor_pool_strategy):
    dataset = _dataset(enabled=False)
    source = dataset.data
    operator = BatchedActorMapper(marker="disabled", batch_size=16, num_proc=2)
    compute = object()
    actor_pool_strategy.return_value = compute

    dataset._run_single_op(operator, cached_columns=set())

    actor_pool_strategy.assert_called_once_with(size=operator.num_proc)
    source.map_batches.assert_called_once_with(
        operator.__class__,
        fn_args=None,
        fn_kwargs=None,
        fn_constructor_args=operator._init_args,
        fn_constructor_kwargs=operator._init_kwargs,
        batch_size=16,
        num_cpus=operator.num_cpus,
        num_gpus=operator.num_gpus,
        compute=compute,
        batch_format="pyarrow",
        runtime_env=operator.runtime_env,
    )


@patch("data_juicer.core.data.ray_dataset.ActorPoolStrategy")
def test_enabled_flag_installs_one_actor_local_adaptive_wrapper(actor_pool_strategy):
    dataset = _dataset(enabled=True)
    source = dataset.data
    operator = BatchedActorMapper(marker="enabled", batch_size=16, num_proc=2)
    compute = object()
    actor_pool_strategy.return_value = compute

    dataset._run_single_op(operator, cached_columns=set())

    actor_pool_strategy.assert_called_once_with(size=operator.num_proc)
    source.map_batches.assert_called_once_with(
        RayAdaptiveMapperActor,
        fn_args=None,
        fn_kwargs=None,
        fn_constructor_kwargs={
            "operator_class": operator.__class__,
            "operator_args": operator._init_args,
            "operator_kwargs": operator._init_kwargs,
            "initial_batch_size": 16,
            "max_batch_size": 16,
            "metrics_sink": dataset._elastic_juicer_metrics_sink,
            "metrics_max_in_flight": 64,
            "job_id": "job-test",
            "op_name": operator.__class__.__name__,
        },
        batch_size=16,
        num_cpus=operator.num_cpus,
        num_gpus=operator.num_gpus,
        compute=compute,
        batch_format="pyarrow",
        runtime_env=operator.runtime_env,
    )


@patch("data_juicer.core.data.ray_dataset.ActorPoolStrategy")
def test_non_batched_mapper_keeps_existing_path_when_flag_is_enabled(actor_pool_strategy):
    dataset = _dataset(enabled=True)
    source = dataset.data
    operator = SingleActorMapper(batch_size=16, num_proc=2)
    compute = object()
    actor_pool_strategy.return_value = compute

    dataset._run_single_op(operator, cached_columns=set())

    actor_pool_strategy.assert_called_once_with(size=operator.num_proc)
    assert source.map_batches.call_args.args[0] is operator.__class__
