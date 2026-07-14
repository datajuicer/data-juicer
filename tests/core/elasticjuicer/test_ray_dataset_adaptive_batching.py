from unittest.mock import MagicMock, patch

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
    return dataset


def test_feature_flag_defaults_to_disabled():
    dataset = RayDataset.__new__(RayDataset)
    dataset.data = MagicMock()
    dataset._auto_proc = False

    dataset._configure_elastic_juicer({})

    assert dataset._elastic_juicer_adaptive_batching is False


@patch("data_juicer.core.data.ray_dataset.get_compute_strategy")
def test_disabled_flag_preserves_existing_actor_map_batches_call(get_compute_strategy):
    dataset = _dataset(enabled=False)
    source = dataset.data
    operator = BatchedActorMapper(marker="disabled", batch_size=16, num_proc=2)
    compute = object()
    get_compute_strategy.return_value = compute

    dataset._run_single_op(operator, cached_columns=set())

    get_compute_strategy.assert_called_once_with(operator.__class__, concurrency=operator.num_proc)
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


@patch("data_juicer.core.data.ray_dataset.get_compute_strategy")
def test_enabled_flag_installs_one_actor_local_adaptive_wrapper(get_compute_strategy):
    dataset = _dataset(enabled=True)
    source = dataset.data
    operator = BatchedActorMapper(marker="enabled", batch_size=16, num_proc=2)
    compute = object()
    get_compute_strategy.return_value = compute

    dataset._run_single_op(operator, cached_columns=set())

    get_compute_strategy.assert_called_once_with(RayAdaptiveMapperActor, concurrency=operator.num_proc)
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
        },
        batch_size=16,
        num_cpus=operator.num_cpus,
        num_gpus=operator.num_gpus,
        compute=compute,
        batch_format="pyarrow",
        runtime_env=operator.runtime_env,
    )


@patch("data_juicer.core.data.ray_dataset.get_compute_strategy")
def test_non_batched_mapper_keeps_existing_path_when_flag_is_enabled(get_compute_strategy):
    dataset = _dataset(enabled=True)
    source = dataset.data
    operator = SingleActorMapper(batch_size=16, num_proc=2)
    compute = object()
    get_compute_strategy.return_value = compute

    dataset._run_single_op(operator, cached_columns=set())

    get_compute_strategy.assert_called_once_with(operator.__class__, concurrency=operator.num_proc)
    assert source.map_batches.call_args.args[0] is operator.__class__
