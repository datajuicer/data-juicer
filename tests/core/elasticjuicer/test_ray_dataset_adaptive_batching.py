import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor
from data_juicer.core.executor.ray_executor import RayExecutor
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
    dataset._elastic_juicer_control_service = MagicMock(name="control_service")
    dataset._elastic_juicer_control_poll_interval_sec = 0.1
    dataset._elastic_juicer_sample_interval_sec = 0.01
    dataset._elastic_juicer_captain_enabled = False
    dataset._elastic_juicer_captain_lifecycle = None
    dataset._elastic_juicer_next_stage_index = 0
    dataset._elastic_juicer_profile_seed_enabled = False
    dataset._elastic_juicer_cfg = None
    return dataset


def test_feature_flag_defaults_to_disabled():
    dataset = RayDataset.__new__(RayDataset)
    dataset.data = MagicMock()
    dataset._auto_proc = False

    dataset._configure_elastic_juicer({})

    assert dataset._elastic_juicer_adaptive_batching is False
    assert dataset._elastic_juicer_metrics_sink is None
    assert dataset._elastic_juicer_metrics_max_in_flight == 64
    assert dataset._elastic_juicer_control_service is None
    assert dataset._elastic_juicer_control_poll_interval_sec == 0.1
    assert dataset._elastic_juicer_sample_interval_sec == 0.01
    assert dataset._elastic_juicer_captain_enabled is False
    assert dataset._elastic_juicer_profile_seed_enabled is False
    assert dataset.elastic_juicer_captain_lifecycle is None
    assert dataset.elastic_juicer_metrics_sink is None
    assert dataset.elastic_juicer_control_service is None


@patch("data_juicer.core.elasticjuicer.control_service.create_ray_control_service")
@patch("data_juicer.core.elasticjuicer.async_metrics_sink.create_ray_metrics_sink")
def test_enabled_datasets_reuse_job_scoped_services_from_shared_config(create_sink, create_control):
    handle = MagicMock(name="metrics_sink")
    control_handle = MagicMock(name="control_service")
    create_sink.return_value = handle
    create_control.return_value = control_handle
    cfg = {"elastic_juicer_adaptive_batching": True, "job_id": "job-a"}
    first = RayDataset.__new__(RayDataset)
    second = RayDataset.__new__(RayDataset)

    first._configure_elastic_juicer(cfg)
    second._configure_elastic_juicer(cfg)

    create_sink.assert_called_once_with("job-a", max_events=2048)
    create_control.assert_called_once_with("job-a", lease_ttl_ms=60_000, profile_ttl_ms=1_800_000)
    assert first._elastic_juicer_metrics_sink is handle
    assert second._elastic_juicer_metrics_sink is handle
    assert first.elastic_juicer_metrics_sink is handle
    assert first.elastic_juicer_control_service is control_handle
    assert second.elastic_juicer_control_service is control_handle


@patch("data_juicer.core.elasticjuicer.async_metrics_sink.create_ray_metrics_sink")
def test_metrics_in_flight_limit_is_configurable_and_validated(create_sink):
    dataset = RayDataset.__new__(RayDataset)
    dataset._configure_elastic_juicer(
        {
            "elastic_juicer_adaptive_batching": True,
            "elastic_juicer_metrics_max_in_flight": 7,
            "job_id": "job-a",
            "_elastic_juicer_control_service": MagicMock(),
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
                "_elastic_juicer_control_service": MagicMock(),
            }
        )
    create_sink.assert_called_once_with("job-a", max_events=2048)


def test_sampler_and_control_intervals_must_be_positive():
    for key in ("elastic_juicer_sample_interval_sec", "elastic_juicer_control_poll_interval_sec"):
        dataset = RayDataset.__new__(RayDataset)
        with pytest.raises(ValueError, match=key):
            dataset._configure_elastic_juicer(
                {
                    "elastic_juicer_adaptive_batching": True,
                    "job_id": "job-a",
                    key: 0,
                }
            )


def test_captain_requires_adaptive_batching():
    dataset = RayDataset.__new__(RayDataset)
    with pytest.raises(ValueError, match="requires elastic_juicer_adaptive_batching"):
        dataset._configure_elastic_juicer(
            {
                "elastic_juicer_adaptive_batching": False,
                "elastic_juicer_captain_enabled": True,
            }
        )


@patch("data_juicer.core.elasticjuicer.captain.create_captain_lifecycle")
def test_product_captain_lifecycle_is_explicit_job_scoped_and_stoppable(create_lifecycle):
    dataset = _dataset(enabled=True)
    dataset._elastic_juicer_captain_enabled = True
    lifecycle = MagicMock()
    create_lifecycle.return_value = lifecycle
    cfg = {}

    assert dataset.start_elastic_juicer_captain(cfg) is lifecycle
    assert dataset.start_elastic_juicer_captain(cfg) is lifecycle
    create_lifecycle.assert_called_once_with(
        dataset._elastic_juicer_metrics_sink,
        dataset._elastic_juicer_control_service,
        cfg,
    )
    assert lifecycle.start.call_count == 2
    dataset.close_elastic_juicer_captain()
    lifecycle.close.assert_called_once_with()


def test_ray_executor_owns_captain_start_and_finally_close():
    source = inspect.getsource(RayExecutor.run)
    assert "start_elastic_juicer_captain" in source
    assert "finally:" in source
    assert "captain_lifecycle.close()" in source


@patch("data_juicer.core.executor.ray_executor.load_ops", return_value=[])
def test_ray_executor_closes_captain_when_dataset_processing_fails(load_ops, tmp_path):
    executor = RayExecutor.__new__(RayExecutor)
    executor.cfg = SimpleNamespace(
        process=[],
        dataset_path=None,
        dataset=None,
        op_fusion=False,
        export_path=str(tmp_path / "out.jsonl"),
    )
    executor.datasetbuilder = MagicMock()
    executor.op_env_manager = None
    executor.pipeline_dag = None
    executor.tracer = None
    executor.executor_type = "ray"
    executor.work_dir = str(tmp_path)
    executor.tmp_dir = str(tmp_path / "temp")
    executor.log_job_start = MagicMock()
    executor._initialize_dag_execution = MagicMock()
    dataset = MagicMock()
    dataset.data.columns.return_value = ["value"]
    dataset.data.count.return_value = 1
    dataset.process.side_effect = RuntimeError("processing failed")
    lifecycle = MagicMock()
    dataset.start_elastic_juicer_captain.return_value = lifecycle
    executor.datasetbuilder.load_dataset.return_value = dataset

    with pytest.raises(RuntimeError, match="processing failed"):
        executor.run(skip_export=True)

    lifecycle.close.assert_called_once_with()


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
            "stage_id": f"stage-000000:{operator.__class__.__name__}",
            "control_service": dataset._elastic_juicer_control_service,
            "control_poll_interval_sec": 0.1,
            "sample_interval_sec": 0.01,
            "profile_seed_enabled": False,
            "profile_seed_timeout_sec": 2.0,
            "partition_id": None,
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


def test_repeated_operator_instances_receive_unique_stable_stage_ids():
    dataset = _dataset(enabled=True)
    first = BatchedActorMapper(marker="first")
    second = BatchedActorMapper(marker="second")

    first_stage = dataset._next_elastic_juicer_stage_id(first)
    second_stage = dataset._next_elastic_juicer_stage_id(second)

    assert first_stage == f"stage-000000:{first.__class__.__name__}"
    assert second_stage == f"stage-000001:{second.__class__.__name__}"
    assert first_stage != second_stage


def test_stage_identity_is_stable_for_same_operator_across_datasets_in_one_job():
    cfg = {"elastic_juicer_adaptive_batching": True, "job_id": "job-a"}
    first_dataset = _dataset(enabled=True)
    second_dataset = _dataset(enabled=True)
    first_dataset._elastic_juicer_cfg = cfg
    second_dataset._elastic_juicer_cfg = cfg
    operator = BatchedActorMapper()
    other_operator = BatchedActorMapper()

    first_stage = first_dataset._next_elastic_juicer_stage_id(operator)
    second_stage = second_dataset._next_elastic_juicer_stage_id(operator)
    other_stage = second_dataset._next_elastic_juicer_stage_id(other_operator)

    # The same operator instance keeps one stable stage id across datasets
    # (e.g. per-partition RayDataset wrappers), so cross-partition profile
    # inheritance and Captain quotas address one logical stage.
    assert first_stage == second_stage
    assert first_stage.startswith("stage-000000:")
    # A distinct instance still receives a distinct identity.
    assert other_stage.startswith("stage-000001:")


def test_executor_stamped_identity_wins_over_the_first_seen_fallback():
    from data_juicer.core.elasticjuicer.stage_identity import assign_stage_identities

    dataset = _dataset(enabled=True)
    first = BatchedActorMapper(marker="first")
    second = BatchedActorMapper(marker="second")
    manifest = assign_stage_identities([first, second])

    # The deterministic executor-assigned identity survives any dataset-side
    # resolution order, so checkpoint-resume slicing cannot drift stage ids.
    assert dataset._next_elastic_juicer_stage_id(second) == manifest["stages"][1]["stage_id"]
    assert dataset._next_elastic_juicer_stage_id(first) == manifest["stages"][0]["stage_id"]
    assert manifest["stages"][0]["stage_id"].startswith("stage-0000-occ0-")
    assert manifest["stages"][0]["stage_id"].endswith(f":{first.__class__.__name__}")
    # Different init kwargs produce different fingerprints, same-config repeats
    # are disambiguated by the occurrence counter instead.
    assert manifest["stages"][0]["op_fingerprint"] != manifest["stages"][1]["op_fingerprint"]

    same_config = [BatchedActorMapper(marker="twin"), BatchedActorMapper(marker="twin")]
    twin_manifest = assign_stage_identities(same_config)
    assert twin_manifest["stages"][0]["occurrence"] == 0
    assert twin_manifest["stages"][1]["occurrence"] == 1
    assert twin_manifest["stages"][0]["stage_id"] != twin_manifest["stages"][1]["stage_id"]
