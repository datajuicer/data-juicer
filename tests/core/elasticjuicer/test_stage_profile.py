import json
from types import SimpleNamespace
from unittest.mock import Mock

import pyarrow as pa
import pytest
from core.elasticjuicer.test_ray_adaptive_mapper import ThresholdMapper, make_batch
from jsonargparse import Namespace

from data_juicer.config.config import build_base_parser, init_setup_from_cfg
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.elasticjuicer.profile_context import make_profile_context
from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor
from data_juicer.core.elasticjuicer.stage_identity import assign_stage_identities
from data_juicer.core.elasticjuicer.stage_profile import (
    StageProfileSession,
    StageProfileStore,
)
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor


class ProfileThresholdMapper(ThresholdMapper):
    _name = "ej_profile_threshold_mapper"
    elastic_juicer_cost_columns = ("cost",)

    def __init__(self, resource_class="fixture", memory_budget_bytes=64, **kwargs):
        super().__init__(**kwargs)
        self.resource_class = resource_class
        self.memory_budget_bytes = memory_budget_bytes

    def elastic_juicer_profile_resource_context(self):
        return {"resource_class": self.resource_class, "memory_budget_bytes": self.memory_budget_bytes}


class LocalHandle:
    """Exercise the actual service/client boundary without starting Ray."""

    def __init__(self, store):
        for method in ("read", "publish", "snapshot", "ready"):
            setattr(self, method, SimpleNamespace(remote=getattr(store, method)))


def profile_batch(n=16, cost="small"):
    batch = make_batch(n)
    batch["cost"] = [cost] * n
    return batch


def profile_op(**kwargs):
    options = dict(batch_size=8, num_proc=1, num_cpus=1, num_gpus=0.25, ray_execution_mode="actor")
    options.update(kwargs)
    return ProfileThresholdMapper(**options)


@pytest.fixture
def setup_profile(monkeypatch, tmp_path):
    import ray

    monkeypatch.setattr(ray, "get", lambda value, timeout: value)
    op = profile_op()
    manifest = assign_stage_identities([op])
    session = StageProfileSession(manifest, str(tmp_path))
    store = StageProfileStore(session.job_id, manifest)
    session.store = LocalHandle(store)

    def actor(operator=None):
        operator = operator or op
        stage = manifest["stages"][0]["stage_id"]
        return RayAdaptiveMapperActor(
            type(operator),
            operator._init_args,
            operator._init_kwargs,
            operator.batch_size,
            stage,
            **session.actor_arguments(operator, stage),
        )

    first = actor()
    context, _ = make_profile_context(profile_batch(), first.op, first.controller, first._profile_resource_envelope)
    request = {**first._profile_request, "context": context}
    return SimpleNamespace(op=op, manifest=manifest, session=session, store=store, actor=actor, request=request)


def publish(store, request, successes=(), ooms=(), actor="source"):
    return store.publish(request, {"successes": list(successes), "ooms": list(ooms)}, actor)


def test_merge_uses_actual_success_below_exclusive_oom_bound(setup_profile):
    fixture = setup_profile
    assert publish(fixture.store, fixture.request, [2, 8], [6])
    assert publish(fixture.store, fixture.request, [3], [4], "second")
    prior = fixture.store.read(fixture.request)
    assert (prior["safe_batch_size"], prior["oom_upper_bound"]) == (3, 4)
    assert prior["source_actor_ids"] == ["source", "second"]
    # No observed success below the failing size: never invent success at 5.
    store = StageProfileStore(fixture.session.job_id, fixture.manifest)
    publish(store, fixture.request, [8], [6])
    assert store.read(fixture.request)["safe_batch_size"] is None


def test_each_evidence_expires_without_rejuvenation_by_other_updates(setup_profile):
    fixture = setup_profile
    clock = [0.0]
    store = StageProfileStore(fixture.session.job_id, fixture.manifest, ttl_seconds=10, clock=lambda: clock[0])
    publish(store, fixture.request, [2], [4])
    clock[0] = 9
    publish(store, fixture.request, [3])
    clock[0] = 10
    prior = store.read(fixture.request)
    assert prior["safe_batch_size"] == 3
    assert prior["oom_upper_bound"] is None
    clock[0] = 20
    publish(store, fixture.request, [8])
    assert store.read(fixture.request)["safe_batch_size"] == 8
    assert store.snapshot()["metrics"]["expired"] == 1
    clock[0] = 30
    assert store.read(fixture.request) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("job_id", "different"),
        ("pipeline_fingerprint", "different"),
        ("stage_id", "different"),
        ("op_fingerprint", "different"),
        ("context", "{}"),
        ("context", None),
    ],
)
def test_incompatible_or_malformed_requests_fail_closed(setup_profile, field, value):
    fixture = setup_profile
    publish(fixture.store, fixture.request, [2], [4])
    request = {**fixture.request, field: value}
    assert fixture.store.read(request) is None
    assert not publish(fixture.store, request, [3])
    assert fixture.store.read(fixture.request)["safe_batch_size"] == 2
    assert sum(count for key, count in fixture.store.metrics.items() if key.startswith("rejected:")) == 2


@pytest.mark.parametrize("successes,ooms", [([True], []), ([9], []), ([0], []), ([], ["4"]), ([], [0]), ([], [])])
def test_invalid_observations_cannot_poison_profile(setup_profile, successes, ooms):
    fixture = setup_profile
    publish(fixture.store, fixture.request, [2], [4])
    assert not publish(fixture.store, fixture.request, successes, ooms)
    prior = fixture.store.read(fixture.request)
    assert (prior["safe_batch_size"], prior["oom_upper_bound"]) == (2, 4)


def test_store_bounds_entries(setup_profile):
    fixture = setup_profile
    store = StageProfileStore(fixture.session.job_id, fixture.manifest, max_entries=2)
    for cost in ("a", "b", "c"):
        actor = fixture.actor()
        context, _ = make_profile_context(
            profile_batch(cost=cost), actor.op, actor.controller, actor._profile_resource_envelope
        )
        publish(store, {**fixture.request, "context": context}, [2], [4])
    assert len(store.snapshot()["profiles"]) == 2
    assert store.metrics["evicted"] == 1


def test_bounded_size_retention_keeps_proven_small_success(setup_profile):
    fixture = setup_profile
    data = json.loads(fixture.request["context"])
    data["limits"] = [1, 128, 128]
    request = {**fixture.request, "context": json.dumps(data, sort_keys=True, separators=(",", ":"))}
    publish(fixture.store, request, range(1, 33))
    publish(fixture.store, request, range(33, 65))
    publish(fixture.store, request, ooms=[5])
    record = next(iter(fixture.store.records.values()))
    assert len(record["successes"]) == 32
    assert fixture.store.read(request)["safe_batch_size"] == 1  # Retained actual proof, never synthesized 4.


@pytest.mark.parametrize("arrow", [False, True])
def test_new_actor_inherits_and_preserves_output(setup_profile, arrow):
    fixture = setup_profile
    batch = pa.Table.from_pydict(profile_batch()) if arrow else profile_batch()
    cold = fixture.actor()
    expected = cold(batch)
    warm = fixture.actor()
    actual = warm(batch)
    assert actual == expected
    assert actual["meta"] == [{"attempts": 1}] * 16
    assert len(cold.op.calls[0]) == 8
    assert len(warm.op.calls[0]) == 2
    assert warm.mapper.oom_retries < cold.mapper.oom_retries
    assert warm.profile_diagnostics["seeded"] == 1
    snapshot = fixture.store.snapshot()
    assert snapshot["metrics"]["hits"] == 1
    assert warm.actor_id in snapshot["profiles"][0]["seeded_actor_ids"]
    assert len(snapshot["profiles"][0]["source_actor_ids"]) == 2


@pytest.mark.parametrize("change", ["cost", "budget", "resource_class", "reservation", "batch_cap"])
def test_context_changes_reject_inheritance(setup_profile, change):
    fixture = setup_profile
    fixture.actor()(profile_batch())
    actor = fixture.actor()
    batch = profile_batch()
    if change == "cost":
        batch["cost"] = ["large"] * 16
    elif change == "budget":
        actor.op.memory_budget_bytes = 32
    elif change == "resource_class":
        actor.op.resource_class = "different-device"
    elif change == "reservation":
        actor._profile_resource_envelope["num_gpus"] = 0.5
    else:
        actor.controller.max_batch_size = 4
        actor.controller.hard_limit = 4
        actor.controller.current_batch_size = 4
    actor(batch)
    assert not actor.profile_diagnostics["seeded"]
    assert fixture.store.metrics["hits"] == 0


@pytest.mark.parametrize("invalid", ["mixed", "missing", "null", "nan", "opaque", "no_contract", "bad_resource"])
def test_unknown_context_neither_reads_nor_publishes(setup_profile, invalid):
    fixture = setup_profile
    actor = fixture.actor()
    batch = profile_batch()
    if invalid == "mixed":
        batch["cost"][-1] = "large"
    elif invalid == "missing":
        batch.pop("cost")
    elif invalid == "null":
        batch["cost"][0] = None
    elif invalid == "nan":
        batch["cost"] = [float("nan")] * 16
    elif invalid == "opaque":
        batch["cost"] = [object()] * 16
    elif invalid == "no_contract":
        actor.op.elastic_juicer_cost_columns = ()
    else:
        actor.op.memory_budget_bytes = True
    actor(batch)
    assert fixture.store.metrics["reads"] == fixture.store.metrics["publishes"] == 0
    assert sum(actor.profile_diagnostics.values()) == 1


def test_only_before_first_local_observation_can_seed_and_local_oom_wins(setup_profile):
    fixture = setup_profile
    fixture.actor()(profile_batch())
    warm = fixture.actor()
    warm.op.threshold = 1  # Undeclared runtime change: the prior remains advisory.
    warm(profile_batch())
    assert warm.profile_diagnostics["seeded"] == 1
    assert warm.controller.state.oom_upper_bound == 2
    assert warm.controller.state.current_batch_size == 1
    warm(profile_batch(cost="large"))
    assert warm.profile_diagnostics["seeded"] == 1
    assert warm.controller.state.oom_upper_bound == 2
    assert fixture.store.metrics["reads"] == 2
    # Cost-unknown first call observes OOM locally; later eligible input cannot
    # replace that fresh evidence with another actor's less conservative prior.
    actor = fixture.actor()
    missing = profile_batch()
    missing.pop("cost")
    actor(missing)
    before = fixture.store.metrics["reads"]
    actor(profile_batch())
    assert fixture.store.metrics["reads"] == before


@pytest.mark.parametrize("failure", ["ordinary", "rows", "tags", "schema", "resource_change"])
def test_failed_or_context_changed_outer_batch_does_not_publish_partial_observations(setup_profile, failure):
    fixture = setup_profile
    actor = fixture.actor()
    actor.op.skip_op_error = True
    if failure in ("ordinary", "rows", "tags"):
        actor.op.failure = failure
    else:
        original = actor.op.process_batched

        def process(batch):
            output = original(batch)
            if failure == "schema" and batch["id"][0] >= 2:
                output["extra"] = [1] * len(batch["id"])
            if failure == "resource_change":
                actor.op.memory_budget_bytes += 1
            return output

        actor.op.process_batched = process
    if failure in ("rows", "tags", "schema"):
        with pytest.raises(ValueError):
            actor(profile_batch())
    else:
        actor(profile_batch())
    assert fixture.store.metrics["publishes"] == 0
    assert fixture.store.snapshot()["profiles"] == []


@pytest.mark.parametrize("mutation", ["schema", "identity", "bounds", "ttl", "bool"])
def test_corrupt_prior_falls_back_without_blocking_data(setup_profile, mutation):
    fixture = setup_profile
    publish(fixture.store, fixture.request, [2], [4])
    prior = fixture.store.read(fixture.request)
    if mutation == "schema":
        prior["schema_version"] = 9
    elif mutation == "identity":
        prior["op_fingerprint"] = "corrupt"
    elif mutation == "bounds":
        prior["safe_batch_size"] = 4
    elif mutation == "ttl":
        prior["remaining_ttl_seconds"] = 0
    else:
        prior["safe_batch_size"] = True
    actor = fixture.actor()
    actor._profile_store.read.remote = lambda request: prior
    assert actor(profile_batch())["id"] == list(range(16))
    assert actor.profile_diagnostics["rejected_prior"] == 1
    assert len(actor.op.calls[0]) == 8


def test_service_outage_uses_local_control_with_one_wait_per_actor(setup_profile):
    actor = setup_profile.actor()
    actor._profile_store.read.remote = Mock(side_effect=TimeoutError)
    assert actor(profile_batch())["id"] == list(range(16))
    assert actor(profile_batch())["id"] == list(range(16))
    assert actor.profile_diagnostics["rpc_errors"] == 1
    assert actor.mapper.oom_retries > 0


def test_publication_failure_does_not_change_delivered_output(setup_profile):
    actor = setup_profile.actor()
    actor._profile_store.publish.remote = Mock(side_effect=RuntimeError("service lost"))
    actual = actor(profile_batch())
    assert actual["id"] == list(range(16))
    assert actual["meta"] == [{"attempts": 1}] * 16
    assert actor.profile_diagnostics["rpc_errors"] == 1
    assert actor._profile_store is None


def test_dispatch_carries_job_and_full_stage_identity_without_changing_resources(setup_profile):
    fixture = setup_profile
    cfg = Namespace(elastic_juicer_adaptive_batching=True, _stage_profile_session=fixture.session)
    dataset = RayDataset(Mock(), cfg=cfg)
    data = dataset.data
    dataset._run_single_op(fixture.op, {"id", "meta", "cost"})
    options = data.map_batches.call_args.kwargs
    args = options["fn_constructor_kwargs"]
    assert args["profile_request"]["op_fingerprint"] == fixture.manifest["stages"][0]["op_fingerprint"]
    assert args["profile_request"]["job_id"] == fixture.session.job_id
    assert args["profile_store"] is fixture.session.store
    assert options["batch_size"] == 8 and options["num_gpus"] == 0.25


def test_profile_switch_defaults_off_and_requires_supported_executor_and_adaptive_batching():
    parser = build_base_parser()
    assert parser.get_defaults().elastic_juicer_profile_seed is False
    assert parser.parse_args(["--auto", "--elastic_juicer_profile_seed", "true"]).elastic_juicer_profile_seed
    for cfg in (
        Namespace(executor_type="ray_partitioned", elastic_juicer_profile_seed=True),
        Namespace(executor_type="ray", elastic_juicer_profile_seed=True, elastic_juicer_adaptive_batching=True),
    ):
        with pytest.raises(ValueError, match="elastic_juicer_profile_seed"):
            init_setup_from_cfg(cfg)


def test_driver_always_closes_owned_session_after_export_error(tmp_path):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.tmp_dir = str(tmp_path / "tmp")
    executor.cfg = Namespace()
    session = Mock()
    session.finish.return_value = {"profiles": []}
    executor._stage_profile_session = session
    executor.cfg._stage_profile_session = session
    executor._run_impl = Mock(side_effect=RuntimeError("export failed"))
    with pytest.raises(RuntimeError, match="export failed"):
        executor.run()
    session.finish.assert_called_once()
    assert executor.cfg._stage_profile_session is None
    assert executor._stage_profile_session is None


def test_finalization_failure_does_not_mask_processing_error(tmp_path):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.tmp_dir = str(tmp_path / "tmp")
    executor.cfg = Namespace()
    executor._stage_profile_session = Mock()
    executor._stage_profile_session.finish.side_effect = OSError("profile write failed")
    executor._run_impl = Mock(side_effect=RuntimeError("processing failed"))
    with pytest.raises(RuntimeError, match="processing failed"):
        executor.run()
    assert executor.cfg._resolved_stage_profile_summary is None
    assert executor._stage_profile_session is None


def test_default_off_does_not_create_service():
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = Namespace(elastic_juicer_profile_seed=False)
    executor._start_stage_profiles([profile_op()])
    assert not hasattr(executor, "_stage_profile_session")


def test_atomic_audit_is_not_a_cross_run_seed(setup_profile, tmp_path, monkeypatch):
    import ray

    fixture = setup_profile
    publish(fixture.store, fixture.request, [2], [4])
    killed = Mock()
    monkeypatch.setattr(ray, "kill", killed)
    snapshot = fixture.session.finish()
    assert json.loads((tmp_path / "elastic_juicer_stage_profiles.json").read_text()) == snapshot
    assert not list(tmp_path.glob(".stage-profile-*"))
    killed.assert_called_once()
    assert fixture.session.store is None
    fresh = StageProfileSession(fixture.manifest, str(tmp_path))
    assert fresh.job_id != fixture.session.job_id
    new_store = StageProfileStore(fresh.job_id, fixture.manifest)
    assert new_store.read({**fixture.request, "job_id": fresh.job_id}) is None
    assert new_store.read(fixture.request) is None


def test_profile_artifact_contains_hashes_instead_of_raw_cost_values(setup_profile):
    fixture = setup_profile
    fixture.actor()(profile_batch(cost="sensitive-input-path"))
    assert "sensitive-input-path" not in json.dumps(fixture.store.snapshot())


def test_failed_atomic_write_keeps_previous_artifact_and_releases_service(setup_profile, tmp_path, monkeypatch):
    import ray

    import data_juicer.core.elasticjuicer.stage_profile as module

    fixture = setup_profile
    path = tmp_path / "elastic_juicer_stage_profiles.json"
    path.write_text("previous audit")
    monkeypatch.setattr(module.os, "replace", Mock(side_effect=OSError("disk error")))
    killed = Mock()
    monkeypatch.setattr(ray, "kill", killed)
    assert fixture.session.finish() is None
    assert path.read_text() == "previous audit"
    assert not list(tmp_path.glob(".stage-profile-*"))
    killed.assert_called_once()
    assert fixture.session.store is None


def test_service_startup_failure_closes_its_actor_without_starting_ray(setup_profile, monkeypatch):
    import ray

    fixture = setup_profile
    remote = Mock()
    handle = Mock()
    remote.return_value.remote.return_value = handle
    monkeypatch.setattr(ray, "remote", Mock(return_value=remote))
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get", Mock(side_effect=TimeoutError))
    init, killed = Mock(), Mock()
    monkeypatch.setattr(ray, "init", init)
    monkeypatch.setattr(ray, "kill", killed)
    session = StageProfileSession(fixture.manifest, fixture.session.work_dir)
    assert not session.start()
    assert session.store is None
    init.assert_not_called()
    killed.assert_called_once_with(handle, no_restart=True)
