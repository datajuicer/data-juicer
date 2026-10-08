"""Real Ray transport/reconstruction and public execution-group reuse.

Logical GPU reservations and injected OOMs; no physical CUDA allocation.
"""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from core.elasticjuicer.test_stage_profile import (
    ProfileThresholdMapper,
    profile_batch,
    profile_op,
)

from data_juicer.config import init_configs
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor
from data_juicer.core.elasticjuicer.stage_identity import assign_stage_identities
from data_juicer.core.elasticjuicer.stage_profile import StageProfileSession
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.ops.base_op import OPERATORS


class InspectableAdaptiveActor(RayAdaptiveMapperActor):
    def diagnostics(self):
        return {
            "actor_id": self.actor_id,
            "first_batch_size": len(self.op.calls[0]),
            "oom_retries": self.mapper.oom_retries,
            "seeded": self.profile_diagnostics["seeded"],
        }


class AuditedProfileMapper(ProfileThresholdMapper):
    _name = "ej_audited_profile_mapper"

    def __init__(self, diagnostics_path=None, **kwargs):
        super().__init__(**kwargs)
        self.diagnostics_path = diagnostics_path

    def process_batched(self, samples):
        # Test-only attempt telemetry; output/model state does not depend on it.
        with open(self.diagnostics_path, "a") as stream:
            stream.write(
                json.dumps(
                    {
                        "stage_id": self._elastic_juicer_stage_identity,
                        "pid": os.getpid(),
                        "batch_size": len(samples["id"]),
                        "oom": len(samples["id"]) > self.threshold,
                    }
                )
                + "\n"
            )
        return super().process_batched(samples)


@pytest.fixture(scope="module")
def local_ray():
    ray = pytest.importorskip("ray")
    if ray.is_initialized():
        pytest.skip("Requires an isolated local Ray cluster")
    ray.init(address="local", num_cpus=4, num_gpus=1, include_dashboard=False, object_store_memory=100 * 1024**2)
    try:
        yield ray
    finally:
        ray.shutdown()


def test_killed_actor_replacement_inherits_from_live_job_service(local_ray, tmp_path):
    ray = local_ray
    op = profile_op()
    manifest = assign_stage_identities([op])
    stage_id = manifest["stages"][0]["stage_id"]
    session = StageProfileSession(manifest, str(tmp_path))
    assert session.start()
    actor_class = ray.remote(num_cpus=1, num_gpus=0.25)(InspectableAdaptiveActor)

    def create_actor(active_session):
        return actor_class.remote(
            type(op),
            op._init_args,
            op._init_kwargs,
            op.batch_size,
            stage_id,
            **active_session.actor_arguments(op, stage_id),
        )

    actors = []
    independent = StageProfileSession(manifest, str(tmp_path / "independent"))
    try:
        cold = create_actor(session)
        actors.append(cold)
        expected = ray.get(cold.__call__.remote(profile_batch()))
        cold_info = ray.get(cold.diagnostics.remote())
        ray.kill(cold, no_restart=True)
        replacement = create_actor(session)
        actors.append(replacement)
        assert ray.get(replacement.__call__.remote(profile_batch())) == expected
        warm_info = ray.get(replacement.diagnostics.remote())
        assert cold_info["actor_id"] != warm_info["actor_id"]
        assert (cold_info["first_batch_size"], warm_info["first_batch_size"]) == (8, 2)
        assert warm_info["oom_retries"] < cold_info["oom_retries"]
        assert warm_info["seeded"] == 1
        snapshot = ray.get(session.store.snapshot.remote())
        assert warm_info["actor_id"] in snapshot["profiles"][0]["seeded_actor_ids"]
        assert independent.start()
        isolated = create_actor(independent)
        actors.append(isolated)
        assert ray.get(isolated.__call__.remote(profile_batch())) == expected
        isolated_info = ray.get(isolated.diagnostics.remote())
        assert isolated_info["first_batch_size"] == 8
        assert isolated_info["seeded"] == 0
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        session.finish()
        independent.finish()


def test_public_execution_groups_seed_on_off_and_completed_checkpoint_reuse(local_ray, tmp_path, monkeypatch):
    OPERATORS.register_module(AuditedProfileMapper._name)(AuditedProfileMapper)
    rows = [{"id": i, "text": str(i), "meta": {"attempts": 0}, "cost": "small"} for i in range(64)]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    results, oom_counts = [], []
    for enabled in (False, True):
        directory = tmp_path / ("seed_on" if enabled else "seed_off")
        directory.mkdir()
        diagnostics = directory / "attempts.jsonl"
        export = directory / "result.jsonl"
        op_cfg = {
            AuditedProfileMapper._name: {
                "batch_size": 8,
                "threshold": 2,
                "num_proc": 1,
                "num_cpus": 1,
                "num_gpus": 0.5,
                "ray_execution_mode": "actor",
                "adaptive_batching": True,
                "diagnostics_path": str(diagnostics),
            }
        }
        recipe = {
            "project_name": "stage-profile-e2e",
            "executor_type": "ray_partitioned",
            "dataset_path": str(source),
            "export_path": str(export),
            "work_dir": str(directory / "work"),
            "strict_preflight": False,
            "auto_op_parallelism": False,
            "elastic_juicer_adaptive_batching": True,
            "elastic_juicer_profile_seed": enabled,
            "partition": {
                "mode": "manual",
                "num_of_partitions": 4,
                "execution_group_size": 1,
                "max_concurrent_partitions": 1,
                "gpu_preflight_enabled": False,
            },
            "checkpoint": {"enabled": True, "strategy": "every_op"},
            "process": [op_cfg, deepcopy(op_cfg)],
        }
        config_path = directory / "recipe.yaml"
        config_path.write_text(yaml.safe_dump(recipe))
        cfg = init_configs(["--config", str(config_path)])
        executor = PartitionedRayExecutor(cfg)
        output = executor.run()
        actual = sorted(output.data.take_all(), key=lambda row: row["id"])
        assert [row["id"] for row in actual] == list(range(64))
        assert [row["value"] for row in actual] == [i * 2 for i in range(64)]
        assert all(row["meta"]["attempts"] == 2 for row in actual)
        assert all("__data_juicer_logical_partition_id__" not in row for row in actual)
        results.append(actual)
        attempts = [json.loads(line) for line in diagnostics.read_text().splitlines()]
        oom_counts.append(sum(attempt["oom"] for attempt in attempts))
        artifact = Path(cfg.work_dir) / "elastic_juicer_stage_profiles.json"
        if enabled:
            snapshot = json.loads(artifact.read_text())
            assert snapshot["metrics"]["hits"] >= 2
            profiles = snapshot["profiles"]
            assert len({profile["stage_id"] for profile in profiles}) == 2
            assert all(len(profile["source_actor_ids"]) >= 2 for profile in profiles)
            assert all(profile["seeded_actor_ids"] for profile in profiles)
            assert cfg._stage_profile_session is None

            def forbidden_reprocessing(*args, **kwargs):
                raise AssertionError("Completed checkpoint rows must not execute again")

            with monkeypatch.context() as patch:
                patch.setattr(RayDataset, "process", forbidden_reprocessing)
                cfg._resume_requested = True
                executor._is_resuming = True
                ops = executor._prepare_operators()
                restored = executor._process_with_simple_partitioning(executor.datasetbuilder.load_dataset(), ops)
                assert sorted(restored.data.take_all(), key=lambda row: row["id"]) == actual
            assert len(diagnostics.read_text().splitlines()) == len(attempts)
        else:
            assert not artifact.exists()
        files = [export] if export.is_file() else list(export.rglob("*.json*"))
        exported = [json.loads(line) for path in files for line in path.read_text().splitlines() if line]
        assert sorted(row["id"] for row in exported) == list(range(64))
    assert results[0] == results[1]
    assert 0 < oom_counts[1] < oom_counts[0]
