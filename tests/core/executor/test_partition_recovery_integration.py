"""End-to-end checks of side-path checkpointing on a local Ray cluster."""

import json
import os
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow as pa
import ray
import yaml
from jsonargparse import Namespace

from data_juicer.config import init_configs
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor import ray_partition_runner
from data_juicer.core.executor.partition_protocol import (
    COMMIT_DIR,
    SOURCE_SEQ,
    Commit,
    LocalStore,
    ProtocolError,
    commit_key,
    with_keys,
    write_tables,
)
from data_juicer.core.executor.ray_executor_partitioned import PartitionedRayExecutor
from data_juicer.ops.base_op import Mapper
from data_juicer.ops.deduplicator.document_deduplicator import DocumentDeduplicator
from data_juicer.ops.filter.alphanumeric_filter import AlphanumericFilter
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.fused_sequential_batch_op import FusedSequentialBatchOp
from data_juicer.ops.mapper.whitespace_normalization_mapper import (
    WhitespaceNormalizationMapper,
)
from data_juicer.ops.pipeline.ray_repartition_pipeline import RayRepartitionPipeline
from data_juicer.utils import logger_utils
from data_juicer.utils.ckpt_utils import RayCheckpointManager
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import TEST_TAG


class DuplicateMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        return {key: [value for item in values for value in (item, item)] for key, values in samples.items()}


class OddMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        keep = [i for i, value in enumerate(samples["id"]) if value % 2]
        output = {key: [values[i] for i in keep] for key, values in samples.items()}
        output["mapped"] = [True] * len(keep)
        return output


class BatchSizeMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        samples["batch_rows"] = [len(samples["id"])] * len(samples["id"])
        return samples


class LoggingMapper(Mapper):
    """Record processed ids; optionally fail once when it sees ``fail_id``."""

    _batched_op = True
    _name = "logging_mapper"

    def __init__(self, log_path, fail_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path
        self.fail_id = fail_id

    def process_batched(self, samples):
        marker = Path(self.log_path + ".failed")
        if self.fail_id in samples["id"] and not marker.exists():
            marker.write_text("x")
            raise RuntimeError("injected mapper failure")
        with open(self.log_path, "a") as stream:
            stream.write(json.dumps(samples["id"]) + "\n")
        return samples


class LoggingTextLengthFilter(TextLengthFilter):
    def __init__(self, log_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = log_path

    def compute_stats_batched(self, samples):
        with open(self.log_path, "a") as stream:
            stream.write(json.dumps(samples["id"]) + "\n")
        return super().compute_stats_batched(samples)


class OutputFormatMapper(Mapper):
    _batched_op = True

    def __init__(self, output_format, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_format = output_format

    def process_batched(self, samples):
        if self.output_format == "generator":
            return ({key: [values[i]] for key, values in samples.items()} for i in range(len(samples["id"])))
        if self.output_format == "dataframe":
            import pandas as pd

            return pd.DataFrame(samples)
        import numpy as np

        samples["tensor"] = np.array([np.full((2, 2), value) for value in samples["id"]])
        return samples


class ExitAfterFirstCommitActor(ray_partition_runner.CheckpointStageActor):
    """Kill the worker after the first committed batch has been yielded."""

    def __call__(self, batch):
        for table in super().__call__(batch):
            yield table
            marker = Path(self.context.store.root, "exited-once")
            try:
                with open(marker, "x"):
                    pass
            except FileExistsError:
                continue
            os._exit(86)


class FailedPublicationStore(LocalStore):
    def __init__(self, root, published=False):
        super().__init__(root)
        self.published = published

    def put_if_absent(self, key, data):
        if key.startswith(COMMIT_DIR + "/"):
            if self.published:
                super().put_if_absent(key, data)
            raise OSError("injected manifest failure")
        return super().put_if_absent(key, data)


class AmbiguousPublicationStore:
    """Remote-like failures do not prove that a write cannot still publish."""

    def __init__(self, root):
        self.delegate = FailedPublicationStore(root)

    def get(self, key):
        return self.delegate.get(key)

    def list(self, key):
        return self.delegate.list(key)

    def put_if_absent(self, key, data):
        return self.delegate.put_if_absent(key, data)

    def delete(self, key):
        return self.delegate.delete(key)

    def failed_write_is_unpublished(self, key):
        return False


def exit_after_first_commit_task(batch, context, plan):
    for table in ray_partition_runner.run_checkpoint_task(batch, context, plan):
        yield table
        try:
            with open(Path(context.store.root, "exited-once"), "x"):
                pass
        except FileExistsError:
            continue
        os._exit(86)


def _executor(tmp_path, *, resume=False, size=2):
    executor = PartitionedRayExecutor.__new__(PartitionedRayExecutor)
    executor.cfg = Namespace(auto_op_parallelism=False)
    executor.ckpt_manager = RayCheckpointManager(str(tmp_path / "checkpoints"), checkpoint_enabled=True)
    executor.job_id = "checkpoint-integration"
    executor.partition_size_cfg = size
    executor._is_resuming = resume
    return executor


def _source(executor, *, changed=False, blocks=1):
    rows = [
        {"text": " A\tB ", "id": 0},
        {"text": "!!!", "id": 1},
        {"text": " C ", "id": 2},
        {"text": "???", "id": 3},
    ]
    if changed:
        rows[0]["text"] = "different input"
    return RayDataset(ray.data.from_items(rows, override_num_blocks=blocks), cfg=executor.cfg)


def _run(executor, ops, source=None):
    dataset, recovery = executor._process_with_partition_recovery(source or _source(executor), ops)
    try:
        rows = dataset.data.take_all()
        recovery.finish()
    finally:
        recovery.close()
    return rows


def _store(tmp_path):
    return LocalStore(str(tmp_path / "checkpoints" / "partition_recovery"))


def _commits(tmp_path):
    store = _store(tmp_path)
    return sorted((Commit.from_bytes(store.get(key)) for key in store.list(COMMIT_DIR)), key=lambda c: c.seq)


def _logged_ids(path):
    if not os.path.exists(path):
        return []
    return [value for line in Path(path).read_text().splitlines() for value in json.loads(line)]


def _by_id(rows):
    return sorted(rows, key=lambda row: row["id"])


def _read_json_rows(path):
    return [json.loads(line) for shard in sorted(Path(path).glob("*.json")) for line in shard.read_text().splitlines()]


def _write_job(tmp_path, process, *, extra=None):
    input_path = tmp_path / "input.jsonl"
    with input_path.open("w") as stream:
        for text in [" A\tB ", "!!!", " C ", " C "]:
            stream.write(json.dumps({"text": text}) + "\n")
    config = {
        "project_name": "checkpoint-integration",
        "executor_type": "ray_partitioned",
        "dataset_path": str(input_path),
        "export_path": str(tmp_path / "output.jsonl"),
        "work_dir": str(tmp_path / "work"),
        "checkpoint_dir": str(tmp_path / "work" / "checkpoints"),
        "auto_op_parallelism": False,
        "strict_preflight": False,
        "checkpoint": {"enabled": True, "strategy": "every_op"},
        "partition": {"size": 2},
        "process": process,
        **(extra or {}),
    }
    config_path = tmp_path / "job.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _exported_texts(tmp_path):
    export_path = tmp_path / "output.jsonl"
    shards = sorted(export_path.glob("*.json")) if export_path.is_dir() else [export_path]
    return sorted(json.loads(line)["text"] for shard in shards for line in shard.read_text().splitlines())


class PartitionRecoveryIntegrationTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="dj-recovery-test-")
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)
        # Keep init_configs from leaving global file sinks in a deleted fixture.
        # The executor's actual checkpoint and event JSONL files are still written.
        patcher = patch.object(logger_utils, "LOGGER_SETUP", True)
        self.addCleanup(patcher.stop)
        patcher.start()
        method = getattr(self, self._testMethodName)
        if "ray" in getattr(method, "__test_tags__", ()):
            # Fault injection must not attach to the CI runner's shared cluster.
            ray.shutdown()
            self.addCleanup(ray.shutdown)
            ray.init(address="local", num_cpus=4, include_dashboard=False, log_to_driver=False)

    def _prepare_fused_filters(self, *, resume=False, min_len=1):
        executor = _executor(self.tmp_path, resume=resume)
        executor.cfg.op_fusion = True
        executor.cfg.fusion_strategy = "greedy"
        executor.cfg.process = [
            {"average_line_length_filter": {"min_len": min_len, "num_proc": 1, "auto_op_parallelism": False}},
            {"maximum_line_length_filter": {"min_len": 1, "num_proc": 1, "auto_op_parallelism": False}},
        ]
        ops = executor._prepare_operators()
        self.assertEqual(len(ops), 1)
        ops[0].num_proc = 1
        return executor, ops

    @TEST_TAG("ray")
    def test_fused_filter_resume_reconstructs_child_operators(self):
        executor, ops = self._prepare_fused_filters()
        expected = _run(executor, ops)
        self.assertEqual(sorted(row["id"] for row in expected), [0, 1, 2, 3])
        commits = _commits(self.tmp_path)

        resumed, rebuilt = self._prepare_fused_filters(resume=True)
        self.assertIsNot(ops[0].fused_filters[0], rebuilt[0].fused_filters[0])
        self.assertEqual(_by_id(_run(resumed, rebuilt)), _by_id(expected))
        self.assertEqual(_commits(self.tmp_path), commits)

        changed, changed_ops = self._prepare_fused_filters(resume=True, min_len=4)
        with self.assertRaisesRegex(ProtocolError, "changed since the checkpoint"):
            _run(changed, changed_ops)

    @TEST_TAG("ray")
    def test_fused_mapper_resume_reconstructs_child_operators(self):
        def make_ops(text_key="text"):
            return [
                FusedSequentialBatchOp(
                    fused_ops=[WhitespaceNormalizationMapper(text_key=text_key, num_proc=1)],
                    group_name="checkpoint-fused-mapper",
                    num_proc=1,
                    auto_op_parallelism=False,
                )
            ]

        original = make_ops()
        expected = _run(_executor(self.tmp_path), original)
        self.assertEqual(_by_id(expected)[0]["text"], "A B")
        rebuilt = make_ops()
        self.assertIsNot(original[0]._init_kwargs["fused_ops"][0], rebuilt[0]._init_kwargs["fused_ops"][0])
        self.assertEqual(_by_id(_run(_executor(self.tmp_path, resume=True), rebuilt)), _by_id(expected))
        with self.assertRaisesRegex(ProtocolError, "changed since the checkpoint"):
            _run(_executor(self.tmp_path, resume=True), make_ops("other_text"))

    def _check_checkpointed_rows_match_native_and_resume_reuses_commits(self, mode):
        log_path = str(self.tmp_path / "calls.jsonl")
        kwargs = {"num_proc": 1, "ray_execution_mode": mode}
        ops = [
            LoggingMapper(log_path, batch_size=1, **kwargs),
            WhitespaceNormalizationMapper(**kwargs),
            TextLengthFilter(min_len=1, max_len=2, **kwargs),
        ]
        native = _source(_executor(self.tmp_path)).process(ops).data.take_all()
        Path(log_path).unlink()

        rows = _run(_executor(self.tmp_path), ops)
        self.assertEqual(_by_id(rows), _by_id(native))
        self.assertEqual([row["id"] for row in rows], [2])
        commits = _commits(self.tmp_path)
        self.assertEqual({commit.stage for commit in commits}, {0, 1, 2})
        self.assertTrue(all((sum((stop - start for _, start, stop in commit.consumed)) <= 2 for commit in commits)))
        self.assertEqual(sorted(_logged_ids(log_path)), [0, 1, 2, 3])

        self.assertEqual(_run(_executor(self.tmp_path, resume=True), ops), rows)
        self.assertEqual(_commits(self.tmp_path), commits)
        self.assertEqual(sorted(_logged_ids(log_path)), [0, 1, 2, 3])

    @TEST_TAG("ray")
    def test_checkpointed_rows_match_native_and_resume_reuses_commits_actor(self):
        self._check_checkpointed_rows_match_native_and_resume_reuses_commits("actor")

    @TEST_TAG("ray")
    def test_checkpointed_rows_match_native_and_resume_reuses_commits_task(self):
        self._check_checkpointed_rows_match_native_and_resume_reuses_commits("task")

    @TEST_TAG("ray")
    def test_failed_run_resumes_only_unconsumed_rows(self):
        log_path = str(self.tmp_path / "calls.jsonl")
        kwargs = {"num_proc": 1, "ray_execution_mode": "actor"}
        ops = [
            LoggingMapper(log_path, batch_size=1, **kwargs),
            LoggingMapper(log_path + ".second", fail_id=3, batch_size=1, **kwargs),
        ]
        with self.assertRaisesRegex(Exception, "injected mapper failure"):
            _run(_executor(self.tmp_path), ops)
        committed_first = sorted(
            index
            for commit in _commits(self.tmp_path)
            if commit.stage == 0
            for _, start, stop in commit.consumed
            for index in range(start, stop)
        )
        first_calls = _logged_ids(log_path)
        self.assertTrue(committed_first and sorted(first_calls) == sorted(set(first_calls)))

        rows = _run(_executor(self.tmp_path, resume=True), ops)
        self.assertEqual(sorted((row["id"] for row in rows)), [0, 1, 2, 3])
        resumed_calls = _logged_ids(log_path)[len(first_calls) :]
        self.assertEqual(sorted(resumed_calls), sorted(set(range(4)) - set(committed_first)))
        # The failed commit batch reran as a whole; committed batches did not.
        second = Counter(_logged_ids(log_path + ".second"))
        self.assertTrue(set(second) == {0, 1, 2, 3} and sum(second.values()) <= 4 + 2)

    def _check_worker_death_after_yield_replays_the_committed_batch(self, mode):
        if mode == "actor":
            patcher = patch.object(ray_partition_runner, "CheckpointStageActor", ExitAfterFirstCommitActor)
        else:
            patcher = patch.object(ray_partition_runner, "run_checkpoint_task", exit_after_first_commit_task)
        self.addCleanup(patcher.stop)
        patcher.start()
        kwargs = {"num_proc": 1, "ray_execution_mode": mode}
        log_path = str(self.tmp_path / "calls.jsonl")
        ops = [
            LoggingMapper(log_path, batch_size=1, **kwargs),
            DuplicateMapper(batch_size=1, **kwargs),
            WhitespaceNormalizationMapper(**kwargs),
        ]
        rows = _run(_executor(self.tmp_path), ops)
        self.assertTrue((self.tmp_path / "checkpoints" / "partition_recovery" / "exited-once").exists())
        self.assertEqual(sorted((row["id"] for row in rows)), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(sorted(_logged_ids(log_path)), [0, 1, 2, 3])

    @TEST_TAG("ray")
    def test_worker_death_after_yield_replays_the_committed_batch_actor(self):
        self._check_worker_death_after_yield_replays_the_committed_batch("actor")

    @TEST_TAG("ray")
    def test_worker_death_after_yield_replays_the_committed_batch_task(self):
        self._check_worker_death_after_yield_replays_the_committed_batch("task")

    def _check_commit_size_does_not_merge_blocks(self, block_rows, max_rows, expected):
        import pyarrow as pa

        from data_juicer.core.data.ray_batch_plan import RayBatchPlan, map_whole_blocks
        from data_juicer.core.executor.partition_protocol import SOURCE_SEQ, with_keys

        store = LocalStore(str(self.tmp_path))
        coordinator = ray_partition_runner._CommitCoordinator.remote(store, 1, 8)
        context = ray_partition_runner.StageContext(store, 0, coordinator, max_rows)
        plan = RayBatchPlan().map_batches(lambda batch: batch, batch_format="pyarrow")
        table = with_keys(pa.table({"id": list(range(8))}), SOURCE_SEQ)
        source = ray.data.from_arrow([table.slice(start, block_rows) for start in range(0, 8, block_rows)])
        try:
            result = map_whole_blocks(
                source,
                ray_partition_runner.run_checkpoint_task,
                fn_args=(context, plan),
                batch_format="pyarrow",
            )
            self.assertEqual(sorted((row["id"] for row in result.take_all())), list(range(8)))
            commits = [Commit.from_bytes(store.get(key)) for key in store.list(COMMIT_DIR)]
            self.assertEqual(sorted((commit.rows for commit in commits)), sorted(expected))
        finally:
            ray.kill(coordinator)

    @TEST_TAG("ray")
    def test_commit_size_does_not_merge_blocks_small_blocks(self):
        self._check_commit_size_does_not_merge_blocks(2, 8, [2, 2, 2, 2])

    @TEST_TAG("ray")
    def test_commit_size_does_not_merge_blocks_oversized_block(self):
        self._check_commit_size_does_not_merge_blocks(8, 3, [3, 3, 2])

    @TEST_TAG("ray")
    def test_gpu_whole_block_plan_has_no_bundling_threshold(self):
        from data_juicer.core.data.ray_batch_plan import map_whole_blocks

        result = map_whole_blocks(ray.data.range(4), lambda batch: batch, num_gpus=1)
        self.assertIs(result._logical_plan.dag.min_rows_per_bundled_input, None)

    @TEST_TAG("ray")
    def test_expansion_shrinking_and_batches_within_commits(self):
        ops = [DuplicateMapper(batch_size=1, num_proc=1), BatchSizeMapper(batch_size=3, num_proc=1)]
        rows = _run(_executor(self.tmp_path, size=4), ops)
        self.assertEqual(sorted((row["id"] for row in rows)), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(max((row["batch_rows"] for row in rows)), 3)

        shrinking = self.tmp_path / "shrinking"
        rows = _run(
            _executor(shrinking), [OddMapper(batch_size=1, num_proc=1), TextLengthFilter(min_len=1, num_proc=1)]
        )
        self.assertEqual(sorted((row["id"] for row in rows)), [1, 3])
        self.assertTrue(all((row["mapped"] is True for row in rows)))

    @TEST_TAG("ray")
    def test_all_filtered_run_completes_and_resumes(self):
        ops = [TextLengthFilter(min_len=100, num_proc=1), WhitespaceNormalizationMapper(num_proc=1)]
        self.assertEqual(_run(_executor(self.tmp_path), ops), [])
        commits = _commits(self.tmp_path)
        self.assertTrue({commit.stage for commit in commits} == {0} and all((commit.rows == 0 for commit in commits)))
        self.assertEqual(_run(_executor(self.tmp_path, resume=True), ops), [])
        self.assertEqual(_commits(self.tmp_path), commits)

    def _check_native_output_formats_survive_checkpoint(self, output_format):
        import numpy as np

        op = OutputFormatMapper(output_format, batch_size=2, num_proc=1, ray_execution_mode="actor")
        expected = _source(_executor(self.tmp_path)).process([op]).data.take_all()
        actual = _run(_executor(self.tmp_path), [op])
        if output_format == "tensor":
            for rows in (actual, expected):
                for row in rows:
                    row["tensor"] = np.asarray(row["tensor"]).tolist()
        self.assertEqual(_by_id(actual), _by_id(expected))

    @TEST_TAG("ray")
    def test_native_output_formats_survive_checkpoint_generator(self):
        self._check_native_output_formats_survive_checkpoint("generator")

    @TEST_TAG("ray")
    def test_native_output_formats_survive_checkpoint_dataframe(self):
        self._check_native_output_formats_survive_checkpoint("dataframe")

    @TEST_TAG("ray")
    def test_native_output_formats_survive_checkpoint_tensor(self):
        self._check_native_output_formats_survive_checkpoint("tensor")

    def _check_stats_export_matches_native_and_republishes_on_resume(self, mode):
        native_path, checkpoint_path = self.tmp_path / "native-stats", self.tmp_path / "ckpt-stats"
        kwargs = {"min_len": 4, "batch_size": 2, "num_proc": 1, "ray_execution_mode": mode}
        native = _source(_executor(self.tmp_path)).process(
            [TextLengthFilter(stats_export_path=str(native_path), **kwargs)]
        )
        native_rows = native.data.take_all()
        ops = [TextLengthFilter(stats_export_path=str(checkpoint_path), **kwargs)]
        self.assertEqual(_by_id(_run(_executor(self.tmp_path), ops)), _by_id(native_rows))
        exported = _by_id(_read_json_rows(checkpoint_path))
        self.assertEqual(exported, _by_id(_read_json_rows(native_path)))
        self.assertEqual([row[Fields.stats]["text_len"] for row in exported], [5, 3, 3, 3])

        _run(_executor(self.tmp_path, resume=True), ops)
        self.assertEqual(_by_id(_read_json_rows(checkpoint_path)), exported)

    @TEST_TAG("ray")
    def test_stats_export_matches_native_and_republishes_on_resume_actor(self):
        self._check_stats_export_matches_native_and_republishes_on_resume("actor")

    @TEST_TAG("ray")
    def test_stats_export_matches_native_and_republishes_on_resume_task(self):
        self._check_stats_export_matches_native_and_republishes_on_resume("task")

    @TEST_TAG("ray")
    def test_stats_publish_failure_resumes_without_recompute(self):
        from data_juicer.core.data.ray_batch_plan import StatsExport

        log_path = str(self.tmp_path / "calls.jsonl")
        stats_path = self.tmp_path / "stats"
        stats_path.mkdir()
        (stats_path / "unrelated.bin").write_bytes(b"keep")
        publish = StatsExport.publish
        calls = []

        def publish_then_fail_once(export, data, identity):
            calls.append(tuple(identity))
            publish(export, data, identity)
            if len(calls) == 1:
                raise RuntimeError("injected failure after stats publication")

        patcher = patch.object(StatsExport, "publish", publish_then_fail_once)
        self.addCleanup(patcher.stop)
        patcher.start()
        ops = [
            LoggingMapper(log_path, batch_size=2, num_proc=1),
            TextLengthFilter(stats_export_path=str(stats_path), min_len=4, num_proc=1),
        ]
        with self.assertRaisesRegex(RuntimeError, "injected failure after stats publication"):
            _run(_executor(self.tmp_path), ops)
        before = _by_id(_read_json_rows(stats_path))
        self.assertEqual([row["id"] for row in before], [0, 1, 2, 3])
        calls_before = _logged_ids(log_path)

        self.assertEqual([row["id"] for row in _run(_executor(self.tmp_path, resume=True), ops)], [0])
        self.assertEqual(_logged_ids(log_path), calls_before)
        self.assertEqual(calls[1], calls[0])
        self.assertEqual(_by_id(_read_json_rows(stats_path)), before)
        self.assertEqual((stats_path / "unrelated.bin").read_bytes(), b"keep")

    @TEST_TAG("ray")
    def test_multiple_stats_stages_share_directory(self):
        stats_path = self.tmp_path / "stats"
        kwargs = dict(stats_export_path=str(stats_path), num_proc=1, ray_execution_mode="actor")
        ops = [TextLengthFilter(min_len=4, **kwargs), AlphanumericFilter(min_ratio=0.1, **kwargs)]
        self.assertEqual([row["id"] for row in _run(_executor(self.tmp_path), ops)], [0])
        expected = Counter({0: 2, 1: 1, 2: 1, 3: 1})
        self.assertEqual(Counter((row["id"] for row in _read_json_rows(stats_path))), expected)
        _run(_executor(self.tmp_path, resume=True), ops)
        self.assertEqual(Counter((row["id"] for row in _read_json_rows(stats_path))), expected)

    @TEST_TAG("ray")
    def test_resume_rejects_changed_input_and_fresh_run_rejects_existing_checkpoint(self):
        ops = [WhitespaceNormalizationMapper(num_proc=1)]
        _run(_executor(self.tmp_path), ops)
        resumed = _executor(self.tmp_path, resume=True)
        with self.assertRaisesRegex(ProtocolError, "Input content, read order, or operator config changed"):
            _run(resumed, ops, _source(resumed, changed=True))
        with self.assertRaisesRegex(ProtocolError, "already exists"):
            _run(_executor(self.tmp_path), ops)
        other_ops = [WhitespaceNormalizationMapper(num_proc=1), TextLengthFilter(min_len=1, num_proc=1)]
        with self.assertRaisesRegex(ProtocolError, "changed since the checkpoint"):
            _run(_executor(self.tmp_path, resume=True), other_ops)

    @TEST_TAG("ray")
    def test_resume_accepts_a_different_commit_size(self):
        log_path = str(self.tmp_path / "calls.jsonl")
        ops = [LoggingMapper(log_path, fail_id=2, batch_size=1, num_proc=1)]
        with self.assertRaisesRegex(Exception, "injected mapper failure"):
            _run(_executor(self.tmp_path, size=1), ops)
        rows = _run(_executor(self.tmp_path, resume=True, size=3), ops)
        self.assertEqual(sorted((row["id"] for row in rows)), [0, 1, 2, 3])
        self.assertEqual(sorted(_logged_ids(log_path)), [0, 1, 2, 3])

    def test_global_operators_end_the_checkpointed_prefix(self):
        class SharedFilter(TextLengthFilter):
            def _prepare_for_ray_map_batches(self):
                return True

        executor = _executor(self.tmp_path)
        mapper = WhitespaceNormalizationMapper(num_proc=1)
        self.assertEqual(executor._checkpoint_prefix_length([mapper, DocumentDeduplicator(), mapper]), 1)
        self.assertEqual(executor._checkpoint_prefix_length([mapper, SharedFilter(num_proc=1)]), 1)
        self.assertEqual(executor._checkpoint_prefix_length([mapper, TextLengthFilter(num_proc=1)]), 2)
        executor.ckpt_manager.checkpoint_enabled = False
        self.assertEqual(executor._checkpoint_prefix_length([mapper]), 0)
        with self.assertRaisesRegex(ProtocolError, "Shared Ray state"):
            RayDataset(None, auto_op_parallelism=False).process(
                [SharedFilter(num_proc=1)], partition_recovery=Namespace(stage_count=1)
            )

    def _check_full_executor_exports_checkpointed_result(self, drop_all):
        config_path = _write_job(
            self.tmp_path,
            [
                {"whitespace_normalization_mapper": {"num_proc": 1, "ray_execution_mode": "actor"}},
                {
                    "text_length_filter": {
                        "min_len": 100 if drop_all else 1,
                        "max_len": 200 if drop_all else 2,
                        "num_proc": 1,
                        "ray_execution_mode": "actor",
                    }
                },
            ],
        )
        PartitionedRayExecutor(init_configs(["--config", str(config_path)])).run(skip_return=True)
        self.assertEqual(_exported_texts(self.tmp_path), [] if drop_all else ["C", "C"])

    @TEST_TAG("ray")
    def test_full_executor_exports_checkpointed_result_false(self):
        self._check_full_executor_exports_checkpointed_result(False)

    @TEST_TAG("ray")
    def test_full_executor_exports_checkpointed_result_true(self):
        self._check_full_executor_exports_checkpointed_result(True)

    @TEST_TAG("ray")
    def test_empty_export_resumes_after_completion_logging_fails(self):
        config = _write_job(self.tmp_path, [{"text_length_filter": {"min_len": 100, "num_proc": 1}}])
        executor = PartitionedRayExecutor(init_configs(["--config", str(config)]))
        with patch.object(executor, "log_job_complete", side_effect=RuntimeError("late completion failure")):
            with self.assertRaisesRegex(RuntimeError, "late completion failure"):
                executor.run(skip_return=True)
        self.assertTrue((self.tmp_path / "output.jsonl").is_dir())
        resumed = PartitionedRayExecutor(init_configs(["--config", str(config), "--resume", executor.job_id]))
        self.assertTrue(resumed.cfg._same_yaml_config)
        resumed.run(skip_return=True)
        self.assertEqual(_exported_texts(self.tmp_path), [])

    @TEST_TAG("ray")
    def test_failed_statistics_publication_keeps_previous_export_until_verified_resume(self):
        from data_juicer.core.data.ray_batch_plan import StatsExport

        config = _write_job(
            self.tmp_path,
            [
                {
                    "text_length_filter": {
                        "min_len": 1,
                        "num_proc": 1,
                        "stats_export_path": str(self.tmp_path / "statistics"),
                    }
                }
            ],
        )
        initial = PartitionedRayExecutor(init_configs(["--config", str(config)]))
        output = self.tmp_path / "output.jsonl"
        publish = StatsExport.publish

        def fail_after_publish(export, data, identity):
            publish(export, data, identity)
            raise RuntimeError("statistics publication failed")

        with patch.object(StatsExport, "publish", fail_after_publish):
            with self.assertRaisesRegex(RuntimeError, "statistics publication failed"):
                initial.run(skip_return=True)
            first_files = set(output.glob("*.json"))
            self.assertTrue(first_files)
            args = ["--config", str(config), "--resume", initial.job_id]
            with self.assertRaisesRegex(RuntimeError, "statistics publication failed"):
                PartitionedRayExecutor(init_configs(args)).run(skip_return=True)
            self.assertTrue(first_files <= set(output.glob("*.json")))
        incomplete_files = set(output.glob("*.json"))
        self.assertGreater(len(incomplete_files), len(first_files))
        PartitionedRayExecutor(init_configs(args)).run(skip_return=True)
        self.assertTrue(incomplete_files.isdisjoint(output.glob("*.json")))
        self.assertEqual(_exported_texts(self.tmp_path), [" A\tB ", " C ", " C ", "!!!"])

    @TEST_TAG("ray")
    def test_manifest_failure_cleanup_requires_definitive_local_nonpublication(self):
        for name, published, remote in [("absent", False, False), ("published", True, False), ("remote", False, True)]:
            with self.subTest(name=name):
                local = FailedPublicationStore(str(self.tmp_path / name), published)
                files = write_tables(local, [pa.table({"v": [1]})])
                stats = write_tables(local, [pa.table({"s": [1]})], kind="stats")
                retained = write_tables(local, [pa.table({"foreign": [2]})])
                store = AmbiguousPublicationStore(local.root) if remote else local
                coordinator = ray_partition_runner._CommitCoordinator.remote(store, 1, 1)
                try:
                    with self.assertRaisesRegex(Exception, "injected manifest failure"):
                        ray.get(coordinator.commit.remote(0, ((SOURCE_SEQ, 0, 1),), files, stats))
                    for item in (*files, *stats):
                        self.assertEqual(local.get(item.key) is not None, published or remote)
                    self.assertIsNotNone(local.get(retained[0].key))
                    self.assertEqual(local.get(commit_key(0, 0)) is not None, published)
                finally:
                    ray.kill(coordinator)

    @TEST_TAG("ray")
    def test_duplicate_attempt_removes_own_files_and_keeps_winning_data_and_stats(self):
        store = LocalStore(str(self.tmp_path))
        winner = write_tables(store, [pa.table({"v": [42]})])
        stats = write_tables(store, [pa.table({"s": [42]})], kind="stats")
        coordinator = ray_partition_runner._CommitCoordinator.remote(store, 1, 1)
        consumed = ((SOURCE_SEQ, 0, 1),)
        try:
            committed = ray.get(coordinator.commit.remote(0, consumed, winner, stats))

            class Worker:
                def process_partition(self, tables, write_stats):
                    write_stats([pa.table({"s": [99]})])
                    yield from tables

            context = ray_partition_runner.StageContext(store, 0, coordinator, 1)
            result, outputs = ray_partition_runner._commit_slice(
                context, Worker(), with_keys(pa.table({"v": [99]}), SOURCE_SEQ), consumed
            )
            self.assertEqual(result, committed)
            self.assertEqual(outputs[0].to_pylist(), [{"v": 42}])
            expected = {item.key for item in (*winner, *stats)}
            self.assertEqual(
                {str(path.relative_to(self.tmp_path)) for path in self.tmp_path.glob("data/**/*.parquet")}, expected
            )
        finally:
            ray.kill(coordinator)

    def test_operator_failure_removes_stats_written_before_commit(self):
        store = LocalStore(str(self.tmp_path))
        retained = write_tables(store, [pa.table({"v": [42]})])

        class Worker:
            def process_partition(self, tables, write_stats):
                write_stats([pa.table({"s": [99]})])
                raise RuntimeError("operator failed after stats")

        context = ray_partition_runner.StageContext(store, 0, None, 1)
        with self.assertRaisesRegex(RuntimeError, "operator failed after stats"):
            ray_partition_runner._commit_slice(context, Worker(), with_keys(pa.table({"v": [99]}), SOURCE_SEQ), ())
        self.assertEqual(
            {str(path.relative_to(self.tmp_path)) for path in self.tmp_path.glob("data/**/*.parquet")},
            {item.key for item in retained},
        )

    @TEST_TAG("ray")
    def test_full_executor_checkpoints_the_prefix_before_a_deduplicator(self):
        config_path = _write_job(
            self.tmp_path,
            [
                {"whitespace_normalization_mapper": {"num_proc": 1}},
                {"ray_document_deduplicator": {}},
                {"text_length_filter": {"min_len": 1, "max_len": 3, "num_proc": 1}},
            ],
        )
        PartitionedRayExecutor(init_configs(["--config", str(config_path)])).run(skip_return=True)
        self.assertEqual(_exported_texts(self.tmp_path), ["!!!", "A B", "C"])
        store = LocalStore(str(self.tmp_path / "work" / "checkpoints" / "partition_recovery"))
        stages = {Commit.from_bytes(store.get(key)).stage for key in store.list(COMMIT_DIR)}
        self.assertEqual(stages, {0})

    @TEST_TAG("ray")
    def test_full_executor_return_reuses_stateful_suffix_result(self):
        config_path = _write_job(
            self.tmp_path,
            [
                {"whitespace_normalization_mapper": {"num_proc": 1}},
                {"ray_document_deduplicator": {"num_proc": 1, "dedup_set_num": 1}},
            ],
        )
        result = PartitionedRayExecutor(init_configs(["--config", str(config_path)])).run()
        expected = _exported_texts(self.tmp_path)
        self.assertEqual(expected, ["!!!", "A B", "C"])
        self.assertEqual(sorted(result.get_column("text")), expected)
        self.assertEqual(sorted(result.get_column("text")), expected)
        self.assertEqual(result.count(), 3)

    def _check_full_executor_suffix_executes_once(self, mode, *, sharded, skip_return):
        config_path = _write_job(
            self.tmp_path,
            [{"whitespace_normalization_mapper": {"num_proc": 1}}],
            extra={"export_shard_size": 1048576 if sharded else 0},
        )
        rows = [{"text": str(i), "id": i} for i in range(4)]
        (self.tmp_path / "input.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        prefix_log, suffix_log = str(self.tmp_path / "prefix.calls"), str(self.tmp_path / "suffix.calls")
        kwargs = {"num_proc": 1, "ray_execution_mode": mode, "batch_size": 1}
        ops = [LoggingMapper(prefix_log, **kwargs), RayRepartitionPipeline(), LoggingMapper(suffix_log, **kwargs)]
        executor = PartitionedRayExecutor(init_configs(["--config", str(config_path)]))
        with patch("data_juicer.core.executor.ray_executor_partitioned.load_ops", return_value=ops):
            result = executor.run(skip_return=skip_return)
        self.assertEqual(_exported_texts(self.tmp_path), ["0", "1", "2", "3"])
        if skip_return:
            self.assertIsNone(result)
        else:
            self.assertEqual(_by_id(result.data.take_all()), rows)
            self.assertEqual(_by_id(result.data.take_all()), rows)
        self.assertEqual(sorted(_logged_ids(prefix_log)), [0, 1, 2, 3])
        self.assertEqual(sorted(_logged_ids(suffix_log)), [0, 1, 2, 3])

    @TEST_TAG("ray")
    def test_full_executor_return_does_not_recompute_suffix_actor(self):
        self._check_full_executor_suffix_executes_once("actor", sharded=False, skip_return=False)

    @TEST_TAG("ray")
    def test_full_executor_return_does_not_recompute_suffix_task(self):
        self._check_full_executor_suffix_executes_once("task", sharded=False, skip_return=False)

    @TEST_TAG("ray")
    def test_full_executor_sharded_export_does_not_recompute_suffix_actor(self):
        self._check_full_executor_suffix_executes_once("actor", sharded=True, skip_return=True)

    @TEST_TAG("ray")
    def test_full_executor_sharded_export_does_not_recompute_suffix_task(self):
        self._check_full_executor_suffix_executes_once("task", sharded=True, skip_return=True)

    def _check_native_stats_export_computes_once(self, mode):
        log_path = str(self.tmp_path / "stats.calls")
        stats_path = self.tmp_path / "stats"
        source = _source(_executor(self.tmp_path))
        result = source.process(
            [
                LoggingTextLengthFilter(
                    log_path, min_len=4, num_proc=1, ray_execution_mode=mode, stats_export_path=str(stats_path)
                )
            ]
        )
        self.assertEqual([row["id"] for row in result.data.take_all()], [0])
        self.assertEqual(sorted(_logged_ids(log_path)), [0, 1, 2, 3])
        self.assertEqual([row["id"] for row in _by_id(_read_json_rows(stats_path))], [0, 1, 2, 3])

    @TEST_TAG("ray")
    def test_native_stats_export_computes_once_actor(self):
        self._check_native_stats_export_computes_once("actor")

    @TEST_TAG("ray")
    def test_native_stats_export_computes_once_task(self):
        self._check_native_stats_export_computes_once("task")

    @TEST_TAG("ray")
    def test_chained_process_does_not_execute_schema_probe(self):
        log_path = str(self.tmp_path / "chain.calls")
        source = _source(_executor(self.tmp_path))
        result = source.process([LoggingMapper(log_path, num_proc=1)])
        result.process([TextLengthFilter(min_len=4, num_proc=1)])
        self.assertEqual(_logged_ids(log_path), [])
        self.assertEqual([row["id"] for row in result.data.take_all()], [0])
        self.assertEqual(sorted(_logged_ids(log_path)), [0, 1, 2, 3])

    @TEST_TAG("ray")
    def test_get_column_does_not_execute_schema_probe(self):
        for k in (None, 1, 10):
            with self.subTest(k=k):
                log_path = str(self.tmp_path / f"column-{k}.calls")
                source = _source(_executor(self.tmp_path))
                result = source.process([LoggingMapper(log_path, num_proc=1)])
                with patch.object(result.data, "materialize", side_effect=AssertionError("Unexpected materialization")):
                    self.assertEqual(result.get_column("id", k=k), [0, 1, 2, 3][:k])
                self.assertEqual(sorted(_logged_ids(log_path)), [0, 1, 2, 3])

    @TEST_TAG("ray")
    def test_get_column_empty_output_preserves_schema(self):
        for k in (None, 0, 1):
            for column in ("text", "missing"):
                with self.subTest(k=k, column=column):
                    log_path = str(self.tmp_path / f"empty-column-{k}-{column}.calls")
                    source = _source(_executor(self.tmp_path))
                    result = source.process([LoggingMapper(log_path, num_proc=1)])
                    result.data = result.data.map_batches(lambda batch: batch.slice(0, 0), batch_format="pyarrow")
                    self.assertIsNone(result.data.columns(fetch_if_missing=False))
                    with patch.object(
                        result.data, "materialize", side_effect=AssertionError("Unexpected materialization")
                    ):
                        if column == "text":
                            self.assertEqual(result.get_column(column, k=k), [])
                        else:
                            with self.assertRaisesRegex(KeyError, "Column 'missing' not found in dataset"):
                                result.get_column(column, k=k)
                    self.assertEqual(sorted(_logged_ids(log_path)), [0, 1, 2, 3])

    @TEST_TAG("ray")
    def test_unsupported_checkpoint_scheme_fails_before_running(self):
        config_path = _write_job(
            self.tmp_path,
            [{"whitespace_normalization_mapper": {"num_proc": 1}}],
            extra={"checkpoint_dir": "gs://bucket/checkpoints"},
        )
        with self.assertRaisesRegex(ValueError, "not supported"):
            PartitionedRayExecutor(init_configs(["--config", str(config_path)]))

    def test_removed_checkpoint_strategies_are_rejected(self):
        config_path = _write_job(self.tmp_path, [{"whitespace_normalization_mapper": {}}])
        with self.assertRaises(BaseException):
            init_configs(["--config", str(config_path), "--checkpoint.strategy", "every_n_ops"])

    @TEST_TAG("ray")
    def test_parquet_round_trip_keeps_nullable_schema_changes(self):
        class NullableMapper(Mapper):
            _batched_op = True

            def process_batched(self, samples):
                samples["extra"] = [str(value) if value == 0 else None for value in samples["id"]]
                return samples

        op = NullableMapper(batch_size=1, num_proc=1)
        rows = _run(_executor(self.tmp_path, size=4), [op])
        self.assertEqual(_by_id(rows)[0]["extra"], "0")
        self.assertEqual(_run(_executor(self.tmp_path, resume=True, size=4), [op]), rows)


if __name__ == "__main__":
    unittest.main()
