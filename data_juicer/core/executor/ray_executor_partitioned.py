"""
Simplified Partitioned Ray Executor for Large Dataset Processing

This module implements a streamlined partitioned execution strategy for Ray mode that:
2. Splits the dataset into manageable partitions at exact row boundaries
3. Processes each partition independently with Ray tasks
4. Merges results back into a single dataset for export
5. Supports convergence points for global operations (like deduplicators)
"""

import copy
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from jsonargparse import Namespace
from loguru import logger
from pydantic import PositiveInt

from data_juicer.core.data.dataset_builder import (
    DatasetBuilder,
    deprecated_load_data_np_kwargs,
)
from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.core.executor import ExecutorBase
from data_juicer.core.executor.dag_execution_mixin import DAGExecutionMixin
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin, EventType
from data_juicer.core.ray_exporter import RayExporter
from data_juicer.ops import Filter, Mapper, load_ops
from data_juicer.ops.op_fusion import fuse_operators
from data_juicer.utils.ckpt_utils import (
    STREAM_MANIFEST_SCHEMA_VERSION,
    CheckpointStrategy,
    RayCheckpointManager,
    atomic_write_json,
    make_row_id_membership,
)
from data_juicer.utils.config_utils import ConfigAccessor
from data_juicer.utils.file_utils import is_remote_path
from data_juicer.utils.lazy_loader import LazyLoader

ray = LazyLoader("ray")

_AUTO_CPU_PARTITION_CAP = 4
_AUTO_GPU_PIPELINE_CPU_FRACTION = 0.80
_AUTO_GPU_PIPELINE_MEMORY_FRACTION = 0.90
_DEFAULT_MAX_GPU_WORKERS_PER_DEVICE = 5
_DEFAULT_GPU_PROBE_WARMUP_BATCHES = 1
_DEFAULT_GPU_PROBE_STEADY_BATCHES = 3
_DEFAULT_GPU_PROBE_SAMPLE_SEED = 42
_DEFAULT_MAX_INITIALIZATION_OVERHEAD_RATIO = 0.10
_LOGICAL_PARTITION_COLUMN = "__data_juicer_logical_partition_id__"
# Stable global read-order row id injected at ingestion for streaming tee-sink
# recovery. Mappers/Filters carry it unchanged; it is the frontier key and is
# dropped just before export. Reserved: rejected if present in the input schema.
_ROW_ID_COLUMN = "__data_juicer_row_id__"
# Known row-EXPANDING (1:many) mappers. A 1:many op copies a parent id onto
# every sub-row. Streaming recovery SUPPORTS this: the row-id frontier commits a
# SET of INPUT ids (presence, decoupled from output count -- the same decoupling
# that makes row-dropping filters safe), and the block-preserving tee
# (batch_size=None) keeps a parent's whole expansion inside one committed block,
# so committing that input id is exactly-once. This list is the best-effort
# auto-detector that flips the tee into expansion mode (relaxing its uniqueness
# guard); a custom expander not listed here can force it via
# partition.stream_allow_row_expansion=true. DJ exposes no cardinality flag, so
# a listed-but-1:1 op is harmless (no dups => guard never triggers anyway).
_ROW_EXPANDING_MAPPERS = frozenset(
    {
        "generate_qa_from_text_mapper",
        "image_diffusion_mapper",
        "nlpaug_en_mapper",
        "nlpcda_zh_mapper",
        "expand_duplicate_mapper",
    }
)
_PARTITION_CONTENT_HASH_ALGORITHM = "sha256-sequence-v1"
_PARTITION_CONTENT_HASH_MODULUS = 1 << 256
_PARTITION_CONTENT_HASH_BASE = int.from_bytes(hashlib.sha256(b"data-juicer-partition-sequence-v1").digest(), "big") | 1


def _canonical_row_bytes(row: Dict) -> bytes:
    """Serialize one row consistently for partition content hashing."""
    try:
        serialized = json.dumps(row, sort_keys=True, default=str, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        serialized = str(row)
    return serialized.encode("utf-8")


def _hash_partition_batch(batch):
    """Return an order-sensitive rolling hash for one Arrow batch."""
    import pyarrow

    sequence_hash = 0
    rows = batch.to_pylist()
    for row in rows:
        digest = int.from_bytes(hashlib.sha256(_canonical_row_bytes(row)).digest(), "big")
        sequence_hash = (sequence_hash * _PARTITION_CONTENT_HASH_BASE + digest) % _PARTITION_CONTENT_HASH_MODULUS

    return pyarrow.table(
        {
            "row_count": [len(rows)],
            "sequence_hash": [f"{sequence_hash:064x}"],
        }
    )


def _combine_partition_hash_partials(partials: List[Dict]) -> tuple:
    """Combine ordered batch hashes without depending on batch boundaries."""
    row_count = 0
    sequence_hash = 0
    for partial in partials:
        partial_count = int(partial["row_count"])
        sequence_hash = (
            sequence_hash * pow(_PARTITION_CONTENT_HASH_BASE, partial_count, _PARTITION_CONTENT_HASH_MODULUS)
            + int(partial["sequence_hash"], 16)
        ) % _PARTITION_CONTENT_HASH_MODULUS
        row_count += partial_count

    payload = f"{_PARTITION_CONTENT_HASH_ALGORITHM}:{row_count}:{sequence_hash:064x}"
    return hashlib.sha256(payload.encode("ascii")).hexdigest(), row_count


def _add_logical_partition_id(batch, partition_id: int):
    """Attach the checkpoint partition identity without changing row order."""
    import pyarrow

    if _LOGICAL_PARTITION_COLUMN in batch.column_names:
        index = batch.column_names.index(_LOGICAL_PARTITION_COLUMN)
        batch = batch.remove_column(index)
    values = pyarrow.array([partition_id] * batch.num_rows, type=pyarrow.int64())
    return batch.append_column(_LOGICAL_PARTITION_COLUMN, values)


def _matches_logical_partition(row: Dict, partition_id: int) -> bool:
    return row[_LOGICAL_PARTITION_COLUMN] == partition_id


def _contiguous_row_id_ranges(row_ids: List[int]) -> List[list]:
    """Compress a set of row ids into sorted, disjoint half-open [lo, hi) ranges.

    Row ids are contiguous in read order, but a Filter upstream may punch holes,
    so we compute the actual runs present in this block rather than assume one range.
    """
    ranges: List[list] = []
    # De-duplicate first: with a row-EXPANDING (1:many) op a parent id is copied
    # onto every sub-row, so this block's ids can repeat. The frontier is a SET
    # of committed input ids (presence, not multiplicity), so collapsing dups
    # here keeps the emitted ranges clean and disjoint; reconcile merges across
    # shards regardless.
    for rid in sorted({int(x) for x in row_ids}):
        if ranges and rid == ranges[-1][1]:
            ranges[-1][1] = rid + 1
        else:
            ranges.append([rid, rid + 1])
    return ranges


class _StreamTeeSink:
    """Pass-through Ray Data actor that durably tees each block, then returns it
    unchanged so the pipeline keeps streaming (no materialize barrier).

    Verified in experiments/seg_grid/ray_stream_probe.py (V1/V2): a hot actor
    pool of this class writes one parquet block + one atomic manifest shard per
    input block without triggering an implicit materialize.

    Commit ordering is data-before-pointer and each step is atomic: the block
    parquet is written to a unique ``*.tmp`` then ``os.replace``-d into its
    content-keyed final name, and only then is the manifest shard written the
    same write-tmp-then-replace way. A crash therefore leaves at most a stray
    ``*.tmp`` (ignored by RayCheckpointManager.reconcile_stream_frontier) and
    never a manifest pointing at a half-written or absent block; the reconciler
    additionally drops any shard whose block file is missing.
    """

    def __init__(
        self,
        ckpt_dir: str,
        data_dir: str,
        manifest_dir: str,
        segment_index: int,
        op_boundary_idx: int,
        op_name: str,
        job_id: str,
        row_id_column: str = _ROW_ID_COLUMN,
        fsync: bool = True,
        allow_expansion: bool = False,
    ):
        self.ckpt_dir = ckpt_dir
        self.data_dir = data_dir
        self.manifest_dir = manifest_dir
        self.segment_index = segment_index
        self.op_boundary_idx = op_boundary_idx
        self.op_name = op_name
        self.job_id = job_id
        self.row_id_column = row_id_column
        self.fsync = fsync
        # When True this segment contains a row-EXPANDING (1:many) op, so the
        # per-block row-id uniqueness guard below is relaxed (a parent id legally
        # repeats across its sub-rows). When False (1:1 / row-dropping pipeline)
        # the guard stays armed as a defensive net against a buggy op silently
        # duplicating ids.
        self.allow_expansion = allow_expansion
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.manifest_dir, exist_ok=True)

    @staticmethod
    def _fsync_path(path):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def __call__(self, batch):
        import pyarrow.parquet as pq

        if batch.num_rows == 0:
            return batch
        if self.row_id_column not in batch.column_names:
            raise RuntimeError(
                f"Stream tee-sink lost the row-id column '{self.row_id_column}'; "
                "an operator must have dropped it, breaking recovery provenance."
            )

        row_ids = [int(x) for x in batch.column(self.row_id_column).to_pylist()]
        sorted_ids = sorted(row_ids)
        # Cardinality guard. A 1:1 / row-dropping op keeps every input row's
        # unique stamped id, so ids stay unique -- a duplicate then means a buggy
        # op, and we fail closed. A row-EXPANDING (1:many) op legally copies a
        # parent id onto each sub-row; the frontier is a SET of committed INPUT
        # ids (presence, not output count), and the block-preserving tee
        # (batch_size=None) keeps a parent's whole expansion inside ONE committed
        # block, so committing that input id is exactly-once. When allow_expansion
        # is set we therefore permit duplicates instead of rejecting them.
        if not self.allow_expansion and len(set(sorted_ids)) != len(sorted_ids):
            raise RuntimeError(
                f"Streaming recovery requires 1:1 / row-dropping ops, but segment {self.segment_index} "
                f"(op '{self.op_name}') produced duplicate row ids unexpectedly -- a row-expanding (1:many) "
                "operator was not declared. Add it to _ROW_EXPANDING_MAPPERS / set "
                "partition.stream_allow_row_expansion=true, or use partition.recovery_mode=partition."
            )

        # Content-key filename: stable across Ray lineage recompute of this
        # side-effecting tee. If Ray recomputes a lost block with the same rows,
        # it overwrites its own file instead of writing a fresh uuid-named
        # duplicate that would double-count on a future resume. Uses the DEDUPED
        # id set so a 1:many block keys on its input-id footprint (the frontier
        # unit), identical whether or not Ray coalesces sub-rows.
        content_key = hashlib.sha1((",".join(map(str, sorted(set(sorted_ids))))).encode()).hexdigest()[:16]
        stem = f"block_{content_key}_{self.segment_index:04d}"
        block_path = os.path.join(self.data_dir, f"{stem}.parquet")
        # Atomic, durable block write (all columns, including the row-id
        # provenance). Write to a per-writer-unique tmp then os.replace into the
        # content-keyed final name: a reader (a resume reconcile) sees either no
        # file or the whole file, never a partial parquet, even if Ray recomputes
        # this side-effecting block or a speculative duplicate races on the same
        # target. The manifest shard below is only written AFTER this returns, so
        # a manifest never points at a half-written block.
        tmp_block = f"{block_path}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
        pq.write_table(batch, tmp_block)
        if self.fsync:
            self._fsync_path(tmp_block)
        os.replace(tmp_block, block_path)
        if self.fsync:
            # Block durable (file + its dir) before the manifest that points at
            # it, and before the batch flows to the next segment.
            self._fsync_path(block_path)
            self._fsync_path(self.data_dir)

        payload = {
            "schema_version": STREAM_MANIFEST_SCHEMA_VERSION,
            "job_id": self.job_id,
            "segment_index": self.segment_index,
            "op_boundary_idx": self.op_boundary_idx,
            "op_name": self.op_name,
            "block_uri": os.path.relpath(block_path, self.ckpt_dir),
            "frontier_key": "row_id",
            "committed_row_id_ranges": _contiguous_row_id_ranges(sorted_ids),
            "row_count": batch.num_rows,
            "written_at": time.time(),
        }
        # atomic_write_json(fsync=...) flushes the shard + manifest dir when on,
        # so on return both block and manifest are durable for this segment.
        atomic_write_json(os.path.join(self.manifest_dir, f"{stem}.json"), payload, fsync=self.fsync)
        # Pass through unchanged: downstream keeps consuming while later blocks tee.
        return batch


class TempDirManager:
    """Context manager for temporary directory cleanup."""

    def __init__(self, tmp_dir):
        self.tmp_dir = tmp_dir

    def __enter__(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.tmp_dir):
            logger.info(f"Removing tmp dir {self.tmp_dir} ...")
            # in some cases, such as we mount OSS bucket with fuse device
            # using Fluid, cleaning up temporary directories via
            # shutil.rmtree() fails, but os.rmdir() succeeds.
            try:
                shutil.rmtree(self.tmp_dir)
            except OSError as e:
                logger.warning(f"Remove tmp dir with shutil.rmtree() failed: {e}, " "will try os.rmdir()")
                os.rmdir(self.tmp_dir)


# Note: Using Ray Data's built-in map_batches for parallel processing instead of custom remote functions


# Simplified classes for basic functionality
@dataclass
class PartitionResult:
    """Simple result container for partition processing."""

    partition_id: int
    dataset: Optional[Any] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class PartitionMetadata:
    """Metadata for a single partition to enable validation on resume.

    Stores information about each partition that can be used to verify
    that re-partitioning produces the same result on job resumption.
    """

    partition_id: int
    row_count: int
    first_row_hash: str  # Hash of first row for validation
    last_row_hash: str  # Hash of last row for validation
    content_hash: str = ""  # Stable hash of the complete partition contents
    start_row: Optional[int] = None  # Inclusive row offset in the ordered input
    end_row: Optional[int] = None  # Exclusive row offset in the ordered input

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "PartitionMetadata":
        return cls(**data)


@dataclass
class PartitioningInfo:
    """Complete partitioning information for a job.

    Stored alongside checkpoints to enable validation that re-partitioning
    on resume produces identical partitions.
    """

    num_partitions: int
    total_rows: int
    partitions: List[PartitionMetadata] = field(default_factory=list)
    deterministic: bool = True  # Whether deterministic splitting was used
    hash_algorithm: str = _PARTITION_CONTENT_HASH_ALGORITHM

    def to_dict(self) -> Dict:
        return {
            "num_partitions": self.num_partitions,
            "total_rows": self.total_rows,
            "deterministic": self.deterministic,
            "hash_algorithm": self.hash_algorithm,
            "partitions": [p.to_dict() for p in self.partitions],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PartitioningInfo":
        partitions = [PartitionMetadata.from_dict(p) for p in data.get("partitions", [])]
        return cls(
            num_partitions=data["num_partitions"],
            total_rows=data["total_rows"],
            deterministic=data.get("deterministic", True),
            hash_algorithm=data.get("hash_algorithm", _PARTITION_CONTENT_HASH_ALGORITHM),
            partitions=partitions,
        )

    def save(self, path: str) -> None:
        """Save partitioning info to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved partitioning info to {path}")

    @classmethod
    def load(cls, path: str) -> Optional["PartitioningInfo"]:
        """Load partitioning info from JSON file."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load partitioning info from {path}: {e}")
            return None


class PartitionedRayExecutor(ExecutorBase, DAGExecutionMixin, EventLoggingMixin):
    """
    Simplified Ray executor with row-balanced dataset partitioning.

    Features:
    - Single DatasetBuilder loads the full dataset
    - Splits at exact row boundaries so partitions stay balanced and non-empty
    - Processes partitions in parallel with Ray tasks
    - Supports convergence points for global operations
    - Merges results back into a single dataset
    """

    def __init__(self, cfg: Optional[Namespace] = None):
        """Initialize the partitioned Ray executor."""
        super().__init__(cfg)

        self.executor_type = "ray_partitioned"
        self.work_dir = self._resolve_local_path(self.cfg.work_dir)
        self.job_id = self.cfg.get("job_id", None)

        # Initialize temporary directory for Ray operations
        self.tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())

        # Initialize EventLoggingMixin for job management and event logging
        EventLoggingMixin.__init__(self, cfg)

        # Initialize DAGExecutionMixin for AST/DAG functionality
        DAGExecutionMixin.__init__(self)

        # Override strategy methods for partitioned execution
        self._override_strategy_methods()

        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="ray")

        # Partition configuration
        self._configure_partitioning()

        # Checkpoint configuration and manager initialization
        checkpoint_cfg = getattr(self.cfg, "checkpoint", None)
        checkpoint_dir = self._resolve_local_path(
            getattr(self.cfg, "checkpoint_dir", os.path.join(self.work_dir, "checkpoints"))
        )

        if checkpoint_cfg:
            # Use ConfigAccessor to handle both dict and object configurations
            checkpoint_enabled = ConfigAccessor.get(checkpoint_cfg, "enabled", True)
            strategy_str = ConfigAccessor.get(checkpoint_cfg, "strategy", "every_op")
            checkpoint_n_ops = ConfigAccessor.get(checkpoint_cfg, "n_ops", 1)
            checkpoint_op_names = ConfigAccessor.get(checkpoint_cfg, "op_names", [])

            # Parse checkpoint strategy with validation
            try:
                checkpoint_strategy = CheckpointStrategy(strategy_str)
            except ValueError:
                logger.warning(f"Unknown checkpoint strategy: {strategy_str}, defaulting to EVERY_OP")
                checkpoint_strategy = CheckpointStrategy.EVERY_OP
        else:
            checkpoint_enabled = False
            checkpoint_strategy = CheckpointStrategy.DISABLED
            checkpoint_n_ops = 1
            checkpoint_op_names = []

        # Initialize Ray checkpoint manager
        self.ckpt_manager = RayCheckpointManager(
            ckpt_dir=checkpoint_dir,
            checkpoint_enabled=checkpoint_enabled,
            checkpoint_strategy=checkpoint_strategy,
            checkpoint_n_ops=checkpoint_n_ops,
            checkpoint_op_names=checkpoint_op_names,
            event_logger=self,
        )

        logger.info(f"Checkpointing: {'enabled' if self.ckpt_manager.checkpoint_enabled else 'disabled'}")
        if self.ckpt_manager.checkpoint_enabled:
            logger.info(f"Checkpoint strategy: {self.ckpt_manager.checkpoint_strategy.value}")
            logger.info(f"Checkpoint directory: {self.ckpt_manager.ckpt_dir}")

        # Initialize RayExporter for final output
        logger.info("Preparing exporter...")
        # Prepare export extra args, including S3 credentials if export_path is S3
        export_extra_args = dict(self.cfg.export_extra_args) if hasattr(self.cfg, "export_extra_args") else {}

        # If export_path is S3, extract AWS credentials with priority:
        # 1. export_aws_credentials (export-specific)
        # 2. dataset config (for backward compatibility)
        # 3. environment variables (handled by exporter)
        if urlparse(self.cfg.export_path).scheme.lower() == "s3":
            # Pass export-specific credentials if provided.
            # The RayExporter will handle falling back to environment variables or other credential mechanisms.
            if hasattr(self.cfg, "export_aws_credentials") and self.cfg.export_aws_credentials:
                export_aws_creds = self.cfg.export_aws_credentials
                if hasattr(export_aws_creds, "aws_access_key_id"):
                    export_extra_args["aws_access_key_id"] = export_aws_creds.aws_access_key_id
                if hasattr(export_aws_creds, "aws_secret_access_key"):
                    export_extra_args["aws_secret_access_key"] = export_aws_creds.aws_secret_access_key
                if hasattr(export_aws_creds, "aws_session_token"):
                    export_extra_args["aws_session_token"] = export_aws_creds.aws_session_token
                if hasattr(export_aws_creds, "aws_region"):
                    export_extra_args["aws_region"] = export_aws_creds.aws_region
                if hasattr(export_aws_creds, "endpoint_url"):
                    export_extra_args["endpoint_url"] = export_aws_creds.endpoint_url

        self.exporter = RayExporter(
            self.cfg.export_path,
            getattr(self.cfg, "export_type", None),
            getattr(self.cfg, "export_shard_size", 0),
            keep_stats_in_res_ds=getattr(self.cfg, "keep_stats_in_res_ds", True),
            keep_hashes_in_res_ds=getattr(self.cfg, "keep_hashes_in_res_ds", False),
            encrypt_before_export=getattr(self.cfg, "encrypt_before_export", False),
            encryption_key_path=getattr(self.cfg, "encryption_key_path", None),
            **export_extra_args,
        )

    @staticmethod
    def _resolve_local_path(path):
        """Convert a local, non-empty path to an absolute path.

        Ray's writers (e.g. write_parquet) run on workers whose working
        directory may differ from the main process, so relative paths like
        './tmp/...' cannot be resolved correctly and lead to empty checkpoint
        directories. Absolute conversion is applied only to local, non-empty
        paths: remote URIs (e.g. s3://, gs://, hdfs://) are left untouched to
        avoid corrupting their scheme, and empty/None values pass through
        unchanged to avoid raising TypeError.
        """
        if path and not is_remote_path(path):
            return os.path.abspath(path)
        return path

    def _configure_partitioning(self):
        """Configure partitioning based on manual or auto mode."""
        # Get partition configuration
        partition_cfg = getattr(self.cfg, "partition", {})

        # Use ConfigAccessor to handle both dict and object configurations
        mode = ConfigAccessor.get(partition_cfg, "mode", "auto")
        num_of_partitions = ConfigAccessor.get(partition_cfg, "num_of_partitions", 4)
        max_concurrent_partitions = ConfigAccessor.get(partition_cfg, "max_concurrent_partitions", "auto")
        max_gpu_workers_per_device = ConfigAccessor.get(
            partition_cfg,
            "max_gpu_workers_per_device",
            _DEFAULT_MAX_GPU_WORKERS_PER_DEVICE,
        )
        max_concurrent_gpu_probes = ConfigAccessor.get(
            partition_cfg,
            "max_concurrent_gpu_probes",
            "auto",
        )
        gpu_preflight_enabled = ConfigAccessor.get(
            partition_cfg,
            "gpu_preflight_enabled",
            True,
        )
        gpu_probe_timeout_seconds = ConfigAccessor.get(
            partition_cfg,
            "gpu_probe_timeout_seconds",
            None,
        )
        gpu_probe_warmup_batches = ConfigAccessor.get(
            partition_cfg,
            "gpu_probe_warmup_batches",
            _DEFAULT_GPU_PROBE_WARMUP_BATCHES,
        )
        gpu_probe_steady_batches = ConfigAccessor.get(
            partition_cfg,
            "gpu_probe_steady_batches",
            _DEFAULT_GPU_PROBE_STEADY_BATCHES,
        )
        gpu_probe_sample_offset = ConfigAccessor.get(partition_cfg, "gpu_probe_sample_offset", 0)
        gpu_probe_sample_shuffle = ConfigAccessor.get(partition_cfg, "gpu_probe_sample_shuffle", False)
        gpu_probe_sample_seed = ConfigAccessor.get(
            partition_cfg,
            "gpu_probe_sample_seed",
            _DEFAULT_GPU_PROBE_SAMPLE_SEED,
        )
        execution_group_size = ConfigAccessor.get(partition_cfg, "execution_group_size", "auto")
        recovery_mode = ConfigAccessor.get(partition_cfg, "recovery_mode", "streaming")
        stream_segments = ConfigAccessor.get(partition_cfg, "stream_segments", 1)
        stream_fsync = ConfigAccessor.get(partition_cfg, "stream_fsync", True)
        # "auto" (default) => enable 1:many streaming recovery iff a known
        # row-expanding op is present; True/False force it on/off (needed for
        # custom expanders not on _ROW_EXPANDING_MAPPERS).
        stream_allow_row_expansion = ConfigAccessor.get(partition_cfg, "stream_allow_row_expansion", "auto")
        max_initialization_overhead_ratio = ConfigAccessor.get(
            partition_cfg,
            "max_initialization_overhead_ratio",
            _DEFAULT_MAX_INITIALIZATION_OVERHEAD_RATIO,
        )
        partition_size = ConfigAccessor.get(partition_cfg, "size", None)
        max_size_mb = ConfigAccessor.get(partition_cfg, "max_size_mb", 64)

        # Fallback to legacy configuration if partition config is not available
        # or if legacy num_partitions is explicitly set
        if (
            not partition_cfg
            or hasattr(self.cfg, "num_partitions")
            and getattr(self.cfg, "num_partitions", None) is not None
        ):
            mode = "manual"
            num_of_partitions = getattr(self.cfg, "num_partitions", 4)
            if not partition_cfg:
                logger.warning("No partition configuration found, using legacy num_partitions")
            else:
                logger.warning("Legacy num_partitions detected, overriding partition configuration")

        # Cluster-aware auto mode: the sentinel "auto" as a partition count
        # requests a count derived from cluster topology (node count and
        # resolved driver concurrency) instead of a fixed integer. A plain
        # integer always wins, so larger/smaller counts stay one config key
        # away. Numeric strings are normalized so YAML/CLI typing quirks do
        # not flip modes silently.
        partitions_per_node = ConfigAccessor.get(partition_cfg, "partitions_per_node", "auto")
        if isinstance(num_of_partitions, str):
            sentinel = num_of_partitions.strip().lower()
            if sentinel == "auto":
                mode = "auto"
                num_of_partitions = 4  # placeholder until dataset/cluster analysis runs
            else:
                try:
                    num_of_partitions = int(sentinel)
                except ValueError:
                    logger.warning(
                        f"Invalid partition.num_of_partitions={num_of_partitions!r}; " "falling back to auto mode"
                    )
                    mode = "auto"
                    num_of_partitions = 4
        if isinstance(num_of_partitions, float):
            num_of_partitions = max(1, int(num_of_partitions))

        self._partitions_per_node_cfg = partitions_per_node
        self.partition_mode = mode
        self.num_partitions = num_of_partitions
        self.max_concurrent_partitions = max_concurrent_partitions
        if isinstance(gpu_preflight_enabled, str):
            normalized = gpu_preflight_enabled.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                self.gpu_preflight_enabled = True
            elif normalized in {"false", "0", "no", "off"}:
                self.gpu_preflight_enabled = False
            else:
                logger.warning(
                    "Invalid partition.gpu_preflight_enabled=" f"{gpu_preflight_enabled!r}; enabling GPU preflight"
                )
                self.gpu_preflight_enabled = True
        else:
            self.gpu_preflight_enabled = bool(gpu_preflight_enabled)
        try:
            self.max_gpu_workers_per_device = max(1, int(max_gpu_workers_per_device))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid partition.max_gpu_workers_per_device="
                f"{max_gpu_workers_per_device!r}; using {_DEFAULT_MAX_GPU_WORKERS_PER_DEVICE}"
            )
            self.max_gpu_workers_per_device = _DEFAULT_MAX_GPU_WORKERS_PER_DEVICE
        if isinstance(max_concurrent_gpu_probes, str) and max_concurrent_gpu_probes.strip().lower() == "auto":
            self.max_concurrent_gpu_probes = None
        else:
            try:
                self.max_concurrent_gpu_probes = max(1, int(max_concurrent_gpu_probes))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid partition.max_concurrent_gpu_probes=" f"{max_concurrent_gpu_probes!r}; using auto"
                )
                self.max_concurrent_gpu_probes = None
        if gpu_probe_timeout_seconds is None:
            self.gpu_probe_timeout_seconds = None
        else:
            try:
                timeout = float(gpu_probe_timeout_seconds)
                if isinstance(gpu_probe_timeout_seconds, bool) or timeout <= 0:
                    raise ValueError
                self.gpu_probe_timeout_seconds = timeout
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid partition.gpu_probe_timeout_seconds="
                    f"{gpu_probe_timeout_seconds!r}; disabling the probe timeout"
                )
                self.gpu_probe_timeout_seconds = None
        try:
            if isinstance(gpu_probe_warmup_batches, bool):
                raise ValueError
            self.gpu_probe_warmup_batches = max(0, int(gpu_probe_warmup_batches))
        except (TypeError, ValueError):
            logger.warning(
                "Invalid partition.gpu_probe_warmup_batches="
                f"{gpu_probe_warmup_batches!r}; using {_DEFAULT_GPU_PROBE_WARMUP_BATCHES}"
            )
            self.gpu_probe_warmup_batches = _DEFAULT_GPU_PROBE_WARMUP_BATCHES
        try:
            if isinstance(gpu_probe_steady_batches, bool) or int(gpu_probe_steady_batches) < 1:
                raise ValueError
            self.gpu_probe_steady_batches = int(gpu_probe_steady_batches)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid partition.gpu_probe_steady_batches="
                f"{gpu_probe_steady_batches!r}; using {_DEFAULT_GPU_PROBE_STEADY_BATCHES}"
            )
            self.gpu_probe_steady_batches = _DEFAULT_GPU_PROBE_STEADY_BATCHES
        try:
            if isinstance(gpu_probe_sample_offset, bool) or int(gpu_probe_sample_offset) < 0:
                raise ValueError
            self.gpu_probe_sample_offset = int(gpu_probe_sample_offset)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid partition.gpu_probe_sample_offset=" f"{gpu_probe_sample_offset!r}; sampling the dataset head"
            )
            self.gpu_probe_sample_offset = 0
        if isinstance(gpu_probe_sample_shuffle, str):
            normalized = gpu_probe_sample_shuffle.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                self.gpu_probe_sample_shuffle = True
            elif normalized in {"false", "0", "no", "off"}:
                self.gpu_probe_sample_shuffle = False
            else:
                logger.warning(
                    "Invalid partition.gpu_probe_sample_shuffle="
                    f"{gpu_probe_sample_shuffle!r}; sampling the dataset head"
                )
                self.gpu_probe_sample_shuffle = False
        else:
            self.gpu_probe_sample_shuffle = bool(gpu_probe_sample_shuffle)
        if gpu_probe_sample_seed is None:
            self.gpu_probe_sample_seed = None
        else:
            try:
                if isinstance(gpu_probe_sample_seed, bool):
                    raise ValueError
                self.gpu_probe_sample_seed = int(gpu_probe_sample_seed)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid partition.gpu_probe_sample_seed="
                    f"{gpu_probe_sample_seed!r}; using {_DEFAULT_GPU_PROBE_SAMPLE_SEED}"
                )
                self.gpu_probe_sample_seed = _DEFAULT_GPU_PROBE_SAMPLE_SEED
        if isinstance(execution_group_size, str) and execution_group_size.strip().lower() == "auto":
            self.execution_group_size = "auto"
        else:
            try:
                if isinstance(execution_group_size, bool) or int(execution_group_size) < 1:
                    raise ValueError
                self.execution_group_size = int(execution_group_size)
            except (TypeError, ValueError):
                logger.warning(f"Invalid partition.execution_group_size={execution_group_size!r}; using auto")
                self.execution_group_size = "auto"
        try:
            ratio = float(max_initialization_overhead_ratio)
            if isinstance(max_initialization_overhead_ratio, bool) or not 0 < ratio < 1:
                raise ValueError
            self.max_initialization_overhead_ratio = ratio
        except (TypeError, ValueError):
            logger.warning(
                "Invalid partition.max_initialization_overhead_ratio="
                f"{max_initialization_overhead_ratio!r}; using "
                f"{_DEFAULT_MAX_INITIALIZATION_OVERHEAD_RATIO}"
            )
            self.max_initialization_overhead_ratio = _DEFAULT_MAX_INITIALIZATION_OVERHEAD_RATIO
        # Recovery mode selects how the GPU + all-(Mapper/Filter) segment is
        # processed. "streaming" (default) uses the tee-sink row-level frontier
        # that decouples actor reuse from crash blast radius; "partition" forces
        # the plain per-partition path (no execution groups), which is the honest
        # A/B baseline. The execution_group barrier path is retired from selection.
        normalized_recovery = str(recovery_mode).strip().lower() if recovery_mode is not None else "streaming"
        if normalized_recovery not in {"streaming", "partition"}:
            logger.warning(f"Invalid partition.recovery_mode={recovery_mode!r}; using 'streaming'")
            normalized_recovery = "streaming"
        self.recovery_mode = normalized_recovery
        # Number of contiguous op-segments; a tee-sink is inserted after each.
        # More segments => finer resume (re-run only ops after the deepest
        # committed segment) at the cost of more committed blocks. Clamped to
        # >= 1 here; further clamped to <= len(ops) when the segments are built.
        # Blocks are always block-preserving (batch_size=None, no rebatch) so a
        # 1:many op's expansion is not split across committed blocks by us; the
        # former stream_block_size knob was redundant with override_num_blocks
        # / the read block_size and is removed.
        try:
            self.stream_segments = max(1, int(stream_segments)) if stream_segments else 1
        except (TypeError, ValueError):
            logger.warning(f"Invalid partition.stream_segments={stream_segments!r}; using 1")
            self.stream_segments = 1
        # fsync committed block + manifest (and dirs) before the tee returns, so
        # segment k is durable before its rows reach segment k+1 (nested frontier
        # holds under power-loss, not just process-kill).
        self.stream_fsync = bool(stream_fsync) if stream_fsync is not None else True
        # Normalize the 1:many toggle: keep the literal "auto" string (resolved
        # against the known-expander list at _should_use_streaming_recovery time)
        # or coerce to an explicit bool override.
        if isinstance(stream_allow_row_expansion, str) and stream_allow_row_expansion.lower() == "auto":
            self.stream_allow_row_expansion = "auto"
        else:
            self.stream_allow_row_expansion = bool(stream_allow_row_expansion)
        self.partition_size_cfg = partition_size
        self.max_size_mb = max_size_mb

        if mode == "manual":
            if self.partition_size_cfg is None:
                logger.info(f"Manual partition mode: using {self.num_partitions} partitions")
            else:
                logger.info(
                    f"Manual partition mode: targeting {self.partition_size_cfg} samples per partition; "
                    "the partition count will be derived after loading the dataset"
                )
        else:  # auto mode
            logger.info("Auto partition mode: will determine optimal partitioning based on data characteristics")
            if self.partition_size_cfg is not None:
                logger.info(f"Optimizer fallback: targeting {self.partition_size_cfg} samples per partition")

    @staticmethod
    def _partition_count_from_size(dataset, partition_size: int, total_samples: Optional[int] = None) -> int:
        """Derive the nearest partition count for a samples-per-partition target.

        ``total_samples`` lets callers reuse a row count they already paid for
        instead of triggering a second Ray ``count()`` over the same dataset.
        """
        if partition_size <= 0:
            raise ValueError(f"partition_size must be positive, got {partition_size}")

        if total_samples is None:
            if hasattr(dataset, "count"):
                try:
                    total_samples = dataset.count()
                except TypeError:
                    total_samples = len(dataset) if hasattr(dataset, "__len__") else None
            elif hasattr(dataset, "__len__"):
                total_samples = len(dataset)

        if total_samples is None:
            raise RuntimeError(
                "Cannot determine dataset size for sample-based partitioning; "
                "use partition.num_of_partitions instead."
            )
        if isinstance(total_samples, int) and isinstance(partition_size, int):
            quotient, remainder = divmod(total_samples, partition_size)
            return max(1, quotient + int(2 * remainder >= partition_size))
        return max(1, int(total_samples / partition_size + 0.5))

    def _configure_auto_partitioning(self, dataset, ops):
        """Configure partitioning using the optimizer and an optional sample-count fallback."""
        recommendations = None
        total_samples = getattr(self, "_auto_total_samples", None)
        try:
            from data_juicer.core.executor.partition_size_optimizer import (
                auto_configure_resources,
            )

            logger.info("🔧 Auto-configuring partition settings based on data characteristics...")
            recommendations = auto_configure_resources(self.cfg, dataset, ops)
        except ImportError as e:
            logger.warning(f"Could not import partition size optimizer: {e}")
        except Exception as e:
            logger.warning(f"Auto partition configuration failed: {e}")

        recommended_size = None
        recommended_workers = getattr(self.cfg, "np", 4)
        if recommendations is not None:
            recommended_size = ConfigAccessor.get(recommendations, "recommended_partition_size", None)
            recommended_workers = ConfigAccessor.get(recommendations, "recommended_worker_count", recommended_workers)

        # A missing or nonsensical recommendation must not silently size the
        # run; fall back to the explicit partition.size target when there is one.
        if not isinstance(recommended_size, (int, float)) or recommended_size <= 0:
            recommended_size = self.partition_size_cfg
            if recommended_size is not None:
                logger.info(f"Using partition.size fallback: {recommended_size} samples per partition")

        if recommended_size is not None:
            try:
                self.num_partitions = self._partition_count_from_size(
                    dataset, recommended_size, total_samples=total_samples
                )

                # Cap auto-mode work submission at 2x the recommended workers.
                max_partitions = max(32, recommended_workers * 2)
                self.num_partitions = min(self.num_partitions, max_partitions)

                logger.info("📊 Dataset analysis complete:")
                logger.info(f"  Recommended partition size: {recommended_size} samples")
                logger.info(f"  Calculated partitions: {self.num_partitions}")
                logger.info(f"  Recommended workers: {recommended_workers}")

                if not hasattr(self.cfg, "np") or self.cfg.np is None:
                    self.cfg.np = recommended_workers
                    logger.info(f"  Updated worker count to: {recommended_workers}")
            except Exception as e:
                logger.warning(f"Could not calculate partition count from the sample target: {e}")
                logger.info(f"Using configured partition count: {self.num_partitions}")
        else:
            logger.info(f"Using configured partition count: {self.num_partitions}")

        # Keep a real row-count ceiling even when the optimizer is unavailable,
        # so cluster-aware planning never asks for more partitions than rows.
        # ``_split_into_row_balanced_partitions`` is what actually keeps every
        # partition non-empty; this ceiling additionally keeps the recorded plan
        # and the execution-group sizing derived from it honest.
        if total_samples is None and dataset is not None:
            try:
                if hasattr(dataset, "count"):
                    total_samples = dataset.count()
                elif hasattr(dataset, "__len__"):
                    total_samples = len(dataset)
            except Exception as e:
                logger.warning(f"Could not determine dataset size for partition ceiling: {e}")

        # Align the partition count with the real cluster topology. This
        # also runs when the optimizer failed above, so the fallback
        # count still benefits from cluster awareness.
        try:
            self._apply_cluster_partition_bounds(ops, total_samples=total_samples)
        except Exception as e:
            logger.warning(f"Cluster-aware partition bounds failed: {e}")

    def _apply_cluster_partition_bounds(self, ops: List, total_samples: Optional[int] = None) -> None:
        """Align the auto partition count with the cluster topology.

        The data-driven count determines how many partitions the workload
        needs. Cluster topology supplies an upper target, not a lower bound:

        - capacity target: ``max(nodes x partitions_per_node, nodes)``;
        - overlap target: roughly 2x resolved concurrency so tail partitions
          can overlap instead of serializing the final stage.

        A small workload must not be expanded merely to fill theoretical
        worker slots. In particular, fractional GPU requests can imply
        hundreds of slots on a large-memory GPU; treating that capacity as a
        partition floor creates empty partitions and actor scheduling stalls.
        When the row count is known, it is also a hard ceiling so automatic
        mode never asks Ray to create empty partitions.

        Requires ``_resolve_max_concurrent_partitions`` to have run first so
        ``max_concurrent_partitions`` is an integer; otherwise concurrency is
        treated as unknown (0) and only the node-based capacity target applies.
        """
        from data_juicer.utils.ray_cluster_utils import detect_cluster_topology

        topology = detect_cluster_topology()

        concurrency = getattr(self, "max_concurrent_partitions", None)
        if isinstance(concurrency, str) or concurrency is None:
            concurrency = 0
        concurrency = max(0, int(concurrency))

        per_node = self._resolve_partitions_per_node(ops, topology)
        capacity_target = max(topology.num_nodes * per_node, topology.num_nodes)
        target = max(2 * concurrency, capacity_target) if concurrency > 0 else capacity_target

        data_driven = max(1, int(self.num_partitions))
        useful_ceiling = data_driven
        if total_samples is not None:
            useful_ceiling = min(useful_ceiling, max(1, int(total_samples)))

        # Cluster resources cap useful outer parallelism. They must never
        # manufacture work that the data-driven optimizer did not request.
        resolved = max(1, min(useful_ceiling, target))
        if resolved != data_driven:
            logger.info(
                f"Cluster-aware partitioning: adjusting num_partitions from {data_driven} to {resolved} "
                f"(samples={total_samples}, nodes={topology.num_nodes}, "
                f"partitions_per_node={per_node}, concurrency={concurrency})"
            )
        self.num_partitions = resolved

        # Expose the decision for auditing and downstream consumers (e.g.
        # elastic control planes reading driver-side planning state).
        try:
            self.cfg._resolved_partition_plan = {
                "num_partitions": resolved,
                "data_driven_count": data_driven,
                "num_nodes": topology.num_nodes,
                "partitions_per_node": per_node,
                "max_concurrent_partitions": concurrency,
                "total_samples": total_samples,
                "cluster_target": target,
            }
        except (AttributeError, TypeError):
            pass

    def _resolve_partitions_per_node(self, ops: List, topology) -> int:
        """Resolve the per-node partition multiplier.

        An explicit positive integer in ``partition.partitions_per_node``
        always wins. Otherwise the multiplier is derived from the tightest
        per-node parallelism implied by operator GPU requests (one partition
        per node-level GPU slot). Pipelines whose operators do not use GPUs
        fall back to a 2x overlap factor even on GPU clusters, so stage
        boundaries can still overlap without spawning GPU-count-sized
        partitions for CPU-only work.
        """
        configured = getattr(self, "_partitions_per_node_cfg", "auto")
        if not (isinstance(configured, str) and configured.strip().lower() == "auto"):
            try:
                value = int(configured)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                pass
            logger.warning(f"Invalid partition.partitions_per_node={configured!r}; using auto")

        gpus_per_node = topology.gpus_per_node
        if gpus_per_node > 0:
            per_stage_gpus = []
            for op in ops:
                num_gpus = getattr(op, "num_gpus", None)
                gpu_per_worker = float(num_gpus) if num_gpus and float(num_gpus) > 0 else None
                if gpu_per_worker is None:
                    try:
                        if op.use_cuda():
                            gpu_per_worker = 1.0
                    except (AttributeError, RuntimeError):
                        pass
                if gpu_per_worker is not None:
                    per_stage_gpus.append(gpu_per_worker)
            if per_stage_gpus:
                tightest = max(per_stage_gpus)
                return max(1, math.floor(gpus_per_node / tightest + 1e-9))
        # No operator uses GPUs (or the cluster has none): CPU-only overlap
        # factor, even when the cluster itself carries GPUs.
        return 2

    def run(self, load_data_np: Optional[PositiveInt] = None, skip_return=False):
        """
        Run the simplified partitioned dataset processing pipeline.

        Args:
            load_data_np: deprecated, and never applied here: no Ray load
                strategy reads `num_proc`. Use `override_num_blocks` to
                control read parallelism.
            skip_return: Whether to skip returning the dataset
            job_id: Optional job ID to resume from checkpoints

        Returns:
            Processed dataset
        """
        # Use TempDirManager to ensure cleanup of temporary files
        with TempDirManager(self.tmp_dir):
            return self._run_impl(load_data_np, skip_return)

    def _run_impl(self, load_data_np: Optional[PositiveInt] = None, skip_return=False):
        """
        Internal implementation of the run method.
        """
        job_start_time = time.time()

        # Check if user provided a job_id (indicating resumption attempt)
        user_provided_job_id = getattr(self.cfg, "_user_provided_job_id", False)
        resume_requested = getattr(self.cfg, "_resume_requested", False)

        if user_provided_job_id and self.job_id:
            logger.info(f"🔄 User provided job_id: {self.job_id} - attempting to resume job")
            resume_result = self._resume_job(self.job_id)
            if resume_result == "completed":
                logger.info("✅ Job is already completed - nothing to do")
                return None  # Exit gracefully
            elif resume_result == "resuming":
                logger.info("✅ Job resumption successful - will use existing checkpoints")
                is_resuming = True
            else:  # resume_result == "failed"
                if resume_requested:
                    raise RuntimeError(
                        f"Unable to resume job {self.job_id}. " "The existing checkpoints were left unchanged."
                    )
                logger.info("❌ Job resumption failed - starting fresh")
                is_resuming = False
        else:
            if self.job_id:
                logger.info(f"🚀 Starting new job with auto-generated job_id: {self.job_id}")
            else:
                logger.info("🚀 Starting new job")
            is_resuming = False
            if self.job_id and self.ckpt_manager.checkpoint_enabled:
                logger.info(
                    f"Resume token: {self.job_id}. "
                    f"Rerun the original command with --resume {self.job_id} to resume this job."
                )

        self._is_resuming = is_resuming

        if not is_resuming:
            logger.info("🚀 Starting simplified partitioned processing...")
        else:
            logger.info("🔄 Resuming partitioned processing from checkpoints...")

        # Log job start event
        self._log_event(
            event_type=EventType.JOB_START,
            message=(
                "Starting partitioned dataset processing"
                if not is_resuming
                else "Resuming partitioned dataset processing"
            ),
            metadata={
                "num_partitions": self.num_partitions,
                "checkpoint_enabled": self.ckpt_manager.checkpoint_enabled,
                "is_resuming": is_resuming,
                "job_id": self.job_id,
                "user_provided_job_id": user_provided_job_id,
                "resume_requested": resume_requested,
            },
        )

        # Note: Config validation is handled in _resume_job() if resuming

        # Load the full dataset using a single DatasetBuilder
        # Preflight stage 1: validate config before any expensive I/O
        if getattr(self.cfg, "strict_preflight", True):
            from data_juicer.core.preflight import pre_instantiation_check

            pre_instantiation_check(self.cfg.process)

        # Load the full dataset using single DatasetBuilder
        logger.info("Loading dataset with single DatasetBuilder...")

        # Ray Dataset captures a copy of DataContext when it is created. This
        # must therefore happen before DatasetBuilder.load_dataset() so saved
        # row boundaries have the same meaning when a job is resumed.
        self._enable_deterministic_execution()
        dataset = self.datasetbuilder.load_dataset(
            **deprecated_load_data_np_kwargs(load_data_np, self.datasetbuilder.executor_type)
        )
        dataset_schema = dataset.schema()
        columns = dataset_schema.columns

        # Prepare operations
        logger.info("Preparing operations...")
        ops = self._prepare_operators()

        # Preflight stage 2: validate ops against dataset schema and environment
        if getattr(self.cfg, "strict_preflight", True):
            from data_juicer.core.preflight import post_instantiation_check

            post_instantiation_check(ops, dataset_schema, self.cfg)

        # Profile unresolved GPU resources and plan per-operator parallelism
        # only after the generic schema/environment validation has succeeded.
        self._configure_pre_partition_resources(dataset, ops)

        # Streaming tee-sink recovery keeps the input as one lazy stream and
        # never manually partitions, so the whole partition-count negotiation
        # below (including the resume partitioning_info.json requirement) is
        # bypassed; its frontier lives in the stream manifest instead.
        streaming_selected = self._should_use_streaming_recovery(dataset, ops)

        # A resumed job must reuse the saved partition count before DAG
        # initialization. Auto mode can otherwise choose a different count if
        # the Ray cluster resources changed between runs.
        if resume_requested and not streaming_selected:
            saved_info = self._load_partitioning_info()
            if saved_info is None:
                raise RuntimeError(
                    "Explicit resume requires saved partitioning_info.json; "
                    "existing checkpoints were left unchanged."
                )
            self.num_partitions = saved_info.num_partitions
            logger.info(f"Using saved partition count for resume: {self.num_partitions}")
        # Handle auto partition mode BEFORE initializing DAG
        # (DAG needs final partition count). At this point GPU preflight and
        # automatic operator parallelism have both populated the real
        # per-worker requests used by the cluster-aware bounds.
        if streaming_selected:
            logger.info("Streaming tee-sink recovery selected; skipping manual partitioning.")
        elif not resume_requested and self.partition_mode == "auto":
            self._configure_auto_partitioning(dataset, ops)
        elif self.partition_size_cfg is not None:
            # split_at_indices() materializes the input too. Retaining that
            # materialization here lets counting and splitting share one stable
            # snapshot instead of executing the lazy input plan repeatedly.
            dataset.data = dataset.data.materialize()
            self.num_partitions = self._partition_count_from_size(dataset, self.partition_size_cfg)
            logger.info(
                f"Manual sample-based partitioning: {self.num_partitions} partitions "
                f"for a target of {self.partition_size_cfg} samples each"
            )

        # Initialize DAG execution planning with final partition count
        # Pass ops to avoid redundant loading
        self._initialize_dag_execution(self.cfg, ops=ops)

        # Log job start with DAG context
        # Handle both dataset_path (string) and dataset (dict) configurations
        dataset_info = {}
        if hasattr(self.cfg, "dataset_path") and self.cfg.dataset_path:
            dataset_info["dataset_path"] = self.cfg.dataset_path
        if hasattr(self.cfg, "dataset") and self.cfg.dataset:
            dataset_info["dataset"] = self.cfg.dataset

        job_config = {
            **dataset_info,
            "work_dir": self.work_dir,
            "executor_type": self.executor_type,
            "partition": {"size": self.partition_size_cfg},
            "dag_node_count": len(self.pipeline_dag.nodes) if self.pipeline_dag else 0,
            "dag_edge_count": len(self.pipeline_dag.edges) if self.pipeline_dag else 0,
            "parallel_groups_count": len(self.pipeline_dag.parallel_groups) if self.pipeline_dag else 0,
        }
        self.log_job_start(job_config, self.num_partitions)

        # Detect convergence points for global operations
        convergence_points = self._detect_convergence_points(self.cfg)

        if convergence_points:
            logger.info(f"Found convergence points at operations: {convergence_points}")
            final_dataset = self._process_with_convergence(dataset, ops, convergence_points)
        else:
            logger.info("No convergence points found, processing with simple partitioning")
            final_dataset = self._process_with_simple_partitioning(dataset, ops)

        # Export final dataset
        logger.info("Exporting final dataset...")
        self.exporter.export(final_dataset.data, columns=columns)

        job_duration = time.time() - job_start_time
        logger.info(f"✅ Job completed successfully in {job_duration:.2f}s")
        logger.info(f"📁 Output saved to: {self.cfg.export_path}")

        # Log job completion with DAG context
        self.log_job_complete(job_duration, self.cfg.export_path)

        if skip_return:
            return None

        return final_dataset

    def cleanup_temp_files(self):
        """Manually clean up temporary files from previous runs."""
        tmp_base_dir = os.path.join(self.work_dir, ".tmp")
        if os.path.exists(tmp_base_dir):
            logger.info(f"Cleaning up temporary files in {tmp_base_dir}")
            try:
                shutil.rmtree(tmp_base_dir)
            except OSError as e:
                logger.warning(f"Remove tmp dir with shutil.rmtree() failed: {e}, " "will try os.rmdir()")
                os.rmdir(tmp_base_dir)
            logger.info("Temporary files cleaned up successfully")
        else:
            logger.info("No temporary files found to clean up")

    def _process_with_simple_partitioning(self, dataset: RayDataset, ops: List):
        """
        Process dataset with real partitioning using Ray Data's split and union.

        Uses deterministic splitting to ensure reproducible partitions for
        checkpoint resumption.
        """
        # Streaming tee-sink row-level recovery replaces manual partitioning
        # entirely (no split, no per-partition materialize barrier). Dispatch
        # before splitting so it never pays for partitions it will not use.
        if self._should_use_streaming_recovery(dataset, ops):
            return self._process_streaming_with_tee_recovery(dataset, ops)

        logger.info("Processing with real partitioning using Ray Data's split and union...")

        # Split the dataset deterministically with metadata collection
        partitions, partitioning_info = self._split_dataset_deterministic(dataset)
        logger.info(
            f"Partitioning complete: {partitioning_info.num_partitions} partitions, "
            f"{partitioning_info.total_rows} total rows"
        )

        if self._should_use_execution_groups(dataset, ops):
            return self._process_partition_execution_groups(partitions, partitioning_info, ops)

        # Process partitions concurrently. Each worker thread drives one Ray
        # Dataset execution; the actual data processing still runs on Ray.
        requested_max_workers = min(len(partitions), self.max_concurrent_partitions)
        if getattr(self, "_throughput_planned_op_ids", set()):
            # Resume and unsupported operator graphs use the legacy per-partition
            # path. Keep it sequential so the global actor plan is never
            # multiplied by the number of driver threads.
            requested_max_workers = 1
        max_workers = self._limit_partition_workers_for_explicit_actors(ops, requested_max_workers)
        logger.info(
            f"Processing {len(partitions)} partitions with up to {max_workers} concurrent "
            "driver threads and checkpointing support..."
        )
        processed_partitions = [None] * len(partitions)
        original_concurrency = self._scale_operator_parallelism(ops, max_workers)

        try:
            # Capture the per-partition resource plan in an isolated template.
            # Restore the original operators before any worker starts so later
            # convergence stages never depend on thread-pool shutdown timing.
            isolate_operators = max_workers > 1
            partition_ops_template = self._clone_partition_operators(ops) if isolate_operators else ops
        finally:
            self._restore_operator_parallelism(original_concurrency)

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ray-partition") as executor:
            futures = {
                executor.submit(
                    self._process_partition,
                    partition,
                    i,
                    len(partitions),
                    partition_ops_template,
                    isolate_operators,
                ): i
                for i, partition in enumerate(partitions)
            }
            try:
                for future in as_completed(futures):
                    partition_id, processed_data = future.result()
                    processed_partitions[partition_id] = processed_data
            except BaseException:
                # Running Ray Dataset executions cannot be forcefully
                # interrupted here, but queued driver work should not start
                # after the job has already failed.
                for future in futures:
                    future.cancel()
                raise

        # Merge all processed partitions back into a single dataset
        logger.info("Merging processed partitions...")
        if len(processed_partitions) == 1:
            merged_dataset = processed_partitions[0]
        else:
            # Union all partitions
            merged_dataset = processed_partitions[0]
            for partition in processed_partitions[1:]:
                merged_dataset = merged_dataset.union(partition)

        # Return as RayDataset wrapper
        return RayDataset(merged_dataset, cfg=self.cfg)

    def _should_use_execution_groups(self, dataset: RayDataset, ops: List) -> bool:
        """The execution-group barrier path is retired from selection.

        recovery_mode=streaming is handled by the tee-sink frontier (dispatched
        before this is ever reached), and recovery_mode=partition is defined as
        the plain per-partition path — "no execution groups" — so this must not
        route a partition-mode run into the group barrier. Keeping the method
        (rather than deleting the branch) preserves the reserved-column guard as
        a single documented decision point should the path ever be revived.
        """
        # Guard against the reserved logical-partition column even though we no
        # longer select the group path, so a revival keeps the invariant.
        try:
            names = self._schema_names(dataset.data)
        except Exception:
            names = set()
        if _LOGICAL_PARTITION_COLUMN in names:
            raise RuntimeError(f"Input dataset contains reserved execution-group column {_LOGICAL_PARTITION_COLUMN!r}.")
        return False

    def _should_use_streaming_recovery(self, dataset: RayDataset, ops: List) -> bool:
        """Return whether this segment should use streaming tee-sink row-level
        recovery instead of manual per-partition checkpointing.

        Cached on first call (both ``_run_impl`` and
        ``_process_with_simple_partitioning`` consult it). Streaming replaces
        #1054's per-partition ``.materialize()`` barrier with one lazy stream +
        one hot pass-through sink actor pool, decoupling model loads (= pool
        size) from crash blast radius (= uncommitted rows).
        """
        cached = getattr(self, "_streaming_selected_cache", None)
        if cached is not None:
            return cached

        def _decide() -> bool:
            if getattr(self, "recovery_mode", "streaming") != "streaming":
                return False
            if self.ckpt_manager is None or self.ckpt_manager.checkpoint_strategy == CheckpointStrategy.DISABLED:
                logger.info("Streaming recovery disabled: checkpointing is off.")
                return False
            # A single lazy segment only keeps a meaningful row-id frontier for
            # row-preserving / row-dropping map-style ops (Phase 1 envelope,
            # identical to execution grouping).
            if any(not isinstance(op, (Mapper, Filter)) for op in ops):
                logger.info("Streaming recovery disabled: segment contains a dataset-level operator.")
                return False
            # Cardinality handling: a row-EXPANDING (1:many) mapper copies a
            # parent id onto each sub-row. The row-id frontier represents this
            # fine -- it commits a SET of INPUT ids, decoupled from output count
            # (same decoupling that already makes row-dropping filters safe), and
            # the block-preserving tee keeps a parent's whole expansion in one
            # committed block. So instead of failing closed we ENABLE expansion
            # mode and relax the tee's uniqueness guard for this run. Detection is
            # best-effort via the known-expander list; a config override forces it
            # for custom expanders. (out>in still requires the atomic-per-block
            # invariant, which batch_size=None gives.)
            force = getattr(self, "stream_allow_row_expansion", "auto")
            expanders = [getattr(op, "_name", "") for op in ops if getattr(op, "_name", "") in _ROW_EXPANDING_MAPPERS]
            if isinstance(force, str) and force.lower() == "auto":
                self._stream_allow_expansion = bool(expanders)
            else:
                self._stream_allow_expansion = bool(force)
            if self._stream_allow_expansion:
                logger.info(
                    "Streaming recovery: row-expanding (1:many) mode ENABLED "
                    f"(detected {expanders or 'via config override'}); tee uniqueness guard relaxed."
                )
            # Block + manifest commits rely on POSIX-atomic os.replace; a remote
            # checkpoint dir gives no such guarantee, so fall back.
            if is_remote_path(self.ckpt_manager.ckpt_dir):
                logger.info("Streaming recovery disabled: checkpoint dir is a remote path.")
                return False
            try:
                names = self._schema_names(dataset.data)
            except Exception:
                names = set()
            if _ROW_ID_COLUMN in names:
                raise RuntimeError(f"Input dataset contains reserved streaming row-id column {_ROW_ID_COLUMN!r}.")
            return True

        result = _decide()
        self._streaming_selected_cache = result
        return result

    def _resolve_tee_sink_concurrency(self, ops: List) -> int:
        """Fixed actor-pool size for the pass-through tee sink.

        The sink is I/O-bound (one parquet block + one atomic manifest shard per
        input block) and must stay a small HOT pool so it never turns into a
        materialize barrier.
        """
        raw = getattr(self, "max_concurrent_partitions", 4)
        try:
            base = int(raw)
        except (TypeError, ValueError):
            base = 4
        return max(1, min(4, base))

    def _stamp_row_ids(self, dataset: RayDataset) -> RayDataset:
        """Inject a dense, unique, read-order-stable global row id.

        preserve_order is already enabled for deterministic execution, so a
        materialized snapshot zipped with ``ray.data.range(N)`` produces
        identical ids across runs (verified: zip block-alignment is dense +
        unique + deterministic in experiments/seg_grid).
        """
        base = dataset.data.materialize()
        n = base.count()
        row_ids = ray.data.range(n).rename_columns({"id": _ROW_ID_COLUMN})
        stamped = base.zip(row_ids)
        return RayDataset(stamped, cfg=self.cfg)

    def _split_ops_into_segments(self, ops: List) -> List[dict]:
        """Split the op chain into ``stream_segments`` contiguous, non-empty,
        as-evenly-sized-as-possible segments.

        Each segment records the ops it runs, the GLOBAL index of its last op
        (the tee boundary), and that op's name for the manifest. The count is
        clamped to ``len(ops)`` so every segment has at least one op.
        """
        n = len(ops)
        s = max(1, min(int(getattr(self, "stream_segments", 1)), n)) if n else 1
        force = getattr(self, "stream_allow_row_expansion", "auto")
        force_auto = isinstance(force, str) and force.lower() == "auto"
        segments: List[dict] = []
        start = 0
        for i in range(s):
            size = n // s + (1 if i < n % s else 0)
            seg_ops = ops[start : start + size]
            boundary_idx = start + size - 1
            op_name = getattr(ops[boundary_idx], "_name", f"op_{boundary_idx}") if seg_ops else "identity"
            # Per-segment cardinality guard (P2): relax the tee's row-id
            # uniqueness check ONLY for the segment(s) that actually contain a
            # row-expanding op. A single global flag would disarm the guard for
            # every 1:1 segment too, hiding a buggy op that duplicates ids
            # outside the declared expander. Under "auto" we detect per segment;
            # an explicit True/False config forces all segments uniformly.
            if force_auto:
                seg_allow = any(getattr(op, "_name", "") in _ROW_EXPANDING_MAPPERS for op in seg_ops)
            else:
                seg_allow = bool(force)
            segments.append(
                {
                    "ops": seg_ops,
                    "boundary_idx": boundary_idx,
                    "op_name": op_name,
                    "allow_expansion": seg_allow,
                }
            )
            start += size
        return segments

    def _thread_segments(self, source_data, segments: List[dict], start_seg: int, concurrency: int):
        """Thread a lazy stream through segments ``start_seg..end``, applying
        each segment's ops then a pass-through tee, with NO materialize barrier
        between segments (one lazy plan)."""
        data = source_data
        for k in range(start_seg, len(segments)):
            seg = segments[k]
            data = RayDataset(data, cfg=self.cfg).process(seg["ops"]).data
            tee_kwargs = dict(
                ckpt_dir=self.ckpt_manager.ckpt_dir,
                data_dir=self.ckpt_manager.stream_data_dir(k),
                manifest_dir=self.ckpt_manager.stream_manifest_dir(k),
                segment_index=k,
                op_boundary_idx=seg["boundary_idx"],
                op_name=seg["op_name"],
                job_id=self.job_id,
                row_id_column=_ROW_ID_COLUMN,
                fsync=getattr(self, "stream_fsync", True),
                allow_expansion=bool(seg.get("allow_expansion", getattr(self, "_stream_allow_expansion", False))),
            )
            # batch_size None => one tee block per upstream block (block-preserving,
            # no rebatch) so a 1:many expansion is never split across our blocks.
            data = data.map_batches(
                _StreamTeeSink,
                fn_constructor_kwargs=tee_kwargs,
                concurrency=concurrency,
                batch_format="pyarrow",
                batch_size=None,
            )
        return data

    def _stream_run_path(self) -> str:
        """Path to the run-level streaming provenance manifest."""
        return os.path.join(self.ckpt_manager.ckpt_dir, "stream_run.json")

    def _compute_stream_run_signature(self, stamped: RayDataset, segments: List[dict], ops: List) -> dict:
        """Capture what the committed frontier is only valid against.

        The row-id frontier commits INPUT row positions. It is only meaningful
        for the SAME input, read in the SAME order, run through the SAME op
        chain split the SAME way. This signature pins all of that so resume can
        fail closed instead of splicing a stale frontier onto different work:

        - ``input_content_hash`` / ``input_row_count``: an order-sensitive
          content hash over the stamped input (row-id included -- it is a pure
          function of read order, so the hash captures both content AND order).
          A changed input file, a reorder, or a row count change all flip it.
        - ``op_names`` / ``segment_layout`` / ``stream_segments``: the op chain
          and how it was cut into tee boundaries. Resuming with an edited
          pipeline would map committed frontiers onto the wrong ops.
        - ``job_id`` / ``allow_expansion`` / ``schema_version``: run identity and
          manifest-format guards.
        """
        content_hash, row_count = self._compute_partition_content_hash(stamped.data)
        return {
            "schema_version": STREAM_MANIFEST_SCHEMA_VERSION,
            "job_id": self.job_id,
            "row_id_column": _ROW_ID_COLUMN,
            "stream_segments": len(segments),
            "op_names": [getattr(op, "_name", f"op_{i}") for i, op in enumerate(ops)],
            "segment_layout": [{"boundary_idx": s["boundary_idx"], "op_name": s["op_name"]} for s in segments],
            "allow_expansion": bool(getattr(self, "_stream_allow_expansion", False)),
            "input_content_hash": content_hash,
            "input_row_count": row_count,
        }

    def _write_or_validate_stream_run(self, signature: dict, resuming: bool) -> None:
        """Fail closed on resume unless the provenance signature matches.

        Fresh run: record ``signature`` durably. Resume: the committed frontier
        we are about to splice in is only trustworthy if the manifest exists and
        every critical field matches what we are running NOW. A missing manifest
        (frontier of unknown provenance) or any mismatch (different input, op
        chain, or segmentation) raises instead of silently emitting a corrupted
        mix of old committed blocks and freshly reprocessed rows.
        """
        path = self._stream_run_path()
        if not resuming:
            atomic_write_json(path, signature, fsync=getattr(self, "stream_fsync", True))
            logger.info(
                f"Streaming provenance recorded: {signature['input_row_count']} input row(s), "
                f"content_hash={signature['input_content_hash'][:16]}..., "
                f"segments={signature['stream_segments']}."
            )
            return

        if not os.path.exists(path):
            raise RuntimeError(
                f"Streaming resume refused: no provenance manifest at {path}. The committed frontier "
                "cannot be trusted against the current input/op-chain. Start a fresh job_id, or use "
                "partition.recovery_mode=partition."
            )
        try:
            with open(path) as f:
                prior = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Streaming resume refused: unreadable provenance manifest {path}: {e}")

        # Critical fields whose change invalidates the committed frontier.
        critical = [
            "schema_version",
            "row_id_column",
            "stream_segments",
            "op_names",
            "segment_layout",
            "input_content_hash",
            "input_row_count",
        ]
        mismatches = []
        for key in critical:
            if prior.get(key) != signature.get(key):
                mismatches.append(f"{key}: checkpoint={prior.get(key)!r} != current={signature.get(key)!r}")
        # job_id is expected to match by construction (ckpt dir is job-scoped);
        # a mismatch means a cross-wired checkpoint dir -- also fatal.
        if prior.get("job_id") != signature.get("job_id"):
            mismatches.append(f"job_id: checkpoint={prior.get('job_id')!r} != current={signature.get('job_id')!r}")
        if mismatches:
            detail = "\n  - ".join(mismatches)
            raise RuntimeError(
                "Streaming resume refused: provenance mismatch -- the committed frontier does not "
                "belong to the current run. Resuming would splice stale committed blocks onto "
                f"different work and corrupt the output. Mismatches:\n  - {detail}\n"
                "Start a fresh job_id, or use partition.recovery_mode=partition."
            )
        logger.info(
            f"Streaming provenance validated on resume: {signature['input_row_count']} input row(s), "
            f"content_hash={signature['input_content_hash'][:16]}..., segments={signature['stream_segments']}."
        )

    def _process_streaming_with_tee_recovery(self, dataset: RayDataset, ops: List) -> RayDataset:
        """Multi-segment, block-level streaming recovery.

        The Mapper/Filter chain is split into ``stream_segments`` contiguous
        segments; a hot pass-through tee actor pool durably commits (block
        parquet + atomic manifest shard) after each segment WITHOUT a
        materialize barrier -- still one lazy stream. Rows carry a stable global
        id, so on resume each segment's committed frontier F_k is reconciled
        (nested F_0 >= F_1 >= ... >= F_{S-1} under kill + local FS; fsync extends
        that to power-loss). A row is re-processed only from the segment AFTER
        the DEEPEST tee that already committed it -- so the crash blast radius is
        (uncommitted rows) x (ops after their deepest committed segment), strictly
        finer than the single-segment whole-chain re-run.
        """
        stamped = self._stamp_row_ids(dataset)
        segments = self._split_ops_into_segments(ops)
        s = len(segments)
        concurrency = self._resolve_tee_sink_concurrency(ops)
        resuming = getattr(self, "_is_resuming", False)
        logger.info(
            f"Streaming tee-sink recovery: segments={s}, concurrency={concurrency}, "
            f"fsync={getattr(self, 'stream_fsync', True)}, resuming={resuming}"
        )

        # Provenance gate (P0): a committed frontier is only valid against the
        # SAME input, order, op chain, and segmentation. Fail closed on resume
        # unless the recorded signature matches exactly; record it on a fresh
        # run. Done BEFORE any reconcile so a stale/foreign frontier is never
        # spliced into the output.
        signature = self._compute_stream_run_signature(stamped, segments, ops)
        self._write_or_validate_stream_run(signature, resuming)

        # Reconcile every segment's committed frontier (resume only).
        frontiers: List[List[Tuple[int, int]]] = [[] for _ in range(s)]
        block_uris: List[List[str]] = [[] for _ in range(s)]
        if resuming:
            for k in range(s):
                fr, bu, sc = self.ckpt_manager.reconcile_stream_frontier(k)
                frontiers[k], block_uris[k] = fr, bu
                logger.info(
                    f"Streaming resume: segment {k} has {sc} committed shard(s), "
                    f"{sum(hi - lo for lo, hi in fr)} committed row(s)."
                )

        # Each row enters the pipeline once, right AFTER the deepest tee that
        # committed it. Partition the work into disjoint streams by that depth
        # and union their endpoints:
        #   - depth S-1 (fully done): restore final output, no reprocessing.
        #   - depth k in [0, S-2]: restore segment k's output for rows in
        #     F_k \ F_{k+1}, run segments k+1..S-1.
        #   - fresh (not in F_0): run all segments from the raw stamped input.
        # (block_uris[k] contains exactly F_k's rows, so filtering out F_{k+1}
        #  yields F_k \ F_{k+1} without an explicit range difference.)
        output_streams = []

        if resuming and block_uris[s - 1]:
            # Already through the whole chain -- restore verbatim, no ops/tee.
            output_streams.append(ray.data.read_parquet(block_uris[s - 1]))

        if resuming:
            for k in range(s - 2, -1, -1):
                if not block_uris[k]:
                    continue
                next_frontier = frontiers[k + 1]
                restored_k = ray.data.read_parquet(block_uris[k])
                if next_frontier:
                    is_committed = make_row_id_membership(next_frontier)
                    restored_k = restored_k.filter(lambda row, m=is_committed: not m(int(row[_ROW_ID_COLUMN])))
                output_streams.append(self._thread_segments(restored_k, segments, k + 1, concurrency))

        # Fresh rows: everything not yet committed by the first tee.
        f0 = frontiers[0] if resuming else []
        if f0:
            is_committed_f0 = make_row_id_membership(f0)
            fresh_src = stamped.data.filter(lambda row, m=is_committed_f0: not m(int(row[_ROW_ID_COLUMN])))
        else:
            fresh_src = stamped.data
        output_streams.append(self._thread_segments(fresh_src, segments, 0, concurrency))

        merged = output_streams[0]
        for stream in output_streams[1:]:
            merged = merged.union(stream)

        # Row-id provenance is internal; strip it before final export.
        merged = merged.drop_columns([_ROW_ID_COLUMN])

        # Terminal materialize == exactly-once execution of this streaming plan.
        # The tee sinks have side effects (block parquet + manifest shard); Ray
        # runs a lazy plan once per terminal action, and the downstream exporter
        # alone triggers TWO (a .columns() schema probe then the write), which
        # would double-commit every block. Collecting here pins one streaming
        # pass -- each tee commits its blocks once and manifests are written
        # incrementally as blocks flow -- while the exporter's later
        # .columns()/write read cached blocks without re-teeing. This is a single
        # end-of-plan barrier, NOT #1054's per-op-group barriers: hot actor
        # pools, row-level crash blast.
        merged = merged.materialize()
        return RayDataset(merged, cfg=self.cfg)

    def _resolve_execution_group_size(
        self,
        partitioning_info: PartitioningInfo,
        ops: List,
    ) -> int:
        """Choose how many logical partitions reuse one set of actor pools."""
        total_partitions = max(1, partitioning_info.num_partitions)
        configured = getattr(self, "execution_group_size", "auto")
        if not (isinstance(configured, str) and configured.lower() == "auto"):
            return min(total_partitions, max(1, int(configured)))

        plan = getattr(self, "_resolved_throughput_actor_plan", None) or {}
        stages = plan.get("operators") or []
        if not stages or partitioning_info.total_rows <= 0:
            return total_partitions

        # Every stage actor pool is constructed once per execution group. The
        # stages themselves are sequential, while actors within one stage are
        # generally initialized concurrently; max(init) per stage is therefore
        # a useful conservative wall-time estimate.
        initialization_seconds = sum(float(stage.get("initialization_seconds", 0) or 0) for stage in stages)
        source_capacity = min(float(stage.get("source_equivalent_rows_per_second", 0) or 0) for stage in stages)
        if initialization_seconds <= 0 or source_capacity <= 0:
            return total_partitions

        rows_per_partition = partitioning_info.total_rows / total_partitions
        processing_seconds = rows_per_partition / source_capacity
        ratio = getattr(
            self,
            "max_initialization_overhead_ratio",
            _DEFAULT_MAX_INITIALIZATION_OVERHEAD_RATIO,
        )
        required_processing = initialization_seconds * (1 - ratio) / ratio
        group_size = math.ceil(required_processing / max(processing_seconds, 1e-9))
        return min(total_partitions, max(1, group_size))

    @staticmethod
    def _tag_execution_partition(partition, partition_id: int):
        return partition.map_batches(
            _add_logical_partition_id,
            fn_kwargs={"partition_id": partition_id},
            batch_format="pyarrow",
        )

    @staticmethod
    def _logical_partition_view(dataset, partition_id: int):
        return dataset.filter(
            _matches_logical_partition,
            fn_kwargs={"partition_id": partition_id},
        ).drop_columns([_LOGICAL_PARTITION_COLUMN])

    @staticmethod
    def _schema_names(dataset) -> set:
        schema = dataset.schema()
        return set(getattr(schema, "names", None) or getattr(schema, "columns", None) or [])

    def _process_partition_execution_groups(
        self,
        partitions: List,
        partitioning_info: PartitioningInfo,
        ops: List,
    ) -> RayDataset:
        """Run multiple checkpoint partitions through one actor-pool lifecycle."""
        group_size = self._resolve_execution_group_size(partitioning_info, ops)
        execution_group_count = math.ceil(len(partitions) / group_size)
        plan = {
            "logical_partitions": len(partitions),
            "execution_group_size": group_size,
            "execution_groups": execution_group_count,
            "actor_pool_initializations_per_operator": execution_group_count,
            "checkpoint_granularity": "logical_partition",
        }
        try:
            self.cfg._resolved_execution_group_plan = plan
        except (AttributeError, TypeError):
            pass
        logger.info(
            f"GPU execution grouping: {len(partitions)} logical checkpoint partitions -> "
            f"{execution_group_count} execution group(s), up to {group_size} partitions per actor lifecycle."
        )

        processed_groups = []
        for start in range(0, len(partitions), group_size):
            group_partitions = partitions[start : start + group_size]
            partition_ids = list(range(start, start + len(group_partitions)))
            for partition_id in partition_ids:
                self._log_event(
                    event_type=EventType.PARTITION_START,
                    message=(
                        f"Starting logical partition {partition_id + 1}/{len(partitions)} "
                        f"in shared GPU execution group"
                    ),
                    partition_id=partition_id,
                )

            tagged = [
                self._tag_execution_partition(partition, partition_id)
                for partition, partition_id in zip(group_partitions, partition_ids)
            ]
            combined = tagged[0]
            for tagged_partition in tagged[1:]:
                combined = combined.union(tagged_partition)
            try:
                processed = self._process_execution_group_with_checkpointing(
                    self._wrap_with_precomputed_parallelism(combined),
                    partition_ids,
                    ops,
                )
            except Exception as error:
                for partition_id in partition_ids:
                    self.log_partition_failed(partition_id, str(error), retry_count=0)
                raise

            processed_groups.append(processed.data)
            for partition_id in partition_ids:
                self._log_event(
                    event_type=EventType.PARTITION_COMPLETE,
                    message=(
                        f"Completed logical partition {partition_id + 1}/{len(partitions)} "
                        f"in shared GPU execution group"
                    ),
                    partition_id=partition_id,
                )

        merged = processed_groups[0]
        for processed in processed_groups[1:]:
            merged = merged.union(processed)
        return RayDataset(merged, cfg=self.cfg)

    def _process_execution_group_with_checkpointing(
        self,
        dataset: RayDataset,
        partition_ids: List[int],
        ops: List,
    ) -> RayDataset:
        """Execute once, while writing independent checkpoints for each tag."""
        if not self.ckpt_manager.checkpoint_enabled:
            current = dataset.process(ops)
            current.data = current.data.materialize()
            if _LOGICAL_PARTITION_COLUMN not in self._schema_names(current.data):
                raise RuntimeError(
                    "An operator removed the internal logical-partition tag; "
                    "shared GPU execution cannot preserve checkpoint identity."
                )
            current.data = current.data.drop_columns([_LOGICAL_PARTITION_COLUMN])
            return current

        op_groups = self.ckpt_manager.group_operations_for_checkpointing(ops)
        current = dataset
        for group_idx, (start_idx, end_idx, group_ops) in enumerate(op_groups):
            if not group_ops:
                continue
            input_rows = current.data.count()
            started = time.time()
            for partition_id in partition_ids:
                if self.pipeline_dag:
                    self._pre_execute_operations_with_dag_monitoring(group_ops, partition_id=partition_id)
            current = current.process(group_ops)
            current.data = current.data.materialize()
            if _LOGICAL_PARTITION_COLUMN not in self._schema_names(current.data):
                raise RuntimeError(f"Operation group {group_idx + 1} removed the internal logical-partition tag.")
            duration = time.time() - started
            output_rows = current.data.count()
            metrics = {"duration": duration, "input_rows": input_rows, "output_rows": output_rows}
            for partition_id in partition_ids:
                if self.pipeline_dag:
                    self._post_execute_operations_with_dag_monitoring(
                        group_ops,
                        partition_id=partition_id,
                        metrics=metrics,
                    )

            last_op_idx = end_idx - 1
            last_op_name = ops[last_op_idx]._name
            if self.ckpt_manager.should_checkpoint(last_op_idx, last_op_name):
                for partition_id in partition_ids:
                    logical_dataset = self._wrap_with_precomputed_parallelism(
                        self._logical_partition_view(current.data, partition_id)
                    )
                    self.ckpt_manager.save_checkpoint(
                        logical_dataset,
                        last_op_idx,
                        last_op_name,
                        partition_id,
                        cfg=self.cfg,
                    )

        current.data = current.data.drop_columns([_LOGICAL_PARTITION_COLUMN])
        return current

    def _process_partition(
        self,
        partition,
        partition_id: int,
        total_partitions: int,
        ops: List,
        isolate_operators: bool = True,
    ):
        """Process one partition while preserving its position in the final union."""
        logger.info(f"Processing partition {partition_id + 1}/{total_partitions}")

        self._log_event(
            event_type=EventType.PARTITION_START,
            message=f"Starting processing of partition {partition_id + 1}/{total_partitions}",
            partition_id=partition_id,
        )

        try:
            # RayDataset may temporarily mutate operator attributes while
            # building an execution plan (for example runtime_env fallback and
            # tracing). Give every concurrent partition its own instances.
            partition_ops = self._clone_partition_operators(ops) if isolate_operators else ops
            partition_dataset = self._wrap_with_precomputed_parallelism(partition)
            processed_partition = self._process_with_checkpointing(partition_dataset, partition_id, partition_ops)
        except Exception as error:
            self.log_partition_failed(partition_id, str(error), retry_count=0)
            raise

        self._log_event(
            event_type=EventType.PARTITION_COMPLETE,
            message=f"Completed processing of partition {partition_id + 1}/{total_partitions}",
            partition_id=partition_id,
        )

        return partition_id, processed_partition.data

    @staticmethod
    def _clone_partition_operators(ops: List) -> List:
        """Create an isolated operator graph for one partition."""
        try:
            return copy.deepcopy(ops)
        except Exception as error:
            raise RuntimeError(
                "Failed to isolate operators for concurrent partition execution. "
                "All partition operators must support deep copying."
            ) from error

    def _configure_operator_parallelism(self, ops: List) -> None:
        """Plan global actor resources once before partition threads.

        A positive explicit actor ``num_proc`` is treated as a global job
        budget, matching its meaning in the non-partitioned Ray executor. Auto
        parallelism is also calculated globally here and divided later.
        """
        self._auto_parallel_op_ids = set()
        self._explicit_actor_op_ids = {
            id(op) for op in ops if op.use_ray_actor() and self._actor_pool_capacity(op.num_proc) is not None
        }

        if not ConfigAccessor.get(self.cfg, "auto_op_parallelism", True):
            return

        auto_ops = [op for op in ops if op.use_auto_proc()]
        if not auto_ops:
            return

        from data_juicer.utils.process_utils import calculate_ray_np

        self._auto_parallel_op_ids = {id(op) for op in auto_ops}
        calculate_ray_np(ops)

    def _configure_pre_partition_resources(self, dataset, ops: List) -> None:
        """Resolve worker resources before partition sizing and execution."""
        if not getattr(self, "gpu_preflight_enabled", True):
            logger.info(
                "GPU preflight is disabled by partition.gpu_preflight_enabled=false; "
                "using the configured operator resources and actor counts."
            )
            self._configure_operator_parallelism(ops)
            self._resolve_max_concurrent_partitions(ops)
            return

        # Resource profiling is independent from the logical partition policy.
        # Manual partition counts still need measured GPU memory and throughput
        # for actor planning.
        from data_juicer.core.executor.gpu_memory_probe import GPUMemoryProbe

        GPUMemoryProbe(
            self.work_dir,
            max_gpu_workers_per_device=getattr(
                self,
                "max_gpu_workers_per_device",
                _DEFAULT_MAX_GPU_WORKERS_PER_DEVICE,
            ),
            max_concurrent_probes=getattr(self, "max_concurrent_gpu_probes", None),
            probe_timeout_seconds=getattr(self, "gpu_probe_timeout_seconds", None),
            warmup_batches=getattr(
                self,
                "gpu_probe_warmup_batches",
                _DEFAULT_GPU_PROBE_WARMUP_BATCHES,
            ),
            steady_batches=getattr(
                self,
                "gpu_probe_steady_batches",
                _DEFAULT_GPU_PROBE_STEADY_BATCHES,
            ),
            sample_offset=getattr(self, "gpu_probe_sample_offset", 0),
            sample_shuffle=getattr(self, "gpu_probe_sample_shuffle", False),
            sample_seed=getattr(self, "gpu_probe_sample_seed", _DEFAULT_GPU_PROBE_SAMPLE_SEED),
        ).resolve(dataset, ops)

        total_samples = self._count_dataset_rows(dataset)
        if total_samples is not None:
            self._auto_total_samples = total_samples

        # Every partition shares the same operator objects. Automatic actor
        # parallelism and outer partition concurrency must see probe results.
        self._configure_operator_parallelism(ops)
        self._configure_throughput_aware_gpu_parallelism(ops, total_samples=total_samples)
        self._resolve_max_concurrent_partitions(ops, total_samples=total_samples)
        self._cap_auto_gpu_operator_parallelism(
            ops,
            getattr(self, "_safe_gpu_workers_per_stage", None),
            total_samples=total_samples,
        )

    def _configure_auto_operator_parallelism(self, ops: List) -> None:
        """Compatibility wrapper for the former auto-only planner."""
        self._configure_operator_parallelism(ops)

    @staticmethod
    def _gpu_actor_requests_fit(
        stage_specs: List[Dict[str, Any]], counts: List[int], total_gpus: float, gpu_devices_per_node=None
    ) -> bool:
        """Check scheduling and measured-memory fractions on individual GPUs.

        Requests are processed largest first and placed with a best-fit policy,
        preserving the largest remaining contiguous shares for later requests.
        """
        device_count = int(math.floor(total_gpus + 1e-9))
        if device_count < 1:
            return False
        requests = []
        for spec, count in zip(stage_specs, counts):
            requests.extend([(spec["num_gpus"], spec["memory_fraction"], spec["max_per_device"])] * count)
        requests.sort(key=lambda item: (max(item[0], item[1]), item[1]), reverse=True)
        node_sizes = gpu_devices_per_node if gpu_devices_per_node is not None else (device_count,)
        devices = [
            {"gpu": 0.0, "memory": 0.0, "workers": 0, "node": node}
            for node, size in enumerate(node_sizes)
            for _ in range(size)
        ]
        for gpu_fraction, memory_fraction, max_per_device in requests:
            if gpu_fraction > 1:
                if not float(gpu_fraction).is_integer():
                    return False
                # Ray reserves whole, same-node devices for integer requests.
                for node in range(len(node_sizes)):
                    free = [device for device in devices if device["node"] == node and device["workers"] == 0]
                    if len(free) >= int(gpu_fraction):
                        for device in free[: int(gpu_fraction)]:
                            device.update(gpu=1.0, memory=1.0, workers=1)
                        break
                else:
                    return False
                continue
            eligible = [
                device
                for device in devices
                if (
                    device["gpu"] + gpu_fraction <= 1.0 + 1e-9
                    and device["memory"] + memory_fraction <= 1.0 + 1e-9
                    and device["workers"] < max_per_device
                )
            ]
            if not eligible:
                return False

            # Best fit avoids fragmenting devices by putting the request in the
            # eligible device with the least remaining dominant resource share.
            device = min(
                eligible,
                key=lambda item: (
                    max(1.0 - item["gpu"] - gpu_fraction, 1.0 - item["memory"] - memory_fraction),
                    2.0 - item["gpu"] - gpu_fraction - item["memory"] - memory_fraction,
                    max_per_device - item["workers"] - 1,
                ),
            )
            device["gpu"] += gpu_fraction
            device["memory"] += memory_fraction
            device["workers"] += 1
        return True

    def _configure_throughput_aware_gpu_parallelism(
        self,
        ops: List,
        *,
        total_samples: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Allocate auto GPU actors to balance measured pipeline throughput.

        Logical partitions are deliberately absent from this calculation. The
        result is one global actor budget for the execution group, constrained
        by CPU, per-device scheduling fractions, measured GPU memory, and the
        amount of useful data available to each stage.
        """
        auto_ids = getattr(self, "_auto_parallel_op_ids", set())
        candidates = [
            op
            for op in ops
            if id(op) in auto_ids
            and self._is_cuda_operator(op)
            and float(getattr(op, "_gpu_rows_per_second", 0) or 0) > 0
            and op.use_ray_actor()
        ]
        if not candidates:
            self._resolved_throughput_actor_plan = None
            self._throughput_planned_op_ids = set()
            profiled_cuda_ops = [op for op in ops if self._is_cuda_operator(op) and hasattr(op, "_gpu_rows_per_second")]
            if profiled_cuda_ops:
                skipped = ", ".join(
                    f"{getattr(op, '_name', type(op).__name__)}=" f"{getattr(op, 'num_proc', None)}"
                    for op in profiled_cuda_ops
                )
                logger.info(
                    "Throughput-aware GPU actor plan skipped despite GPU probe records: "
                    "no profiled CUDA operator is eligible for automatic actor planning "
                    f"(num_proc: {skipped})."
                )
            return None

        from data_juicer.utils.ray_cluster_utils import detect_cluster_topology

        topology = detect_cluster_topology()
        max_per_device = getattr(
            self,
            "max_gpu_workers_per_device",
            _DEFAULT_MAX_GPU_WORKERS_PER_DEVICE,
        )

        candidate_ids = {id(op) for op in candidates}
        # All other actor pools keep their existing plan, including CPU pools
        # and unprofiled GPU stages. Their resources remain resident alongside
        # the candidates in Ray's streaming pipeline.
        fixed_actor_ops = [op for op in ops if id(op) not in candidate_ids and op.use_ray_actor()]
        fixed_specs = []
        fixed_counts = []
        fixed_gpu_counts_by_id = {}
        fixed_cpu = 0.0
        for op in fixed_actor_ops:
            count = self._actor_pool_capacity(op.num_proc) or 1
            fixed_cpu += count * float(getattr(op, "num_cpus", None) or 1.0)
            if not self._is_cuda_operator(op):
                continue
            fixed_counts.append(count)
            fixed_gpu_counts_by_id[id(op)] = count
            fixed_specs.append(
                {
                    "num_gpus": float(getattr(op, "num_gpus", None) or 1.0),
                    "memory_fraction": float(
                        getattr(op, "_gpu_memory_fraction", 0) or getattr(op, "num_gpus", None) or 1.0
                    ),
                    "max_per_device": max_per_device,
                }
            )

        # Apply the input-pipeline reserve to the remaining capacity, so an
        # explicit CPU actor budget is counted without being silently reduced.
        remaining_cpus = max(0.0, topology.total_cpus - fixed_cpu)
        cpu_budget = min(
            topology.total_cpus,
            fixed_cpu + max(1.0, math.floor(remaining_cpus * _AUTO_GPU_PIPELINE_CPU_FRACTION)),
        )
        gpu_nodes = getattr(topology, "gpu_devices_per_node", ())
        if not gpu_nodes:
            if getattr(topology, "num_nodes", 1) > 1 and any(spec["num_gpus"] > 1 for spec in fixed_specs):
                raise RuntimeError("Multi-GPU actor planning requires the GPU capacity of each Ray node.")
            gpu_nodes = None

        source_ratio = 1.0
        stage_specs = []
        candidate_by_id = {id(op): op for op in candidates}
        for op in ops:
            if id(op) in candidate_by_id:
                batch_size = max(1, int(getattr(op, "batch_size", 1) or 1))
                stage_rows = None if total_samples is None else max(1, math.ceil(total_samples * source_ratio))
                stage_specs.append(
                    {
                        "op": op,
                        "name": getattr(op, "_name", type(op).__name__),
                        "throughput": float(op._gpu_rows_per_second),
                        "source_ratio": max(source_ratio, 1e-9),
                        "data_cap": (None if stage_rows is None else max(1, math.ceil(stage_rows / batch_size))),
                        "num_cpus": float(getattr(op, "num_cpus", None) or 1.0),
                        "num_gpus": float(getattr(op, "num_gpus", None) or 1.0),
                        "memory_fraction": float(
                            getattr(op, "_gpu_memory_fraction", 0) or getattr(op, "num_gpus", None) or 1.0
                        ),
                        "max_per_device": max_per_device,
                    }
                )
            measured_ratio = getattr(op, "_gpu_output_ratio", None)
            if measured_ratio is not None and float(measured_ratio) >= 0:
                source_ratio *= float(measured_ratio)

        counts = [1] * len(stage_specs)

        def cpu_used(values):
            return fixed_cpu + sum(value * spec["num_cpus"] for value, spec in zip(values, stage_specs))

        def resources_fit(values):
            if cpu_used(values) > cpu_budget + 1e-9:
                return False
            return self._gpu_actor_requests_fit(
                fixed_specs + stage_specs,
                fixed_counts + values,
                topology.total_gpus,
                gpu_nodes,
            )

        if not resources_fit(counts):
            names = ", ".join(spec["name"] for spec in stage_specs)
            raise RuntimeError(
                "The Ray cluster cannot host the minimum throughput-aware GPU actor plan "
                f"(one actor for each profiled stage: {names}). Reduce per-actor GPU memory, "
                "reduce explicit actor budgets, or add GPU/CPU resources."
            )

        while True:
            current_capacities = [
                value * item["throughput"] / item["source_ratio"] for value, item in zip(counts, stage_specs)
            ]
            bottleneck_capacity = min(current_capacities)
            feasible = []
            for index, spec in enumerate(stage_specs):
                # Scaling a stage that is already faster than the bottleneck
                # cannot improve pipeline throughput. Once no bottleneck stage
                # can grow, retain the available GPU capacity instead of loading
                # unnecessary model replicas.
                if current_capacities[index] > bottleneck_capacity + 1e-9:
                    continue
                if spec["data_cap"] is not None and counts[index] >= spec["data_cap"]:
                    continue
                trial = list(counts)
                trial[index] += 1
                if not resources_fit(trial):
                    continue
                capacities = [
                    value * item["throughput"] / item["source_ratio"] for value, item in zip(trial, stage_specs)
                ]
                feasible.append((min(capacities), -capacities[index], -index, index, trial))
            if not feasible:
                break
            counts = max(feasible)[-1]

        for count, spec in zip(counts, stage_specs):
            spec["op"].num_proc = count
        self._throughput_planned_op_ids = {id(spec["op"]) for spec in stage_specs}

        actor_counts_by_id = {id(spec["op"]): count for count, spec in zip(counts, stage_specs)}
        actor_counts_by_id.update(fixed_gpu_counts_by_id)
        profiled_stages = []
        source_ratio = 1.0
        for op in ops:
            throughput = float(getattr(op, "_gpu_rows_per_second", 0) or 0)
            count = actor_counts_by_id.get(id(op))
            if count is not None and self._is_cuda_operator(op) and op.use_ray_actor() and throughput > 0:
                profiled_stages.append(
                    {
                        "name": getattr(op, "_name", type(op).__name__),
                        "actors": count,
                        "steady_rows_per_second": throughput,
                        "source_input_ratio": max(source_ratio, 1e-9),
                        "source_equivalent_rows_per_second": count * throughput / max(source_ratio, 1e-9),
                        "data_cap": next(
                            (spec["data_cap"] for spec in stage_specs if spec["op"] is op),
                            None,
                        ),
                        "num_cpus": float(getattr(op, "num_cpus", None) or 1.0),
                        "num_gpus": float(getattr(op, "num_gpus", None) or 1.0),
                        "memory_fraction": float(
                            getattr(op, "_gpu_memory_fraction", 0) or getattr(op, "num_gpus", None) or 1.0
                        ),
                        "initialization_seconds": float(getattr(op, "_gpu_init_seconds", 0) or 0),
                    }
                )
            measured_ratio = getattr(op, "_gpu_output_ratio", None)
            if measured_ratio is not None and float(measured_ratio) >= 0:
                source_ratio *= float(measured_ratio)

        plan = {
            "policy": "balanced_source_throughput",
            "total_samples": total_samples,
            "cluster_cpus": topology.total_cpus,
            "cluster_gpus": topology.total_gpus,
            "cpu_budget": cpu_budget,
            "cpu_used": cpu_used(counts),
            "operators": profiled_stages,
        }
        self._resolved_throughput_actor_plan = plan
        try:
            self.cfg._resolved_throughput_actor_plan = plan
        except (AttributeError, TypeError):
            pass
        logger.info(
            "Throughput-aware GPU actor plan: "
            + ", ".join(f"{spec['name']}={count}" for count, spec in zip(counts, stage_specs))
        )
        return plan

    @staticmethod
    def _count_dataset_rows(dataset) -> Optional[int]:
        """Return the exact row count when the dataset exposes one."""
        if dataset is None:
            return None
        try:
            if hasattr(dataset, "count"):
                return max(0, int(dataset.count()))
            if hasattr(dataset, "__len__"):
                return max(0, int(len(dataset)))
        except Exception as error:
            logger.warning(f"Could not determine dataset row count for worker planning: {error}")
        return None

    @staticmethod
    def _is_cuda_operator(op) -> bool:
        try:
            return bool(op.use_cuda())
        except (AttributeError, RuntimeError):
            return False

    def _resolve_gpu_pipeline_worker_plan(self, ops: List, topology, total_samples: Optional[int] = None):
        """Calculate a conservative common worker limit for all GPU stages."""
        gpu_ops = [op for op in ops if self._is_cuda_operator(op)]
        if not gpu_ops:
            return None

        cpu_per_pipeline = sum(float(getattr(op, "num_cpus", None) or 1.0) for op in gpu_ops)
        gpu_per_pipeline = sum(float(getattr(op, "num_gpus", None) or 1.0) for op in gpu_ops)
        memory_fractions = [float(getattr(op, "_gpu_memory_fraction", 0) or 0) for op in gpu_ops]

        cpu_budget = max(1, math.floor(topology.total_cpus * _AUTO_GPU_PIPELINE_CPU_FRACTION))
        cpu_capacity = math.floor(cpu_budget / cpu_per_pipeline + 1e-9)
        gpu_capacity = math.floor(topology.total_gpus / gpu_per_pipeline + 1e-9)

        memory_capacity = None
        if memory_fractions and all(value > 0 for value in memory_fractions):
            memory_capacity = math.floor(
                topology.total_gpus * _AUTO_GPU_PIPELINE_MEMORY_FRACTION / sum(memory_fractions) + 1e-9
            )

        data_capacity = None
        if total_samples is not None:
            data_capacity = min(
                max(1, math.ceil(total_samples / max(1, int(getattr(op, "batch_size", 1) or 1)))) for op in gpu_ops
            )

        capacities = [cpu_capacity, gpu_capacity]
        if memory_capacity is not None:
            capacities.append(memory_capacity)
        if data_capacity is not None:
            capacities.append(data_capacity)
        safe_limit = min(capacities)
        if safe_limit < 1:
            logger.warning(
                "The GPU pipeline resource sum cannot host one concurrent copy "
                "of every GPU stage; using a worker limit of 1 so the existing "
                "resource validation can surface the scheduling error."
            )
            safe_limit = 1

        plan = {
            "safe_workers_per_stage": safe_limit,
            "gpu_operator_count": len(gpu_ops),
            "gpu_operators": [getattr(op, "_name", type(op).__name__) for op in gpu_ops],
            "total_samples": total_samples,
            "cpu_budget": cpu_budget,
            "cpu_per_pipeline": cpu_per_pipeline,
            "cpu_capacity": cpu_capacity,
            "gpu_per_pipeline": gpu_per_pipeline,
            "gpu_capacity": gpu_capacity,
            "memory_fraction_per_pipeline": sum(memory_fractions),
            "gpu_memory_budget_fraction": _AUTO_GPU_PIPELINE_MEMORY_FRACTION,
            "memory_capacity": memory_capacity,
            "data_capacity": data_capacity,
            "max_gpu_workers_per_device": getattr(
                self,
                "max_gpu_workers_per_device",
                _DEFAULT_MAX_GPU_WORKERS_PER_DEVICE,
            ),
        }
        self._safe_gpu_workers_per_stage = safe_limit
        try:
            self.cfg._resolved_gpu_worker_plan = plan
        except (AttributeError, TypeError):
            pass
        logger.info(
            "Safe GPU worker plan: "
            f"workers_per_stage={safe_limit} "
            f"(cpu={cpu_capacity}, gpu={gpu_capacity}, memory={memory_capacity}, "
            f"data={data_capacity}, stages={len(gpu_ops)})"
        )
        return plan

    @staticmethod
    def _cap_concurrency(concurrency, limit: int):
        """Cap a Ray actor/task concurrency value without changing its shape."""
        if isinstance(concurrency, bool) or concurrency is None:
            return concurrency
        if isinstance(concurrency, int):
            return min(concurrency, limit) if concurrency > 0 else concurrency
        if isinstance(concurrency, (tuple, list)):
            capped = [
                min(value, limit) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else value
                for value in concurrency
            ]
            if len(capped) >= 2 and isinstance(capped[0], int) and isinstance(capped[1], int):
                capped[0] = min(capped[0], capped[1])
            if (
                len(capped) == 3
                and isinstance(capped[0], int)
                and isinstance(capped[1], int)
                and isinstance(capped[2], int)
            ):
                capped[2] = min(max(capped[2], capped[0]), capped[1])
            return tuple(capped)
        return concurrency

    def _cap_auto_gpu_operator_parallelism(
        self,
        ops: List,
        safe_limit: Optional[int],
        *,
        total_samples: Optional[int] = None,
    ) -> None:
        """Apply the global safe limit to automatically planned GPU workers."""
        if safe_limit is None:
            return
        auto_parallel_op_ids = getattr(self, "_auto_parallel_op_ids", set())
        throughput_planned_op_ids = getattr(self, "_throughput_planned_op_ids", set())
        for op in ops:
            if (
                id(op) not in auto_parallel_op_ids
                or id(op) in throughput_planned_op_ids
                or not self._is_cuda_operator(op)
            ):
                continue
            op_limit = safe_limit
            if total_samples is not None:
                batch_size = max(1, int(getattr(op, "batch_size", 1) or 1))
                op_limit = min(op_limit, max(1, math.ceil(total_samples / batch_size)))
            original = op.num_proc
            op.num_proc = self._cap_concurrency(op.num_proc, op_limit)
            if op.num_proc != original:
                logger.info(
                    f"Op[{op._name}] automatic concurrency capped by the safe GPU "
                    f"worker plan: {original} -> {op.num_proc}"
                )

    def _resolve_max_concurrent_partitions(
        self,
        ops: List,
        total_samples: Optional[int] = None,
    ) -> int:
        """Resolve automatic driver concurrency after operator planning.

        A GPU pipeline is bounded by the joint resource cost of all GPU stages
        that can coexist in Ray's streaming execution. CPU-only pipelines use
        a small outer-pipeline cap because Ray Data can already parallelize
        work inside each partition.
        """
        raw_value = self.max_concurrent_partitions
        if not (isinstance(raw_value, str) and raw_value.lower() == "auto"):
            self.max_concurrent_partitions = max(1, int(raw_value))
            return self.max_concurrent_partitions

        from data_juicer.utils.ray_cluster_utils import detect_cluster_topology

        topology = detect_cluster_topology()
        total_cpus = topology.total_cpus
        total_gpus = topology.total_gpus

        if total_cpus <= 0:
            logger.warning(
                "Ray cluster reports no CPU resources for automatic partition " "concurrency; falling back to 1."
            )
            self.max_concurrent_partitions = 1
            return self.max_concurrent_partitions

        capacities = []
        gpu_operator_names = []
        for op in ops:
            num_cpus = getattr(op, "num_cpus", None)
            cpu_per_worker = float(num_cpus) if num_cpus and float(num_cpus) > 0 else 1.0
            capacity = math.floor(total_cpus / cpu_per_worker + 1e-9)

            num_gpus = getattr(op, "num_gpus", None)
            gpu_per_worker = float(num_gpus) if num_gpus and float(num_gpus) > 0 else None
            if gpu_per_worker is None:
                try:
                    if op.use_cuda():
                        gpu_per_worker = 1.0
                except (AttributeError, RuntimeError):
                    pass

            if gpu_per_worker is not None:
                gpu_operator_names.append(getattr(op, "_name", type(op).__name__))
                gpu_capacity = math.floor(total_gpus / gpu_per_worker + 1e-9)
                capacity = min(capacity, gpu_capacity)

            capacities.append(capacity)

        resource_capacity = min(capacities) if capacities else math.floor(total_cpus)
        if resource_capacity < 1:
            logger.warning(
                "Ray cluster resources cannot host one worker for every operator "
                "with the current resource requests; using one partition pipeline "
                "so Ray can surface the underlying scheduling error."
            )
            resource_capacity = 1

        if gpu_operator_names:
            pipeline_plan = self._resolve_gpu_pipeline_worker_plan(
                ops,
                topology,
                total_samples=total_samples,
            )
            resolved = min(resource_capacity, pipeline_plan["safe_workers_per_stage"])
            workload = f"GPU operators: {', '.join(gpu_operator_names)}"
        else:
            resolved = min(resource_capacity, _AUTO_CPU_PARTITION_CAP)
            workload = f"CPU-only pipeline cap: {_AUTO_CPU_PARTITION_CAP}"

        self.max_concurrent_partitions = max(1, resolved)
        logger.info(
            "Auto-configured max_concurrent_partitions="
            f"{self.max_concurrent_partitions} from Ray resources "
            f"(CPU={total_cpus:g}, GPU={total_gpus:g}; {workload})"
        )
        return self.max_concurrent_partitions

    @staticmethod
    def _actor_pool_capacity(concurrency) -> Optional[int]:
        """Return the maximum size represented by an actor concurrency value."""
        if isinstance(concurrency, bool):
            return None
        if isinstance(concurrency, int) and concurrency > 0:
            return concurrency
        if isinstance(concurrency, (tuple, list)) and len(concurrency) in (2, 3):
            max_size = concurrency[1]
            if isinstance(max_size, int) and not isinstance(max_size, bool) and max_size > 0:
                return max_size
        return None

    def _limit_partition_workers_for_explicit_actors(self, ops: List, requested_max_workers: int) -> int:
        """Keep concurrent partitions within every explicit actor-pool budget."""
        explicit_actor_op_ids = getattr(self, "_explicit_actor_op_ids", set())
        actor_budgets = [
            (op._name, self._actor_pool_capacity(op.num_proc)) for op in ops if id(op) in explicit_actor_op_ids
        ]
        actor_budgets = [(name, budget) for name, budget in actor_budgets if budget is not None]
        if not actor_budgets:
            return requested_max_workers

        explicit_limit = min(budget for _, budget in actor_budgets)
        max_workers = max(1, min(requested_max_workers, explicit_limit))
        if max_workers < requested_max_workers:
            budget_summary = ", ".join(f"{name}={budget}" for name, budget in actor_budgets)
            logger.warning(
                f"Reducing concurrent partition workers from {requested_max_workers} to {max_workers} "
                f"to honor explicit global actor num_proc budget(s): {budget_summary}"
            )
        return max_workers

    @staticmethod
    def _scale_concurrency_for_partitions(concurrency, max_workers: int, *, round_up: bool = True):
        """Share a global actor pool size across concurrent partitions."""
        if max_workers <= 1 or concurrency is None:
            return concurrency

        def _scale(value):
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return value
            if round_up:
                return max(1, math.ceil(value / max_workers))
            return max(1, value // max_workers)

        if isinstance(concurrency, (tuple, list)):
            scaled = [_scale(value) for value in concurrency]
            if len(scaled) >= 2 and isinstance(scaled[0], int) and isinstance(scaled[1], int):
                scaled[0] = min(scaled[0], scaled[1])
            if (
                len(scaled) == 3
                and isinstance(scaled[0], int)
                and isinstance(scaled[1], int)
                and isinstance(scaled[2], int)
            ):
                scaled[2] = min(max(scaled[2], scaled[0]), scaled[1])
            return tuple(scaled)
        if isinstance(concurrency, int) and concurrency > 0:
            return _scale(concurrency)
        return concurrency

    def _scale_operator_parallelism(self, ops: List, max_workers: int):
        """Convert global auto and explicit actor budgets to per-partition values."""
        original_concurrency = []
        auto_parallel_op_ids = getattr(self, "_auto_parallel_op_ids", set())
        explicit_actor_op_ids = getattr(self, "_explicit_actor_op_ids", set())
        for op in ops:
            if id(op) in auto_parallel_op_ids:
                scaling_mode = "automatic"
                round_up = True
            elif id(op) in explicit_actor_op_ids:
                scaling_mode = "explicit"
                round_up = False
            else:
                continue

            original_concurrency.append((op, op.num_proc))
            op.num_proc = self._scale_concurrency_for_partitions(
                op.num_proc,
                max_workers,
                round_up=round_up,
            )
            if op.num_proc != original_concurrency[-1][1]:
                logger.info(
                    f"Op[{op._name}] {scaling_mode} global concurrency: "
                    f"{original_concurrency[-1][1]} -> {op.num_proc} "
                    f"across {max_workers} concurrent partitions"
                )
        return original_concurrency

    def _scale_auto_operator_parallelism(self, ops: List, max_workers: int):
        """Compatibility wrapper for the former auto-only scaler."""
        return self._scale_operator_parallelism(ops, max_workers)

    @staticmethod
    def _restore_operator_parallelism(original_concurrency) -> None:
        """Restore the global plan for convergence or later execution stages."""
        for op, concurrency in original_concurrency:
            op.num_proc = concurrency

    @staticmethod
    def _restore_auto_operator_parallelism(original_concurrency) -> None:
        """Compatibility wrapper for the former auto-only restorer."""
        PartitionedRayExecutor._restore_operator_parallelism(original_concurrency)

    def _wrap_with_precomputed_parallelism(self, dataset) -> RayDataset:
        """Wrap data without recalculating shared operator resources."""
        wrapped = RayDataset(dataset, cfg=self.cfg)
        wrapped._auto_proc = False
        return wrapped

    def _process_with_convergence(self, dataset: RayDataset, ops: List, convergence_points: List[int]):
        """
        Process dataset with convergence support for global operations.
        """
        logger.info("Processing with convergence support for global operations...")

        # Find the first convergence point
        first_convergence = min(convergence_points)
        logger.info(f"First convergence point at operation {first_convergence}")

        # Split operations into pre-convergence and post-convergence
        pre_convergence_ops = ops[:first_convergence]
        post_convergence_ops = ops[first_convergence:]

        logger.info(f"Pre-convergence operations: {len(pre_convergence_ops)}")
        logger.info(f"Post-convergence operations: {len(post_convergence_ops)}")

        # Process partitions up to convergence point
        if pre_convergence_ops:
            logger.info("Processing partitions up to convergence point...")
            processed_dataset = self._process_with_simple_partitioning(dataset, pre_convergence_ops)
        else:
            logger.info("No pre-convergence operations, using original dataset...")
            processed_dataset = dataset

        # Merge partitions for global operations
        logger.info("Merging partitions for global operations...")
        merged_dataset = processed_dataset.data

        # Process merged dataset with post-convergence operations
        if post_convergence_ops:
            logger.info("Processing merged dataset with global operations...")
            merged_ray_dataset = self._wrap_with_precomputed_parallelism(merged_dataset)

            # Pre-execute DAG monitoring (log operation start events)
            if self.pipeline_dag:
                self._pre_execute_operations_with_dag_monitoring(post_convergence_ops, partition_id=0)

            # Execute operations
            final_dataset = merged_ray_dataset.process(post_convergence_ops)

            # Post-execute DAG monitoring (log operation completion events)
            if self.pipeline_dag:
                self._post_execute_operations_with_dag_monitoring(post_convergence_ops, partition_id=0)

            logger.info("Global operations completed. Final dataset ready for export")
            return final_dataset
        else:
            # No post-convergence operations, just return the merged result
            return RayDataset(merged_dataset, cfg=self.cfg)

    def _process_with_checkpointing(self, dataset: RayDataset, partition_id: int, ops: List) -> RayDataset:
        """
        Process dataset with checkpointing support.
        Groups operations and checkpoints between groups based on strategy.
        """
        logger.info(f"Processing partition {partition_id} with checkpointing support...")

        if not self.ckpt_manager.checkpoint_enabled:
            logger.info(f"Checkpointing disabled, processing all operations at once for partition {partition_id}")

            # Get input row count before processing
            input_rows = dataset.data.count()
            start_time = time.time()

            # Pre-execute DAG monitoring (log operation start events)
            if self.pipeline_dag:
                self._pre_execute_operations_with_dag_monitoring(ops, partition_id=partition_id)

            # Execute operations (lazy)
            processed_dataset = dataset.process(ops)

            # Force materialization to get real execution (required for union anyway)
            processed_dataset.data = processed_dataset.data.materialize()

            # Get metrics after execution
            duration = time.time() - start_time
            output_rows = processed_dataset.data.count()

            logger.info(f"Partition {partition_id}: Processed {input_rows}→{output_rows} rows in {duration:.2f}s")

            # Post-execute DAG monitoring with real metrics
            if self.pipeline_dag:
                metrics = {"duration": duration, "input_rows": input_rows, "output_rows": output_rows}
                self._post_execute_operations_with_dag_monitoring(ops, partition_id=partition_id, metrics=metrics)

            return processed_dataset

        # check the latest checkpoint for the partition
        latest_checkpoint = self.ckpt_manager.find_latest_checkpoint(partition_id)

        # Group operations based on checkpoint strategy
        op_groups = self.ckpt_manager.group_operations_for_checkpointing(ops)
        logger.info(f"Grouped {len(ops)} operations into {len(op_groups)} groups for checkpointing")
        logger.info(f"Detailed op groups: {op_groups}")

        current_dataset = dataset

        for group_idx, (start_idx, end_idx, group_ops) in enumerate(op_groups):
            logger.info(
                f"Processing partition {partition_id}, group {group_idx + 1}/{len(op_groups)}: operations {start_idx}-{end_idx-1}"
            )

            if latest_checkpoint and latest_checkpoint[0] >= end_idx:
                logger.info(
                    f"Partition {partition_id}: All operations in group {group_idx + 1} already processed (checkpoint at op {latest_checkpoint[0]}, group ends at {end_idx-1}), skipping"
                )
                continue

            if latest_checkpoint and latest_checkpoint[0] >= start_idx:
                logger.info(f"Partition {partition_id}: Resuming from checkpoint at operation {latest_checkpoint[0]}")
                current_dataset = self.ckpt_manager.load_checkpoint(
                    latest_checkpoint[0], latest_checkpoint[1], partition_id, cfg=self.cfg
                )
                if current_dataset is None:
                    logger.warning(f"Partition {partition_id}: Failed to load checkpoint, starting from beginning")
                    current_dataset = dataset
                    group_ops = ops[start_idx:end_idx]  # Start from beginning of group
                    logger.info(
                        f"Partition {partition_id}: Will process {len(group_ops)} operations from beginning of group"
                    )
                else:
                    current_dataset._auto_proc = False
                    logger.info(
                        f"Partition {partition_id}: Successfully loaded checkpoint, resuming from operation {latest_checkpoint[0] + 1}"
                    )
                    group_ops = ops[latest_checkpoint[0] + 1 : end_idx]  # Resume from checkpoint
                    if not group_ops:
                        logger.info(
                            f"Partition {partition_id}: All operations in this group already processed, skipping"
                        )
                        continue
                    else:
                        logger.info(
                            f"Partition {partition_id}: Will process {len(group_ops)} remaining operations from checkpoint"
                        )

            # Process the group of operations
            if group_ops:
                logger.info(
                    f"Partition {partition_id}: Processing {len(group_ops)} operations in group {group_idx + 1}"
                )

                # Get input row count before processing
                input_rows = current_dataset.data.count()
                start_time = time.time()

                # Pre-execute DAG monitoring (log operation start events)
                if self.pipeline_dag:
                    self._pre_execute_operations_with_dag_monitoring(group_ops, partition_id=partition_id)

                # Execute operations (lazy)
                current_dataset = current_dataset.process(group_ops)

                # Force materialization (required for checkpointing anyway)
                current_dataset.data = current_dataset.data.materialize()

                # Get metrics after execution
                duration = time.time() - start_time
                output_rows = current_dataset.data.count()

                logger.info(
                    f"Partition {partition_id}, group {group_idx + 1}: Processed {input_rows}→{output_rows} rows in {duration:.2f}s"
                )

                # Post-execute DAG monitoring with real metrics
                if self.pipeline_dag:
                    metrics = {"duration": duration, "input_rows": input_rows, "output_rows": output_rows}
                    self._post_execute_operations_with_dag_monitoring(
                        group_ops, partition_id=partition_id, metrics=metrics
                    )

            # Checkpoint after the last operation in the group
            if group_ops:
                last_op_idx = end_idx - 1
                last_op_name = ops[last_op_idx]._name
                if self.ckpt_manager.should_checkpoint(last_op_idx, last_op_name):
                    logger.info(
                        f"Partition {partition_id}: Creating checkpoint after operation {last_op_idx}: {last_op_name}"
                    )
                    # Data already materialized above, safe to checkpoint
                    self.ckpt_manager.save_checkpoint(
                        current_dataset, last_op_idx, last_op_name, partition_id, cfg=self.cfg
                    )

        return current_dataset

    def _find_work_directory(self, job_id: str) -> Optional[str]:
        """Find the work directory based on job_id."""
        # Check if the current work_dir already contains the job_id
        current_work_dir = Path(self.work_dir)
        logger.info(f"Checking if current work_dir contains job_id: {current_work_dir}")

        if job_id in str(current_work_dir):
            # Current work_dir already contains job_id, check if it's a valid work directory
            logger.info(f"Current work_dir contains job_id '{job_id}', checking if it's a valid work directory")

            # Check if this directory has events files (indicating it's a work directory)
            latest_events_file = self.event_logger.find_latest_events_file(str(current_work_dir))
            if latest_events_file:
                logger.info(f"Found events file in current work_dir: {latest_events_file}")
                return str(current_work_dir)

            logger.warning(f"No events file found in current work_dir: {current_work_dir}")

        logger.warning(f"No directory found containing job_id '{job_id}' with events files")
        return None

    def _check_job_completion(self, work_dir: str, job_id: str) -> bool:
        """Check if the job is already completed."""
        latest_events_file = self.event_logger.find_latest_events_file(work_dir)
        if not latest_events_file:
            logger.info(f"No events file found in work directory: {work_dir}")
            return False

        is_completed = self.event_logger.check_job_completion(latest_events_file)
        if is_completed:
            logger.info(f"Job {job_id} is already completed - no need to resume")
        else:
            logger.info(f"Job {job_id} is not completed - resumption possible")

        return is_completed

    def _resume_job(self, job_id: str) -> str:
        """Resume a job from checkpoints.

        Returns:
            "completed": Job is already completed
            "resuming": Job can be resumed
            "failed": Job resumption failed
        """
        logger.info(f"Attempting to resume job: {job_id}")

        # Find work directory
        work_dir = self._find_work_directory(job_id)
        if not work_dir:
            logger.error(f"Work directory not found for job_id: {job_id}")
            return "failed"

        logger.info(f"Found work directory: {work_dir}")

        # Check if config validation passed (done during config initialization)
        if not getattr(self.cfg, "_same_yaml_config", False):
            logger.error("Config validation failed - configurations don't match")
            return "failed"

        # Check if job is already completed
        if self._check_job_completion(work_dir, job_id):
            return "completed"  # Job already completed

        # Update checkpoint directory to use the work directory's checkpoint directory
        work_checkpoint_dir = os.path.join(work_dir, "checkpoints")
        if os.path.exists(work_checkpoint_dir):
            self.ckpt_manager.ckpt_dir = work_checkpoint_dir
            logger.info(f"Using checkpoint directory from work directory: {self.ckpt_manager.ckpt_dir}")
        else:
            logger.warning(f"No checkpoint directory found in work directory: {work_checkpoint_dir}")

        return "resuming"

    def _prepare_operators(self):
        """Prepare process operators."""
        ops = load_ops(self.cfg.process)

        # Check for op_fusion configuration with safe attribute access
        if hasattr(self.cfg, "op_fusion") and self.cfg.op_fusion:
            logger.info(f"Start OP fusion and reordering with strategy [{self.cfg.fusion_strategy}]...")
            ops = fuse_operators(
                ops,
                mapper_fusion=getattr(self.cfg, "mapper_fusion", True),
                mapper_fusion_vram_limit=getattr(self.cfg, "mapper_fusion_vram_limit", 0.9),
            )

        return ops

    def _override_strategy_methods(self):
        """Override strategy methods for partitioned execution."""
        # Override DAG-related methods for partitioned execution
        # Note: Partition count is determined by the executor (self.num_partitions),
        # not by the DAG mixin, so we don't override _determine_partition_count here
        # Note: _detect_convergence_points is reused from DAGExecutionMixin (no override needed)
        self._get_dag_node_for_operation = self._get_dag_node_for_operation_partitioned

    def _get_dag_node_for_operation_partitioned(
        self, op_name: str, op_idx: int, partition_id: int = 0, **kwargs
    ) -> Optional[str]:
        """Get DAG node ID for partitioned operation."""
        if not self.dag_execution_strategy:
            return None

        return self.dag_execution_strategy.get_dag_node_id(op_name, op_idx, partition_id=partition_id, **kwargs)

    # ========== Deterministic Partitioning Methods ==========

    def _enable_deterministic_execution(self) -> None:
        """Enable deterministic execution order in Ray Data.

        This keeps the global row order stable so saved row boundaries can be
        reapplied with split_at_indices() during explicit resume.
        """
        try:
            ctx = ray.data.DataContext.get_current()
            ctx.execution_options.preserve_order = True
            logger.info("Enabled deterministic execution (preserve_order=True)")
        except Exception as e:
            logger.warning(f"Could not enable deterministic execution: {e}")

    def _compute_row_hash(self, row: Dict) -> str:
        """Compute a hash of a row for partition validation.

        Uses a stable JSON serialization to ensure consistent hashing.
        """
        # Sort keys for deterministic serialization
        try:
            row_str = json.dumps(row, sort_keys=True, default=str)
            return hashlib.md5(row_str.encode()).hexdigest()[:16]
        except Exception:
            # Fallback for non-serializable rows
            return hashlib.md5(str(row).encode()).hexdigest()[:16]

    def _collect_partition_metadata(
        self,
        partition,
        partition_id: int,
        start_row: Optional[int] = None,
        compute_content_hash: bool = True,
    ) -> PartitionMetadata:
        """Collect metadata from a partition for validation on resume.

        The complete content hash is independent of Ray block boundaries but
        sensitive to row order within the partition. It is used for strict
        explicit resume validation; the first-row hash remains for backward
        compatibility.
        """
        if compute_content_hash:
            content_hash, row_count = self._compute_partition_content_hash(partition)
        else:
            content_hash = ""
            row_count = partition.count()

        # Get first row for hashing (cheap operation)
        first_row_hash = ""

        try:
            first_rows = partition.take(1)
            if first_rows:
                first_row_hash = self._compute_row_hash(first_rows[0])
        except Exception as e:
            logger.warning(f"Could not compute row hash for partition {partition_id}: {e}")

        return PartitionMetadata(
            partition_id=partition_id,
            row_count=row_count,
            first_row_hash=first_row_hash,
            last_row_hash="",  # Skip last_row_hash for efficiency
            content_hash=content_hash,
            start_row=start_row,
            end_row=start_row + row_count if start_row is not None else None,
        )

    def _compute_partition_content_hash(self, partition) -> tuple:
        """Compute an order-sensitive, block-boundary-independent hash."""
        # preserve_order is enabled before the source Dataset is created, so
        # take_all() returns these one-row partials in partition row order.
        partials = partition.map_batches(_hash_partition_batch, batch_format="pyarrow").take_all()
        return _combine_partition_hash_partials(partials)

    def _get_partitioning_info_path(self) -> str:
        """Get the path to the partitioning info file."""
        return os.path.join(self.ckpt_manager.ckpt_dir, "partitioning_info.json")

    def _save_partitioning_info(self, info: PartitioningInfo) -> None:
        """Save partitioning info alongside checkpoints."""
        os.makedirs(self.ckpt_manager.ckpt_dir, exist_ok=True)
        info.save(self._get_partitioning_info_path())

    def _load_partitioning_info(self) -> Optional[PartitioningInfo]:
        """Load partitioning info from checkpoint directory."""
        return PartitioningInfo.load(self._get_partitioning_info_path())

    def _validate_partitions(self, partitions: List, saved_info: PartitioningInfo) -> bool:
        """Validate that current partitions match saved partitioning info.

        Returns True if partitions match (safe to use checkpoints),
        False if there's a mismatch (must restart from scratch).

        Validation checks:
        1. Partition count matches
        2. Row count per partition matches
        3. Complete partition content hash matches when available
        4. First row hash matches for backward compatibility
        """
        if len(partitions) != saved_info.num_partitions:
            logger.error(f"Partition count mismatch: current={len(partitions)}, " f"saved={saved_info.num_partitions}")
            return False

        for i, partition in enumerate(partitions):
            saved_meta = saved_info.partitions[i] if i < len(saved_info.partitions) else None

            if saved_meta is None:
                logger.error(f"No saved metadata for partition {i}")
                return False

            if saved_meta.content_hash:
                current_hash, current_count = self._compute_partition_content_hash(partition)
            else:
                current_hash = ""
                current_count = partition.count()

            if current_count != saved_meta.row_count:
                logger.error(
                    f"Partition {i} row count mismatch: current={current_count}, " f"saved={saved_meta.row_count}"
                )
                return False

            if saved_meta.content_hash and current_hash != saved_meta.content_hash:
                logger.error(
                    f"Partition {i} content hash mismatch: " f"current={current_hash}, saved={saved_meta.content_hash}"
                )
                return False

            # Validate first row hash (skip if not available)
            if saved_meta.first_row_hash:
                try:
                    first_rows = partition.take(1)
                    if first_rows:
                        current_hash = self._compute_row_hash(first_rows[0])
                        if current_hash != saved_meta.first_row_hash:
                            logger.error(
                                f"Partition {i} first row hash mismatch: "
                                f"current={current_hash}, saved={saved_meta.first_row_hash}"
                            )
                            return False
                except Exception as e:
                    logger.warning(f"Could not validate partition {i} hash: {e}")

        logger.info("Partition validation passed - safe to use checkpoints")
        return True

    def _split_at_saved_boundaries(self, dataset: RayDataset, saved_info: PartitioningInfo) -> List:
        """Recreate partitions using row boundaries saved by the first run."""
        if saved_info.num_partitions != self.num_partitions:
            raise RuntimeError(
                "Cannot resume with a different partition count: "
                f"current={self.num_partitions}, saved={saved_info.num_partitions}"
            )
        if len(saved_info.partitions) != saved_info.num_partitions:
            raise RuntimeError(
                "Saved partition metadata is incomplete: "
                f"expected={saved_info.num_partitions}, actual={len(saved_info.partitions)}"
            )

        split_indices = []
        expected_start = 0
        for partition_id, meta in enumerate(saved_info.partitions):
            start_row = meta.start_row if meta.start_row is not None else expected_start
            end_row = meta.end_row if meta.end_row is not None else start_row + meta.row_count
            if start_row != expected_start or end_row - start_row != meta.row_count:
                raise RuntimeError(
                    f"Saved row boundaries are invalid for partition {partition_id}: "
                    f"start={start_row}, end={end_row}, rows={meta.row_count}, "
                    f"expected_start={expected_start}"
                )
            expected_start = end_row
            if partition_id < saved_info.num_partitions - 1:
                split_indices.append(end_row)

        if expected_start != saved_info.total_rows:
            raise RuntimeError(
                "Saved row boundaries do not cover the saved total row count: "
                f"boundaries_end={expected_start}, total_rows={saved_info.total_rows}"
            )

        if saved_info.num_partitions == 1:
            return [dataset.data.materialize()]

        logger.info(f"Recreating partitions at saved row boundaries: {split_indices}")
        return dataset.data.split_at_indices(split_indices)

    @staticmethod
    def _balanced_split_indices(total_rows: int, num_partitions: int) -> List[int]:
        """Return row boundaries that spread ``total_rows`` as evenly as possible.

        The first ``total_rows % num_partitions`` partitions take one extra row,
        so no row is dropped and no partition is empty as long as
        ``num_partitions <= total_rows``.
        """
        base, remainder = divmod(total_rows, num_partitions)
        indices = []
        boundary = 0
        for partition_id in range(num_partitions - 1):
            boundary += base + (1 if partition_id < remainder else 0)
            indices.append(boundary)
        return indices

    def _split_into_row_balanced_partitions(self, dataset: RayDataset) -> List:
        """Split the input into partitions holding nearly equal row counts.

        ``Dataset.split(n)`` hands out whole *blocks* via ``np.array_split``
        rather than cutting at row boundaries. Fewer blocks than ``n`` therefore
        produces trailing empty partitions -- one block of ten rows split four
        ways yields ``[10, 0, 0, 0]`` -- and unevenly sized blocks skew the
        partitions even when the block count is sufficient. Both are harmful
        here: an empty partition still builds an operator graph, takes an actor
        lifecycle and writes a checkpoint, while skew invalidates the uniform
        ``total_rows / num_partitions`` assumption behind execution-group sizing.
        ``split(n, equal=True)`` is not a usable alternative because it silently
        drops up to ``n - 1`` rows.

        Boundaries computed from the exact row count avoid all of that and make
        a fresh split use the same row-addressed primitive as an explicit
        resume, so partition identity no longer depends on Ray's block layout.
        """
        # ``split()`` executes the plan anyway; materializing here makes the
        # row count metadata-cheap and keeps the boundaries stable.
        data = dataset.data.materialize()
        total_rows = data.count()

        if total_rows <= 0:
            logger.warning("Input dataset has no rows; creating a single empty partition.")
            self.num_partitions = 1
            return [data]

        if self.num_partitions > total_rows:
            raise ValueError(
                f"partition.num_of_partitions={self.num_partitions} exceeds the {total_rows} row(s) in "
                "the input dataset, so at least one partition would be empty. An empty partition still "
                "builds an operator graph, takes an actor lifecycle and writes a checkpoint, and "
                "lowering the count here would silently disagree with the count a resume was asked "
                f"for. Set num_of_partitions to at most {total_rows}, or use partition.mode: auto to "
                "let the executor size the partitions from the dataset itself."
            )

        if self.num_partitions == 1:
            return [data]

        split_indices = self._balanced_split_indices(total_rows, self.num_partitions)
        logger.info(
            f"Splitting {total_rows} rows into {self.num_partitions} partitions at row boundaries: {split_indices}"
        )
        return data.split_at_indices(split_indices)

    def _split_dataset_deterministic(self, dataset: RayDataset) -> tuple:
        """Split dataset deterministically and collect metadata.

        Returns:
            tuple: (partitions, partitioning_info)
        """
        # Enable deterministic execution
        self._enable_deterministic_execution()

        # Check for existing partitioning info (resumption case)
        saved_info = self._load_partitioning_info()
        resume_requested = getattr(self.cfg, "_resume_requested", False)

        if resume_requested:
            if saved_info is None:
                raise RuntimeError(
                    "Explicit resume requires saved partitioning_info.json; "
                    "existing checkpoints were left unchanged."
                )
            if saved_info.hash_algorithm != _PARTITION_CONTENT_HASH_ALGORITHM:
                raise RuntimeError(
                    "Explicit resume cannot validate the saved partition hash algorithm: "
                    f"saved={saved_info.hash_algorithm}, "
                    f"supported={_PARTITION_CONTENT_HASH_ALGORITHM}. "
                    "Existing checkpoints were left unchanged."
                )
            if any(
                not meta.content_hash or meta.start_row is None or meta.end_row is None
                for meta in saved_info.partitions
            ):
                raise RuntimeError(
                    "Explicit resume requires content hashes and row boundaries "
                    "created by this version. Existing checkpoints were left unchanged."
                )

            logger.info("Explicit resume requested; recreating saved partition row boundaries...")
            partitions = self._split_at_saved_boundaries(dataset, saved_info)
            logger.info(f"Recreated {len(partitions)} partitions from saved boundaries")
            if not self._validate_partitions(partitions, saved_info):
                raise RuntimeError(
                    "Saved partition content hashes do not match the current input. "
                    "Refusing to resume; existing checkpoints were left unchanged."
                )
            logger.info("Saved partition hashes validated successfully - resuming checkpoints")
            return partitions, saved_info

        # Split the dataset
        partitions = self._split_into_row_balanced_partitions(dataset)
        logger.info(f"Created {len(partitions)} partitions (deterministic mode)")

        # If resuming, validate partitions match
        if saved_info is not None:
            logger.info("Found existing partitioning info, validating...")
            if self._validate_partitions(partitions, saved_info):
                logger.info("Partitions validated successfully - resuming with existing checkpoints")
                return partitions, saved_info
            else:
                logger.warning(
                    "Partition validation FAILED - partitions don't match saved info. "
                    "This can happen if the input data changed or Ray's internal state differs. "
                    "Clearing checkpoints and starting fresh."
                )
                self._clear_invalid_checkpoints()
                saved_info = None

        # Collect metadata for new partitions
        logger.info("Collecting partition metadata for checkpoint validation...")
        partition_metadata = []
        next_start_row = 0
        compute_content_hash = self.ckpt_manager.checkpoint_enabled

        for i, partition in enumerate(partitions):
            meta = self._collect_partition_metadata(
                partition,
                i,
                start_row=next_start_row,
                compute_content_hash=compute_content_hash,
            )
            partition_metadata.append(meta)
            next_start_row = meta.end_row
            logger.debug(f"Partition {i}: {meta.row_count} rows, hash={meta.first_row_hash[:8]}...")

        total_rows = sum(meta.row_count for meta in partition_metadata)

        partitioning_info = PartitioningInfo(
            num_partitions=self.num_partitions,
            total_rows=total_rows,
            partitions=partition_metadata,
            deterministic=True,
        )

        # Save partitioning info
        self._save_partitioning_info(partitioning_info)

        return partitions, partitioning_info

    def _clear_invalid_checkpoints(self) -> None:
        """Clear checkpoints when partition validation fails."""
        if os.path.exists(self.ckpt_manager.ckpt_dir):
            logger.warning(f"Clearing invalid checkpoints in {self.ckpt_manager.ckpt_dir}")
            try:
                shutil.rmtree(self.ckpt_manager.ckpt_dir)
            except OSError as e:
                logger.warning(f"Remove ckpt dir with shutil.rmtree() failed: {e}, " "will try os.rmdir()")
                os.rmdir(self.ckpt_manager.ckpt_dir)
            os.makedirs(self.ckpt_manager.ckpt_dir, exist_ok=True)
