import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Tuple

from loguru import logger

# ===== Streaming tee-sink recovery (row-level manifest) =====
# Schema version for per-block stream manifest shards. Bump on incompatible
# manifest layout changes so resume can fail closed instead of misreading.
STREAM_MANIFEST_SCHEMA_VERSION = 1


def atomic_write_json(path: str, payload: dict, fsync: bool = False) -> None:
    """Write ``payload`` as JSON atomically (write-tmp then os.replace).

    Crash-safe and concurrency-safe on POSIX: a reader either sees the old file
    or the fully-written new one, never a partial. A half-written ``.tmp`` left
    by a crash is ignored by the reconciler.

    The temp file name is made unique per writer (pid + random suffix) so two
    concurrent writers of the SAME target path (e.g. a Ray task retry / a
    speculative duplicate re-committing the same content-keyed block) never
    share one ``.tmp`` and clobber each other's partial write; each renames its
    own fully-written temp into place. The reconciler ignores every ``*.tmp``.

    ``os.replace`` gives atomicity, not durability: after a process-kill the
    page cache preserves the write, but a power-loss/NFS crash can lose or
    reorder it. Pass ``fsync=True`` to flush the file and its directory to
    stable storage before returning, which additionally guarantees ordering
    against a later write in the same directory.
    """
    tmp = f"{path}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(payload, f)
        if fsync:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
    if fsync:
        dir_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def merge_row_id_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge a list of half-open [lo, hi) row-id ranges into sorted, disjoint ones."""
    normalized = sorted((int(lo), int(hi)) for lo, hi in ranges if int(hi) > int(lo))
    merged: List[Tuple[int, int]] = []
    for lo, hi in normalized:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def row_id_in_ranges(row_id: int, ranges: List[Tuple[int, int]]) -> bool:
    """Membership test for a row id against sorted, disjoint [lo, hi) ranges.

    Rebuilds the ``lo`` index on every call: O(K) per row. For hot filter paths
    that test many rows against the SAME ranges, use :func:`make_row_id_membership`
    which hoists that index out of the per-row call.
    """
    import bisect

    if not ranges:
        return False
    los = [lo for lo, _ in ranges]
    idx = bisect.bisect_right(los, row_id) - 1
    return idx >= 0 and ranges[idx][0] <= row_id < ranges[idx][1]


def make_row_id_membership(ranges: List[Tuple[int, int]]):
    """Build a fast membership predicate over fixed, sorted, disjoint [lo, hi) ranges.

    Precomputes the ``lo`` index once so each returned-closure call is a single
    O(log K) bisect instead of rebuilding the index O(K) per row. Use in Ray
    ``filter`` closures that test every row against the same committed frontier.
    """
    import bisect

    los = [int(lo) for lo, _ in ranges]
    tuples = [(int(lo), int(hi)) for lo, hi in ranges]

    def _member(row_id: int) -> bool:
        if not tuples:
            return False
        idx = bisect.bisect_right(los, row_id) - 1
        return idx >= 0 and tuples[idx][0] <= row_id < tuples[idx][1]

    return _member


class CheckpointManagerBase(ABC):
    """
    Base class for checkpoint managers.

    Provides common functionality for managing checkpoint directories and
    defines the interface that checkpoint managers should implement.
    """

    def __init__(self, ckpt_dir: str):
        """
        Initialize base checkpoint manager.

        :param ckpt_dir: Directory to save and load checkpoints
        """
        self.ckpt_dir = ckpt_dir
        # Ensure checkpoint directory exists
        os.makedirs(self.ckpt_dir, exist_ok=True)

    @abstractmethod
    def save_checkpoint(self, dataset: Any, **kwargs) -> str:
        """
        Save a dataset checkpoint.

        :param dataset: Dataset to save
        :param kwargs: Additional arguments specific to the implementation
        :return: Path to saved checkpoint
        """
        pass

    @abstractmethod
    def load_checkpoint(self, **kwargs) -> Optional[Any]:
        """
        Load a dataset checkpoint.

        :param kwargs: Arguments specific to the implementation (e.g., op_idx, partition_id)
        :return: Loaded dataset or None if checkpoint doesn't exist
        """
        pass

    def checkpoint_exists(self, checkpoint_path: str) -> bool:
        """
        Check if a checkpoint file/directory exists.

        :param checkpoint_path: Path to checkpoint
        :return: True if checkpoint exists, False otherwise
        """
        return os.path.exists(checkpoint_path)


class CheckpointManager(CheckpointManagerBase):
    """
    This class is used to save the latest version of dataset to checkpoint
    directory or load it from checkpoint directory, a bit like cache management
    Rerun the same config will reload the checkpoint and skip ops before it.

    If any args of operator in process list is changed, all ops will be
    rerun from the beginning.
    """

    def __init__(self, ckpt_dir, original_process_list, num_proc=1):
        """
        Initialization method.

        :param ckpt_dir: path to save and load checkpoint
        :param original_process_list: process list in config
        :param num_proc: number of process workers when saving dataset
        """
        super().__init__(ckpt_dir)
        self.ckpt_ds_dir = os.path.join(self.ckpt_dir, "latest")
        self.ckpt_op_record = os.path.join(self.ckpt_dir, "ckpt_op.json")
        self.process_list = original_process_list
        self.num_proc = num_proc
        self.op_record = []

        self.ckpt_available = self.check_ckpt()

    def get_left_process_list(self):
        """
        Get left process list of ops for processing dataset, when checkpoint is
        available, remove some ops from process list, otherwise keep it
        unchanged.

        :return: process list of left ops
        """
        return self.process_list

    def check_ckpt(self):
        """
        Check if checkpoint is available.

        :return: True when checkpoint is available, else False
        """
        if (
            os.path.exists(self.ckpt_ds_dir)
            and os.path.isdir(self.ckpt_ds_dir)
            and os.path.exists(self.ckpt_op_record)
            and os.path.isfile(self.ckpt_op_record)
            and self.check_ops_to_skip()
        ):
            return True
        else:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            return False

    def record(self, op_cfg: dict):
        """Save op name and args to op record, which is used to compare with
        the process list from config to decide if a checkpoint is available."""
        self.op_record.append(op_cfg)

    def check_ops_to_skip(self):
        """
        Check which ops need to be skipped in the process list.

        If op record list from checkpoint are the same as the prefix
        part of process list, then skip these ops and start processing
        from the checkpoint. Otherwise, process the original dataset
        from scratch.

        :return: whether to skip some ops or not
        """

        # load op records
        with open(self.ckpt_op_record, "r") as fin:
            self.op_record = json.load(fin)

        # check whether the op records are exactly the same
        # with prefix of process list
        # 1. same: remove these ops from process list
        # 2. different: cleanup op record, and keep process list unchanged
        recorded_op_num = len(self.op_record)
        process_op_num = len(self.process_list)
        if process_op_num < recorded_op_num:
            logger.warning(
                f"Current config ops ({process_op_num}) are fewer than "
                f"checkpoint ops ({recorded_op_num}). Cannot reuse checkpoint;"
                f" all ops will be processed from the beginning."
            )
            self.op_record = []
            return False

        prefix_process = self.process_list[:recorded_op_num]
        all_the_same = True
        dif1, dif2 = None, None

        for record_op, config_op in zip(self.op_record, prefix_process):
            if record_op != config_op:
                all_the_same = False
                dif1, dif2 = record_op, config_op
                break
        if all_the_same:
            for op in self.op_record:
                op_name = list(op.keys())[0]
                logger.info(f"Skip op [{op_name}].")
            self.process_list = self.process_list[recorded_op_num:]
            return True
        else:
            logger.warning(
                f"Processed ops of checkpoint are different from "
                f"current configs: checkpoint-{dif1} vs. config-"
                f"{dif2}. All ops will be processed from the "
                f"beginning."
            )
            self.op_record = []
            return False

    def save_ckpt(self, ds):
        """
        Save dataset to checkpoint directory and dump processed ops list.
        Alias for save_checkpoint for backward compatibility.

        :param ds: input dataset to save
        """
        return self.save_checkpoint(ds)

    def save_checkpoint(self, ds, **kwargs):
        """
        Save dataset to checkpoint directory and dump processed ops list.

        :param ds: input dataset to save
        :param kwargs: Additional arguments (not used, kept for interface compatibility)
        :return: Path to checkpoint directory
        """
        left_sample_num = len(ds)
        if left_sample_num > 0:
            ds.save_to_disk(self.ckpt_ds_dir, num_proc=min(self.num_proc, left_sample_num))
        else:
            # Empty dataset: skip save_to_disk to avoid ZeroDivisionError in
            # datasets._estimate_nbytes when the Arrow table has 0 rows.
            logger.warning("Checkpoint skipped: dataset is empty.")

        with open(self.ckpt_op_record, "w") as fout:
            json.dump(self.op_record, fout)

        return self.ckpt_ds_dir

    def load_ckpt(self):
        """
        Load dataset from a checkpoint file.
        Alias for load_checkpoint for backward compatibility.

        :return: a dataset stored in checkpoint file.
        """
        return self.load_checkpoint()

    def load_checkpoint(self, **kwargs):
        """
        Load dataset from a checkpoint file.

        :param kwargs: Additional arguments (not used, kept for interface compatibility)
        :return: a dataset stored in checkpoint file.
        """
        from data_juicer.core.data import NestedDataset

        ds = NestedDataset.load_from_disk(self.ckpt_ds_dir)
        return ds


class CheckpointStrategy(Enum):
    """Checkpoint strategies for controlling when to create checkpoints."""

    EVERY_OP = "every_op"  # Checkpoint after every operation
    EVERY_N_OPS = "every_n_ops"  # Checkpoint after every N operations
    MANUAL = "manual"  # Checkpoint only after specified operations
    DISABLED = "disabled"  # Disable checkpointing entirely


class RayCheckpointManager(CheckpointManagerBase):
    """
    Checkpoint manager for Ray Data with per-partition checkpointing support.

    This class manages checkpoints for Ray Data datasets using Parquet format,
    supporting per-partition checkpointing and various checkpoint strategies.
    """

    def __init__(
        self,
        ckpt_dir: str,
        checkpoint_enabled: bool = True,
        checkpoint_strategy: CheckpointStrategy = CheckpointStrategy.EVERY_OP,
        checkpoint_n_ops: int = 1,
        checkpoint_op_names: Optional[List[str]] = None,
        event_logger=None,
    ):
        """
        Initialize Ray checkpoint manager.

        :param ckpt_dir: Directory to save and load checkpoints
        :param checkpoint_enabled: Whether checkpointing is enabled
        :param checkpoint_strategy: Strategy for when to create checkpoints
        :param checkpoint_n_ops: Number of operations between checkpoints (for EVERY_N_OPS strategy)
        :param checkpoint_op_names: List of operation names to checkpoint (for MANUAL strategy)
        :param event_logger: Optional event logger for checkpoint events
        """
        super().__init__(ckpt_dir)
        self.checkpoint_enabled = checkpoint_enabled
        self.checkpoint_strategy = checkpoint_strategy
        self.checkpoint_n_ops = checkpoint_n_ops
        self.checkpoint_op_names = set(checkpoint_op_names or [])
        self.event_logger = event_logger

        # If strategy is DISABLED, disable checkpointing regardless of enabled flag
        if self.checkpoint_strategy == CheckpointStrategy.DISABLED:
            self.checkpoint_enabled = False

    def resolve_checkpoint_filename(self, op_idx: int, partition_id: int) -> str:
        """Resolve checkpoint filename using consistent format."""
        return f"checkpoint_op_{op_idx:04d}_partition_{partition_id:04d}.parquet"

    def should_checkpoint(self, op_idx: int, op_name: str) -> bool:
        """Determine if checkpoint should be created based on configuration strategy."""
        if not self.checkpoint_enabled:
            return False

        if self.checkpoint_strategy == CheckpointStrategy.EVERY_OP:
            return True
        elif self.checkpoint_strategy == CheckpointStrategy.EVERY_N_OPS:
            return (op_idx + 1) % self.checkpoint_n_ops == 0
        elif self.checkpoint_strategy == CheckpointStrategy.MANUAL:
            return op_name in self.checkpoint_op_names
        elif self.checkpoint_strategy == CheckpointStrategy.DISABLED:
            return False
        else:
            logger.warning(f"Unknown checkpoint strategy: {self.checkpoint_strategy}, defaulting to every_op")
            return True

    def save_checkpoint(
        self,
        dataset: Any,  # RayDataset or ray.data.Dataset
        op_idx: int,
        op_name: Optional[str] = None,
        partition_id: int = 0,
        cfg: Optional[Any] = None,
    ) -> str:
        """
        Save dataset checkpoint to parquet format.

        :param dataset: RayDataset or ray.data.Dataset to save
        :param op_idx: Operation index
        :param op_name: Operation name (optional)
        :param partition_id: Partition ID
        :param cfg: Optional config for RayDataset wrapper
        :return: Path to saved checkpoint
        """
        checkpoint_filename = self.resolve_checkpoint_filename(op_idx, partition_id)
        checkpoint_path = os.path.join(self.ckpt_dir, checkpoint_filename)

        # Ensure directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Extract ray.data.Dataset if it's wrapped in RayDataset
        ray_data = dataset.data if hasattr(dataset, "data") else dataset

        # Save as parquet
        ray_data.write_parquet(checkpoint_path)

        # Log checkpoint save event if event logger is available
        if self.event_logger and hasattr(self.event_logger, "_log_event"):
            from data_juicer.core.executor.event_logging_mixin import EventType

            self.event_logger._log_event(
                event_type=EventType.CHECKPOINT_SAVE,
                message=f"Saved checkpoint after operation {op_idx}: {op_name}",
                partition_id=partition_id,
                operation_name=op_name,
                operation_idx=op_idx,
                metadata={"checkpoint_path": checkpoint_path},
            )

        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(
        self,
        op_idx: int,
        op_name: Optional[str] = None,
        partition_id: int = 0,
        cfg: Optional[Any] = None,
    ) -> Optional[Any]:  # Returns RayDataset or None
        """
        Load dataset checkpoint from parquet format.

        :param op_idx: Operation index
        :param op_name: Operation name (optional)
        :param partition_id: Partition ID
        :param cfg: Optional config for RayDataset wrapper
        :return: RayDataset or None if checkpoint doesn't exist
        """
        checkpoint_filename = self.resolve_checkpoint_filename(op_idx, partition_id)
        checkpoint_path = os.path.join(self.ckpt_dir, checkpoint_filename)

        if not os.path.exists(checkpoint_path):
            return None

        try:
            # Lazy import ray to avoid dependency if not using Ray
            from data_juicer.utils.lazy_loader import LazyLoader

            ray = LazyLoader("ray")

            # Load from parquet
            ray_dataset = ray.data.read_parquet(checkpoint_path)

            # Log checkpoint load event if event logger is available
            if self.event_logger and hasattr(self.event_logger, "_log_event"):
                from data_juicer.core.executor.event_logging_mixin import EventType

                self.event_logger._log_event(
                    event_type=EventType.CHECKPOINT_LOAD,
                    message=f"Loaded checkpoint from operation {op_idx}",
                    partition_id=partition_id,
                    operation_name=op_name or f"op_{op_idx:04d}",
                    operation_idx=op_idx,
                    metadata={"checkpoint_path": checkpoint_path},
                )

            # Wrap in RayDataset if cfg is provided
            if cfg is not None:
                from data_juicer.core.data.ray_dataset import RayDataset

                return RayDataset(ray_dataset, cfg=cfg)
            else:
                return ray_dataset

        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None

    def find_latest_checkpoint(self, partition_id: int = 0) -> Optional[Tuple[int, str, str]]:
        """
        Find the latest checkpoint for a partition.

        :param partition_id: Partition ID
        :return: Tuple of (op_idx, op_name, checkpoint_path) or None if no checkpoint found
        """
        checkpoint_files = []

        if not os.path.exists(self.ckpt_dir):
            return None

        for filename in os.listdir(self.ckpt_dir):
            if filename.startswith("checkpoint_op_") and filename.endswith(f"_partition_{partition_id:04d}.parquet"):
                try:
                    # Parse filename: checkpoint_op_XXXX_partition_YYYY.parquet
                    parts = filename.replace(".parquet", "").split("_")
                    if len(parts) >= 4:
                        op_idx = int(parts[2])
                        # For backward compatibility, we'll use a generic op_name
                        op_name = f"op_{op_idx:04d}"
                        checkpoint_files.append((op_idx, op_name, os.path.join(self.ckpt_dir, filename)))
                except (ValueError, IndexError):
                    continue

        if not checkpoint_files:
            return None

        # Return the latest checkpoint (highest op_idx)
        latest = max(checkpoint_files, key=lambda x: x[0])
        return latest

    def group_operations_for_checkpointing(self, ops: List[Any]) -> List[Tuple[int, int, List[Any]]]:
        """
        Group operations based on checkpoint strategy.

        :param ops: List of operations
        :return: List of (start_idx, end_idx, group_ops) tuples
        """
        groups = []
        current_start = 0

        for i, op in enumerate(ops):
            op_name = getattr(op, "_name", f"op_{i}")
            if self.should_checkpoint(i, op_name):
                # This operation should trigger a checkpoint
                groups.append((current_start, i + 1, ops[current_start : i + 1]))
                current_start = i + 1

        # Add remaining operations as the last group
        if current_start < len(ops):
            groups.append((current_start, len(ops), ops[current_start:]))

        return groups

    # ===== Streaming tee-sink recovery =====
    # Durable layout under ckpt_dir (never collides with the legacy
    # checkpoint_op_*_partition_*.parquet files that find_latest_checkpoint globs):
    #   stream_data/segment_{S:04d}/block_<uuid>.parquet   -- teed output blocks
    #   stream_manifest/segment_{S:04d}/block_<uuid>.json  -- per-block manifest shards

    def stream_data_dir(self, segment_index: int) -> str:
        """Directory holding teed output block parquet files for a segment."""
        return os.path.join(self.ckpt_dir, "stream_data", f"segment_{segment_index:04d}")

    def stream_manifest_dir(self, segment_index: int) -> str:
        """Directory holding per-block manifest shards for a segment."""
        return os.path.join(self.ckpt_dir, "stream_manifest", f"segment_{segment_index:04d}")

    def reconcile_stream_frontier(self, segment_index: int) -> Tuple[List[Tuple[int, int]], List[str], int]:
        """Reconstruct the committed frontier of a segment from durable manifest shards.

        Lists the manifest dir, ignores ``.tmp`` shards left by a crash, skips
        unparseable JSON, and unions ``committed_row_id_ranges`` across all valid
        shards. Shards whose ``schema_version`` differs are skipped defensively
        (the caller enforces fail-closed resume separately). A shard whose
        referenced block parquet file is MISSING on disk is skipped in full
        (ranges included): the block that would supply those rows is gone, so
        treating them as committed would silently drop them from the output.

        :return: (merged_ranges, block_uris, shard_count) where merged_ranges is a
            sorted list of disjoint half-open [lo, hi) row-id ranges, block_uris are
            absolute parquet paths to restore, and shard_count is the number of
            valid shards reconciled.
        """
        manifest_dir = self.stream_manifest_dir(segment_index)
        ranges: List[Tuple[int, int]] = []
        block_uris: List[str] = []
        shard_count = 0
        if not os.path.isdir(manifest_dir):
            return ranges, block_uris, shard_count

        for name in sorted(os.listdir(manifest_dir)):
            if name.endswith(".tmp") or not name.endswith(".json"):
                continue
            shard_path = os.path.join(manifest_dir, name)
            try:
                with open(shard_path) as f:
                    shard = json.load(f)
            except Exception as e:  # partial/corrupt shard: ignore, do not resume past it
                logger.warning(f"Skipping unparseable stream manifest shard {shard_path}: {e}")
                continue
            if shard.get("schema_version") != STREAM_MANIFEST_SCHEMA_VERSION:
                logger.warning(
                    f"Stream manifest shard {shard_path} has schema_version="
                    f"{shard.get('schema_version')} != {STREAM_MANIFEST_SCHEMA_VERSION}; skipping"
                )
                continue
            block_uri = shard.get("block_uri")
            # A manifest shard is only trustworthy if its data block still
            # exists: the ranges it claims are "committed" are only recoverable
            # by restoring that block on resume. If the block file is gone
            # (crash between block delete/move and manifest cleanup, partial
            # copy, etc.), skip the WHOLE shard so its row-ids never enter the
            # frontier and get silently excluded from the output.
            abs_block = os.path.join(self.ckpt_dir, block_uri) if block_uri else None
            if not abs_block or not os.path.exists(abs_block):
                logger.warning(
                    f"Stream manifest shard {shard_path} references missing block "
                    f"{block_uri!r}; skipping shard (its row-ids will be reprocessed)"
                )
                continue
            for lo, hi in shard.get("committed_row_id_ranges", []):
                ranges.append((int(lo), int(hi)))
            block_uris.append(abs_block)
            shard_count += 1

        return merge_row_id_ranges(ranges), block_uris, shard_count
