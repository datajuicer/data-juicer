import base64
import json
import os
import shutil
import tempfile
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, List, Optional, Tuple

from loguru import logger


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

    def resolve_checkpoint_filename(self, op_idx: int, partition_id: Optional[int]) -> str:
        """Resolve checkpoint filename using consistent format."""
        if partition_id is None:
            return f"checkpoint_op_{op_idx:04d}_global.parquet"
        return f"checkpoint_op_{op_idx:04d}_partition_{partition_id:04d}.parquet"

    def begin_checkpoint(
        self, op_idx: int, op_name: Optional[str], partition_ids: List[Optional[int]]
    ) -> "CheckpointTransaction":
        """Create an unpublished transaction on the same filesystem as its destination."""
        return CheckpointTransaction(self, op_idx, op_name, partition_ids)

    def for_partitions(self, partition_ids, op_offset=0, partition_column=None, on_start=None, on_complete=None):
        return StreamingCheckpointer(self, partition_ids, op_offset, partition_column, on_start, on_complete)

    def _read_manifest(self, path):
        """Validate a published v2 checkpoint without discovering uncommitted files."""
        with open(os.path.join(path, "manifest.json"), encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest["version"] != 2 or not isinstance(manifest["files"], list):
            raise ValueError("Unsupported checkpoint manifest")
        pid = manifest["partition_id"]
        if manifest["scope"] != ("global" if pid is None else "partition") or not isinstance(manifest["op_idx"], int):
            raise ValueError("Invalid checkpoint scope")
        rows = 0
        names = set()
        for entry in manifest["files"]:
            name = entry["path"]
            if name != os.path.basename(name) or name in names or entry["format"] not in ("parquet", "arrow"):
                raise ValueError("Invalid checkpoint file entry")
            names.add(name)
            if entry["rows"] < 0 or os.path.getsize(os.path.join(path, name)) != entry["size_bytes"]:
                raise ValueError("Missing or truncated checkpoint file")
            rows += entry["rows"]
        if rows != manifest["rows"] or rows < 0:
            raise ValueError("Invalid checkpoint row count")
        if manifest.get("schema"):
            import pyarrow as pa

            pa.ipc.read_schema(pa.BufferReader(base64.b64decode(manifest["schema"])))
        return manifest

    def _complete_checkpoint(self, path):
        if not os.path.exists(path):
            return False
        if os.path.isdir(path) and any(
            os.path.exists(os.path.join(path, name)) for name in ("_checkpoint_v2", "manifest.json")
        ):
            try:
                self._read_manifest(path)
            except (OSError, ValueError, KeyError, TypeError):
                return False
        return True

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
        partition_id: Optional[int] = 0,
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
        from data_juicer.core.data.ray_checkpoint_adapter import insert_checkpoint

        ray_data = dataset.data if hasattr(dataset, "data") else dataset
        saved = insert_checkpoint(ray_data, self, op_idx, op_name, [partition_id]).materialize()
        if hasattr(dataset, "data"):
            dataset.data = saved
        return os.path.join(self.ckpt_dir, self.resolve_checkpoint_filename(op_idx, partition_id))

    def _log_checkpoint_save(self, checkpoint_path, op_idx, op_name, partition_id):
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

    def load_checkpoint(
        self,
        op_idx: int,
        op_name: Optional[str] = None,
        partition_id: Optional[int] = 0,
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

        if not self._complete_checkpoint(checkpoint_path):
            return None

        try:
            # Lazy import ray to avoid dependency if not using Ray
            from data_juicer.utils.lazy_loader import LazyLoader

            ray = LazyLoader("ray")

            # Load from parquet
            if any(os.path.exists(os.path.join(checkpoint_path, name)) for name in ("_checkpoint_v2", "manifest.json")):
                from data_juicer.core.data.ray_checkpoint_adapter import read_checkpoint

                manifest = self._read_manifest(checkpoint_path)
                if manifest["op_idx"] != op_idx or manifest["partition_id"] != partition_id:
                    raise ValueError("Checkpoint scope or operator does not match")
                ray_dataset = read_checkpoint(checkpoint_path, manifest)
            else:
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

    def find_latest_checkpoint(self, partition_id: Optional[int] = 0) -> Optional[Tuple[int, str, str]]:
        """
        Find the latest checkpoint for a partition.

        :param partition_id: Partition ID
        :return: Tuple of (op_idx, op_name, checkpoint_path) or None if no checkpoint found
        """
        checkpoint_files = []

        if not os.path.exists(self.ckpt_dir):
            return None

        suffix = "_global.parquet" if partition_id is None else f"_partition_{partition_id:04d}.parquet"
        for filename in os.listdir(self.ckpt_dir):
            if filename.startswith("checkpoint_op_") and filename.endswith(suffix):
                try:
                    # Parse filename: checkpoint_op_XXXX_partition_YYYY.parquet
                    parts = filename.replace(".parquet", "").split("_")
                    if len(parts) >= 4:
                        op_idx = int(parts[2])
                        # For backward compatibility, we'll use a generic op_name
                        op_name = f"op_{op_idx:04d}"
                        path = os.path.join(self.ckpt_dir, filename)
                        if self._complete_checkpoint(path):
                            if os.path.exists(os.path.join(path, "manifest.json")):
                                manifest = self._read_manifest(path)
                                if manifest["op_idx"] != op_idx or manifest["partition_id"] != partition_id:
                                    continue
                                op_name = manifest.get("op_name") or op_name
                            checkpoint_files.append((op_idx, op_name, path))
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


class StreamingCheckpointer:
    """Bind checkpoint policy and completion notifications to one lazy pipeline."""

    def __init__(self, manager, partition_ids, op_offset, partition_column, on_start, on_complete):
        self.manager = manager
        self.partition_ids = partition_ids
        self.op_offset = op_offset
        self.partition_column = partition_column
        self.on_start = on_start
        self.on_complete = on_complete
        self.pending = []

    def before_op(self, op, index):
        index += self.op_offset
        if self.on_start:
            self.on_start(op, index)
        self.pending.append((op, index))

    def _complete(self, operations):
        if self.on_complete:
            for op, index in operations:
                self.on_complete(op, index)

    def after_op(self, dataset, op, index, schema=None):
        from data_juicer.core.data.ray_checkpoint_adapter import insert_checkpoint

        index += self.op_offset
        if self.manager.should_checkpoint(index, op._name):
            operations = tuple(self.pending)
            self.pending.clear()
            return insert_checkpoint(
                dataset,
                self.manager,
                index,
                op._name,
                self.partition_ids,
                self.partition_column,
                lambda: self._complete(operations),
                schema,
            )
        return dataset

    def finish(self):
        self._complete(self.pending)
        self.pending.clear()


class CheckpointTransaction:
    """Driver-side metadata for one complete operator boundary.

    Workers write unique attempt files. Only successful task returns are committed;
    neither retries nor files left by interrupted attempts enter the manifest.
    """

    def __init__(self, manager, op_idx, op_name, partition_ids):
        self.manager = manager
        self.op_idx = op_idx
        self.op_name = op_name
        self.partition_ids = list(partition_ids)
        self.root = tempfile.mkdtemp(prefix=".checkpoint-", dir=os.path.abspath(manager.ckpt_dir))
        self.directories = {}
        self.entries = {pid: [] for pid in partition_ids}
        self.schemas = {pid: [] for pid in partition_ids}
        self.published = False
        for pid in partition_ids:
            directory = os.path.join(self.root, "global" if pid is None else str(pid))
            os.mkdir(directory)
            with open(os.path.join(directory, "_checkpoint_v2"), "w"):
                pass
            self.directories[pid] = directory

    def add_result(self, result):
        for pid, entry, schema in result:
            if pid not in self.entries:
                raise ValueError(f"Unexpected checkpoint partition {pid}")
            self.entries[pid].append(entry)
            if schema:
                self.schemas[pid].append(schema)

    def publish(self, fallback_schema=None):
        import pyarrow as pa

        # Prepare all manifests before publishing any logical partition.
        for pid, directory in self.directories.items():
            schemas = [pa.ipc.read_schema(pa.BufferReader(base64.b64decode(s))) for s in self.schemas[pid]]
            if not any(schema.names for schema in schemas) and isinstance(fallback_schema, pa.Schema):
                schemas = [fallback_schema]
            schema = pa.unify_schemas(schemas, promote_options="permissive") if schemas else None
            manifest = {
                "version": 2,
                "op_idx": self.op_idx,
                "op_name": self.op_name,
                "scope": "global" if pid is None else "partition",
                "partition_id": pid,
                "files": self.entries[pid],
                "rows": sum(entry["rows"] for entry in self.entries[pid]),
                "schema": base64.b64encode(schema.serialize()).decode("ascii") if schema is not None else None,
            }
            with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
                stream.flush()
                os.fsync(stream.fileno())
        for pid, directory in self.directories.items():
            final = os.path.join(self.manager.ckpt_dir, self.manager.resolve_checkpoint_filename(self.op_idx, pid))
            os.rename(directory, final)
            self.manager._log_checkpoint_save(final, self.op_idx, self.op_name, pid)
        self.published = True
        self.abort()

    def abort(self):
        """Remove only this transaction's unpublished data, after writers stop."""
        shutil.rmtree(self.root, ignore_errors=True)


def write_checkpoint_block(block, directories, partition_column=None, offsets=None):
    """Persist a native block on a worker and return file metadata only."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from ray.data.block import BlockAccessor

    table = BlockAccessor.for_block(block).to_arrow()
    if offsets:
        table = table.slice(offsets[0], offsets[1] - offsets[0])
    if partition_column:
        if partition_column not in table.column_names:
            if table.num_rows:
                raise ValueError("An operator removed the internal logical-partition tag")
            table = table.append_column(partition_column, pa.array([], type=pa.int64()))
        ids = set(table[partition_column].to_pylist())
        if not ids.issubset(directories):
            raise ValueError(f"Unexpected logical partition tags: {ids}")
        tables = [(pid, table.filter(pc.equal(table[partition_column], pid))) for pid in directories]
        tables = [(pid, part.drop([partition_column])) for pid, part in tables]
    else:
        if len(directories) != 1:
            raise ValueError("Multiple logical partitions require a partition column")
        tables = [(next(iter(directories)), table)]
    result = []
    for pid, part in tables:
        # Empty files retain schema, including for partitions filtered to zero rows.
        filename = f"{uuid.uuid4().hex}.parquet"
        path = os.path.join(directories[pid], filename)
        file_format = "parquet"
        try:
            pq.write_table(part, path)
        except pa.ArrowNotImplementedError:
            # Parquet cannot represent empty structs (e.g. DJ metadata). Arrow IPC
            # preserves these and nullable schemas without altering user columns.
            filename = f"{uuid.uuid4().hex}.arrow"
            path = os.path.join(directories[pid], filename)
            file_format = "arrow"
            with pa.OSFile(path, "wb") as sink, pa.ipc.new_file(sink, part.schema) as writer:
                writer.write_table(part)
        result.append(
            (
                pid,
                {"path": filename, "format": file_format, "rows": part.num_rows, "size_bytes": os.path.getsize(path)},
                base64.b64encode(part.schema.serialize()).decode("ascii"),
            )
        )
    return result
