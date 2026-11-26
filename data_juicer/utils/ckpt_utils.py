import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional

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


class RayCheckpointManager:
    """Checkpoint settings of the ray_partitioned executor.

    Storage is created lazily by the commit log, so remote locations are
    never touched as local directories.
    """

    def __init__(self, ckpt_dir: str, checkpoint_enabled: bool = True, event_logger=None):
        self.ckpt_dir = ckpt_dir
        self.checkpoint_enabled = checkpoint_enabled
        self.event_logger = event_logger
