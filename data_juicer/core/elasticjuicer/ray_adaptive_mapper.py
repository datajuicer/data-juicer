"""Optional microbatch execution inside PR1054's existing Ray actor pools."""

import gc
import sys

import pyarrow as pa
from loguru import logger

from data_juicer.ops import Mapper
from data_juicer.utils.constant import Fields

from .adaptive_mapper import AdaptiveBatchContractError, OOMSafeAdaptiveMapper
from .batch_controller import AdaptiveBatchController
from .oom import is_oom_error
from .stage_identity import STAGE_IDENTITY_ATTR

# PR1054 execution-group routing column; avoid importing the executor here.
_PARTITION_COLUMN = "__data_juicer_logical_partition_id__"


def adaptive_batching_enabled(op, enabled=False):
    """Validate an opted-in, row-preserving, slice-independent GPU Mapper."""
    if not enabled:
        return False
    requested = getattr(op, "adaptive_batching", None)
    if requested is None:
        requested = getattr(op, "_supports_adaptive_batching", False)
    if not requested:
        return False
    name = getattr(op, "_name", None) or type(op).__name__
    if not isinstance(op, Mapper) or not op.is_batched_op():
        raise ValueError(f"{name}: adaptive batching requires a batched Mapper")
    if op.accelerator != "cuda" or op.ray_execution_mode == "task":
        raise ValueError(f"{name}: adaptive batching requires a CUDA Ray actor")
    if op.num_gpus is not None and not 0 < op.num_gpus <= 1:
        raise ValueError(f"{name}: adaptive batching supports at most one GPU per actor")
    if isinstance(op.batch_size, bool) or not isinstance(op.batch_size, int) or op.batch_size < 1:
        raise ValueError(f"{name}: adaptive batching requires a positive integer batch_size")
    return True


def _cleanup_cuda():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


class RayAdaptiveMapperActor:
    """Keep one model and local batch-size state across Ray outer batches.

    Ray resource reservations and outer batch size stay fixed. Opt-in asserts
    row/order preservation, slice independence and retry safety. External side
    effects and operator-owned OOM swallowing are not compatible.
    """

    def __init__(self, op_class, op_args, op_kwargs, max_batch_size, stage_id=None):
        self.op = op_class(*op_args, **op_kwargs)
        if stage_id:
            setattr(self.op, STAGE_IDENTITY_ATTR, stage_id)
        self.stage_id = stage_id or self.op._name or op_class.__name__
        self.controller = AdaptiveBatchController(initial_batch_size=max_batch_size, max_batch_size=max_batch_size)
        self.mapper = OOMSafeAdaptiveMapper(
            self._process_slice, self.controller, oom_cleanup=_cleanup_cuda, label=self.stage_id
        )

    def _process_slice(self, batch):
        if isinstance(batch, pa.Table):
            batch = batch.to_pydict()
        tags = list(batch[_PARTITION_COLUMN]) if _PARTITION_COLUMN in batch else None
        # Bypass Mapper's generic skip wrapper so OOM reaches the controller.
        output = self.op.process_batched(batch)
        if isinstance(output, pa.Table):
            output = output.to_pydict()
        if tags is not None:
            if (
                not isinstance(output, dict)
                or _PARTITION_COLUMN not in output
                or list(output[_PARTITION_COLUMN]) != tags
            ):
                raise AdaptiveBatchContractError(f"{self.stage_id}: mapper changed logical partition tags")
        return output

    def __call__(self, batch):
        try:
            return self.mapper(batch)
        except Exception as error:
            if is_oom_error(error) or isinstance(error, AdaptiveBatchContractError) or not self.op.skip_op_error:
                raise
            # Match the original policy: an ordinary error skips the entire
            # outer batch, including any previously successful microbatches.
            logger.exception(f"An error occurred in {self.stage_id}; skipping the outer batch")
            keys = batch.column_names if isinstance(batch, pa.Table) else batch.keys()
            result = {key: [] for key in keys}
            result[Fields.stats] = []
            result[Fields.source_file] = []
            return result
