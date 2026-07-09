"""Compatibility module for the old FusedParallelMapper import path."""

from data_juicer.ops.fused_sequential_batch_op import (
    FusedParallelMapper,
    FusedSequentialBatchOp,
)

__all__ = ["FusedParallelMapper", "FusedSequentialBatchOp"]
