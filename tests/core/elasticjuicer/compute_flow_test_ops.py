"""CPU arithmetic operators using logical GPU resources in transport tests."""

import time
from pathlib import Path

from data_juicer.ops.base_op import OPERATORS, TAGGING_OPS, Mapper
from data_juicer.utils.constant import Fields


@OPERATORS.register_module("compute_flow_test_mapper")
@TAGGING_OPS.register_module("compute_flow_test_mapper")
class ComputeFlowTestMapper(Mapper):
    _batched_op = True
    _accelerator = "cuda"
    _supports_adaptive_batching = True

    def __init__(self, scale=2, threshold=2, delay=0.0, failure=None, failure_file=None, **kwargs):
        super().__init__(**kwargs)
        self.scale, self.threshold, self.delay, self.failure = scale, threshold, delay, failure
        self.failure_file = failure_file

    def process_batched(self, samples):
        time.sleep(self.delay)
        # Mutation before the injected failure tests replay isolation.
        for meta in samples[Fields.meta]:
            meta["calls"] = meta.get("calls", 0) + 1
        if len(samples["id"]) > self.threshold:
            raise RuntimeError("CUDA out of memory (injected CPU fixture)")
        if self.failure == "transient" and samples["id"][0] >= 18 and Path(self.failure_file).exists():
            raise ValueError("injected transient failure")
        if self.failure == "ordinary" and samples["id"][0] >= 18:
            raise ValueError("injected late failure")
        if self.failure == "row_order":
            samples["__data_juicer_compute_flow_row_id__"].reverse()
        if self.failure == "partition":
            samples.pop("__data_juicer_logical_partition_id__")
        samples["value"] = [x * self.scale for x in samples["value"]]
        return samples
