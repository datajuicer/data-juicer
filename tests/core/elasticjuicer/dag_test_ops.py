"""Replay-safe DJ operators for the resident DAG contract tests (no GPU work)."""

import time

from data_juicer.ops.base_op import OPERATORS, Mapper


@OPERATORS.register_module("ej_test_dag_mapper")
class DAGTestMapper(Mapper):
    _batched_op = True
    elastic_juicer_dag_safe = True

    def __init__(
        self,
        inputs,
        output,
        scale=1,
        offset=0,
        keep_even=False,
        oom_above=None,
        sleep_seconds=0,
        corrupt_id=False,
        reverse=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.inputs, self.output = inputs, output
        self.scale, self.offset, self.keep_even = scale, offset, keep_even
        self.oom_above, self.sleep_seconds = oom_above, sleep_seconds
        self.corrupt_id, self.reverse = corrupt_id, reverse

    def process_batched(self, batch):
        if self.oom_above is not None and len(batch["id"]) > self.oom_above:
            raise RuntimeError("CUDA out of memory (explicit CPU fixture)")
        time.sleep(self.sleep_seconds)
        positions = [i for i, key in enumerate(batch["id"]) if not self.keep_even or key % 2 == 0]
        if self.reverse:
            positions.reverse()
        return {
            "id": [batch["id"][i] + int(self.corrupt_id) * 100000 for i in positions],
            self.output: [(sum(batch[k][i] for k in self.inputs) * self.scale + self.offset) for i in positions],
        }
