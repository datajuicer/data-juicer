"""Contracts shared by native Ray plans and checkpoint batch workers."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import ray
from ray.data import ActorPoolStrategy, TaskPoolStrategy

from data_juicer.core.data.ray_batch_plan import BatchPlanWorker, RayBatchPlan
from data_juicer.core.executor.partition_protocol import ProtocolError
from data_juicer.utils.unittest_utils import TEST_TAG


class ContractMapper:
    def __init__(self, log, bias, *, offset, output):
        with open(log, "a") as stream:
            stream.write("constructed\n")
        self.bias = bias + offset
        self.output = output

    def __call__(self, batch, add, *, multiply):
        values = (batch["value"] + self.bias + add) * multiply
        if self.output == "arrow":
            return pa.table({"value": values})
        if self.output == "pandas":
            import pandas as pd

            return pd.DataFrame({"value": values})
        if self.output == "generator":
            return ({"value": values[i : i + 1]} for i in range(len(values)))
        return {"value": values}


class RayBatchPlanTest(unittest.TestCase):
    def test_checkpoint_resources_come_from_the_declared_operator(self):
        helper = TaskPoolStrategy(size=3)
        operator = ActorPoolStrategy(size=1)
        plan = RayBatchPlan().map_batches(lambda batch: batch, compute=helper)
        plan.map_operator(lambda batch: batch, compute=operator, num_cpus=0.5, runtime_env={"env_vars": {"K": "V"}})
        dataset = MagicMock()
        recovery = SimpleNamespace(context=object(), register_stats=lambda step: None)
        plan.apply(dataset, recovery)
        options = dataset._map_batches_without_batch_size_validation.call_args.kwargs
        self.assertIs(options["compute"], operator)
        self.assertEqual(options["num_cpus"], 0.5)
        self.assertEqual(options["runtime_env"], {"env_vars": {"K": "V"}})
        with self.assertRaisesRegex(ValueError, "only one operator"):
            plan.map_operator(lambda batch: batch, compute=helper)

    def test_checkpoint_plan_rejects_missing_resource_owner(self):
        plan = RayBatchPlan().map_batches(lambda batch: batch, compute=TaskPoolStrategy(size=1))
        with self.assertRaisesRegex(ProtocolError, "explicit operator resources"):
            plan.apply(None, SimpleNamespace(register_stats=lambda step: None))

    def test_unknown_steps_do_not_silently_materialize(self):
        plan = RayBatchPlan()
        plan.steps.append("typo")
        dataset = MagicMock()
        with self.assertRaisesRegex(TypeError, "Unknown batch plan step"):
            plan.apply(dataset)
        dataset.materialize.assert_not_called()
        with self.assertRaisesRegex(TypeError, "Unknown batch plan step"):
            list(BatchPlanWorker(plan).process_partition([]))

    @TEST_TAG("ray")
    def test_native_and_checkpoint_workers_share_udf_contract(self):
        ray.shutdown()
        self.addCleanup(ray.shutdown)
        ray.init(address="local", num_cpus=2, include_dashboard=False, log_to_driver=False)
        with TemporaryDirectory(prefix="dj-batch-contract-") as directory:
            for output in ("numpy", "arrow", "pandas", "generator"):
                with self.subTest(output=output):
                    root = Path(directory)

                    def plan(log):
                        return RayBatchPlan().map_operator(
                            ContractMapper,
                            compute=ActorPoolStrategy(size=1),
                            num_cpus=1,
                            fn_constructor_args=(str(log), 5),
                            fn_constructor_kwargs={"offset": 7, "output": output},
                            fn_args=(3,),
                            fn_kwargs={"multiply": 4},
                            batch_format="numpy",
                            batch_size=2,
                            zero_copy_batch=False,
                        )

                    native_log, checkpoint_log = root / (output + "-native"), root / (output + "-checkpoint")
                    table = pa.table({"value": np.arange(4)})
                    native = plan(native_log).apply(ray.data.from_arrow([table.slice(0, 2), table.slice(2)])).take_all()
                    worker = BatchPlanWorker(plan(checkpoint_log))
                    checkpoint = [
                        row
                        for part in (table.slice(0, 2), table.slice(2))
                        for block in worker.process_partition([part])
                        for row in block.to_pylist()
                    ]
                    expected = [{"value": value} for value in (60, 64, 68, 72)]
                    self.assertEqual(sorted(native, key=lambda row: row["value"]), expected)
                    self.assertEqual(checkpoint, expected)
                    self.assertEqual(native_log.read_text().splitlines(), ["constructed"])
                    self.assertEqual(checkpoint_log.read_text().splitlines(), ["constructed"])

    def test_checkpoint_worker_uses_ray_output_validation(self):
        from ray.data._internal.planner.plan_udf_map_op import (
            _generate_transform_fn_for_map_batches,
        )

        def invalid(batch):
            return [1, 2]

        table = pa.table({"value": [1]})
        native_transform = _generate_transform_fn_for_map_batches(invalid)
        with self.assertRaises(ValueError) as native:
            list(native_transform([table], None))
        worker = BatchPlanWorker(RayBatchPlan().map_batches(invalid, batch_format="pyarrow"))
        with self.assertRaises(ValueError) as checkpoint:
            list(worker.process_partition([table]))
        self.assertEqual(str(checkpoint.exception), str(native.exception))


if __name__ == "__main__":
    unittest.main()
