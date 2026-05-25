import importlib
import unittest

import pyarrow as pa

_register_extension_type = pa.register_extension_type


def _register_extension_type_once(extension_type):
    try:
        _register_extension_type(extension_type)
    except pa.ArrowKeyError:
        if not extension_type.extension_name.startswith("datasets.features.features."):
            raise


pa.register_extension_type = _register_extension_type_once

NestedDataset = importlib.import_module("data_juicer.core.data").NestedDataset
load_ops = importlib.import_module("data_juicer.ops.load").load_ops
RepartitionPipeline = importlib.import_module("data_juicer.ops.pipeline.repartition_pipeline").RepartitionPipeline

pa.register_extension_type = _register_extension_type


class FakeRayDataset:
    def __init__(self):
        self.repartition_kwargs = None

    def repartition(self, **kwargs):
        self.repartition_kwargs = kwargs
        return self


class RepartitionPipelineTest(unittest.TestCase):
    def test_ray_dataset_repartitions_without_shuffle_by_default(self):
        dataset = FakeRayDataset()

        output = RepartitionPipeline(num_blocks=128).run(dataset)

        self.assertIs(output, dataset)
        self.assertEqual(dataset.repartition_kwargs, {"num_blocks": 128, "shuffle": False})

    def test_defaults_to_one_block(self):
        dataset = FakeRayDataset()

        output = RepartitionPipeline().run(dataset)

        self.assertIs(output, dataset)
        self.assertEqual(dataset.repartition_kwargs, {"num_blocks": 1, "shuffle": False})

    def test_ray_dataset_repartitions_with_shuffle(self):
        dataset = FakeRayDataset()

        output = RepartitionPipeline(num_blocks=64, shuffle=True).run(dataset)

        self.assertIs(output, dataset)
        self.assertEqual(dataset.repartition_kwargs, {"num_blocks": 64, "shuffle": True})

    def test_local_nested_dataset_fails_fast(self):
        dataset = NestedDataset.from_list([{"id": 1}])

        with self.assertRaisesRegex(RuntimeError, "requires Ray executor"):
            RepartitionPipeline(num_blocks=2).run(dataset)

    def test_constructor_validates_num_blocks(self):
        for invalid_num_blocks in [0, -1, True, 1.5, "2"]:
            with self.subTest(num_blocks=invalid_num_blocks):
                with self.assertRaisesRegex(ValueError, "num_blocks"):
                    RepartitionPipeline(num_blocks=invalid_num_blocks)

    def test_load_ops_can_load_repartition_pipeline(self):
        ops = load_ops([{"repartition_pipeline": {"num_blocks": 32, "shuffle": True}}])

        self.assertEqual(len(ops), 1)
        self.assertIsInstance(ops[0], RepartitionPipeline)
        self.assertEqual(ops[0].num_blocks, 32)
        self.assertTrue(ops[0].shuffle)


if __name__ == "__main__":
    unittest.main()
