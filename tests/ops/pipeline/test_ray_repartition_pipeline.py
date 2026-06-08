import unittest

from data_juicer.core.data import NestedDataset
from data_juicer.ops.load import load_ops
from data_juicer.ops.pipeline.ray_repartition_pipeline import \
    RayRepartitionPipeline
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class FakeRayDataset:
    def __init__(self):
        self.repartition_kwargs = None

    def repartition(self, **kwargs):
        self.repartition_kwargs = kwargs
        return self


class RayRepartitionPipelineTest(DataJuicerTestCaseBase):

    @TEST_TAG('ray')
    def test_ray_dataset_repartitions_without_shuffle_by_default(self):
        dataset = FakeRayDataset()

        output = RayRepartitionPipeline(num_blocks=128).run(dataset)

        self.assertIs(output, dataset)
        self.assertEqual(dataset.repartition_kwargs,
                         {"num_blocks": 128, "shuffle": False})

    @TEST_TAG('ray')
    def test_defaults_to_one_block(self):
        dataset = FakeRayDataset()

        output = RayRepartitionPipeline().run(dataset)

        self.assertIs(output, dataset)
        self.assertEqual(dataset.repartition_kwargs,
                         {"num_blocks": 1, "shuffle": False})

    @TEST_TAG('ray')
    def test_ray_dataset_repartitions_with_shuffle(self):
        dataset = FakeRayDataset()

        output = RayRepartitionPipeline(num_blocks=64,
                                        shuffle=True).run(dataset)

        self.assertIs(output, dataset)
        self.assertEqual(dataset.repartition_kwargs,
                         {"num_blocks": 64, "shuffle": True})

    def test_local_nested_dataset_fails_fast(self):
        dataset = NestedDataset.from_list([{"id": 1}])

        with self.assertRaisesRegex(RuntimeError, "requires Ray executor"):
            RayRepartitionPipeline(num_blocks=2).run(dataset)

    def test_constructor_validates_num_blocks(self):
        for invalid_num_blocks in [0, -1, True, 1.5, "2"]:
            with self.subTest(num_blocks=invalid_num_blocks):
                with self.assertRaisesRegex(ValueError, "num_blocks"):
                    RayRepartitionPipeline(num_blocks=invalid_num_blocks)

    @TEST_TAG('ray')
    def test_load_ops_can_load_ray_repartition_pipeline(self):
        ops = load_ops([{"ray_repartition_pipeline":
                         {"num_blocks": 32, "shuffle": True}}])

        self.assertEqual(len(ops), 1)
        self.assertIsInstance(ops[0], RayRepartitionPipeline)
        self.assertEqual(ops[0].num_blocks, 32)
        self.assertTrue(ops[0].shuffle)


if __name__ == '__main__':
    unittest.main()
