import os
import unittest

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops.pipeline.llm_inference_with_ray_vllm_pipeline import LLMRayVLLMEnginePipeline
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from data_juicer.utils.constant import RAY_JOB_ENV_VAR
from data_juicer.utils.unittest_utils import TEST_TAG, FROM_FORK


class LLMRayVLLMEnginePipelineTest(DataJuicerTestCaseBase):

    def setUp(self):
        os.environ[RAY_JOB_ENV_VAR] = '1'
        return super().setUp()

    def tearDown(self):
        os.environ[RAY_JOB_ENV_VAR] = '0'
        return super().tearDown()

    @TEST_TAG('ray')
    def test_hf_model(self):
        import ray
        import ray.data

        ds_list = [
            {'query': '糖醋排骨怎么做?',},
            {'query': 'Where is the capital of China?',}
        ]
        ray_ds = ray.data.from_items(ds_list)
        ds = RayDataset(ray_ds)
        op = LLMRayVLLMEnginePipeline(
            api_or_hf_model='Qwen/Qwen2.5-0.5B' ,
            num_proc=1
            )
        ds = ds.process([op])
        res = ds.data.take_all()
        self.assertEqual(len(res), len(ds_list))
        for item in res:
            self.assertTrue(len(item['response']) > 0)

    @unittest.skipIf(FROM_FORK, "Skipping API-based test because running from a fork repo")
    @TEST_TAG('ray')
    def test_api_model(self):
        import ray
        import ray.data

        ds_list = [
            {'query': '糖醋排骨怎么做?',},
            {'query': 'Where is the capital of China?',}
        ]
        ray_ds = ray.data.from_items(ds_list)
        ds = RayDataset(ray_ds)
        op = LLMRayVLLMEnginePipeline(
            api_or_hf_model='qwen2.5-72b-instruct',
            is_hf_model=False,
            sampling_params=dict(
                temperature=0.0,
                max_tokens=150,
            ),
            num_proc=1)
        ds = ds.process([op])
        res = ds.data.take_all()

        self.assertEqual(len(res), len(ds_list))
        for item in res:
            self.assertIn('query', item)
            self.assertTrue(len(item['response']) > 0)


if __name__ == '__main__':
    unittest.main()
