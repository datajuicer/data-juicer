import unittest
import os
import shutil
import numpy as np
from PIL import Image

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator.ray_bts_image_minhash_deduplicator import (
    RayImageBTSMinhashDeduplicator,
    RayImageBTSMinhashDeduplicatorWithUid,
)
from data_juicer.utils.constant import HashKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


class RayImageBTSMinhashDeduplicatorTest(DataJuicerTestCaseBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cur_dir = os.path.dirname(os.path.abspath(__file__))
        cls.work_dir = os.path.join(cls.cur_dir, "image_dedup_test")
        if not os.path.exists(cls.work_dir):
            os.makedirs(cls.work_dir)

        # 准备测试图片路径
        cls.img_paths = [
            os.path.join(cls.work_dir, "img1.jpg"),
            os.path.join(cls.work_dir, "img2.jpg"),
            os.path.join(cls.work_dir, "img3.jpg"),
            os.path.join(cls.work_dir, "img4.jpg"),
        ]
        cls._generate_test_images()

    @classmethod
    def _generate_test_images(cls):
        img1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
        img1.save(cls.img_paths[0])
        img1.save(cls.img_paths[1])
        img3_np = np.array(img1).copy()
        img3_np[0, 0] = [254, 1, 1]
        Image.fromarray(img3_np).save(cls.img_paths[2])
        img4 = Image.new("RGB", (224, 224), color=(0, 0, 255))
        img4.save(cls.img_paths[3])

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.work_dir):
            shutil.rmtree(cls.work_dir)
        super().tearDownClass()

    def _run_minhash_dedup(self, dataset: Dataset, target_list, op):
        check_keys = [op.image_key] if hasattr(op, "image_key") else ["images"]
        res_list = self.run_single_op(dataset, op, check_keys)
        self.assertEqual(len(res_list), len(target_list))

    @TEST_TAG("ray")
    def test_image_path_deduplication(self):
        ds_list = [{"images": [p]} for p in self.img_paths]
        tgt_list = [{"images": [self.img_paths[0]]}, {"images": [self.img_paths[3]]}]

        dataset = self.generate_dataset(ds_list)
        op = RayImageBTSMinhashDeduplicator(jaccard_threshold=0.85, work_dir=self.work_dir, minhash_batch_size=2)
        self._run_minhash_dedup(dataset, tgt_list, op)

    @TEST_TAG("ray")
    def test_image_bytes_deduplication(self):
        ds_list = []
        for p in self.img_paths:
            with open(p, "rb") as f:
                ds_list.append({"image_bytes": f.read()})

        tgt_list = ds_list[:1] + ds_list[3:4]

        dataset = self.generate_dataset(ds_list)
        op = RayImageBTSMinhashDeduplicator(jaccard_threshold=0.85, work_dir=self.work_dir)

        check_keys = [op.image_bytes_key] if hasattr(op, "image_bytes_key") else ["image_bytes"]
        res_list = self.run_single_op(dataset, op, check_keys)
        self.assertEqual(len(res_list), len(tgt_list))


if __name__ == "__main__":
    unittest.main()
