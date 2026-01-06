import unittest
import os
import os.path as osp
import shutil
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.mm_utils import load_image, load_image_byte
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

# Assuming the operator is located at this path
from data_juicer.ops.mapper.s3_download_file_mapper import S3DownloadFileMapper


class S3DownloadFileMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        # 1. Setup patching for boto3.client
        self.patcher = patch("boto3.client")
        self.mock_boto_client = self.patcher.start()

        # 2. Create the mock S3 client instance and link it
        self.mock_s3 = MagicMock()
        self.mock_boto_client.return_value = self.mock_s3

        # 3. Setup temporary workspace and data paths
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), "..", "data"))

        # 4. Prepare test resources (local images and their S3 URLs)
        self.img1_path = osp.join(self.data_path, "img1.png")
        self.img1_s3_url = "s3://test-bucket/img1.png"
        self.img1_content = load_image_byte(self.img1_path)

        self.img2_path = osp.join(self.data_path, "img2.jpg")
        self.img2_s3_url = "s3://test-bucket/img2.jpg"
        self.img2_content = load_image_byte(self.img2_path)

        os.environ["AWS_ACCESS_KEY_ID"] = "fake_id"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "fake_key"
        os.environ["AWS_REGION"] = "us-east-1"
        os.environ["AWS_SESSION_TOKEN"] = "fake_token"

        # 5. Define default mock behaviors for the S3 client
        self._set_default_mock_behavior()

    def tearDown(self):
        # IMPORTANT: Stop the patcher to clean up the global namespace
        self.patcher.stop()

        if osp.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        del os.environ["AWS_ACCESS_KEY_ID"]
        del os.environ["AWS_SECRET_ACCESS_KEY"]
        del os.environ["AWS_REGION"]
        del os.environ["AWS_SESSION_TOKEN"]
        super().tearDown()

    def _set_default_mock_behavior(self):
        """Helper to set up standard successful S3 responses."""

        # Default behavior for get_object (memory download)
        def mock_get_object(Bucket, Key):
            content = self.img1_content if "img1" in Key else self.img2_content
            return {"Body": MagicMock(read=lambda: content)}

        self.mock_s3.get_object.side_effect = mock_get_object

        # Default behavior for download_file (disk download)
        def mock_download_file(Bucket, Key, Filename):
            source_path = self.img1_path if "img1" in Key else self.img2_path
            shutil.copy(source_path, Filename)

        self.mock_s3.download_file.side_effect = mock_download_file

    def _run_op_and_verify(self, ds_list, op_params):
        """Unified runner to execute the operator and return results."""
        op = S3DownloadFileMapper(**op_params)
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)

        res_list = dataset.to_list()
        return sorted(res_list, key=lambda x: x["id"])

    def test_s3_download_to_dir(self):
        """Test standard S3 download to a local directory."""
        ds_list = [{"images": [self.img1_s3_url], "id": 1}, {"images": [self.img2_s3_url], "id": 2}]

        params = {
            "save_dir": self.temp_dir,
            "download_field": "images",
            "aws_access_key_id": "fake_id",
            "aws_secret_access_key": "fake_key",
        }

        res_list = self._run_op_and_verify(ds_list, params)

        for res in res_list:
            r_path = res["images"][0]
            self.assertEqual(osp.dirname(r_path), self.temp_dir)
            self.assertTrue(osp.exists(r_path))
            # Data integrity check
            actual_img = np.array(load_image(r_path))
            self.assertIsNotNone(actual_img)

    def test_s3_download_to_memory(self):
        """Test downloading content directly into a field (bytes)."""
        ds_list = [
            {"images": self.img1_s3_url, "id": 1},
        ]

        params = {
            "download_field": "images",
            "save_field": "image_bytes",
            "aws_access_key_id": "fake_id",
            "aws_secret_access_key": "fake_key",
        }

        res_list = self._run_op_and_verify(ds_list, params)

        self.assertEqual(len(res_list), 1)
        self.assertIn("image_bytes", res_list[0])
        self.assertEqual(res_list[0]["image_bytes"], self.img1_content)

    def test_s3_resume_logic_adaptive(self):
        """Test Resume Download: Verify adaptive skip logic ensures
        existing files do not trigger S3 downloads."""
        os.makedirs(self.temp_dir, exist_ok=True)

        local_file_1 = osp.join(self.temp_dir, "img1.png")
        dummy_content = b"existing_local_data"

        with open(local_file_1, "wb") as f:
            f.write(dummy_content)

        img1_url = "s3://bucket/img1.png"
        img2_url = "s3://bucket/img2.png"

        self.mock_s3.download_file.reset_mock()

        ds_list = [{"images": [img1_url, img2_url], "id": 1}]
        params = {
            "save_dir": self.temp_dir,
            "download_field": "images",
            "resume_download": True,
        }

        self._run_op_and_verify(ds_list, params)

        # Verify point A: img1 should be skipped (exists)
        self.assertEqual(self.mock_s3.download_file.call_count, 1)

        # Verify point B: Ensure img2 was downloaded
        self.assertTrue(osp.exists(osp.join(self.temp_dir, "img2.png")))

        # Verify point C: Ensure img1's content has not been altered (verify resume is effective)
        with open(local_file_1, "rb") as f:
            self.assertEqual(f.read(), dummy_content)

    def test_s3_resume_logic(self):
        """Test that existing local files are skipped when resume_download=True."""
        # 1. Pre-create a dummy file to simulate an existing download
        os.makedirs(self.temp_dir, exist_ok=True)
        local_file = osp.join(self.temp_dir, "img1.png")
        dummy_content = b"old_data"
        with open(local_file, "wb") as f:
            f.write(dummy_content)

        # 2. Reset mock to track new calls
        self.mock_s3.download_file.reset_mock()

        ds_list = [{"images": [self.img1_s3_url], "id": 1}]
        params = {
            "save_dir": self.temp_dir,
            "download_field": "images",
            "resume_download": True,
            "aws_access_key_id": "fake_id",
            "aws_secret_access_key": "fake_key",
        }

        self._run_op_and_verify(ds_list, params)

        # 3. Verification: download_file should NOT have been called
        self.assertFalse(self.mock_s3.download_file.called)
        with open(local_file, "rb") as f:
            self.assertEqual(f.read(), dummy_content)

    def test_s3_download_failure(self):
        """Test handling of S3 client errors (e.g., file not found)."""
        # Override default behavior to simulate failure
        self.mock_s3.download_file.side_effect = Exception("S3 Object Not Found")

        ds_list = [{"images": [self.img1_s3_url], "id": 1}]
        params = {
            "save_dir": self.temp_dir,
            "download_field": "images",
        }

        res_list = self._run_op_and_verify(ds_list, params)

        # According to operator logic, it should return the original URL on failure
        self.assertEqual(res_list[0]["images"][0], self.img1_s3_url)


if __name__ == "__main__":
    unittest.main()
