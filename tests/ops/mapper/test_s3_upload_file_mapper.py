import unittest
import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import botocore.exceptions
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

# Ensure the path points to your operator file
from data_juicer.ops.mapper.s3_upload_file_mapper import S3UploadFileMapper


class S3UploadFileMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()

        # 1. Patch boto3 client
        self.patcher = patch("boto3.client")
        self.mock_boto = self.patcher.start()

        # 2. Setup Mock S3 object
        self.mock_s3 = MagicMock()
        self.mock_boto.return_value = self.mock_s3

        # [Core Fix]: Default head_object to raise 404 (Not Found).
        # This ensures that by default, files are considered "new" and upload_file is called.
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        self.mock_s3.head_object.side_effect = botocore.exceptions.ClientError(error_response, "HeadObject")

        # 3. Create temporary workspace
        self.temp_dir = tempfile.mkdtemp()

        # 4. Create dummy local files
        self.file1_name = "file1.txt"
        self.file1_path = os.path.join(self.temp_dir, self.file1_name)
        with open(self.file1_path, "w") as f:
            f.write("content 1")

        self.file2_name = "file2.jpg"
        self.file2_path = os.path.join(self.temp_dir, self.file2_name)
        with open(self.file2_path, "w") as f:
            f.write("content 2")

        # 5. Set AWS credentials in environment variables
        os.environ["AWS_ACCESS_KEY_ID"] = "test_key_id"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"
        os.environ["AWS_REGION"] = "us-east-1"
        os.environ["AWS_SESSION_TOKEN"] = "test_token"

        # 6. Common parameters
        self.bucket = "test-bucket"
        self.prefix = "dataset/"
        self.base_params = {
            "upload_field": "files",
            "s3_bucket": self.bucket,
            "s3_prefix": self.prefix,
        }

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        # Clean up environment variables
        del os.environ["AWS_ACCESS_KEY_ID"]
        del os.environ["AWS_SECRET_ACCESS_KEY"]
        del os.environ["AWS_REGION"]
        del os.environ["AWS_SESSION_TOKEN"]
        super().tearDown()

    def _run_op(self, ds_list, **kwargs):
        """Helper function to run the operator."""
        params = self.base_params.copy()
        params.update(kwargs)
        op = S3UploadFileMapper(**params)

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, batch_size=2)
        res_list = dataset.to_list()
        return sorted(res_list, key=lambda x: x["id"])

    def test_upload_basic(self):
        """Test basic upload: local path should be updated to S3 URL."""
        ds_list = [{"files": [self.file1_path], "id": 1}]

        res_list = self._run_op(ds_list)

        # Expected S3 URL
        expected_url = f"s3://{self.bucket}/{self.prefix}{self.file1_name}"

        # 1. Verify path update
        self.assertEqual(res_list[0]["files"][0], expected_url)

        # 2. Verify boto3 upload_file was called
        self.mock_s3.upload_file.assert_called_with(self.file1_path, self.bucket, self.prefix + self.file1_name)

    def test_upload_nested_structure(self):
        """Test nested list structure preservation (List of multiple files)."""
        # Fix: Input should be a list of strings [path1, path2], not list of lists [[p1], [p2]]
        # The operator expects the elements inside the list to be file paths (strings).
        ds_list = [{"files": [self.file1_path, self.file2_path], "id": 1}]

        res_list = self._run_op(ds_list)

        res_files = res_list[0]["files"]
        expected_url1 = f"s3://{self.bucket}/{self.prefix}{self.file1_name}"
        expected_url2 = f"s3://{self.bucket}/{self.prefix}{self.file2_name}"

        # 1. Verify it remains a list
        self.assertIsInstance(res_files, list)

        # 2. Verify elements are now strings (S3 URLs)
        self.assertIsInstance(res_files[0], str)
        self.assertIsInstance(res_files[1], str)

        # 3. Verify content
        self.assertEqual(res_files[0], expected_url1)
        self.assertEqual(res_files[1], expected_url2)

    def test_skip_existing_true(self):
        """Test skip_existing=True: do not upload if file exists on S3."""
        # [Core Modification]: Simulate file exists (head_object returns success)
        self.mock_s3.head_object.side_effect = None

        ds_list = [{"files": [self.file1_path], "id": 1}]

        res_list = self._run_op(ds_list, skip_existing=True)

        # Verify path is still updated to S3 URL
        expected_url = f"s3://{self.bucket}/{self.prefix}{self.file1_name}"
        self.assertEqual(res_list[0]["files"][0], expected_url)

        # Key Verification: upload_file should NOT be called
        self.mock_s3.upload_file.assert_not_called()
        # head_object should be called
        self.mock_s3.head_object.assert_called()

    def test_skip_existing_false(self):
        """Test skip_existing=False: force upload even if file exists on S3."""
        # Simulate file exists
        self.mock_s3.head_object.side_effect = None

        ds_list = [{"files": [self.file1_path], "id": 1}]

        self._run_op(ds_list, skip_existing=False)

        # Key Verification: even if file exists, skip_existing=False forces upload
        self.mock_s3.upload_file.assert_called()

    def test_remove_local_after_upload(self):
        """Test remove_local=True: delete local file after successful upload."""
        ds_list = [{"files": [self.file1_path], "id": 1}]

        # Confirm file exists initially
        self.assertTrue(os.path.exists(self.file1_path))

        self._run_op(ds_list, remove_local=True)

        # Verify boto3 was called
        self.mock_s3.upload_file.assert_called()

        # Verify local file is deleted
        self.assertFalse(os.path.exists(self.file1_path))

    def test_already_s3_url(self):
        """Test input is already S3 URL: keep as is and do not upload."""
        existing_url = "s3://other-bucket/file.txt"
        ds_list = [{"files": [existing_url], "id": 1}]

        self.mock_s3.upload_file.reset_mock()

        res_list = self._run_op(ds_list)

        # Verify URL unchanged
        self.assertEqual(res_list[0]["files"][0], existing_url)

        # Verify no upload occurred
        self.mock_s3.upload_file.assert_not_called()

    def test_local_file_not_found(self):
        """Test local file not found."""
        missing_path = os.path.join(self.temp_dir, "ghost.txt")
        ds_list = [{"files": [missing_path], "id": 1}]

        res_list = self._run_op(ds_list)

        # Should preserve original path on failure
        self.assertEqual(res_list[0]["files"][0], missing_path)

        # Verify no upload called
        self.mock_s3.upload_file.assert_not_called()

    def test_upload_failure(self):
        """Test S3 upload exception handling."""
        # Simulate upload failure
        self.mock_s3.upload_file.side_effect = Exception("Network Error")

        ds_list = [{"files": [self.file1_path], "id": 1}]

        # Run op
        res_list = self._run_op(ds_list)

        # Should preserve local path on failure
        self.assertEqual(res_list[0]["files"][0], self.file1_path)


if __name__ == "__main__":
    unittest.main()
