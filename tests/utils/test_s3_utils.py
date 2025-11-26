"""
Test cases for S3 utilities, focusing on environment variable priority logic.
"""
import os
import unittest
from unittest.mock import patch, MagicMock

from data_juicer.utils.s3_utils import (
    get_aws_credentials,
    create_pyarrow_s3_filesystem,
    validate_s3_path,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestS3Utils(DataJuicerTestCaseBase):
    """Test cases for S3 utility functions"""

    def setUp(self):
        """Save original environment variables before each test"""
        super().setUp()
        self.original_env = {
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
            "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN"),
            "AWS_REGION": os.environ.get("AWS_REGION"),
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION"),
        }

    def tearDown(self):
        """Restore original environment variables after each test"""
        # Clear all AWS-related env vars first
        for key in [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
        ]:
            if key in os.environ:
                del os.environ[key]

        # Restore original values
        for key, value in self.original_env.items():
            if value is not None:
                os.environ[key] = value

        super().tearDown()

    def test_get_aws_credentials_from_env_only(self):
        """Test getting credentials from environment variables only"""
        os.environ["AWS_ACCESS_KEY_ID"] = "env_key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "env_secret"
        os.environ["AWS_SESSION_TOKEN"] = "env_token"
        os.environ["AWS_REGION"] = "us-west-2"

        access_key, secret_key, session_token, region = get_aws_credentials({})

        self.assertEqual(access_key, "env_key")
        self.assertEqual(secret_key, "env_secret")
        self.assertEqual(session_token, "env_token")
        self.assertEqual(region, "us-west-2")

    def test_get_aws_credentials_from_config_only(self):
        """Test getting credentials from config when env vars are not set"""
        ds_config = {
            "aws_access_key_id": "config_key",
            "aws_secret_access_key": "config_secret",
            "aws_session_token": "config_token",
            "aws_region": "us-east-1",
        }

        access_key, secret_key, session_token, region = get_aws_credentials(ds_config)

        self.assertEqual(access_key, "config_key")
        self.assertEqual(secret_key, "config_secret")
        self.assertEqual(session_token, "config_token")
        self.assertEqual(region, "us-east-1")

    def test_get_aws_credentials_env_overrides_config(self):
        """Test that environment variables take priority over config"""
        os.environ["AWS_ACCESS_KEY_ID"] = "env_key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "env_secret"
        os.environ["AWS_REGION"] = "us-west-2"

        ds_config = {
            "aws_access_key_id": "config_key",
            "aws_secret_access_key": "config_secret",
            "aws_region": "us-east-1",
        }

        access_key, secret_key, session_token, region = get_aws_credentials(ds_config)

        # Environment variables should take precedence
        self.assertEqual(access_key, "env_key")
        self.assertEqual(secret_key, "env_secret")
        self.assertEqual(region, "us-west-2")
        # Session token not in env, should be None
        self.assertIsNone(session_token)

    def test_get_aws_credentials_partial_env_partial_config(self):
        """Test mixed scenario: some credentials from env, some from config"""
        os.environ["AWS_ACCESS_KEY_ID"] = "env_key"
        # Secret key not in env, should come from config

        ds_config = {
            "aws_secret_access_key": "config_secret",
            "aws_region": "us-east-1",
        }

        access_key, secret_key, session_token, region = get_aws_credentials(ds_config)

        self.assertEqual(access_key, "env_key")  # From env
        self.assertEqual(secret_key, "config_secret")  # From config
        self.assertEqual(region, "us-east-1")  # From config
        self.assertIsNone(session_token)

    def test_get_aws_credentials_aws_default_region_fallback(self):
        """Test that AWS_DEFAULT_REGION is used when AWS_REGION is not set"""
        os.environ["AWS_DEFAULT_REGION"] = "eu-west-1"
        # Don't set AWS_REGION

        access_key, secret_key, session_token, region = get_aws_credentials({})

        self.assertEqual(region, "eu-west-1")

    def test_get_aws_credentials_aws_region_precedence(self):
        """Test that AWS_REGION takes precedence over AWS_DEFAULT_REGION"""
        os.environ["AWS_REGION"] = "us-east-1"
        os.environ["AWS_DEFAULT_REGION"] = "us-west-2"

        access_key, secret_key, session_token, region = get_aws_credentials({})

        self.assertEqual(region, "us-east-1")

    def test_get_aws_credentials_no_credentials(self):
        """Test when no credentials are provided"""
        access_key, secret_key, session_token, region = get_aws_credentials({})

        self.assertIsNone(access_key)
        self.assertIsNone(secret_key)
        self.assertIsNone(session_token)
        self.assertIsNone(region)

    def test_get_aws_credentials_empty_config(self):
        """Test with empty config dict"""
        os.environ["AWS_ACCESS_KEY_ID"] = "env_key"

        access_key, secret_key, session_token, region = get_aws_credentials({})

        self.assertEqual(access_key, "env_key")
        self.assertIsNone(secret_key)
        self.assertIsNone(session_token)
        self.assertIsNone(region)

    def test_validate_s3_path_valid(self):
        """Test S3 path validation with valid paths"""
        valid_paths = [
            "s3://bucket/file.jsonl",
            "s3://bucket/path/to/file.jsonl",
            "s3://my-bucket-name/data/file.json",
        ]

        for path in valid_paths:
            # Should not raise
            try:
                validate_s3_path(path)
            except ValueError:
                self.fail(f"validate_s3_path raised ValueError for valid path: {path}")

    def test_validate_s3_path_invalid(self):
        """Test S3 path validation with invalid paths"""
        invalid_paths = [
            "https://bucket/file.jsonl",
            "file://bucket/file.jsonl",
            "/local/path/file.jsonl",
            "bucket/file.jsonl",
            "",
        ]

        for path in invalid_paths:
            with self.assertRaises(ValueError) as ctx:
                validate_s3_path(path)
            self.assertIn("s3://", str(ctx.exception).lower())

    @patch("data_juicer.utils.s3_utils.pyarrow.fs.S3FileSystem")
    def test_create_pyarrow_s3_filesystem_with_env_credentials(self, mock_s3_fs):
        """Test creating PyArrow S3FileSystem with environment variable credentials"""
        os.environ["AWS_ACCESS_KEY_ID"] = "env_key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "env_secret"
        os.environ["AWS_SESSION_TOKEN"] = "env_token"
        os.environ["AWS_REGION"] = "us-west-2"

        create_pyarrow_s3_filesystem({})

        # Verify S3FileSystem was called with credentials from env
        mock_s3_fs.assert_called_once()
        call_kwargs = mock_s3_fs.call_args[1]
        self.assertEqual(call_kwargs["access_key"], "env_key")
        self.assertEqual(call_kwargs["secret_key"], "env_secret")
        self.assertEqual(call_kwargs["session_token"], "env_token")
        self.assertEqual(call_kwargs["region"], "us-west-2")

    @patch("data_juicer.utils.s3_utils.pyarrow.fs.S3FileSystem")
    def test_create_pyarrow_s3_filesystem_with_config_credentials(self, mock_s3_fs):
        """Test creating PyArrow S3FileSystem with config credentials"""
        ds_config = {
            "aws_access_key_id": "config_key",
            "aws_secret_access_key": "config_secret",
            "aws_region": "us-east-1",
        }

        create_pyarrow_s3_filesystem(ds_config)

        # Verify S3FileSystem was called with credentials from config
        mock_s3_fs.assert_called_once()
        call_kwargs = mock_s3_fs.call_args[1]
        self.assertEqual(call_kwargs["access_key"], "config_key")
        self.assertEqual(call_kwargs["secret_key"], "config_secret")
        self.assertEqual(call_kwargs["region"], "us-east-1")

    @patch("data_juicer.utils.s3_utils.pyarrow.fs.S3FileSystem")
    def test_create_pyarrow_s3_filesystem_env_overrides_config(self, mock_s3_fs):
        """Test that environment variables override config in PyArrow filesystem creation"""
        os.environ["AWS_ACCESS_KEY_ID"] = "env_key"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "env_secret"

        ds_config = {
            "aws_access_key_id": "config_key",
            "aws_secret_access_key": "config_secret",
        }

        create_pyarrow_s3_filesystem(ds_config)

        # Verify env vars were used, not config
        mock_s3_fs.assert_called_once()
        call_kwargs = mock_s3_fs.call_args[1]
        self.assertEqual(call_kwargs["access_key"], "env_key")
        self.assertEqual(call_kwargs["secret_key"], "env_secret")

    @patch("data_juicer.utils.s3_utils.pyarrow.fs.S3FileSystem")
    def test_create_pyarrow_s3_filesystem_with_endpoint_url(self, mock_s3_fs):
        """Test creating PyArrow S3FileSystem with custom endpoint URL"""
        ds_config = {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "endpoint_url": "https://s3.custom.com",
        }

        create_pyarrow_s3_filesystem(ds_config)

        mock_s3_fs.assert_called_once()
        call_kwargs = mock_s3_fs.call_args[1]
        self.assertEqual(call_kwargs["endpoint_override"], "https://s3.custom.com")

    @patch("data_juicer.utils.s3_utils.pyarrow.fs.S3FileSystem")
    def test_create_pyarrow_s3_filesystem_no_credentials(self, mock_s3_fs):
        """Test creating PyArrow S3FileSystem without explicit credentials (uses default chain)"""
        create_pyarrow_s3_filesystem({})

        # Should be called with no arguments to use default credential chain
        mock_s3_fs.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

