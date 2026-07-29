import copy
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.fs_utils import HDFS_FS_KEYS, S3_FS_KEYS, create_filesystem_for_path
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class CreateFilesystemForPathTest(DataJuicerTestCaseBase):
    """Test cases for create_filesystem_for_path"""

    def test_consumed_key_constants(self):
        """The consumed-key constants are bound to the keys accepted by
        s3_utils/hdfs_utils filesystem factories."""
        self.assertEqual(
            set(S3_FS_KEYS),
            {
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_region",
                "endpoint_url",
            },
        )
        self.assertEqual(
            set(HDFS_FS_KEYS),
            {
                "hdfs_host",
                "hdfs_port",
                "hdfs_user",
                "hdfs_kerb_ticket",
                "hdfs_extra_conf",
            },
        )

    def test_local_path_returns_none_fs_and_new_dict(self):
        """Local paths return (None, copy of extra_args); the copy is a new
        dict object equal to the input."""
        extra_args = {"min_rows_per_file": 10, "compression": "gzip"}
        fs, remaining = create_filesystem_for_path("/tmp/output_dir/res.jsonl", extra_args)
        self.assertIsNone(fs)
        self.assertEqual(remaining, extra_args)
        self.assertIsNot(remaining, extra_args)

    def test_local_relative_path(self):
        """Relative local paths are treated as local, no filesystem created."""
        fs, remaining = create_filesystem_for_path("outputs/demo/res.jsonl", {"a": 1})
        self.assertIsNone(fs)
        self.assertEqual(remaining, {"a": 1})

    def test_file_scheme_path_is_local(self):
        """file:// paths are treated as local: (None, new equal dict)."""
        extra_args = {"min_rows_per_file": 10}
        fs, remaining = create_filesystem_for_path("file:///tmp/output.jsonl", extra_args)
        self.assertIsNone(fs)
        self.assertEqual(remaining, extra_args)
        self.assertIsNot(remaining, extra_args)

    @patch("data_juicer.utils.s3_utils.create_pyarrow_s3_filesystem")
    def test_s3_path(self, mock_create_s3_fs):
        """s3:// paths delegate to create_pyarrow_s3_filesystem with exactly
        the S3 keys; remaining keeps the others; input is not mutated."""
        mock_fs = MagicMock(name="s3_fs")
        mock_create_s3_fs.return_value = mock_fs
        extra_args = {
            "aws_access_key_id": "ak",
            "aws_secret_access_key": "sk",
            "aws_session_token": "st",
            "aws_region": "us-east-1",
            "endpoint_url": "http://localhost:9000",
            "min_rows_per_file": 5,
        }
        original = copy.deepcopy(extra_args)

        fs, remaining = create_filesystem_for_path("s3://bucket/prefix/res.jsonl", extra_args)

        self.assertIs(fs, mock_fs)
        mock_create_s3_fs.assert_called_once_with(
            {
                "aws_access_key_id": "ak",
                "aws_secret_access_key": "sk",
                "aws_session_token": "st",
                "aws_region": "us-east-1",
                "endpoint_url": "http://localhost:9000",
            }
        )
        self.assertEqual(remaining, {"min_rows_per_file": 5})
        # the original extra_args must not be mutated
        self.assertEqual(extra_args, original)

    @patch("data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem")
    def test_hdfs_path(self, mock_create_hdfs_fs):
        """hdfs:// paths delegate to create_pyarrow_hdfs_filesystem with the
        hdfs_* keys plus the path; remaining keeps the others; input is not
        mutated."""
        mock_fs = MagicMock(name="hdfs_fs")
        mock_create_hdfs_fs.return_value = mock_fs
        extra_args = {
            "hdfs_host": "namenode",
            "hdfs_port": 8020,
            "hdfs_user": "dj",
            "hdfs_kerb_ticket": "/tmp/krb5cc_dj",
            "hdfs_extra_conf": {"dfs.replication": "1"},
            "min_rows_per_file": 5,
        }
        original = copy.deepcopy(extra_args)
        path = "hdfs://namenode:8020/user/data/res.jsonl"

        fs, remaining = create_filesystem_for_path(path, extra_args)

        self.assertIs(fs, mock_fs)
        mock_create_hdfs_fs.assert_called_once_with(
            {
                "hdfs_host": "namenode",
                "hdfs_port": 8020,
                "hdfs_user": "dj",
                "hdfs_kerb_ticket": "/tmp/krb5cc_dj",
                "hdfs_extra_conf": {"dfs.replication": "1"},
                "path": path,
            }
        )
        self.assertEqual(remaining, {"min_rows_per_file": 5})
        # the original extra_args must not be mutated
        self.assertEqual(extra_args, original)

    @patch("data_juicer.utils.s3_utils.create_pyarrow_s3_filesystem")
    def test_s3_path_with_query_params(self, mock_create_s3_fs):
        """s3:// paths with query parameters still dispatch to the S3
        branch."""
        mock_fs = MagicMock(name="s3_fs")
        mock_create_s3_fs.return_value = mock_fs

        fs, remaining = create_filesystem_for_path("s3://bucket/prefix?foo=bar", {"aws_region": "us-east-1", "x": 1})

        self.assertIs(fs, mock_fs)
        mock_create_s3_fs.assert_called_once_with({"aws_region": "us-east-1"})
        self.assertEqual(remaining, {"x": 1})

    @patch("data_juicer.utils.s3_utils.create_pyarrow_s3_filesystem")
    def test_uppercase_s3_scheme(self, mock_create_s3_fs):
        """'S3://...' is dispatched case-insensitively to the S3 branch."""
        mock_fs = MagicMock(name="s3_fs")
        mock_create_s3_fs.return_value = mock_fs

        fs, remaining = create_filesystem_for_path("S3://bucket/res.jsonl", {"aws_access_key_id": "ak", "x": 1})

        self.assertIs(fs, mock_fs)
        mock_create_s3_fs.assert_called_once_with({"aws_access_key_id": "ak"})
        self.assertEqual(remaining, {"x": 1})

    @patch("data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem")
    def test_uppercase_hdfs_scheme(self, mock_create_hdfs_fs):
        """'HDFS://...' is dispatched case-insensitively to the HDFS
        branch."""
        mock_fs = MagicMock(name="hdfs_fs")
        mock_create_hdfs_fs.return_value = mock_fs
        path = "HDFS://namenode:8020/x.jsonl"

        fs, remaining = create_filesystem_for_path(path, {"hdfs_user": "dj", "x": 1})

        self.assertIs(fs, mock_fs)
        mock_create_hdfs_fs.assert_called_once_with({"hdfs_user": "dj", "path": path})
        self.assertEqual(remaining, {"x": 1})

    def test_malformed_s3_path_raises_value_error(self):
        """'s3:/...' (missing slash) is detected as S3-intent and rejected."""
        with self.assertRaises(ValueError) as ctx:
            create_filesystem_for_path("s3:/bucket/res.jsonl")
        self.assertIn("s3://", str(ctx.exception))

    def test_malformed_hdfs_path_raises_value_error(self):
        """'hdfs:/...' (missing slash) is detected as HDFS-intent and
        rejected."""
        with self.assertRaises(ValueError) as ctx:
            create_filesystem_for_path("hdfs:/user/data/res.jsonl")
        self.assertIn("hdfs://", str(ctx.exception))

    def test_empty_path_raises_value_error(self):
        """Empty path is invalid and raises ValueError."""
        with self.assertRaises(ValueError):
            create_filesystem_for_path("")

    def test_none_extra_args_local(self):
        """extra_args=None with a local path returns (None, {})."""
        fs, remaining = create_filesystem_for_path("/tmp/output_dir/res.jsonl", None)
        self.assertIsNone(fs)
        self.assertEqual(remaining, {})

    @patch("data_juicer.utils.s3_utils.create_pyarrow_s3_filesystem")
    def test_none_extra_args_s3(self, mock_create_s3_fs):
        """extra_args=None with an s3:// path returns (fs, {}) and the S3
        factory receives an empty conf."""
        mock_fs = MagicMock(name="s3_fs")
        mock_create_s3_fs.return_value = mock_fs

        fs, remaining = create_filesystem_for_path("s3://bucket/res.jsonl", None)

        self.assertIs(fs, mock_fs)
        mock_create_s3_fs.assert_called_once_with({})
        self.assertEqual(remaining, {})

    @patch("data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem")
    def test_none_extra_args_hdfs(self, mock_create_hdfs_fs):
        """extra_args=None with an hdfs:// path returns (fs, {}) and the
        HDFS factory receives exactly {'path': <path>}."""
        mock_fs = MagicMock(name="hdfs_fs")
        mock_create_hdfs_fs.return_value = mock_fs
        path = "hdfs://namenode:8020/user/data/res.jsonl"

        fs, remaining = create_filesystem_for_path(path, None)

        self.assertIs(fs, mock_fs)
        mock_create_hdfs_fs.assert_called_once_with({"path": path})
        self.assertEqual(remaining, {})


if __name__ == "__main__":
    unittest.main()
