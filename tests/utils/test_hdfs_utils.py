"""
Test cases for HDFS utilities, focusing on path parsing, validation,
and filesystem creation logic.
"""

import unittest
from unittest.mock import patch

from data_juicer.utils.hdfs_utils import (
    create_pyarrow_hdfs_filesystem,
    parse_hdfs_path,
    strip_hdfs_scheme,
    validate_hdfs_path,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TestHdfsUtils(DataJuicerTestCaseBase):
    """Test cases for HDFS utility functions"""

    # ── validate_hdfs_path ──────────────────────────────────────────

    def test_validate_hdfs_path_valid(self):
        """Test HDFS path validation with valid paths"""
        valid_paths = [
            "hdfs://namenode:8020/user/data/file.jsonl",
            "hdfs://namenode/user/data",
            "hdfs:///user/data",
            "hdfs://localhost/data",
        ]

        for path in valid_paths:
            try:
                validate_hdfs_path(path)
            except ValueError:
                self.fail(f"validate_hdfs_path raised ValueError for valid path: {path}")

    def test_validate_hdfs_path_invalid(self):
        """Test HDFS path validation with invalid paths (no hdfs:// prefix)"""
        invalid_paths = [
            "s3://bucket/file.jsonl",
            "https://example.com/file.jsonl",
            "/local/path/file.jsonl",
            "file:///tmp/data",
            "",
            "hdfs-without-scheme/path",
        ]

        for path in invalid_paths:
            with self.assertRaises(ValueError) as ctx:
                validate_hdfs_path(path)
            self.assertIn("hdfs://", str(ctx.exception))

    # ── parse_hdfs_path ─────────────────────────────────────────────

    def test_parse_hdfs_path_with_host_and_port(self):
        """Test parsing HDFS URI with explicit host and port"""
        host, port = parse_hdfs_path("hdfs://namenode:8020/user/data")
        self.assertEqual(host, "namenode")
        self.assertEqual(port, 8020)

    def test_parse_hdfs_path_with_host_only(self):
        """Test parsing HDFS URI with host but no port"""
        host, port = parse_hdfs_path("hdfs://namenode/user/data")
        self.assertEqual(host, "namenode")
        self.assertIsNone(port)

    def test_parse_hdfs_path_default_fs(self):
        """Test parsing HDFS URI that relies on default filesystem (no authority)"""
        host, port = parse_hdfs_path("hdfs:///user/data")
        self.assertIsNone(host)
        self.assertIsNone(port)

    def test_parse_hdfs_path_nested_path(self):
        """Test parsing HDFS URI with deeply nested path"""
        host, port = parse_hdfs_path("hdfs://nn:9000/a/b/c/d/e.jsonl")
        self.assertEqual(host, "nn")
        self.assertEqual(port, 9000)

    # ── strip_hdfs_scheme ───────────────────────────────────────────

    def test_strip_hdfs_scheme_with_authority(self):
        """Test stripping scheme from a full HDFS URI"""
        result = strip_hdfs_scheme("hdfs://namenode:8020/user/data/file.jsonl")
        self.assertEqual(result, "/user/data/file.jsonl")

    def test_strip_hdfs_scheme_without_port(self):
        """Test stripping scheme when port is omitted"""
        result = strip_hdfs_scheme("hdfs://namenode/user/data")
        self.assertEqual(result, "/user/data")

    def test_strip_hdfs_scheme_default_fs(self):
        """Test stripping scheme from default-filesystem URI"""
        result = strip_hdfs_scheme("hdfs:///user/data")
        self.assertEqual(result, "/user/data")

    def test_strip_hdfs_scheme_fallback(self):
        """Test strip_hdfs_scheme falls back to original path when urlparse gives empty path"""
        # urlparse on a bare path without scheme gives empty path
        result = strip_hdfs_scheme("/user/data")
        self.assertEqual(result, "/user/data")

    # ── create_pyarrow_hdfs_filesystem ──────────────────────────────

    @patch("data_juicer.utils.hdfs_utils.pyarrow.fs.HadoopFileSystem")
    def test_create_hdfs_fs_with_host_port_in_path(self, mock_hdfs_fs):
        """Test filesystem creation when host/port come from the path field"""
        ds_config = {"path": "hdfs://namenode:8020/user/data"}
        create_pyarrow_hdfs_filesystem(ds_config)

        mock_hdfs_fs.assert_called_once()
        call_kwargs = mock_hdfs_fs.call_args[1]
        self.assertEqual(call_kwargs["host"], "namenode")
        self.assertEqual(call_kwargs["port"], 8020)

    @patch("data_juicer.utils.hdfs_utils.pyarrow.fs.HadoopFileSystem")
    def test_create_hdfs_fs_with_explicit_host_port(self, mock_hdfs_fs):
        """Test that explicit hdfs_host/hdfs_port override path-parsed values"""
        ds_config = {
            "path": "hdfs://old-nn:8020/user/data",
            "hdfs_host": "new-nn",
            "hdfs_port": 9000,
        }
        create_pyarrow_hdfs_filesystem(ds_config)

        mock_hdfs_fs.assert_called_once()
        call_kwargs = mock_hdfs_fs.call_args[1]
        self.assertEqual(call_kwargs["host"], "new-nn")
        self.assertEqual(call_kwargs["port"], 9000)

    @patch("data_juicer.utils.hdfs_utils.pyarrow.fs.HadoopFileSystem")
    def test_create_hdfs_fs_default(self, mock_hdfs_fs):
        """Test filesystem creation falls back to 'default' when no host is provided"""
        ds_config = {"path": "hdfs:///user/data"}
        create_pyarrow_hdfs_filesystem(ds_config)

        mock_hdfs_fs.assert_called_once()
        call_kwargs = mock_hdfs_fs.call_args[1]
        self.assertEqual(call_kwargs["host"], "default")

    @patch("data_juicer.utils.hdfs_utils.pyarrow.fs.HadoopFileSystem")
    def test_create_hdfs_fs_with_user(self, mock_hdfs_fs):
        """Test filesystem creation includes user when specified"""
        ds_config = {
            "path": "hdfs://namenode:8020/user/data",
            "hdfs_user": "data_juicer",
        }
        create_pyarrow_hdfs_filesystem(ds_config)

        mock_hdfs_fs.assert_called_once()
        call_kwargs = mock_hdfs_fs.call_args[1]
        self.assertEqual(call_kwargs["user"], "data_juicer")

    @patch("data_juicer.utils.hdfs_utils.pyarrow.fs.HadoopFileSystem")
    def test_create_hdfs_fs_with_kerb_ticket(self, mock_hdfs_fs):
        """Test filesystem creation includes Kerberos ticket when specified"""
        ds_config = {
            "path": "hdfs://namenode:8020/user/data",
            "hdfs_kerb_ticket": "/tmp/krb5cc_1000",
        }
        create_pyarrow_hdfs_filesystem(ds_config)

        mock_hdfs_fs.assert_called_once()
        call_kwargs = mock_hdfs_fs.call_args[1]
        self.assertEqual(call_kwargs["kerb_ticket"], "/tmp/krb5cc_1000")

    @patch("data_juicer.utils.hdfs_utils.pyarrow.fs.HadoopFileSystem")
    def test_create_hdfs_fs_with_extra_conf(self, mock_hdfs_fs):
        """Test filesystem creation includes extra Hadoop configurations"""
        extra_conf = {"dfs.replication": "3", "dfs.block.size": "134217728"}
        ds_config = {
            "path": "hdfs://namenode:8020/user/data",
            "hdfs_extra_conf": extra_conf,
        }
        create_pyarrow_hdfs_filesystem(ds_config)

        mock_hdfs_fs.assert_called_once()
        call_kwargs = mock_hdfs_fs.call_args[1]
        self.assertEqual(call_kwargs["extra_conf"], extra_conf)

    @patch("data_juicer.utils.hdfs_utils.pyarrow.fs.HadoopFileSystem")
    def test_create_hdfs_fs_host_default_when_empty_string(self, mock_hdfs_fs):
        """Test that an empty hdfs_host string falls back to 'default'"""
        ds_config = {
            "path": "hdfs:///user/data",
            "hdfs_host": "",
        }
        create_pyarrow_hdfs_filesystem(ds_config)

        mock_hdfs_fs.assert_called_once()
        call_kwargs = mock_hdfs_fs.call_args[1]
        self.assertEqual(call_kwargs["host"], "default")

    @patch("data_juicer.utils.hdfs_utils.pyarrow.fs.HadoopFileSystem")
    def test_create_hdfs_fs_explicit_port_overrides_path_port(self, mock_hdfs_fs):
        """Test that explicit hdfs_port overrides the port parsed from path"""
        ds_config = {
            "path": "hdfs://namenode:8020/user/data",
            "hdfs_port": 9999,
        }
        create_pyarrow_hdfs_filesystem(ds_config)

        mock_hdfs_fs.assert_called_once()
        call_kwargs = mock_hdfs_fs.call_args[1]
        self.assertEqual(call_kwargs["port"], 9999)

    @patch("data_juicer.utils.hdfs_utils.pyarrow.fs.HadoopFileSystem")
    def test_create_hdfs_fs_no_path_field(self, mock_hdfs_fs):
        """Test filesystem creation with empty config (no path field)"""
        create_pyarrow_hdfs_filesystem({})

        mock_hdfs_fs.assert_called_once()
        call_kwargs = mock_hdfs_fs.call_args[1]
        self.assertEqual(call_kwargs["host"], "default")


if __name__ == "__main__":
    unittest.main()
