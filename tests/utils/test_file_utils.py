import json
import os
import tempfile
import unittest
import regex as re
import gzip

import pandas as pd

from data_juicer.utils.file_utils import (
    Sizes,
    byte_size_to_size_str,
    find_files_with_suffix,
    get_all_files_paths_under,
    is_absolute_path,
    is_remote_path,
    add_suffix_to_filename,
    create_directory_if_not_exists,
    expand_outdir_and_mkdir,
    read_single_partition,
    single_partition_write_with_filename,
    transfer_filename,
    copy_data,
)
from data_juicer.utils.mm_utils import Fields

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class FileUtilsTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.temp_output_path = "tmp/test_file_utils/"
        os.makedirs(self.temp_output_path)

    def tearDown(self):
        if os.path.exists(self.temp_output_path):
            os.system(f"rm -rf {self.temp_output_path}")
        super().tearDown()

    def test_find_files_with_suffix(self):
        # prepare test files
        fn_list = ["test1.txt", "test2.txt", "test3.md"]
        for fn in fn_list:
            with open(os.path.join(self.temp_output_path, fn), "w") as f:
                f.write(fn)

        self.assertEqual(
            find_files_with_suffix(os.path.join(self.temp_output_path, "test1.txt")),
            {".txt": [os.path.join(self.temp_output_path, "test1.txt")]},
        )
        result = find_files_with_suffix(self.temp_output_path)
        expected = {
            ".txt": sorted([os.path.join(self.temp_output_path, "test1.txt"), os.path.join(self.temp_output_path, "test2.txt")]),
            ".md": [os.path.join(self.temp_output_path, "test3.md")],
        }
        for suffix in result:
            result[suffix] = sorted(result[suffix])
        self.assertEqual(result, expected)

        result_txt = find_files_with_suffix(self.temp_output_path, "txt")
        expected_txt = {
            ".txt": sorted([os.path.join(self.temp_output_path, "test1.txt"), os.path.join(self.temp_output_path, "test2.txt")])
        }
        for suffix in result_txt:
            result_txt[suffix] = sorted(result_txt[suffix])
        self.assertEqual(result_txt, expected_txt)

    def test_is_absolute_path(self):
        self.assertFalse(is_absolute_path(self.temp_output_path))
        self.assertTrue(is_absolute_path(os.path.abspath(self.temp_output_path)))

    def test_add_suffix_to_filename(self):
        self.assertEqual(add_suffix_to_filename("test.txt", "_suffix"), "test_suffix.txt")
        self.assertEqual(add_suffix_to_filename("test.txt", ""), "test.txt")
        self.assertEqual(add_suffix_to_filename("test", "_suffix"), "test_suffix")
        self.assertEqual(add_suffix_to_filename(".git", "_suffix"), ".git_suffix")

    def test_create_directory_if_not_exists(self):
        self.assertTrue(os.path.exists(self.temp_output_path))
        create_directory_if_not_exists(self.temp_output_path)
        self.assertTrue(os.path.exists(self.temp_output_path))
        os.rmdir(self.temp_output_path)
        self.assertFalse(os.path.exists(self.temp_output_path))
        create_directory_if_not_exists(self.temp_output_path)
        self.assertTrue(os.path.exists(self.temp_output_path))

    def test_transfer_filename(self):
        # test existing file
        with open(os.path.join(self.temp_output_path, "abc.jpg"), "w") as f:
            f.write("test")
        self.assertTrue(
            re.match(
                os.path.join(self.temp_output_path, Fields.multimodal_data_output_dir, "op1", "abc__dj_hash_#(.*?)#.jpg"),
                transfer_filename(os.path.join(self.temp_output_path, "abc.jpg"), "op1"),
            )
        )
        # test non-existing file
        self.assertTrue(
            re.match(
                os.path.join(self.temp_output_path, "non-existing.jpg"),
                transfer_filename(os.path.join(self.temp_output_path, "non-existing.jpg"), "op1"),
            )
        )
        # test save_dir
        self.temp_output_path = os.path.abspath(self.temp_output_path)
        self.assertTrue(
            re.match(
                os.path.join(self.temp_output_path, "tmp_save_dir", "abc__dj_hash_#(.*?)#.jpg"),
                transfer_filename(
                    os.path.join(self.temp_output_path, "abc.jpg"),
                    "op1",
                    save_dir=os.path.join(self.temp_output_path, "tmp_save_dir"),
                ),
            )
        )
        # test env dir
        try:
            ori_env_dir = os.environ.get("DJ_PRODUCED_DATA_DIR", None)
            test_env_dir = os.path.join(self.temp_output_path, "tmp_env_dir")
            os.environ["DJ_PRODUCED_DATA_DIR"] = test_env_dir

            transfer_filename(os.path.join(self.temp_output_path, "abc.jpg"), "op1")
            self.assertTrue(
                re.match(
                    os.path.join(test_env_dir, "op1", "abc__dj_hash_#(.*?)#.jpg"),
                    transfer_filename(os.path.join(self.temp_output_path, "abc.jpg"), "op1"),
                )
            )
        finally:
            if ori_env_dir:
                os.environ["DJ_PRODUCED_DATA_DIR"] = ori_env_dir
            elif "DJ_PRODUCED_DATA_DIR" in os.environ:
                del os.environ["DJ_PRODUCED_DATA_DIR"]

    def test_copy_data(self):
        tgt_fn = "test.txt"
        ori_dir = os.path.join(self.temp_output_path, "test1")
        tgt_dir = os.path.join(self.temp_output_path, "test2")

        self.assertFalse(copy_data(ori_dir, tgt_dir, tgt_fn))

        os.makedirs(ori_dir, exist_ok=True)
        with open(os.path.join(ori_dir, tgt_fn), "w") as f:
            f.write("test")

        self.assertTrue(copy_data(ori_dir, tgt_dir, tgt_fn))
        self.assertTrue(os.path.exists(os.path.join(tgt_dir, tgt_fn)))

    def test_find_files_with_suffix_gzip(self):
        # create a gzip compressed jsonl file and ensure it is detected as '.jsonl.gz'
        content = '{"text": "gzip test"}\n'
        gz_path = os.path.join(self.temp_output_path, "demo-dataset.jsonl.gz")
        with gzip.open(gz_path, "wb") as f:
            f.write(content.encode("utf-8"))

        result = find_files_with_suffix(self.temp_output_path)

        # normalize lists for comparison
        for suffix in result:
            result[suffix] = sorted(result[suffix])

        self.assertIn(".jsonl.gz", result)
        self.assertEqual(result[".jsonl.gz"], [gz_path])


class ByteSizeToSizeStrTest(DataJuicerTestCaseBase):
    """Test byte_size_to_size_str: converts byte count to human-readable string."""

    def test_bytes_range(self):
        self.assertEqual(byte_size_to_size_str(0), "0.00 Bytes")
        self.assertEqual(byte_size_to_size_str(512), "512.00 Bytes")
        self.assertEqual(byte_size_to_size_str(1023), "1023.00 Bytes")

    def test_kib_range(self):
        self.assertEqual(byte_size_to_size_str(Sizes.KiB), "1.00 KiB")
        self.assertEqual(byte_size_to_size_str(int(1.5 * Sizes.KiB)), "1.50 KiB")

    def test_mib_range(self):
        self.assertEqual(byte_size_to_size_str(Sizes.MiB), "1.00 MiB")
        self.assertEqual(byte_size_to_size_str(5 * Sizes.MiB), "5.00 MiB")

    def test_gib_range(self):
        self.assertEqual(byte_size_to_size_str(Sizes.GiB), "1.00 GiB")
        self.assertEqual(byte_size_to_size_str(int(2.5 * Sizes.GiB)), "2.50 GiB")

    def test_tib_range(self):
        self.assertEqual(byte_size_to_size_str(Sizes.TiB), "1.00 TiB")
        self.assertEqual(byte_size_to_size_str(3 * Sizes.TiB), "3.00 TiB")

    def test_boundary_kib(self):
        """Exactly at KiB boundary should show KiB, not Bytes."""
        self.assertEqual(byte_size_to_size_str(1024), "1.00 KiB")

    def test_boundary_mib(self):
        """Exactly at MiB boundary should show MiB, not KiB."""
        self.assertEqual(byte_size_to_size_str(1024 * 1024), "1.00 MiB")


class IsRemotePathTest(DataJuicerTestCaseBase):
    """Test is_remote_path: detects http/https/s3/gs/hdfs URLs."""

    def test_http(self):
        self.assertTrue(is_remote_path("http://example.com/data.json"))

    def test_https(self):
        self.assertTrue(is_remote_path("https://example.com/data.json"))

    def test_s3(self):
        self.assertTrue(is_remote_path("s3://bucket/key"))

    def test_gs(self):
        self.assertTrue(is_remote_path("gs://bucket/key"))

    def test_hdfs(self):
        self.assertTrue(is_remote_path("hdfs://cluster/path"))

    def test_local_absolute(self):
        self.assertFalse(is_remote_path("/home/user/data.json"))

    def test_local_relative(self):
        self.assertFalse(is_remote_path("relative/path.json"))

    def test_remote_path_in_is_absolute(self):
        """is_absolute_path should return True for remote paths."""
        self.assertTrue(is_absolute_path("s3://bucket/key"))
        self.assertTrue(is_absolute_path("https://example.com/file"))


class GetAllFilesPathsUnderTest(DataJuicerTestCaseBase):
    """Test get_all_files_paths_under: lists files recursively or flat."""

    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()
        # Create structure:
        #   root/a.txt
        #   root/b.txt
        #   root/sub/c.txt
        for name in ["a.txt", "b.txt"]:
            with open(os.path.join(self.root, name), "w") as f:
                f.write(name)
        subdir = os.path.join(self.root, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "c.txt"), "w") as f:
            f.write("c")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)
        super().tearDown()

    def test_recurse(self):
        result = get_all_files_paths_under(self.root, recurse_subdirectories=True)
        basenames = [os.path.basename(p) for p in result]
        self.assertIn("a.txt", basenames)
        self.assertIn("b.txt", basenames)
        self.assertIn("c.txt", basenames)

    def test_no_recurse(self):
        result = get_all_files_paths_under(self.root, recurse_subdirectories=False)
        basenames = [os.path.basename(p) for p in result]
        self.assertIn("a.txt", basenames)
        self.assertIn("b.txt", basenames)
        # sub/ is a directory entry, c.txt should not appear at top level
        self.assertNotIn("c.txt", [os.path.basename(p) for p in result if os.path.isfile(p) and "sub" not in p])

    def test_sorted_output(self):
        result = get_all_files_paths_under(self.root, recurse_subdirectories=True)
        self.assertEqual(result, sorted(result))


class SinglePartitionWriteTest(DataJuicerTestCaseBase):
    """Test single_partition_write_with_filename: writes DataFrame partitions to disk."""

    def setUp(self):
        super().setUp()
        self.outdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.outdir, ignore_errors=True)
        super().tearDown()

    def test_write_jsonl(self):
        df = pd.DataFrame({
            "text": ["hello", "world"],
            "filename": ["part1.jsonl", "part1.jsonl"],
        })
        result = single_partition_write_with_filename(df, self.outdir, output_type="jsonl")
        # Non-empty partition → returns Series([False]) (empty_partition=False)
        self.assertEqual(list(result), [False])

        out_file = os.path.join(self.outdir, "part1.jsonl")
        self.assertTrue(os.path.exists(out_file))
        with open(out_file) as f:
            lines = [json.loads(line) for line in f]
        texts = [row["text"] for row in lines]
        self.assertEqual(sorted(texts), ["hello", "world"])
        # filename column should be dropped by default
        for row in lines:
            self.assertNotIn("filename", row)

    def test_write_parquet(self):
        df = pd.DataFrame({
            "text": ["alpha", "beta"],
            "filename": ["data.parquet", "data.parquet"],
        })
        result = single_partition_write_with_filename(
            df, self.outdir, output_type="parquet")
        self.assertEqual(list(result), [False])

        out_file = os.path.join(self.outdir, "data.parquet")
        self.assertTrue(os.path.exists(out_file))
        read_back = pd.read_parquet(out_file)
        self.assertEqual(list(read_back["text"]), ["alpha", "beta"])

    def test_keep_filename_column(self):
        df = pd.DataFrame({
            "text": ["keep"],
            "filename": ["out.jsonl"],
        })
        single_partition_write_with_filename(
            df, self.outdir, keep_filename_column=True, output_type="jsonl")
        out_file = os.path.join(self.outdir, "out.jsonl")
        with open(out_file) as f:
            row = json.loads(f.readline())
        self.assertIn("filename", row)

    def test_empty_partition(self):
        df = pd.DataFrame({"text": [], "filename": []})
        result = single_partition_write_with_filename(df, self.outdir)
        # Empty partition → returns Series([True])
        self.assertEqual(list(result), [True])
        # No file should be written
        self.assertEqual(os.listdir(self.outdir), [])

    def test_unknown_output_type_raises(self):
        df = pd.DataFrame({
            "text": ["x"],
            "filename": ["f.csv"],
        })
        with self.assertRaises(ValueError):
            single_partition_write_with_filename(
                df, self.outdir, output_type="csv")

    def test_multiple_filenames(self):
        """Rows with different filenames should be split into separate files."""
        df = pd.DataFrame({
            "text": ["a1", "b1", "a2"],
            "filename": ["file_a.jsonl", "file_b.jsonl", "file_a.jsonl"],
        })
        single_partition_write_with_filename(df, self.outdir, output_type="jsonl")
        self.assertTrue(os.path.exists(os.path.join(self.outdir, "file_a.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.outdir, "file_b.jsonl")))


class ReadSinglePartitionTest(DataJuicerTestCaseBase):
    """Test read_single_partition: reads jsonl/json/parquet files into DataFrame."""

    def setUp(self):
        super().setUp()
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        super().tearDown()

    def _write_jsonl(self, filename, rows):
        path = os.path.join(self.tmpdir, filename)
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    def test_read_jsonl(self):
        path = self._write_jsonl("data.jsonl", [
            {"text": "hello", "score": 1},
            {"text": "world", "score": 2},
        ])
        df = read_single_partition([path], filetype="jsonl")
        self.assertEqual(len(df), 2)
        self.assertIn("text", df.columns)
        self.assertIn("score", df.columns)

    def test_read_jsonl_add_filename(self):
        path = self._write_jsonl("data.jsonl", [{"text": "row1"}])
        df = read_single_partition([path], filetype="jsonl", add_filename=True)
        self.assertIn("filename", df.columns)
        self.assertEqual(df["filename"].iloc[0], "data.jsonl")

    def test_read_jsonl_with_columns_filter(self):
        path = self._write_jsonl("data.jsonl", [
            {"text": "a", "score": 1, "extra": "x"},
        ])
        df = read_single_partition(
            [path], filetype="jsonl", columns=["text", "score"])
        self.assertIn("text", df.columns)
        self.assertIn("score", df.columns)
        self.assertNotIn("extra", df.columns)

    def test_read_parquet(self):
        path = os.path.join(self.tmpdir, "data.parquet")
        pd.DataFrame({"text": ["a", "b"], "num": [1, 2]}).to_parquet(path)
        df = read_single_partition([path], filetype="parquet")
        self.assertEqual(len(df), 2)
        self.assertIn("text", df.columns)

    def test_read_parquet_with_columns(self):
        path = os.path.join(self.tmpdir, "data.parquet")
        pd.DataFrame({"text": ["a"], "num": [1], "extra": ["x"]}).to_parquet(path)
        df = read_single_partition(
            [path], filetype="parquet", columns=["text"])
        self.assertIn("text", df.columns)
        self.assertNotIn("extra", df.columns)

    def test_read_multiple_jsonl_files(self):
        p1 = self._write_jsonl("a.jsonl", [{"text": "row1"}])
        p2 = self._write_jsonl("b.jsonl", [{"text": "row2"}])
        df = read_single_partition([p1, p2], filetype="jsonl")
        self.assertEqual(len(df), 2)

    def test_unknown_filetype_raises(self):
        with self.assertRaises(RuntimeError):
            read_single_partition(["dummy.csv"], filetype="csv")

    def test_columns_sorted(self):
        """Output columns should be alphabetically sorted."""
        path = self._write_jsonl("data.jsonl", [
            {"z_col": 1, "a_col": 2, "m_col": 3},
        ])
        df = read_single_partition([path], filetype="jsonl")
        self.assertEqual(list(df.columns), sorted(df.columns))


class ExpandOutdirAndMkdirTest(DataJuicerTestCaseBase):
    """Test expand_outdir_and_mkdir: expands path and creates directory."""

    def test_creates_and_returns_absolute(self):
        tmpdir = tempfile.mkdtemp()
        import shutil
        shutil.rmtree(tmpdir)
        self.assertFalse(os.path.exists(tmpdir))

        result = expand_outdir_and_mkdir(tmpdir)
        self.assertTrue(os.path.exists(result))
        self.assertTrue(os.path.isabs(result))

        shutil.rmtree(result, ignore_errors=True)

    def test_existing_dir(self):
        tmpdir = tempfile.mkdtemp()
        result = expand_outdir_and_mkdir(tmpdir)
        self.assertTrue(os.path.exists(result))
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
