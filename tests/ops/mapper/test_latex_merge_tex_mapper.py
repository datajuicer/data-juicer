import io
import os
import shutil
import tarfile
import tempfile
import unittest
import zipfile

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.latex_merge_tex_mapper import LatexMergeTexMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

TEX_A = r"""\documentclass{article}
\begin{document}
Hello from main.
\end{document}
"""

TEX_B = r"""\section{Intro}
Some intro text.
"""

TEX_C = r"""\section{Method}
Some method text.
"""


class LatexMergeTexMapperTest(DataJuicerTestCaseBase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_tar_gz(self, files, name="proj.tar.gz"):
        """Create a tar.gz with str values (UTF-8 encoded) or raw bytes."""
        path = os.path.join(self._tmpdir, name)
        with tarfile.open(path, "w:gz") as tf:
            for fname, content in files.items():
                data = content.encode("utf-8") \
                    if isinstance(content, str) else content
                info = tarfile.TarInfo(name=fname)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return path

    def _make_tar(self, files, name="proj.tar"):
        """Create a plain (uncompressed) tar archive."""
        path = os.path.join(self._tmpdir, name)
        with tarfile.open(path, "w") as tf:
            for fname, content in files.items():
                data = content.encode("utf-8") \
                    if isinstance(content, str) else content
                info = tarfile.TarInfo(name=fname)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
        return path

    def _make_zip(self, files, name="proj.zip"):
        """Create a zip with str values (UTF-8 encoded) or raw bytes."""
        path = os.path.join(self._tmpdir, name)
        with zipfile.ZipFile(path, "w") as zf:
            for fname, content in files.items():
                zf.writestr(fname, content)
        return path

    def _run(self, samples, op):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process)
        return list(dataset)

    def _assert_joined_in_either_order(self, result, part_a, part_b, sep):
        """Assert *result* equals part_a + sep + part_b (in either order)."""
        option1 = part_a + sep + part_b
        option2 = part_b + sep + part_a
        self.assertTrue(
            result == option1 or result == option2,
            f"Result does not match either ordering.\n"
            f"Got:\n{result!r}\n"
            f"Expected one of:\n{option1!r}\n--- or ---\n{option2!r}"
        )

    def _sample(self, archive_path):
        """Build a sample dict with separate compressed_file and text keys."""
        return {"compressed_file": archive_path, "text": ""}

    def test_tar_gz_multiple_tex(self):
        archive = self._make_tar_gz({
            "main.tex": TEX_A,
            "intro.tex": TEX_B,
        })
        results = self._run([self._sample(archive)], LatexMergeTexMapper())
        self._assert_joined_in_either_order(
            results[0]["text"], TEX_A, TEX_B, "\n\n")

    def test_zip_multiple_tex(self):
        archive = self._make_zip({
            "main.tex": TEX_A,
            "method.tex": TEX_C,
        })
        results = self._run([self._sample(archive)], LatexMergeTexMapper())
        self._assert_joined_in_either_order(
            results[0]["text"], TEX_A, TEX_C, "\n\n")

    def test_plain_tar_multiple_tex(self):
        archive = self._make_tar({
            "main.tex": TEX_A,
            "intro.tex": TEX_B,
        })
        results = self._run([self._sample(archive)], LatexMergeTexMapper())
        self._assert_joined_in_either_order(
            results[0]["text"], TEX_A, TEX_B, "\n\n")

    def test_tgz_multiple_tex(self):
        archive = self._make_tar_gz({
            "main.tex": TEX_A,
            "method.tex": TEX_C,
        }, name="proj.tgz")
        results = self._run([self._sample(archive)], LatexMergeTexMapper())
        self._assert_joined_in_either_order(
            results[0]["text"], TEX_A, TEX_C, "\n\n")

    def test_unsupported_extension(self):
        path = os.path.join(self._tmpdir, "paper.gz")
        with open(path, "wb") as f:
            f.write(b"dummy")
        results = self._run([self._sample(path)], LatexMergeTexMapper())
        self.assertEqual(results[0]["text"], "")

    def test_no_tex_in_archive(self):
        archive = self._make_tar_gz({
            "readme.md": "# Hello",
            "fig.png": "fake-png-bytes",
        })
        results = self._run([self._sample(archive)], LatexMergeTexMapper())
        self.assertEqual(results[0]["text"], "")

    def test_custom_separator(self):
        archive = self._make_tar_gz({
            "a.tex": TEX_A,
            "b.tex": TEX_B,
        })
        sep = "\n%%% FILE BOUNDARY %%%\n"
        op = LatexMergeTexMapper(separator=sep)
        results = self._run([self._sample(archive)], op)
        self._assert_joined_in_either_order(
            results[0]["text"], TEX_A, TEX_B, sep)

    def test_custom_compressed_file_key(self):
        archive = self._make_tar_gz({
            "main.tex": TEX_A,
        })
        samples = [{"text": "", "archive_path": archive}]
        op = LatexMergeTexMapper(compressed_file_key="archive_path")
        results = self._run(samples, op)
        self.assertIn(TEX_A.strip(), results[0]["text"])

    def test_multiple_samples(self):
        archive1 = self._make_tar_gz(
            {"a.tex": TEX_A}, name="p1.tar.gz")
        archive2 = self._make_zip(
            {"b.tex": TEX_B}, name="p2.zip")
        samples = [self._sample(archive1), self._sample(archive2)]
        results = self._run(samples, LatexMergeTexMapper())
        self.assertIn(TEX_A.strip(), results[0]["text"])
        self.assertIn(TEX_B.strip(), results[1]["text"])

    def test_invalid_path(self):
        results = self._run(
            [self._sample("/nonexistent/path/foo.tar.gz")],
            LatexMergeTexMapper())
        self.assertEqual(results[0]["text"], "")

    def test_latin1_encoding_tar(self):
        latin1_tex = b"\\section{R\xe9sum\xe9}\nCaf\xe9 na\xefve \xfcber.\n"
        archive = self._make_tar_gz(
            {"paper.tex": latin1_tex}, name="latin1.tar.gz")
        results = self._run([self._sample(archive)], LatexMergeTexMapper())
        text = results[0]["text"]
        self.assertIn("\\section{R", text)
        self.assertIn("\ufffd", text,
                      "Non-UTF-8 bytes should be replaced with U+FFFD")
        self.assertNotIn("\xe9", text,
                         "Raw Latin-1 chars must not survive as-is")

    def test_latin1_encoding_zip(self):
        latin1_tex = b"\\begin{document}\nStra\xdfe na\xefve.\n\\end{document}\n"
        archive = self._make_zip(
            {"doc.tex": latin1_tex}, name="latin1.zip")
        results = self._run([self._sample(archive)], LatexMergeTexMapper())
        text = results[0]["text"]
        self.assertIn("\\begin{document}", text)
        self.assertIn("\ufffd", text,
                      "Non-UTF-8 bytes should be replaced with U+FFFD")
        self.assertNotIn("\xdf", text,
                         "Raw Latin-1 chars must not survive as-is")

    def test_max_file_size_tar(self):
        small = "ok"
        big = "x" * 200
        archive = self._make_tar_gz({
            "small.tex": small,
            "big.tex": big,
        }, name="size_limit.tar.gz")
        op = LatexMergeTexMapper(max_file_size=100)
        results = self._run([self._sample(archive)], op)
        self.assertIn(small, results[0]["text"])
        self.assertNotIn(big, results[0]["text"])

    def test_max_file_size_zip(self):
        small = "ok"
        big = "x" * 200
        archive = self._make_zip({
            "small.tex": small,
            "big.tex": big,
        }, name="size_limit.zip")
        op = LatexMergeTexMapper(max_file_size=100)
        results = self._run([self._sample(archive)], op)
        self.assertIn(small, results[0]["text"])
        self.assertNotIn(big, results[0]["text"])


if __name__ == "__main__":
    unittest.main()
