import subprocess
import sys
import unittest
from pathlib import Path


class VideoOptionalDependenciesTest(unittest.TestCase):
    def test_text_mapper_import_and_execution_without_opencv(self):
        script = """
import sys
sys.modules["cv2"] = None
from data_juicer.utils.lazy_loader import LazyLoader

def reject_install(*args, **kwargs):
    raise AssertionError("Text mapper import must not install optional dependencies")
LazyLoader._install_package = reject_install
from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper
op = WhitespaceNormalizationMapper()
assert op.process_batched({"text": ["  a\\tb  "]})["text"] == ["a b"]
assert sys.modules["cv2"] is None
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
