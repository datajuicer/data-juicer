"""Unit tests for data_juicer/utils/availability_utils.py."""

import importlib.metadata
import importlib.util
import os
import unittest
from unittest.mock import MagicMock, patch

from data_juicer.utils.availability_utils import (
    _is_package_available,
    _torch_check_and_set,
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class IsPackageAvailableTest(DataJuicerTestCaseBase):
    """Tests for _is_package_available."""

    def test_existing_package_returns_true(self):
        """A known installed package should return True."""
        result = _is_package_available("datasets")
        self.assertTrue(result)

    def test_nonexistent_package_returns_false(self):
        """A package that does not exist should return False."""
        result = _is_package_available("nonexistent_package_xyz_12345")
        self.assertFalse(result)

    def test_existing_package_with_version(self):
        """return_version=True should return (True, version_string)."""
        exists, version = _is_package_available("datasets", return_version=True)
        self.assertTrue(exists)
        self.assertIsInstance(version, str)
        self.assertNotEqual(version, "N/A")

    def test_nonexistent_package_with_version(self):
        """return_version=True for missing package returns (False, 'N/A')."""
        exists, version = _is_package_available(
            "nonexistent_package_xyz_12345", return_version=True
        )
        self.assertFalse(exists)
        self.assertEqual(version, "N/A")

    def test_spec_exists_but_no_metadata(self):
        """Package whose spec exists but metadata.version raises PackageNotFoundError."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch(
                "importlib.metadata.version",
                side_effect=importlib.metadata.PackageNotFoundError("fake"),
            ):
                result = _is_package_available("fake_pkg")
                self.assertFalse(result)

    def test_spec_exists_but_no_metadata_with_version(self):
        """Same as above but with return_version=True."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            with patch(
                "importlib.metadata.version",
                side_effect=importlib.metadata.PackageNotFoundError("fake"),
            ):
                exists, version = _is_package_available("fake_pkg", return_version=True)
                self.assertFalse(exists)
                self.assertEqual(version, "N/A")


class TorchCheckAndSetTest(DataJuicerTestCaseBase):
    """Tests for _torch_check_and_set."""

    def setUp(self):
        super().setUp()
        # Save and reset global flag
        import data_juicer.utils.availability_utils as mod
        self._mod = mod
        self._original_flag = mod.CHECK_SYSTEM_INFO_ONCE
        mod.CHECK_SYSTEM_INFO_ONCE = True
        self._original_omp = os.environ.get("OMP_NUM_THREADS")

    def tearDown(self):
        self._mod.CHECK_SYSTEM_INFO_ONCE = self._original_flag
        if self._original_omp is not None:
            os.environ["OMP_NUM_THREADS"] = self._original_omp
        elif "OMP_NUM_THREADS" in os.environ:
            del os.environ["OMP_NUM_THREADS"]
        super().tearDown()

    def test_sets_torch_num_threads(self):
        """On a system with torch, _torch_check_and_set calls torch.set_num_threads(1)."""
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                # Not on Mac py3.8 path
                with patch("sys.version_info", (3, 11, 0)):
                    with patch("platform.system", return_value="Linux"):
                        _torch_check_and_set()
                        mock_torch.set_num_threads.assert_called_once_with(1)

    def test_mac_py38_sets_omp_env(self):
        """On Mac + Python 3.8, should set OMP_NUM_THREADS=1."""
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                with patch(
                    "data_juicer.utils.availability_utils.sys"
                ) as mock_sys:
                    mock_sys.version_info = (3, 8, 0)
                    with patch("platform.system", return_value="Darwin"):
                        self._mod.CHECK_SYSTEM_INFO_ONCE = True
                        _torch_check_and_set()
                        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")
                        self.assertFalse(self._mod.CHECK_SYSTEM_INFO_ONCE)

    def test_no_torch_does_nothing(self):
        """When torch is not installed, _torch_check_and_set does nothing."""
        with patch("importlib.util.find_spec", return_value=None):
            # Should not raise
            _torch_check_and_set()
            # Flag unchanged
            self.assertTrue(self._mod.CHECK_SYSTEM_INFO_ONCE)

    def test_check_system_info_once_false_skips(self):
        """When CHECK_SYSTEM_INFO_ONCE is False, torch is not found, function skips."""
        self._mod.CHECK_SYSTEM_INFO_ONCE = False
        with patch("importlib.util.find_spec", return_value=None):
            _torch_check_and_set()
            # Remains False
            self.assertFalse(self._mod.CHECK_SYSTEM_INFO_ONCE)


if __name__ == "__main__":
    unittest.main()
