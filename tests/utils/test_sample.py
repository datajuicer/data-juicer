import unittest

from datasets import Dataset

from data_juicer.utils.sample import random_sample
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RandomSampleTest(DataJuicerTestCaseBase):
    """Test random_sample: subset selection by weight or number."""

    def _make_dataset(self, num_rows):
        return Dataset.from_dict({"text": [f"row{i}" for i in range(num_rows)]})

    def test_sample_by_weight(self):
        ds = self._make_dataset(10)
        result = random_sample(ds, weight=0.5)
        self.assertEqual(len(result), 5)

    def test_sample_by_number(self):
        ds = self._make_dataset(10)
        result = random_sample(ds, sample_number=3)
        self.assertEqual(len(result), 3)

    def test_number_overrides_weight(self):
        """When sample_number > 0, it takes precedence over weight."""
        ds = self._make_dataset(10)
        result = random_sample(ds, weight=0.1, sample_number=7)
        self.assertEqual(len(result), 7)

    def test_full_dataset_returns_same(self):
        """weight=1.0 with full dataset returns the original dataset."""
        ds = self._make_dataset(10)
        result = random_sample(ds, weight=1.0)
        self.assertIs(result, ds)

    def test_default_seed_is_42(self):
        """Two calls without seed should produce identical results."""
        ds = self._make_dataset(20)
        r1 = random_sample(ds, weight=0.5)
        r2 = random_sample(ds, weight=0.5)
        self.assertEqual(r1["text"], r2["text"])

    def test_different_seeds_differ(self):
        ds = self._make_dataset(20)
        r1 = random_sample(ds, weight=0.5, seed=1)
        r2 = random_sample(ds, weight=0.5, seed=2)
        self.assertNotEqual(r1["text"], r2["text"])

    def test_upsample_repeats(self):
        """sample_number > dataset size should repeat rows."""
        ds = self._make_dataset(3)
        result = random_sample(ds, sample_number=7)
        self.assertEqual(len(result), 7)

    def test_weight_zero(self):
        """weight=0 with sample_number=0 → ceil(0)=0 → empty subset."""
        ds = self._make_dataset(10)
        result = random_sample(ds, weight=0.0, sample_number=0)
        self.assertEqual(len(result), 0)

    def test_fractional_weight_rounds_up(self):
        """np.ceil ensures partial rows round up."""
        ds = self._make_dataset(10)
        result = random_sample(ds, weight=0.15)
        # ceil(10 * 0.15) = ceil(1.5) = 2
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
