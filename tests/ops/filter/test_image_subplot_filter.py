import os
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset

from data_juicer.ops.filter.image_subplot_filter import ImageSubplotFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ImageSubplotFilterTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                              'data')
    subplot_img_path = os.path.join(data_path, 'image_subplot.jpg')
    nosubplot_img_path = os.path.join(data_path, 'image_nosubplot.jpg')

    def _run_image_subplot_filter(self, dataset: Dataset, target_list,
                                  op):
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                          column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats)
        dataset = dataset.filter(op.process)
        dataset = dataset.select_columns(column_names=[op.image_key])
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_detect_subplot_image(self):
        """Test that images with subplots are correctly detected and filtered."""
        ds_list = [{'images': [self.subplot_img_path]}]
        tgt_list = []  # Should be filtered out
        dataset = Dataset.from_list(ds_list)
        op = ImageSubplotFilter(min_confidence=0.5)
        self._run_image_subplot_filter(dataset, tgt_list, op)

    def test_detect_no_subplot_image(self):
        """Test that images without subplots are kept."""
        ds_list = [{'images': [self.nosubplot_img_path]}]
        tgt_list = [{'images': [self.nosubplot_img_path]}]  # Should be kept
        dataset = Dataset.from_list(ds_list)
        op = ImageSubplotFilter(min_confidence=0.5)
        self._run_image_subplot_filter(dataset, tgt_list, op)

    def test_confidence_threshold(self):
        """Test different confidence thresholds."""
        ds_list = [{'images': [self.subplot_img_path]}]
        
        # High threshold - might not filter
        tgt_list_high = [{'images': [self.subplot_img_path]}]
        dataset_high = Dataset.from_list(ds_list)
        op_high = ImageSubplotFilter(min_confidence=0.9)
        self._run_image_subplot_filter(dataset_high, tgt_list_high, op_high)
        
        # Low threshold - should filter
        tgt_list_low = []
        dataset_low = Dataset.from_list(ds_list)
        op_low = ImageSubplotFilter(min_confidence=0.1)
        self._run_image_subplot_filter(dataset_low, tgt_list_low, op_low)

    def test_any_strategy(self):
        """Test 'any' strategy for multi-image samples."""
        ds_list = [{'images': [self.nosubplot_img_path, self.subplot_img_path]}]
        tgt_list = []  # Should be filtered due to any subplot presence
        dataset = Dataset.from_list(ds_list)
        op = ImageSubplotFilter(any_or_all='any', min_confidence=0.5)
        self._run_image_subplot_filter(dataset, tgt_list, op)

    def test_all_strategy(self):
        """Test 'all' strategy for multi-image samples."""
        ds_list = [{'images': [self.nosubplot_img_path, self.subplot_img_path]}]
        tgt_list = [{'images': [self.nosubplot_img_path, self.subplot_img_path]}]  # Should be kept
        dataset = Dataset.from_list(ds_list)
        op = ImageSubplotFilter(any_or_all='all', min_confidence=0.5)
        self._run_image_subplot_filter(dataset, tgt_list, op)


    def test_min_lines(self):
        """Test different minimum line requirements."""
        ds_list = [{'images': [self.subplot_img_path]}]
        
        # High line requirements - might not meet criteria
        tgt_list_high = [{'images': [self.subplot_img_path]}]
        dataset_high = Dataset.from_list(ds_list)
        op_high = ImageSubplotFilter(min_horizontal_lines=5, 
                                   min_vertical_lines=5, 
                                   min_confidence=0.5)
        self._run_image_subplot_filter(dataset_high, tgt_list_high, op_high)
        
        # Low line requirements - should meet criteria
        tgt_list_low = []
        dataset_low = Dataset.from_list(ds_list)
        op_low = ImageSubplotFilter(min_horizontal_lines=1, 
                                  min_vertical_lines=1, 
                                  min_confidence=0.5)
        self._run_image_subplot_filter(dataset_low, tgt_list_low, op_low)

    def test_no_images(self):
        """Test sample with no images."""
        ds_list = [{'images': []}]
        tgt_list = [{'images': []}]  # Should be kept
        dataset = Dataset.from_list(ds_list)
        op = ImageSubplotFilter(min_confidence=0.5)
        self._run_image_subplot_filter(dataset, tgt_list, op)

    def test_single_image(self):
        """Test sample with single image."""
        ds_list = [{'images': [self.nosubplot_img_path]}]
        tgt_list = [{'images': [self.nosubplot_img_path]}]  # Should be kept
        dataset = Dataset.from_list(ds_list)
        op = ImageSubplotFilter(min_confidence=0.5)
        self._run_image_subplot_filter(dataset, tgt_list, op)

    def test_stats_computation(self):
        """Test that statistics are correctly computed."""
        ds_list = [{'images': [self.subplot_img_path]}]
        dataset = Dataset.from_list(ds_list)
        op = ImageSubplotFilter(min_confidence=0.5)
        
        # Use the same pattern as _run_image_subplot_filter to ensure stats field exists
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats,
                                          column=[{}] * dataset.num_rows)
        
        # Compute statistics
        dataset = dataset.map(op.compute_stats)
        stats = dataset[0][Fields.stats]
        
        # Verify all required stats keys are present
        self.assertIn(StatsKeys.image_subplot_confidence, stats)
        self.assertIn(StatsKeys.horizontal_peak_count, stats)
        self.assertIn(StatsKeys.vertical_peak_count, stats)
        self.assertIn(StatsKeys.subplot_detected, stats)
        
        # Verify data types and ranges
        self.assertIsInstance(stats[StatsKeys.image_subplot_confidence], list)
        self.assertIsInstance(stats[StatsKeys.horizontal_peak_count], list)
        self.assertIsInstance(stats[StatsKeys.vertical_peak_count], list)
        self.assertIsInstance(stats[StatsKeys.subplot_detected], bool)
        
        # Confidence should be between 0 and 1
        confidences = stats[StatsKeys.image_subplot_confidence]
        if confidences:
            self.assertGreaterEqual(confidences[0], 0.0)
            self.assertLessEqual(confidences[0], 1.0)

    def test_multiple_images(self):
        """Test sample with multiple images."""
        ds_list = [{'images': [self.nosubplot_img_path, self.subplot_img_path, self.nosubplot_img_path]}]
        
        # With any strategy, should be filtered
        tgt_list_any = []
        dataset_any = Dataset.from_list(ds_list)
        op_any = ImageSubplotFilter(any_or_all='any', min_confidence=0.5)
        self._run_image_subplot_filter(dataset_any, tgt_list_any, op_any)
        
        # With all strategy, should be kept
        tgt_list_all = [{'images': [self.nosubplot_img_path, self.subplot_img_path, self.nosubplot_img_path]}]
        dataset_all = Dataset.from_list(ds_list)
        op_all = ImageSubplotFilter(any_or_all='all', min_confidence=0.5)
        self._run_image_subplot_filter(dataset_all, tgt_list_all, op_all)

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid any_or_all parameter
        with self.assertRaises(ValueError):
            ImageSubplotFilter(any_or_all='invalid')

if __name__ == '__main__':
    unittest.main()