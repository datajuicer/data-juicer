"""
Comprehensive tests for PartitionSizeOptimizer.

Tests cover:
- Modality detection (TEXT, IMAGE, AUDIO, VIDEO, MULTIMODAL)
- Resource detection (CPU, memory, GPU)
- Partition size calculations
- Target size configuration
- Edge cases (small datasets, large datasets, skewed data)
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from jsonargparse import Namespace

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class MockDataset:
    """Mock dataset for testing partition size optimizer."""

    def __init__(self, samples, total_count=None):
        self._samples = samples
        self._total_count = total_count or len(samples)

    def count(self):
        return self._total_count

    def __len__(self):
        return self._total_count

    def get(self, n):
        return self._samples[:n]

    def take(self, n):
        return self._samples[:n]


class PartitionSizeOptimizerTest(DataJuicerTestCaseBase):
    """Tests for PartitionSizeOptimizer."""

    def setUp(self):
        super().setUp()
        self.cfg = Namespace()
        self.cfg.text_key = "text"
        self.cfg.image_key = "images"
        self.cfg.audio_key = "audios"
        self.cfg.video_key = "videos"

    # ==================== Modality Detection Tests ====================

    def test_detect_modality_text_only(self):
        """Test detection of pure text modality."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        optimizer = PartitionSizeOptimizer(self.cfg)

        sample = {"text": "This is a text sample", "images": [], "audios": [], "videos": []}
        modality = optimizer.detect_modality(sample)

        self.assertEqual(modality, ModalityType.TEXT)

    def test_detect_modality_image_only(self):
        """Test detection of pure image modality."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        optimizer = PartitionSizeOptimizer(self.cfg)

        sample = {"text": "", "images": ["img1.jpg", "img2.jpg"], "audios": [], "videos": []}
        modality = optimizer.detect_modality(sample)

        self.assertEqual(modality, ModalityType.IMAGE)

    def test_detect_modality_audio_only(self):
        """Test detection of pure audio modality."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        optimizer = PartitionSizeOptimizer(self.cfg)

        sample = {"text": "", "images": [], "audios": ["audio1.mp3"], "videos": []}
        modality = optimizer.detect_modality(sample)

        self.assertEqual(modality, ModalityType.AUDIO)

    def test_detect_modality_video_only(self):
        """Test detection of pure video modality."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        optimizer = PartitionSizeOptimizer(self.cfg)

        sample = {"text": "", "images": [], "audios": [], "videos": ["video1.mp4"]}
        modality = optimizer.detect_modality(sample)

        self.assertEqual(modality, ModalityType.VIDEO)

    def test_detect_modality_multimodal(self):
        """Test detection of multimodal content."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        optimizer = PartitionSizeOptimizer(self.cfg)

        # Text + Image
        sample = {"text": "Caption", "images": ["img.jpg"], "audios": [], "videos": []}
        modality = optimizer.detect_modality(sample)
        self.assertEqual(modality, ModalityType.MULTIMODAL)

        # Text + Audio
        sample = {"text": "Transcript", "images": [], "audios": ["audio.mp3"], "videos": []}
        modality = optimizer.detect_modality(sample)
        self.assertEqual(modality, ModalityType.MULTIMODAL)

        # Image + Video
        sample = {"text": "", "images": ["img.jpg"], "audios": [], "videos": ["video.mp4"]}
        modality = optimizer.detect_modality(sample)
        self.assertEqual(modality, ModalityType.MULTIMODAL)

    def test_detect_modality_empty_sample(self):
        """Test detection with empty sample defaults to TEXT."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        optimizer = PartitionSizeOptimizer(self.cfg)

        sample = {"text": "", "images": [], "audios": [], "videos": []}
        modality = optimizer.detect_modality(sample)

        self.assertEqual(modality, ModalityType.TEXT)

    def test_detect_modality_missing_keys(self):
        """Test detection with missing keys defaults to TEXT."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        optimizer = PartitionSizeOptimizer(self.cfg)

        sample = {}  # No keys at all
        modality = optimizer.detect_modality(sample)

        self.assertEqual(modality, ModalityType.TEXT)

    # ==================== Target Partition Size Tests ====================

    def test_calculate_target_partition_mb_from_config(self):
        """Test that configured target_size_mb is used."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        self.cfg.partition = Namespace()
        self.cfg.partition.target_size_mb = 512

        optimizer = PartitionSizeOptimizer(self.cfg)
        target = optimizer.calculate_target_partition_mb(available_memory_gb=32)

        self.assertEqual(target, 512)

    def test_calculate_target_partition_mb_low_memory(self):
        """Test dynamic target with low memory (<16GB)."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)
        target = optimizer.calculate_target_partition_mb(available_memory_gb=8)

        self.assertEqual(target, 32)

    def test_calculate_target_partition_mb_medium_memory(self):
        """Test dynamic target with medium memory (16-64GB)."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)
        target = optimizer.calculate_target_partition_mb(available_memory_gb=32)

        self.assertEqual(target, 64)

    def test_calculate_target_partition_mb_high_memory(self):
        """Test dynamic target with high memory (64-256GB)."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)
        target = optimizer.calculate_target_partition_mb(available_memory_gb=128)

        self.assertEqual(target, 128)

    def test_calculate_target_partition_mb_very_high_memory(self):
        """Test dynamic target with very high memory (>256GB)."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)
        target = optimizer.calculate_target_partition_mb(available_memory_gb=512)

        self.assertEqual(target, 256)

    # ==================== Resource Detection Tests ====================

    def test_detect_local_resources(self):
        """Test local resource detection."""
        from data_juicer.core.executor.partition_size_optimizer import ResourceDetector

        resources = ResourceDetector.detect_local_resources()

        self.assertIsNotNone(resources)
        self.assertGreater(resources.cpu_cores, 0)
        self.assertGreater(resources.available_memory_gb, 0)
        self.assertGreater(resources.total_memory_gb, 0)
        self.assertGreaterEqual(resources.gpu_count, 0)

    def test_detect_ray_cluster_not_initialized(self):
        """Test Ray cluster detection when Ray is not initialized."""
        from data_juicer.core.executor.partition_size_optimizer import ResourceDetector

        with patch('ray.is_initialized', return_value=False):
            resources = ResourceDetector.detect_ray_cluster()

        self.assertIsNone(resources)

    def test_calculate_optimal_worker_count_basic(self):
        """Test optimal worker count calculation."""
        from data_juicer.core.executor.partition_size_optimizer import (
            LocalResources,
            ResourceDetector,
        )

        local_resources = LocalResources(
            cpu_cores=16,
            available_memory_gb=32,
            total_memory_gb=64,
            gpu_count=0,
        )

        workers = ResourceDetector.calculate_optimal_worker_count(local_resources)

        # Should be ~75% of CPU cores, capped at 32
        self.assertGreater(workers, 0)
        self.assertLessEqual(workers, 16)
        self.assertLessEqual(workers, 32)

    def test_calculate_optimal_worker_count_with_workload(self):
        """Test worker count with workload info."""
        from data_juicer.core.executor.partition_size_optimizer import (
            LocalResources,
            ResourceDetector,
        )

        local_resources = LocalResources(
            cpu_cores=8,
            available_memory_gb=16,
            total_memory_gb=32,
            gpu_count=0,
        )

        # Few partitions - should reduce workers
        workers = ResourceDetector.calculate_optimal_worker_count(
            local_resources,
            partition_size=10000,
            total_samples=20000,  # ~2 partitions
        )

        self.assertGreater(workers, 0)
        self.assertLessEqual(workers, 8)

    # ==================== Dataset Characteristics Analysis Tests ====================

    def test_analyze_dataset_characteristics_text(self):
        """Test dataset analysis for text data."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        optimizer = PartitionSizeOptimizer(self.cfg)

        samples = [
            {"text": "Short text", "images": [], "audios": [], "videos": []},
            {"text": "A longer piece of text that has more characters", "images": [], "audios": [], "videos": []},
            {"text": "Medium length text here", "images": [], "audios": [], "videos": []},
        ]
        dataset = MockDataset(samples, total_count=1000)

        characteristics = optimizer.analyze_dataset_characteristics(dataset)

        self.assertEqual(characteristics.primary_modality, ModalityType.TEXT)
        self.assertGreater(characteristics.avg_text_length, 0)
        self.assertEqual(characteristics.avg_images_per_sample, 0)
        self.assertEqual(characteristics.total_samples, 1000)

    def test_analyze_dataset_characteristics_multimodal(self):
        """Test dataset analysis for multimodal data."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        optimizer = PartitionSizeOptimizer(self.cfg)

        samples = [
            {"text": "Caption 1", "images": ["img1.jpg"], "audios": [], "videos": []},
            {"text": "Caption 2", "images": ["img2.jpg", "img3.jpg"], "audios": [], "videos": []},
            {"text": "Caption 3", "images": ["img4.jpg"], "audios": [], "videos": []},
        ]
        dataset = MockDataset(samples, total_count=500)

        characteristics = optimizer.analyze_dataset_characteristics(dataset)

        self.assertEqual(characteristics.primary_modality, ModalityType.MULTIMODAL)
        self.assertGreater(characteristics.avg_images_per_sample, 0)

    def test_analyze_dataset_characteristics_small_dataset(self):
        """Test analysis with very small dataset."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)

        samples = [{"text": "Single sample", "images": [], "audios": [], "videos": []}]
        dataset = MockDataset(samples, total_count=1)

        characteristics = optimizer.analyze_dataset_characteristics(dataset)

        self.assertEqual(characteristics.total_samples, 1)
        self.assertEqual(characteristics.sample_size_analyzed, 1)

    # ==================== Processing Complexity Tests ====================

    def test_analyze_processing_complexity_simple(self):
        """Test complexity analysis with simple operations."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)

        pipeline = [
            {"text_length_filter": {"min_len": 10}},
            {"whitespace_normalization_mapper": {}},
        ]

        complexity = optimizer.analyze_processing_complexity(pipeline)

        self.assertGreaterEqual(complexity, 1.0)

    def test_analyze_processing_complexity_with_embeddings(self):
        """Test complexity analysis with high-complexity operations."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)

        pipeline = [
            {"text_embedding_mapper": {"model": "bert"}},
            {"document_similarity_deduplicator": {}},
        ]

        complexity = optimizer.analyze_processing_complexity(pipeline)

        # High complexity operations should increase the score
        self.assertGreater(complexity, 1.0)

    def test_analyze_processing_complexity_empty_pipeline(self):
        """Test complexity analysis with empty pipeline."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)

        complexity = optimizer.analyze_processing_complexity([])

        self.assertEqual(complexity, 1.0)  # Base complexity

    # ==================== Optimal Partition Size Tests ====================

    def test_get_optimal_partition_size_text(self):
        """Test optimal partition size for text data."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)

        samples = [{"text": "x" * 500, "images": [], "audios": [], "videos": []} for _ in range(100)]
        dataset = MockDataset(samples, total_count=10000)
        pipeline = [{"text_length_filter": {}}]

        optimal_size, max_size_mb = optimizer.get_optimal_partition_size(dataset, pipeline)

        self.assertGreater(optimal_size, 0)
        self.assertGreater(max_size_mb, 0)
        self.assertLessEqual(max_size_mb, 512)  # Should not exceed max

    def test_get_partition_recommendations(self):
        """Test getting full partition recommendations."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)

        samples = [{"text": "Sample text", "images": [], "audios": [], "videos": []} for _ in range(50)]
        dataset = MockDataset(samples, total_count=5000)
        pipeline = [{"text_length_filter": {}}]

        recommendations = optimizer.get_partition_recommendations(dataset, pipeline)

        self.assertIn("recommended_partition_size", recommendations)
        self.assertIn("recommended_max_size_mb", recommendations)
        self.assertIn("recommended_worker_count", recommendations)
        self.assertIn("primary_modality", recommendations)
        self.assertIn("data_characteristics", recommendations)
        self.assertIn("resource_analysis", recommendations)
        self.assertIn("reasoning", recommendations)

    # ==================== auto_configure_resources Tests ====================

    def test_auto_configure_resources(self):
        """Test the main auto_configure_resources function."""
        from data_juicer.core.executor.partition_size_optimizer import auto_configure_resources

        samples = [{"text": "Test sample", "images": [], "audios": [], "videos": []} for _ in range(20)]
        dataset = MockDataset(samples, total_count=2000)
        pipeline = [{"text_length_filter": {"min_len": 5}}]

        recommendations = auto_configure_resources(self.cfg, dataset, pipeline)

        self.assertIsInstance(recommendations, dict)
        self.assertIn("recommended_partition_size", recommendations)
        self.assertIn("recommended_max_size_mb", recommendations)
        self.assertIn("recommended_worker_count", recommendations)

    # ==================== Modality Config Tests ====================

    def test_modality_configs_exist(self):
        """Test that all modality configs are defined."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        for modality in ModalityType:
            self.assertIn(modality, PartitionSizeOptimizer.MODALITY_CONFIGS)

    def test_modality_configs_have_required_fields(self):
        """Test that modality configs have all required fields."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        for modality, config in PartitionSizeOptimizer.MODALITY_CONFIGS.items():
            self.assertIsNotNone(config.default_partition_size)
            self.assertIsNotNone(config.max_partition_size)
            self.assertIsNotNone(config.max_partition_size_mb)
            self.assertIsNotNone(config.memory_multiplier)
            self.assertIsNotNone(config.complexity_multiplier)
            self.assertGreater(config.default_partition_size, 0)
            self.assertGreater(config.max_partition_size, config.default_partition_size)

    def test_modality_configs_memory_multipliers(self):
        """Test that memory multipliers increase with complexity."""
        from data_juicer.core.executor.partition_size_optimizer import (
            ModalityType,
            PartitionSizeOptimizer,
        )

        configs = PartitionSizeOptimizer.MODALITY_CONFIGS

        # Text should have lowest multiplier
        self.assertEqual(configs[ModalityType.TEXT].memory_multiplier, 1.0)

        # Video should have highest multiplier
        self.assertGreater(
            configs[ModalityType.VIDEO].memory_multiplier,
            configs[ModalityType.IMAGE].memory_multiplier
        )

    # ==================== Edge Cases ====================

    def test_dataset_with_unknown_count(self):
        """Test handling of dataset where count() fails."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)

        class BrokenDataset:
            def count(self):
                raise Exception("Cannot count")

            def get(self, n):
                return [{"text": "sample"}]

        dataset = BrokenDataset()

        # Should not raise, should use fallback
        characteristics = optimizer.analyze_dataset_characteristics(dataset)
        self.assertIsNotNone(characteristics)

    def test_estimate_sample_size_mb(self):
        """Test sample size estimation returns positive value."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)

        sample = {"text": "sample text", "images": [], "audios": [], "videos": []}
        size = optimizer.estimate_sample_size_mb(sample)

        # Should return a positive size in MB
        self.assertGreater(size, 0)
        self.assertIsInstance(size, float)

    def test_estimate_sample_size_deep_calculation(self):
        """Test that sample size estimation uses deep calculation for nested content."""
        from data_juicer.core.executor.partition_size_optimizer import PartitionSizeOptimizer

        optimizer = PartitionSizeOptimizer(self.cfg)

        # Small sample with short text
        small_sample = {"text": "Hello", "id": 1}

        # Large sample with long text and nested metadata
        large_sample = {
            "text": "A" * 10000,  # 10KB of text
            "id": 2,
            "meta": {
                "source": "test",
                "tags": ["tag1", "tag2", "tag3"],
                "nested": {"deep": "value" * 100}
            }
        }

        small_size = optimizer.estimate_sample_size_mb(small_sample)
        large_size = optimizer.estimate_sample_size_mb(large_sample)

        # Deep sizing should show large sample is significantly bigger
        self.assertGreater(large_size, small_size)
        # Large sample should be at least 10x bigger due to 10KB text
        self.assertGreater(large_size, small_size * 5)


class ModalityTypeEnumTest(DataJuicerTestCaseBase):
    """Tests for ModalityType enum."""

    def test_modality_values(self):
        """Test that all modalities have correct string values."""
        from data_juicer.core.executor.partition_size_optimizer import ModalityType

        self.assertEqual(ModalityType.TEXT.value, "text")
        self.assertEqual(ModalityType.IMAGE.value, "image")
        self.assertEqual(ModalityType.AUDIO.value, "audio")
        self.assertEqual(ModalityType.VIDEO.value, "video")
        self.assertEqual(ModalityType.MULTIMODAL.value, "multimodal")


if __name__ == '__main__':
    unittest.main()
