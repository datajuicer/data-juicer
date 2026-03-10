"""
Partition Size Optimizer for DataJuicer

This module automatically configures optimal partition sizes based on:
1. Data modality (text, image, audio, video, multimodal)
2. Dataset characteristics (file sizes, complexity)
3. Available system resources (CPU, memory, GPU)
4. Processing pipeline complexity
5. Ray cluster configuration
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import psutil
import ray
from loguru import logger


class ModalityType(Enum):
    """Supported data modalities."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


@dataclass
class LocalResources:
    """Local system resources."""

    cpu_cores: int
    available_memory_gb: float
    total_memory_gb: float
    gpu_count: int
    gpu_memory_gb: Optional[float] = None
    disk_space_gb: Optional[float] = None


@dataclass
class ClusterResources:
    """Ray cluster resources."""

    num_nodes: int
    total_cpu_cores: int
    total_memory_gb: float
    available_cpu_cores: int
    available_memory_gb: float
    gpu_resources: Dict[str, int]


@dataclass
class DataCharacteristics:
    """Data characteristics from sampling."""

    primary_modality: ModalityType
    modality_distribution: Dict[ModalityType, int]
    avg_text_length: float
    avg_images_per_sample: float
    avg_audio_per_sample: float
    avg_video_per_sample: float
    total_samples: int
    sample_size_analyzed: int
    memory_per_sample_mb: float
    processing_complexity_score: float
    data_skew_factor: float  # 0-1, higher means more variance


@dataclass
class ModalityConfig:
    """Configuration for a specific modality."""

    modality: ModalityType
    default_partition_size: int
    max_partition_size: int
    max_partition_size_mb: int
    memory_multiplier: float  # Memory usage multiplier compared to text
    complexity_multiplier: float  # Processing complexity multiplier
    description: str


class ResourceDetector:
    """Detect available system and cluster resources."""

    @staticmethod
    def detect_local_resources() -> LocalResources:
        """Detect local system resources."""
        # CPU
        cpu_cores = psutil.cpu_count(logical=True)

        # Memory
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        total_memory_gb = memory.total / (1024**3)

        # GPU (basic detection)
        gpu_count = 0
        gpu_memory_gb = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            pass

        # Disk space
        disk_space_gb = None
        try:
            disk_usage = psutil.disk_usage("/")
            disk_space_gb = disk_usage.free / (1024**3)
        except Exception as e:
            logger.warning(f"Could not detect disk space: {e}")
            pass

        return LocalResources(
            cpu_cores=cpu_cores,
            available_memory_gb=available_memory_gb,
            total_memory_gb=total_memory_gb,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            disk_space_gb=disk_space_gb,
        )

    @staticmethod
    def detect_ray_cluster() -> Optional[ClusterResources]:
        """Detect Ray cluster resources."""
        try:
            if not ray.is_initialized():
                return None

            # Get cluster resources
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()

            # Parse resources
            total_cpu = cluster_resources.get("CPU", 0)
            total_memory = cluster_resources.get("memory", 0) / (1024**3)  # Convert to GB
            available_cpu = available_resources.get("CPU", 0)
            available_memory = available_resources.get("memory", 0) / (1024**3)

            # Count nodes (approximate)
            num_nodes = max(1, int(total_cpu / 8))  # Assume 8 cores per node

            # GPU resources
            gpu_resources = {}
            for key, value in cluster_resources.items():
                if key.startswith("GPU"):
                    gpu_resources[key] = value

            return ClusterResources(
                num_nodes=num_nodes,
                total_cpu_cores=int(total_cpu),
                total_memory_gb=total_memory,
                available_cpu_cores=int(available_cpu),
                available_memory_gb=available_memory,
                gpu_resources=gpu_resources,
            )
        except Exception as e:
            logger.warning(f"Could not detect Ray cluster resources: {e}")
            return None

    @staticmethod
    def calculate_optimal_worker_count(
        local_resources: LocalResources,
        cluster_resources: Optional[ClusterResources] = None,
        partition_size: int = None,
        total_samples: int = None,
    ) -> int:
        """
        Calculate optimal number of Ray workers based on available resources.

        Args:
            local_resources: Local system resources
            cluster_resources: Ray cluster resources (optional)
            partition_size: Size of each partition (for workload estimation)
            total_samples: Total number of samples (for workload estimation)

        Returns:
            Optimal number of workers
        """
        # Determine available CPU cores
        if cluster_resources:
            available_cores = min(local_resources.cpu_cores, cluster_resources.available_cpu_cores)
        else:
            available_cores = local_resources.cpu_cores

        # Base calculation: use 75% of available cores to leave room for system processes
        base_workers = max(1, int(available_cores * 0.75))

        # Adjust based on workload characteristics
        if partition_size and total_samples:
            estimated_partitions = total_samples / partition_size

            # We want enough workers to process partitions efficiently
            # But not so many that we have too much overhead
            if estimated_partitions < base_workers:
                # Few partitions - reduce workers to avoid overhead
                optimal_workers = max(1, int(estimated_partitions * 0.8))
            elif estimated_partitions > base_workers * 2:
                # Many partitions - can use more workers
                optimal_workers = min(available_cores, int(base_workers * 1.2))
            else:
                # Balanced workload - use base calculation
                optimal_workers = base_workers
        else:
            # No workload info - use base calculation
            optimal_workers = base_workers

        # Ensure we don't exceed available cores
        optimal_workers = min(optimal_workers, available_cores)

        # Minimum of 1 worker, cap at available cores (no arbitrary limit)
        optimal_workers = max(1, optimal_workers)

        logger.info(f"Worker count calculation:")
        logger.info(f"  Available CPU cores: {available_cores}")
        logger.info(f"  Base workers (75% of cores): {base_workers}")
        if partition_size and total_samples:
            logger.info(f"  Estimated partitions: {total_samples / partition_size:.1f}")
        logger.info(f"  Optimal workers: {optimal_workers}")

        return optimal_workers


class PartitionSizeOptimizer:
    """Automatically optimizes partition sizes based on data characteristics and available resources."""

    def calculate_target_partition_mb(self, available_memory_gb: float) -> int:
        """Calculate target partition size in MB based on available memory and config.

        Uses config.partition.target_size_mb if available, otherwise falls back to
        dynamic sizing based on available memory (32MB - 256MB).
        """
        # Use configured target if available
        if hasattr(self.cfg, "partition") and hasattr(self.cfg.partition, "target_size_mb"):
            configured_size = self.cfg.partition.target_size_mb
            logger.info(f"Using configured target partition size: {configured_size} MB")
            return configured_size

        # Fall back to dynamic calculation based on available memory
        if available_memory_gb < 16:
            return 32
        elif available_memory_gb < 64:
            return 64
        elif available_memory_gb < 256:
            return 128
        else:
            return 256

    # Default configurations for different modalities
    MODALITY_CONFIGS = {
        ModalityType.TEXT: ModalityConfig(
            modality=ModalityType.TEXT,
            default_partition_size=10000,  # Increased for 256MB target
            max_partition_size=50000,  # Increased for larger partitions
            max_partition_size_mb=256,  # Default 256MB per partition (configurable)
            memory_multiplier=1.0,
            complexity_multiplier=1.0,
            description="Text data - efficient processing, low memory usage, target 256MB partitions (configurable)",
        ),
        ModalityType.IMAGE: ModalityConfig(
            modality=ModalityType.IMAGE,
            default_partition_size=2000,  # Increased for 256MB target
            max_partition_size=10000,  # Increased for larger partitions
            max_partition_size_mb=256,  # Default 256MB per partition (configurable)
            memory_multiplier=5.0,
            complexity_multiplier=3.0,
            description="Image data - moderate memory usage, target 256MB partitions (configurable)",
        ),
        ModalityType.AUDIO: ModalityConfig(
            modality=ModalityType.AUDIO,
            default_partition_size=1000,  # Increased for 256MB target
            max_partition_size=4000,  # Increased for larger partitions
            max_partition_size_mb=256,  # Default 256MB per partition (configurable)
            memory_multiplier=8.0,
            complexity_multiplier=5.0,
            description="Audio data - high memory usage, target 256MB partitions (configurable)",
        ),
        ModalityType.VIDEO: ModalityConfig(
            modality=ModalityType.VIDEO,
            default_partition_size=400,  # Increased for 256MB target
            max_partition_size=2000,  # Increased for larger partitions
            max_partition_size_mb=256,  # Default 256MB per partition (configurable)
            memory_multiplier=20.0,
            complexity_multiplier=15.0,
            description="Video data - very high memory usage, target 256MB partitions (configurable)",
        ),
        ModalityType.MULTIMODAL: ModalityConfig(
            modality=ModalityType.MULTIMODAL,
            default_partition_size=1600,  # Increased for 256MB target
            max_partition_size=6000,  # Increased for larger partitions
            max_partition_size_mb=256,  # Default 256MB per partition (configurable)
            memory_multiplier=10.0,
            complexity_multiplier=8.0,
            description="Multimodal data - combination of multiple modalities, target 256MB partitions (configurable)",
        ),
    }

    def __init__(self, cfg):
        """Initialize the optimizer with configuration."""
        self.cfg = cfg
        self.text_key = getattr(cfg, "text_key", "text")
        self.image_key = getattr(cfg, "image_key", "images")
        self.audio_key = getattr(cfg, "audio_key", "audios")
        self.video_key = getattr(cfg, "video_key", "videos")
        self.resource_detector = ResourceDetector()

    def detect_modality(self, sample: Dict) -> ModalityType:
        """Detect the primary modality of a sample."""
        modalities = []

        # Check for text
        if self.text_key in sample and sample[self.text_key]:
            modalities.append(ModalityType.TEXT)

        # Check for images
        if sample.get(self.image_key):
            modalities.append(ModalityType.IMAGE)

        # Check for audio
        if sample.get(self.audio_key):
            modalities.append(ModalityType.AUDIO)

        # Check for video
        if sample.get(self.video_key):
            modalities.append(ModalityType.VIDEO)

        # Determine primary modality
        if len(modalities) > 1:
            return ModalityType.MULTIMODAL
        elif len(modalities) == 1:
            return modalities[0]
        else:
            # Default to text if no modality detected
            return ModalityType.TEXT

    def analyze_dataset_characteristics(self, dataset) -> DataCharacteristics:
        """Analyze dataset characteristics to inform partition sizing."""
        logger.info("Analyzing dataset characteristics for partition optimization...")

        # Get dataset size
        try:
            if hasattr(dataset, "count"):
                total_samples = dataset.count()
            elif hasattr(dataset, "__len__"):
                total_samples = len(dataset)
            else:
                total_samples = 1000
                logger.warning("Could not determine dataset size, using estimate of 1000 samples")
        except Exception as e:
            logger.warning(f"Could not determine dataset size: {e}, using estimate of 1000 samples")
            total_samples = 1000

        # Adaptive sampling: minimum 0.1% for large datasets
        if total_samples < 1000:
            sample_size = total_samples
        elif total_samples < 100000:
            sample_size = min(1000, total_samples // 100)  # 1%
        else:
            sample_size = min(10000, total_samples // 1000)  # 0.1%, cap at 10k

        try:
            # Sample dataset for analysis
            if hasattr(dataset, "get"):
                # RayDataset with get() method
                samples = dataset.get(sample_size)
                logger.info(f"Successfully sampled {len(samples)} samples using get()")
            elif hasattr(dataset, "take"):
                # Datasets with take() method
                samples = list(dataset.take(sample_size))
                logger.info(f"Successfully sampled {len(samples)} samples using take()")
            elif hasattr(dataset, "__getitem__"):
                # Handle list-like datasets
                samples = list(dataset[:sample_size])
                logger.info(f"Successfully sampled {len(samples)} samples from list-like dataset")
            else:
                # Fallback: try to iterate
                samples = []
                for i, sample in enumerate(dataset):
                    if i >= sample_size:
                        break
                    samples.append(sample)
                logger.info(f"Successfully sampled {len(samples)} samples by iteration")
        except Exception as e:
            logger.warning(f"Could not sample dataset: {e}, using default analysis")
            import traceback

            logger.debug(f"Sampling error traceback: {traceback.format_exc()}")
            return DataCharacteristics(
                primary_modality=ModalityType.TEXT,
                modality_distribution={ModalityType.TEXT: 1},
                avg_text_length=500,
                avg_images_per_sample=0,
                avg_audio_per_sample=0,
                avg_video_per_sample=0,
                total_samples=total_samples,
                sample_size_analyzed=0,
                memory_per_sample_mb=0.002,
                processing_complexity_score=1.0,
                data_skew_factor=0.5,
            )

        # Analyze samples
        modality_counts = {modality: 0 for modality in ModalityType}
        text_lengths = []
        image_counts = []
        audio_counts = []
        video_counts = []
        sample_sizes = []

        for sample in samples:
            # Detect modality
            modality = self.detect_modality(sample)
            modality_counts[modality] += 1

            # Analyze text
            text_length = 0
            if self.text_key in sample and sample[self.text_key]:
                if isinstance(sample[self.text_key], str):
                    text_length = len(sample[self.text_key])
                elif isinstance(sample[self.text_key], list):
                    text_length = sum(len(t) for t in sample[self.text_key])
            text_lengths.append(text_length)

            # Count media files
            image_count = len(sample.get(self.image_key, []))
            audio_count = len(sample.get(self.audio_key, []))
            video_count = len(sample.get(self.video_key, []))

            image_counts.append(image_count)
            audio_counts.append(audio_count)
            video_counts.append(video_count)

            # Estimate sample size in MB
            sample_size_mb = self.estimate_sample_size_mb(sample)
            sample_sizes.append(sample_size_mb)

        # Calculate statistics
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        avg_images_per_sample = sum(image_counts) / len(image_counts) if image_counts else 0
        avg_audio_per_sample = sum(audio_counts) / len(audio_counts) if audio_counts else 0
        avg_video_per_sample = sum(video_counts) / len(video_counts) if video_counts else 0

        # Calculate percentile-based memory estimates (p90 is more robust than mean)
        if sample_sizes and len(sample_sizes) > 1:
            sorted_sizes = sorted(sample_sizes)
            p90_idx = int(len(sorted_sizes) * 0.9)
            p90_memory = sorted_sizes[p90_idx]
            mean_size = sum(sample_sizes) / len(sample_sizes)
            variance = sum((x - mean_size) ** 2 for x in sample_sizes) / (len(sample_sizes) - 1)
            std_dev = variance**0.5
            data_skew_factor = min(1.0, std_dev / mean_size if mean_size > 0 else 0)
            # Use p90 for conservative sizing
            avg_memory_per_sample_mb = p90_memory
        else:
            avg_memory_per_sample_mb = sample_sizes[0] if sample_sizes else 0.002
            data_skew_factor = 0.5

        # Determine primary modality
        primary_modality = max(modality_counts.items(), key=lambda x: x[1])[0]

        characteristics = DataCharacteristics(
            primary_modality=primary_modality,
            modality_distribution=modality_counts,
            avg_text_length=avg_text_length,
            avg_images_per_sample=avg_images_per_sample,
            avg_audio_per_sample=avg_audio_per_sample,
            avg_video_per_sample=avg_video_per_sample,
            total_samples=total_samples,
            sample_size_analyzed=len(samples),
            memory_per_sample_mb=avg_memory_per_sample_mb,
            processing_complexity_score=1.0,  # Will be calculated later
            data_skew_factor=data_skew_factor,
        )

        logger.info(f"Dataset analysis complete:")
        logger.info(f"  Primary modality: {primary_modality.value}")
        logger.info(f"  Modality distribution: {modality_counts}")
        logger.info(f"  Avg text length: {avg_text_length:.0f} chars")
        logger.info(f"  Avg images per sample: {avg_images_per_sample:.1f}")
        logger.info(f"  Avg audio per sample: {avg_audio_per_sample:.1f}")
        logger.info(f"  Avg video per sample: {avg_video_per_sample:.1f}")
        logger.info(f"  Avg memory per sample: {avg_memory_per_sample_mb:.3f} MB")
        logger.info(f"  Data skew factor: {data_skew_factor:.2f}")

        return characteristics

    def estimate_sample_size_mb(self, sample: Dict) -> float:
        """Measure actual memory size of a sample in MB.

        Uses deep size calculation to include all nested objects (strings, lists, etc.)
        rather than just the shallow dict overhead.
        """
        return self._deep_getsizeof(sample) / (1024 * 1024)

    def _deep_getsizeof(self, obj, seen: set = None) -> int:
        """Recursively calculate the deep memory size of an object.

        This properly accounts for nested objects like strings in dicts,
        lists of values, etc. Uses a seen set to avoid counting shared
        objects multiple times.

        Args:
            obj: Object to measure
            seen: Set of object ids already counted (for cycle detection)

        Returns:
            Total memory size in bytes
        """
        import sys

        if seen is None:
            seen = set()

        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        size = sys.getsizeof(obj)

        if isinstance(obj, dict):
            size += sum(self._deep_getsizeof(k, seen) + self._deep_getsizeof(v, seen) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            size += sum(self._deep_getsizeof(item, seen) for item in obj)
        elif isinstance(obj, str):
            # String size is already included in getsizeof
            pass
        elif isinstance(obj, bytes):
            # Bytes size is already included in getsizeof
            pass
        elif hasattr(obj, "__dict__"):
            size += self._deep_getsizeof(obj.__dict__, seen)
        elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
            try:
                size += sum(self._deep_getsizeof(item, seen) for item in obj)
            except TypeError:
                pass  # Not iterable after all

        return size

    def analyze_processing_complexity(self, process_pipeline: List) -> float:
        """Analyze the complexity of the processing pipeline using linear scoring."""
        COMPLEXITY_WEIGHTS = {
            "high": 0.3,  # embedding, model, neural
            "medium": 0.2,  # filter, deduplicator
            "low": 0.1,  # text cleaning
        }

        # Count operations by complexity level
        high_ops = medium_ops = low_ops = 0
        for op in process_pipeline:
            if isinstance(op, dict):
                op_name = list(op.keys())[0].lower()
                if any(kw in op_name for kw in ["embedding", "similarity", "model", "neural", "vision", "audio"]):
                    high_ops += 1
                elif any(kw in op_name for kw in ["filter", "deduplicator", "mapper"]):
                    medium_ops += 1
                else:
                    low_ops += 1

        # Linear complexity scoring
        complexity_score = 1.0 + (
            high_ops * COMPLEXITY_WEIGHTS["high"]
            + medium_ops * COMPLEXITY_WEIGHTS["medium"]
            + low_ops * COMPLEXITY_WEIGHTS["low"]
        )

        logger.info(f"Processing complexity: {high_ops} high, {medium_ops} med, {low_ops} low = {complexity_score:.2f}")
        return complexity_score

    def get_optimal_partition_size(self, dataset, process_pipeline: List) -> Tuple[int, int]:
        """Get optimal partition size and max size based on data characteristics and available resources."""

        # Analyze dataset
        characteristics = self.analyze_dataset_characteristics(dataset)

        # Analyze processing complexity
        complexity_multiplier = self.analyze_processing_complexity(process_pipeline)
        characteristics.processing_complexity_score = complexity_multiplier

        # Detect available resources
        local_resources = self.resource_detector.detect_local_resources()
        cluster_resources = self.resource_detector.detect_ray_cluster()

        logger.info(f"Resource analysis:")
        logger.info(f"  Local CPU cores: {local_resources.cpu_cores}")
        logger.info(f"  Local available memory: {local_resources.available_memory_gb:.1f} GB")
        if cluster_resources:
            logger.info(f"  Cluster CPU cores: {cluster_resources.total_cpu_cores}")
            logger.info(f"  Cluster available memory: {cluster_resources.available_memory_gb:.1f} GB")

        # Calculate optimal partition size
        optimal_size = self.calculate_resource_aware_partition_size(
            characteristics, local_resources, cluster_resources, complexity_multiplier
        )

        # Calculate optimal max size in MB
        optimal_max_size_mb = self.calculate_optimal_max_size_mb(
            characteristics, local_resources, cluster_resources, complexity_multiplier
        )

        logger.info(f"Optimal partition configuration:")
        logger.info(f"  Size: {optimal_size} samples")
        logger.info(f"  Max size: {optimal_max_size_mb} MB")
        logger.info(f"  Based on: {characteristics.primary_modality.value} modality")
        logger.info(f"  Complexity multiplier: {complexity_multiplier:.2f}")
        logger.info(f"  Data skew factor: {characteristics.data_skew_factor:.2f}")

        return optimal_size, optimal_max_size_mb

    def calculate_resource_aware_partition_size(
        self,
        characteristics: DataCharacteristics,
        local_resources: LocalResources,
        cluster_resources: Optional[ClusterResources],
        complexity_multiplier: float,
    ) -> int:
        """
        Calculate partition size based on data characteristics and available resources.

        Primary goal: Target partition size based on config (default 256MB).
        Secondary goals: Ensure sufficient parallelism and respect resource constraints.
        """

        # Get base configuration for the modality
        base_config = self.MODALITY_CONFIGS[characteristics.primary_modality]

        # Step 1: Calculate dynamic target based on available memory
        available_memory_gb = self._get_available_memory(local_resources, cluster_resources)
        target_memory_mb = self.calculate_target_partition_mb(available_memory_gb)

        if characteristics.primary_modality == ModalityType.TEXT:
            target_size = self.calculate_text_partition_size_simple(
                characteristics.avg_text_length, complexity_multiplier, target_memory_mb
            )
        else:
            # For media, use memory-per-sample to calculate target
            if characteristics.memory_per_sample_mb > 0:
                target_size = int(target_memory_mb / (characteristics.memory_per_sample_mb * complexity_multiplier))
            else:
                target_size = base_config.default_partition_size
            target_size = max(10, min(target_size, base_config.max_partition_size))

        # Step 2: Check if this fits in available memory
        max_partition_memory_mb = (available_memory_gb * 1024 * 0.8) / 4  # Allow 4 concurrent partitions

        if target_size * characteristics.memory_per_sample_mb * 2 > max_partition_memory_mb:
            # Doesn't fit - scale down
            safe_size = int(max_partition_memory_mb / (characteristics.memory_per_sample_mb * 2))
            logger.warning(f"Memory constraint: reducing partition size from {target_size} to {safe_size} samples")
            target_size = max(10, safe_size)

        # Step 3: Ensure sufficient parallelism for large datasets
        min_partitions_needed = self._calculate_min_partitions(
            characteristics.total_samples, local_resources, cluster_resources
        )

        if characteristics.total_samples / target_size < min_partitions_needed:
            # Too few partitions - reduce size for better parallelism
            parallelism_size = int(characteristics.total_samples / min_partitions_needed)
            logger.info(
                f"Parallelism optimization: reducing partition size from {target_size} to {parallelism_size} "
                f"to create {min_partitions_needed} partitions"
            )
            target_size = max(10, parallelism_size)

        # Step 4: Adjust for data skew
        if characteristics.data_skew_factor > 0.7:
            # High variance - use smaller partitions for better load balancing
            skew_adjusted_size = int(target_size * 0.8)
            logger.info(f"Data skew adjustment: reducing partition size from {target_size} to {skew_adjusted_size}")
            target_size = skew_adjusted_size

        # Step 5: Apply final bounds
        final_size = max(10, min(target_size, base_config.max_partition_size))

        logger.info(f"Final partition size: {final_size} samples")
        logger.info(f"  Estimated memory per partition: {final_size * characteristics.memory_per_sample_mb:.1f} MB")
        logger.info(f"  Estimated total partitions: {characteristics.total_samples / final_size:.0f}")

        return final_size

    def _get_available_memory(
        self, local_resources: LocalResources, cluster_resources: Optional[ClusterResources]
    ) -> float:
        """Get available memory in GB."""
        if cluster_resources:
            return min(local_resources.available_memory_gb, cluster_resources.available_memory_gb)
        return local_resources.available_memory_gb

    def _calculate_min_partitions(
        self,
        total_samples: int,
        local_resources: LocalResources,
        cluster_resources: Optional[ClusterResources],
    ) -> int:
        """Calculate minimum number of partitions needed for good parallelism."""
        # Only enforce minimum partitions for large datasets (>10k samples)
        if total_samples <= 10000:
            return 1  # Small datasets - prioritize 64MB target over parallelism

        # For large datasets, aim for at least 1.5x CPU cores in partitions
        available_cores = local_resources.cpu_cores
        if cluster_resources:
            available_cores = min(available_cores, cluster_resources.available_cpu_cores)

        return max(1, int(available_cores * 1.5))

    def calculate_text_partition_size_simple(
        self, avg_text_length: float, complexity_score: float, target_memory_mb: float
    ) -> int:
        """Calculate text partition size targeting specified memory size."""
        # Estimate bytes per sample (conservative: 2 bytes per char + overhead)
        bytes_per_sample = avg_text_length * 2.0
        mb_per_sample = bytes_per_sample / (1024 * 1024)

        # Calculate samples for target, adjusted for complexity
        if mb_per_sample > 0:
            target_samples = int(target_memory_mb / (mb_per_sample * complexity_score))
        else:
            target_samples = self.MODALITY_CONFIGS[ModalityType.TEXT].default_partition_size

        # Apply bounds from MODALITY_CONFIGS
        text_config = self.MODALITY_CONFIGS[ModalityType.TEXT]
        target_samples = max(100, min(target_samples, text_config.max_partition_size))

        logger.info(f"Text partition calculation:")
        logger.info(f"  Target: {target_memory_mb}MB, Avg text: {avg_text_length:.0f} chars")
        logger.info(f"  Estimated: {mb_per_sample:.3f} MB/sample")
        logger.info(f"  Result: {target_samples} samples (~{target_samples * mb_per_sample:.1f} MB)")

        return target_samples

    def calculate_optimal_max_size_mb(
        self,
        characteristics: DataCharacteristics,
        local_resources: LocalResources,
        cluster_resources: Optional[ClusterResources],
        complexity_multiplier: float,
    ) -> int:
        """Calculate optimal max partition size in MB based on available memory."""
        # Calculate dynamic target based on available memory
        available_memory_gb = local_resources.available_memory_gb
        if cluster_resources:
            available_memory_gb = min(available_memory_gb, cluster_resources.available_memory_gb)

        target_max_size_mb = self.calculate_target_partition_mb(available_memory_gb)

        # Adjust for processing complexity
        complexity_adjusted_size = int(target_max_size_mb / complexity_multiplier)

        # Don't exceed 25% of available memory per partition
        max_size_by_memory = int(available_memory_gb * 1024 * 0.25)

        # Apply bounds
        optimal_max_size_mb = min(complexity_adjusted_size, max_size_by_memory)
        optimal_max_size_mb = max(32, optimal_max_size_mb)
        optimal_max_size_mb = min(512, optimal_max_size_mb)  # Increased max from 128MB

        logger.info(f"Max partition size calculation:")
        logger.info(f"  Target size: {target_max_size_mb} MB (dynamic based on {available_memory_gb:.1f} GB)")
        logger.info(f"  Complexity adjusted: {complexity_adjusted_size} MB")
        logger.info(f"  Max by memory (25%): {max_size_by_memory} MB")
        logger.info(f"  Optimal max size: {optimal_max_size_mb} MB")

        return optimal_max_size_mb

    def get_partition_recommendations(self, dataset, process_pipeline: List) -> Dict:
        """Get comprehensive partition recommendations."""
        optimal_size, optimal_max_size_mb = self.get_optimal_partition_size(dataset, process_pipeline)
        characteristics = self.analyze_dataset_characteristics(dataset)

        # Detect resources
        local_resources = self.resource_detector.detect_local_resources()
        cluster_resources = self.resource_detector.detect_ray_cluster()

        # Calculate optimal worker count
        optimal_workers = self.resource_detector.calculate_optimal_worker_count(
            local_resources, cluster_resources, optimal_size, characteristics.total_samples
        )

        recommendations = {
            "recommended_partition_size": optimal_size,
            "recommended_max_size_mb": optimal_max_size_mb,
            "recommended_worker_count": optimal_workers,
            "primary_modality": characteristics.primary_modality.value,
            "data_characteristics": {
                "avg_text_length": characteristics.avg_text_length,
                "avg_images_per_sample": characteristics.avg_images_per_sample,
                "avg_audio_per_sample": characteristics.avg_audio_per_sample,
                "avg_video_per_sample": characteristics.avg_video_per_sample,
                "memory_per_sample_mb": characteristics.memory_per_sample_mb,
                "data_skew_factor": characteristics.data_skew_factor,
                "total_samples": characteristics.total_samples,
            },
            "resource_analysis": {
                "local_cpu_cores": local_resources.cpu_cores,
                "local_available_memory_gb": local_resources.available_memory_gb,
                "cluster_available_cpu_cores": cluster_resources.available_cpu_cores if cluster_resources else None,
                "cluster_available_memory_gb": cluster_resources.available_memory_gb if cluster_resources else None,
            },
            "reasoning": {
                "modality": f"Based on {characteristics.primary_modality.value} modality",
                "complexity": f"Processing complexity factor: {characteristics.processing_complexity_score:.2f}",
                "dataset_size": f"Dataset size: {characteristics.total_samples} samples",
                "text_length": f"Average text length: {characteristics.avg_text_length:.0f} characters",
                "data_skew": f"Data skew factor: {characteristics.data_skew_factor:.2f}",
                "memory_constraints": f"Memory per sample: {characteristics.memory_per_sample_mb:.3f} MB",
                "worker_count": f"Optimal workers: {optimal_workers} (based on {local_resources.cpu_cores} available cores)",
            },
            "modality_configs": {
                modality.value: {
                    "default_size": config.default_partition_size,
                    "max_size": config.max_partition_size,
                    "max_size_mb": config.max_partition_size_mb,
                    "description": config.description,
                }
                for modality, config in self.MODALITY_CONFIGS.items()
            },
        }

        return recommendations


def auto_configure_resources(cfg, dataset, process_pipeline: List) -> Dict:
    """
    Analyze dataset and return resource configuration recommendations.

    Does NOT mutate cfg - caller should apply recommendations as needed.

    Args:
        cfg: Configuration object (read-only)
        dataset: Dataset to analyze
        process_pipeline: List of processing operations

    Returns:
        Dict with recommended resource configuration
    """
    logger.info("Starting resource optimization...")
    optimizer = PartitionSizeOptimizer(cfg)
    recommendations = optimizer.get_partition_recommendations(dataset, process_pipeline)

    logger.info("Resource optimization completed:")
    logger.info(f"  Recommended partition.size: {recommendations['recommended_partition_size']}")
    logger.info(f"  Recommended partition.max_size_mb: {recommendations['recommended_max_size_mb']}")
    logger.info(f"  Recommended worker count: {recommendations['recommended_worker_count']}")

    return recommendations
