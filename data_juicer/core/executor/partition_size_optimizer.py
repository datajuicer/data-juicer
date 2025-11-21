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

        # Minimum of 1 worker, maximum reasonable limit
        optimal_workers = max(1, min(optimal_workers, 32))  # Cap at 32 workers

        logger.info(f"Worker count calculation:")
        logger.info(f"  Available CPU cores: {available_cores}")
        logger.info(f"  Base workers (75% of cores): {base_workers}")
        if partition_size and total_samples:
            logger.info(f"  Estimated partitions: {total_samples / partition_size:.1f}")
        logger.info(f"  Optimal workers: {optimal_workers}")

        return optimal_workers


class PartitionSizeOptimizer:
    """Automatically optimizes partition sizes based on data characteristics and available resources."""

    # Default configurations for different modalities
    MODALITY_CONFIGS = {
        ModalityType.TEXT: ModalityConfig(
            modality=ModalityType.TEXT,
            default_partition_size=5000,  # Increased from 200
            max_partition_size=20000,  # Increased from 1000
            max_partition_size_mb=64,  # Target 64MB per partition
            memory_multiplier=1.0,
            complexity_multiplier=1.0,
            description="Text data - efficient processing, low memory usage, target 64MB partitions",
        ),
        ModalityType.IMAGE: ModalityConfig(
            modality=ModalityType.IMAGE,
            default_partition_size=1000,  # Increased from 50
            max_partition_size=5000,  # Increased from 200
            max_partition_size_mb=64,  # Target 64MB per partition
            memory_multiplier=5.0,
            complexity_multiplier=3.0,
            description="Image data - moderate memory usage, target 64MB partitions",
        ),
        ModalityType.AUDIO: ModalityConfig(
            modality=ModalityType.AUDIO,
            default_partition_size=500,  # Increased from 30
            max_partition_size=2000,  # Increased from 100
            max_partition_size_mb=64,  # Target 64MB per partition
            memory_multiplier=8.0,
            complexity_multiplier=5.0,
            description="Audio data - high memory usage, target 64MB partitions",
        ),
        ModalityType.VIDEO: ModalityConfig(
            modality=ModalityType.VIDEO,
            default_partition_size=200,  # Increased from 10
            max_partition_size=1000,  # Increased from 50
            max_partition_size_mb=64,  # Target 64MB per partition
            memory_multiplier=20.0,
            complexity_multiplier=15.0,
            description="Video data - very high memory usage, target 64MB partitions",
        ),
        ModalityType.MULTIMODAL: ModalityConfig(
            modality=ModalityType.MULTIMODAL,
            default_partition_size=800,  # Increased from 20
            max_partition_size=3000,  # Increased from 100
            max_partition_size_mb=64,  # Target 64MB per partition
            memory_multiplier=10.0,
            complexity_multiplier=8.0,
            description="Multimodal data - combination of multiple modalities, target 64MB partitions",
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

        # Adaptive sampling based on dataset size
        if total_samples < 100:
            sample_size = total_samples
        elif total_samples < 1000:
            sample_size = min(200, total_samples)
        else:
            sample_size = min(500, total_samples // 10)

        try:
            # Sample dataset for analysis
            if hasattr(dataset, "take"):
                samples = dataset.take(sample_size)
                logger.info(f"Successfully sampled {len(samples)} samples from Ray Dataset")
            elif hasattr(dataset, "__getitem__"):
                # Handle list-like datasets
                samples = dataset[:sample_size]
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
        avg_memory_per_sample_mb = sum(sample_sizes) / len(sample_sizes) if sample_sizes else 0.002

        # Calculate data skew factor (coefficient of variation)
        if sample_sizes and len(sample_sizes) > 1:
            mean_size = sum(sample_sizes) / len(sample_sizes)
            variance = sum((x - mean_size) ** 2 for x in sample_sizes) / (len(sample_sizes) - 1)
            std_dev = variance**0.5
            data_skew_factor = min(1.0, std_dev / mean_size if mean_size > 0 else 0)
        else:
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
        """Estimate the memory size of a sample in MB."""
        size_mb = 0.0

        # Text size
        if self.text_key in sample and sample[self.text_key]:
            if isinstance(sample[self.text_key], str):
                size_mb += len(sample[self.text_key]) / (1024 * 1024)  # Rough estimate
            elif isinstance(sample[self.text_key], list):
                size_mb += sum(len(t) for t in sample[self.text_key]) / (1024 * 1024)

        # Media size estimates
        images = sample.get(self.image_key, [])
        if images:
            size_mb += len(images) * 0.5  # Assume 0.5MB per image

        audios = sample.get(self.audio_key, [])
        if audios:
            size_mb += len(audios) * 2.0  # Assume 2MB per audio file

        videos = sample.get(self.video_key, [])
        if videos:
            size_mb += len(videos) * 10.0  # Assume 10MB per video file

        return max(0.001, size_mb)  # Minimum 1KB

    def analyze_processing_complexity(self, process_pipeline: List) -> float:
        """Analyze the complexity of the processing pipeline."""
        complexity_score = 1.0

        # Count operations by type
        op_counts = {}
        for op in process_pipeline:
            if isinstance(op, dict):
                op_name = list(op.keys())[0]
                op_counts[op_name] = op_counts.get(op_name, 0) + 1

        # Adjust complexity based on operation types
        for op_name, count in op_counts.items():
            # High complexity operations
            if any(
                keyword in op_name.lower()
                for keyword in ["embedding", "similarity", "model", "neural", "vision", "audio"]
            ):
                complexity_score *= 1.2**count
            # Medium complexity operations
            elif any(keyword in op_name.lower() for keyword in ["filter", "deduplicator", "mapper"]):
                complexity_score *= 1.1**count
            # Low complexity operations (text cleaning, etc.)
            else:
                complexity_score *= 1.05**count

        logger.info(f"Processing complexity score: {complexity_score:.2f}")
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
        """Calculate partition size based on data characteristics and available resources."""

        # Set total samples for CPU constraints calculation
        self.estimated_total_samples = characteristics.total_samples

        # Get base configuration for the modality
        base_config = self.MODALITY_CONFIGS[characteristics.primary_modality]

        # Start with modality-based size
        if characteristics.primary_modality == ModalityType.TEXT:
            base_size = self.calculate_text_partition_size(
                characteristics.avg_text_length, characteristics.total_samples, complexity_multiplier
            )
        else:
            base_size = int(base_config.default_partition_size / complexity_multiplier)
            base_size = max(10, min(base_size, base_config.max_partition_size))

        # Adjust for memory constraints
        memory_constrained_size = self.adjust_for_memory_constraints(
            base_size, characteristics, local_resources, cluster_resources
        )

        # Adjust for CPU constraints
        cpu_constrained_size = self.adjust_for_cpu_constraints(
            memory_constrained_size, local_resources, cluster_resources
        )

        # Adjust for data skew
        if characteristics.data_skew_factor > 0.7:
            # High variance - use smaller partitions for better load balancing
            final_size = int(cpu_constrained_size * 0.7)
        else:
            final_size = cpu_constrained_size

        # Apply bounds
        final_size = max(10, min(final_size, base_config.max_partition_size))

        return final_size

    def adjust_for_memory_constraints(
        self,
        base_size: int,
        characteristics: DataCharacteristics,
        local_resources: LocalResources,
        cluster_resources: Optional[ClusterResources],
    ) -> int:
        """Adjust partition size based on available memory."""

        # Calculate memory needed per partition
        memory_per_partition_mb = base_size * characteristics.memory_per_sample_mb * 2  # 2x buffer

        # Check local memory constraints
        available_memory_gb = local_resources.available_memory_gb
        if cluster_resources:
            # Use cluster memory if available
            available_memory_gb = min(available_memory_gb, cluster_resources.available_memory_gb)

        # Reserve 20% of memory for system and other processes
        usable_memory_gb = available_memory_gb * 0.8

        # Calculate how many partitions we can fit
        max_partitions_by_memory = int((usable_memory_gb * 1024) / memory_per_partition_mb)

        if max_partitions_by_memory < 1:
            # Not enough memory - reduce partition size
            memory_constrained_size = int(base_size * 0.5)
            logger.warning(f"Memory constrained: reducing partition size to {memory_constrained_size}")
        else:
            memory_constrained_size = base_size

        return memory_constrained_size

    def adjust_for_cpu_constraints(
        self, base_size: int, local_resources: LocalResources, cluster_resources: Optional[ClusterResources]
    ) -> int:
        """
        Adjust partition size based on available CPU cores.
        Prioritize 64MB target over excessive parallelism for small datasets.
        """

        # Get available CPU cores
        available_cores = local_resources.cpu_cores
        if cluster_resources:
            available_cores = min(available_cores, cluster_resources.available_cpu_cores)

        # Estimate total partitions needed
        if hasattr(self, "estimated_total_samples"):
            total_samples = self.estimated_total_samples
        else:
            total_samples = 10000  # Default estimate

        estimated_partitions = total_samples / base_size

        # Only adjust if we have too few partitions AND the dataset is large enough
        # For small datasets, prioritize 64MB target over parallelism
        min_partitions_for_large_datasets = available_cores * 1.5  # Reduced from 2x

        if estimated_partitions < min_partitions_for_large_datasets and total_samples > 10000:
            # Only reduce size for large datasets with too few partitions
            cpu_constrained_size = int(base_size * (estimated_partitions / min_partitions_for_large_datasets))

            # Don't reduce below reasonable minimum for 64MB target
            min_reasonable_size = 1000
            if cpu_constrained_size < min_reasonable_size:
                cpu_constrained_size = min_reasonable_size

            logger.info(
                f"CPU optimization: reducing partition size to {cpu_constrained_size} for better parallelism (large dataset)"
            )
        else:
            # Keep the base size (prioritize 64MB target)
            cpu_constrained_size = base_size
            if total_samples <= 10000:
                logger.info(
                    f"CPU optimization: keeping partition size {cpu_constrained_size} (prioritizing 64MB target for small dataset)"
                )
            else:
                logger.info(f"CPU optimization: keeping partition size {cpu_constrained_size} (sufficient parallelism)")

        return cpu_constrained_size

    def calculate_optimal_max_size_mb(
        self,
        characteristics: DataCharacteristics,
        local_resources: LocalResources,
        cluster_resources: Optional[ClusterResources],
        complexity_multiplier: float,
    ) -> int:
        """
        Calculate optimal max partition size in MB.
        Target: 64MB per partition for optimal memory usage and processing efficiency.
        """

        base_config = self.MODALITY_CONFIGS[characteristics.primary_modality]

        # Target 64MB per partition (from modality config)
        target_max_size_mb = base_config.max_partition_size_mb  # Should be 64MB

        # Adjust for processing complexity
        # More complex operations may need smaller partitions
        complexity_adjusted_size = int(target_max_size_mb / complexity_multiplier)

        # Adjust for available memory
        available_memory_gb = local_resources.available_memory_gb
        if cluster_resources:
            available_memory_gb = min(available_memory_gb, cluster_resources.available_memory_gb)

        # Don't exceed 25% of available memory per partition
        # This ensures we can have multiple partitions in memory simultaneously
        max_size_by_memory = int(available_memory_gb * 1024 * 0.25)

        # Apply bounds
        optimal_max_size_mb = min(complexity_adjusted_size, max_size_by_memory)
        optimal_max_size_mb = max(32, optimal_max_size_mb)  # Minimum 32MB
        optimal_max_size_mb = min(128, optimal_max_size_mb)  # Maximum 128MB

        logger.info(f"Max partition size calculation (targeting 64MB):")
        logger.info(f"  Target size: {target_max_size_mb} MB")
        logger.info(f"  Complexity adjusted: {complexity_adjusted_size} MB")
        logger.info(f"  Available memory: {available_memory_gb:.1f} GB")
        logger.info(f"  Max by memory (25%): {max_size_by_memory} MB")
        logger.info(f"  Optimal max size: {optimal_max_size_mb} MB")

        return optimal_max_size_mb

    def calculate_text_partition_size(self, avg_text_length: float, total_samples: int, complexity_score: float) -> int:
        """
        Calculate optimal text partition size based on actual data characteristics.
        Target: ~64MB per partition for optimal memory usage and processing efficiency.

        Factors considered:
        1. Text length (longer text = smaller partitions)
        2. Dataset size (larger datasets can use larger partitions)
        3. Processing complexity (complex operations = smaller partitions)
        4. Memory constraints (target ~64MB per partition)
        """
        # Target 64MB per partition
        target_memory_mb = 64.0

        # Estimate memory per sample based on text length
        # Rough estimate: 1 character ≈ 1-2 bytes, plus overhead
        estimated_bytes_per_char = 2.0  # Conservative estimate
        estimated_sample_size_mb = (avg_text_length * estimated_bytes_per_char) / (1024 * 1024)

        # Calculate samples needed for 64MB
        if estimated_sample_size_mb > 0:
            target_samples = int(target_memory_mb / estimated_sample_size_mb)
        else:
            target_samples = 5000  # Fallback for very small text

        # Base partition size targeting 64MB
        base_size = target_samples

        # Adjust for text length (fine-tuning)
        if avg_text_length > 10000:
            # Very long text (articles, documents) - reduce slightly
            length_factor = 0.8
        elif avg_text_length > 5000:
            # Long text (paragraphs) - slight reduction
            length_factor = 0.9
        elif avg_text_length > 1000:
            # Medium text (sentences) - no adjustment
            length_factor = 1.0
        elif avg_text_length < 100:
            # Very short text (tweets, labels) - can use more samples
            length_factor = 1.2
        else:
            # Normal text length
            length_factor = 1.0

        # Adjust for dataset size
        if total_samples > 1000000:
            # Very large dataset - can use larger partitions
            size_factor = 1.3
        elif total_samples > 100000:
            # Large dataset - moderate increase
            size_factor = 1.1
        elif total_samples < 1000:
            # Small dataset - use smaller partitions for better granularity
            size_factor = 0.8
        else:
            # Medium dataset
            size_factor = 1.0

        # Adjust for processing complexity
        complexity_factor = 1.0 / complexity_score

        # Calculate optimal size
        optimal_size = int(base_size * length_factor * size_factor * complexity_factor)

        # Apply bounds (much more reasonable for 64MB target)
        min_size = 1000  # Minimum 1000 samples
        max_size = 20000  # Maximum 20000 samples

        optimal_size = max(min_size, min(optimal_size, max_size))

        logger.info(f"Text partition size calculation (targeting 64MB):")
        logger.info(f"  Target memory: {target_memory_mb} MB")
        logger.info(f"  Estimated sample size: {estimated_sample_size_mb:.3f} MB")
        logger.info(f"  Base size (64MB target): {base_size} samples")
        logger.info(f"  Avg text length: {avg_text_length:.0f} chars (factor: {length_factor:.2f})")
        logger.info(f"  Dataset size: {total_samples} samples (factor: {size_factor:.2f})")
        logger.info(f"  Complexity score: {complexity_score:.2f} (factor: {complexity_factor:.2f})")
        logger.info(f"  Optimal size: {optimal_size} samples")
        logger.info(f"  Estimated partition size: {optimal_size * estimated_sample_size_mb:.1f} MB")

        return optimal_size

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


def auto_configure_partition_size(cfg, dataset, process_pipeline: List) -> Dict:
    """
    Automatically configure partition size and worker count based on dataset characteristics and available resources.

    Args:
        cfg: Configuration object
        dataset: Dataset to analyze
        process_pipeline: List of processing operations

    Returns:
        Dict with recommended partition and worker configuration
    """
    optimizer = PartitionSizeOptimizer(cfg)
    recommendations = optimizer.get_partition_recommendations(dataset, process_pipeline)

    # Update configuration with recommendations
    if not hasattr(cfg, "partition"):
        cfg.partition = {}

    cfg.partition["size"] = recommendations["recommended_partition_size"]
    cfg.partition["max_size_mb"] = recommendations["recommended_max_size_mb"]

    # Update worker count
    cfg.np = recommendations["recommended_worker_count"]

    logger.info("Auto-configured settings:")
    logger.info(f"  partition.size: {cfg.partition['size']}")
    logger.info(f"  partition.max_size_mb: {cfg.partition['max_size_mb']}")
    logger.info(f"  np (worker count): {cfg.np}")

    return recommendations


def auto_configure_resources(cfg, dataset, process_pipeline: List) -> Dict:
    """
    Automatically configure all resource-dependent settings based on dataset characteristics and available resources.

    Args:
        cfg: Configuration object
        dataset: Dataset to analyze
        process_pipeline: List of processing operations

    Returns:
        Dict with recommended resource configuration
    """
    try:
        logger.info("Starting resource optimization...")

        optimizer = PartitionSizeOptimizer(cfg)
        recommendations = optimizer.get_partition_recommendations(dataset, process_pipeline)

        logger.info(f"Got recommendations: {recommendations}")

        # Update configuration with recommendations
        # Handle case where cfg.partition might be None
        if not hasattr(cfg, "partition") or cfg.partition is None:
            logger.info("Creating new partition configuration")
            cfg.partition = {}

        # Ensure cfg.partition is a dictionary
        if not isinstance(cfg.partition, dict):
            logger.info("Converting partition configuration to dictionary")
            cfg.partition = {}

        logger.info(f"Current cfg.partition: {cfg.partition}")
        logger.info(f"Setting partition.size to: {recommendations['recommended_partition_size']}")
        logger.info(f"Setting partition.max_size_mb to: {recommendations['recommended_max_size_mb']}")
        logger.info(f"Setting np to: {recommendations['recommended_worker_count']}")

        # Update partition configuration with new structure
        cfg.partition["size"] = recommendations["recommended_partition_size"]
        cfg.partition["max_size_mb"] = recommendations["recommended_max_size_mb"]

        # Update worker count
        cfg.np = recommendations["recommended_worker_count"]

        logger.info("Resource optimization completed:")
        logger.info(f"  partition.size: {cfg.partition['size']}")
        logger.info(f"  partition.max_size_mb: {cfg.partition['max_size_mb']}")
        logger.info(f"  np (worker count): {cfg.np}")

        return recommendations

    except Exception as e:
        logger.error(f"Resource optimization failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
