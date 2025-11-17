#!/usr/bin/env python3
"""
Comprehensive workload suite for benchmarking different scenarios.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class WorkloadDefinition:
    """Definition of a benchmark workload."""

    name: str
    description: str
    dataset_path: str
    config_path: str
    expected_samples: int
    modality: str  # text, image, video, audio, multimodal
    complexity: str  # simple, medium, complex
    estimated_duration_minutes: int
    resource_requirements: Dict[str, Any]

    def __post_init__(self):
        """Validate workload definition."""
        if not Path(self.dataset_path).exists():
            logger.warning(f"Dataset path does not exist: {self.dataset_path}")
        if not Path(self.config_path).exists():
            logger.warning(f"Config path does not exist: {self.config_path}")


class WorkloadSuite:
    """Comprehensive suite of benchmark workloads."""

    def __init__(self):
        self.workloads = {}
        self._initialize_workloads()

    def _initialize_workloads(self):
        """Initialize all available workloads using production datasets and configs."""

        # Text workloads - Production Wikipedia dataset
        self.workloads["text_simple"] = WorkloadDefinition(
            name="text_simple",
            description="Simple text processing with basic filters",
            dataset_path="perf_bench_data/text/wiki-10k.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=10000,
            modality="text",
            complexity="simple",
            estimated_duration_minutes=5,
            resource_requirements={"memory_gb": 2, "cpu_cores": 2},
        )

        self.workloads["text_production"] = WorkloadDefinition(
            name="text_production",
            description="Production text processing with ML operations",
            dataset_path="perf_bench_data/text/wiki-10k.jsonl",
            config_path="tests/benchmark_performance/configs/text.yaml",
            expected_samples=10000,
            modality="text",
            complexity="complex",
            estimated_duration_minutes=40,
            resource_requirements={"memory_gb": 8, "cpu_cores": 12, "gpu": True},
        )

        # Image workloads - Production image dataset
        self.workloads["image_simple"] = WorkloadDefinition(
            name="image_simple",
            description="Simple image processing",
            dataset_path="perf_bench_data/image/10k.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=10000,
            modality="image",
            complexity="simple",
            estimated_duration_minutes=10,
            resource_requirements={"memory_gb": 4, "cpu_cores": 4, "gpu": True},
        )

        self.workloads["image_production"] = WorkloadDefinition(
            name="image_production",
            description="Production image processing with ML models",
            dataset_path="perf_bench_data/image/10k.jsonl",
            config_path="tests/benchmark_performance/configs/image.yaml",
            expected_samples=10000,
            modality="image",
            complexity="complex",
            estimated_duration_minutes=30,
            resource_requirements={"memory_gb": 16, "cpu_cores": 12, "gpu": True},
        )

        # Video workloads - Production MSR-VTT dataset
        self.workloads["video_simple"] = WorkloadDefinition(
            name="video_simple",
            description="Simple video processing",
            dataset_path="perf_bench_data/video/msr_vtt_train.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=1000,
            modality="video",
            complexity="simple",
            estimated_duration_minutes=20,
            resource_requirements={"memory_gb": 8, "cpu_cores": 8, "gpu": True},
        )

        self.workloads["video_production"] = WorkloadDefinition(
            name="video_production",
            description="Production video processing with frame analysis",
            dataset_path="perf_bench_data/video/msr_vtt_train.jsonl",
            config_path="tests/benchmark_performance/configs/video.yaml",
            expected_samples=1000,
            modality="video",
            complexity="complex",
            estimated_duration_minutes=60,
            resource_requirements={"memory_gb": 32, "cpu_cores": 16, "gpu": True},
        )

        # Audio workloads - Production audio dataset
        self.workloads["audio_simple"] = WorkloadDefinition(
            name="audio_simple",
            description="Simple audio processing",
            dataset_path="perf_bench_data/audio/audio-10k.jsonl",
            config_path="configs/demo/process.yaml",
            expected_samples=10000,
            modality="audio",
            complexity="simple",
            estimated_duration_minutes=15,
            resource_requirements={"memory_gb": 4, "cpu_cores": 4},
        )

        self.workloads["audio_production"] = WorkloadDefinition(
            name="audio_production",
            description="Production audio processing with quality filters",
            dataset_path="perf_bench_data/audio/audio-10k.jsonl",
            config_path="tests/benchmark_performance/configs/audio.yaml",
            expected_samples=10000,
            modality="audio",
            complexity="complex",
            estimated_duration_minutes=25,
            resource_requirements={"memory_gb": 8, "cpu_cores": 8},
        )

        # C4 dataset stress tests - Local vs Ray execution
        self.workloads["stress_test_text_c4_local"] = WorkloadDefinition(
            name="stress_test_text_c4_local",
            description="C4 dataset stress test with local execution (16 processes)",
            dataset_path="perf_bench_data/text/c4-train.00000-of-01024.jsonl",
            config_path="tests/benchmark_performance/configs/text-c4-local.yaml",
            expected_samples=100000,  # C4 dataset is much larger
            modality="text",
            complexity="complex",
            estimated_duration_minutes=120,  # Longer due to C4 dataset size
            resource_requirements={"memory_gb": 64, "cpu_cores": 16, "gpu": False},
        )

        self.workloads["stress_test_text_c4_ray"] = WorkloadDefinition(
            name="stress_test_text_c4_ray",
            description="C4 dataset stress test with Ray distributed execution",
            dataset_path="perf_bench_data/text/c4-train.00000-of-01024.jsonl",
            config_path="tests/benchmark_performance/configs/text-c4-ray.yaml",
            expected_samples=100000,  # C4 dataset is much larger
            modality="text",
            complexity="complex",
            estimated_duration_minutes=90,  # Ray should be faster than local
            resource_requirements={"memory_gb": 64, "cpu_cores": 32, "gpu": False},
        )

        # Operation Reordering Showcase
        self.workloads["op_reorder_showcase"] = WorkloadDefinition(
            name="op_reorder_showcase",
            description="Showcase workload designed to demonstrate operation reordering benefits",
            dataset_path="perf_bench_data/text/c4-train.00000-of-01024.jsonl",
            config_path="configs/optimization/op_reorder_showcase.yaml",
            expected_samples=10000,
            modality="text",
            complexity="complex",
            estimated_duration_minutes=15,  # Moderate complexity
            resource_requirements={"memory_gb": 8, "cpu_cores": 8, "gpu": False},
        )

    def get_workload(self, name: str) -> Optional[WorkloadDefinition]:
        """Get a specific workload by name."""
        return self.workloads.get(name)

    def get_workloads_by_modality(self, modality: str) -> List[WorkloadDefinition]:
        """Get all workloads for a specific modality."""
        return [w for w in self.workloads.values() if w.modality == modality]

    def get_workloads_by_complexity(self, complexity: str) -> List[WorkloadDefinition]:
        """Get all workloads for a specific complexity level."""
        return [w for w in self.workloads.values() if w.complexity == complexity]

    def get_all_workloads(self) -> List[WorkloadDefinition]:
        """Get all available workloads."""
        return list(self.workloads.values())

    def get_workload_names(self) -> List[str]:
        """Get names of all available workloads."""
        return list(self.workloads.keys())

    def validate_workloads(self) -> Dict[str, List[str]]:
        """Validate all workloads and return any issues."""
        issues = {}

        for name, workload in self.workloads.items():
            workload_issues = []

            if not Path(workload.dataset_path).exists():
                workload_issues.append(f"Dataset not found: {workload.dataset_path}")

            if not Path(workload.config_path).exists():
                workload_issues.append(f"Config not found: {workload.config_path}")

            if workload.expected_samples <= 0:
                workload_issues.append("Expected samples must be positive")

            if workload.estimated_duration_minutes <= 0:
                workload_issues.append("Estimated duration must be positive")

            if workload_issues:
                issues[name] = workload_issues

        return issues


# Global workload suite instance
WORKLOAD_SUITE = WorkloadSuite()
