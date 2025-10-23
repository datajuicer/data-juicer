#!/usr/bin/env python3
"""
Main benchmark runner for executing performance tests.
"""

import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..utils.config_manager import ConfigManager
from .metrics_collector import BenchmarkMetrics, MetricsCollector


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    dataset_path: str
    config_path: str
    output_dir: str
    iterations: int = 3
    warmup_runs: int = 1
    timeout_seconds: int = 3600
    strategy_name: str = "baseline"
    strategy_config: Dict[str, Any] = None
    sample_ratio: float = 1.0
    sample_method: str = "random"


class BenchmarkRunner:
    """Main benchmark execution engine."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.config_manager = ConfigManager()

        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_benchmark(self) -> BenchmarkMetrics:
        """Run a single benchmark iteration."""
        logger.info(f"Starting benchmark: {self.config.strategy_name}")

        # Start metrics collection
        self.metrics_collector.start_monitoring()

        try:
            # Prepare configuration
            config_file = self._prepare_config()

            # Run the benchmark
            start_time = time.time()
            result = self._execute_benchmark(config_file)
            end_time = time.time()

            # Stop metrics collection
            metrics = self.metrics_collector.stop_monitoring()

            # Enhance metrics with benchmark-specific data
            if metrics:
                metrics.total_wall_time = end_time - start_time
                metrics.strategy_name = self.config.strategy_name
                metrics.config_hash = self._get_config_hash()

                # Add benchmark-specific metrics
                if result:
                    metrics.samples_processed = result.get("samples_processed", 0)
                    metrics.samples_retained = result.get("samples_retained", 0)
                    metrics.samples_per_second = (
                        metrics.samples_processed / metrics.total_wall_time if metrics.total_wall_time > 0 else 0
                    )
                    metrics.data_retention_rate = (
                        metrics.samples_retained / metrics.samples_processed if metrics.samples_processed > 0 else 0
                    )

            return metrics

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            # Still try to get partial metrics
            metrics = self.metrics_collector.stop_monitoring()
            if metrics:
                metrics.strategy_name = self.config.strategy_name
                metrics.config_hash = self._get_config_hash()
            return metrics

    def run_benchmark_suite(self) -> List[BenchmarkMetrics]:
        """Run multiple iterations of the benchmark."""
        logger.info(f"Running benchmark suite: {self.config.iterations} iterations")

        all_metrics = []

        # Warmup runs (not counted in results)
        for i in range(self.config.warmup_runs):
            logger.info(f"Warmup run {i+1}/{self.config.warmup_runs}")
            self.run_benchmark()

        # Actual benchmark runs
        for i in range(self.config.iterations):
            logger.info(f"Benchmark iteration {i+1}/{self.config.iterations}")
            metrics = self.run_benchmark()
            if metrics:
                all_metrics.append(metrics)

        return all_metrics

    def _prepare_config(self) -> str:
        """Prepare configuration file for the benchmark."""
        # Load base configuration
        base_config = self.config_manager.load_config(self.config.config_path)

        # Apply strategy-specific modifications
        if self.config.strategy_config:
            # Check if this is a core optimizer strategy
            if self.config.strategy_config.get("_benchmark_optimizer_enabled"):
                # For core optimizer strategies, we use environment variables instead of config keys
                # This avoids config validation issues
                logger.info("🔧 Core optimizer strategy detected - will use environment variables")
            else:
                # For regular config strategies, apply them directly
                base_config = self.config_manager.apply_strategy_config(base_config, self.config.strategy_config)

        # Apply sampling if needed
        if self.config.sample_ratio < 1.0:
            base_config = self._apply_sampling_config(base_config)

        # Save modified configuration
        config_output_path = os.path.join(self.config.output_dir, f"config_{self.config.strategy_name}.yaml")
        self.config_manager.save_config(base_config, config_output_path)

        return config_output_path

    def _apply_sampling_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply sampling configuration to the config."""
        # If sampling is enabled, we'll need to create a sampled dataset
        if self.config.sample_ratio < 1.0:
            sampled_dataset_path = self._create_sampled_dataset()
            config["dataset_path"] = sampled_dataset_path

        return config

    def _create_sampled_dataset(self) -> str:
        """Create a sampled version of the dataset."""
        import random

        # Read the original dataset
        with open(self.config.dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_samples = len(lines)
        sample_size = int(total_samples * self.config.sample_ratio)

        # Sample the data based on method
        if self.config.sample_method == "random":
            sampled_lines = random.sample(lines, sample_size)
        elif self.config.sample_method == "first":
            sampled_lines = lines[:sample_size]
        elif self.config.sample_method == "last":
            sampled_lines = lines[-sample_size:]
        else:
            raise ValueError(f"Unknown sampling method: {self.config.sample_method}")

        # Create sampled dataset file
        sampled_path = os.path.join(
            self.config.output_dir, f"sampled_dataset_{self.config.sample_ratio}_{self.config.sample_method}.jsonl"
        )

        with open(sampled_path, "w", encoding="utf-8") as f:
            f.writelines(sampled_lines)

        logger.info(f"Created sampled dataset: {sampled_path} ({sample_size}/{total_samples} samples)")
        return sampled_path

    def _execute_benchmark(self, config_file: str) -> Optional[Dict[str, Any]]:
        """Execute the actual benchmark using data-juicer."""
        try:
            # Clean up output directory before running benchmark
            self._cleanup_output_directory()

            # Check if core optimizer should be enabled
            if self.config.strategy_config and self.config.strategy_config.get("_benchmark_optimizer_enabled"):
                # Use custom executor that applies core optimizer
                return self._execute_with_core_optimizer(config_file)
            else:
                # Use standard subprocess execution
                return self._execute_standard_benchmark(config_file)

        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out after {self.config.timeout_seconds} seconds")
            return None
        except Exception as e:
            logger.error(f"Error executing benchmark: {e}")
            return None

    def _execute_standard_benchmark(self, config_file: str) -> Optional[Dict[str, Any]]:
        """Execute benchmark using standard subprocess."""
        # Build command
        cmd = [
            "python",
            "-m",
            "data_juicer.tools.process_data",
            "--config",
            config_file,
            "--export_path",
            os.path.join(self.config.output_dir, "output.jsonl"),
        ]

        # Only add dataset_path if it's provided (config might have it instead)
        if self.config.dataset_path:
            cmd.extend(["--dataset_path", self.config.dataset_path])

        logger.debug(f"Executing command: {' '.join(cmd)}")

        # Run the benchmark
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.config.timeout_seconds, cwd=os.getcwd()
        )

        if result.returncode != 0:
            logger.error(f"Benchmark execution failed: {result.stderr}")
            return None

        # Log the subprocess output for debugging
        logger.info("=== Subprocess STDOUT ===")
        logger.info(result.stdout)
        logger.info("=== Subprocess STDERR ===")
        logger.info(result.stderr)
        logger.info("=== End Subprocess Output ===")

        # Parse output for metrics (data-juicer logs to stderr, not stdout)
        return self._parse_benchmark_output(result.stdout)

    def _execute_with_core_optimizer(self, config_file: str) -> Optional[Dict[str, Any]]:
        """Execute benchmark with core optimizer applied."""
        try:
            # Get the enabled optimizer strategies
            enabled_strategies = self.config.strategy_config.get("_benchmark_optimizer_strategies", [])
            logger.info(f"🔧 Applying core optimizer with strategies: {enabled_strategies}")

            # Use environment variables to pass optimizer information
            env = os.environ.copy()
            env["DJ_ENABLE_CORE_OPTIMIZER"] = "true"
            env["DJ_OPTIMIZER_STRATEGIES"] = ",".join(enabled_strategies)

            # Build command with the original config
            cmd = [
                "python",
                "-m",
                "data_juicer.tools.process_data",
                "--config",
                config_file,
                "--export_path",
                os.path.join(self.config.output_dir, "output.jsonl"),
            ]

            # Only add dataset_path if it's provided (config might have it instead)
            if self.config.dataset_path:
                cmd.extend(["--dataset_path", self.config.dataset_path])

            logger.debug(f"Executing command with core optimizer: {' '.join(cmd)}")

            # Run the benchmark with environment variables
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout_seconds, cwd=os.getcwd(), env=env
            )

            # No cleanup needed since we're using environment variables

            if result.returncode != 0:
                logger.error(f"Benchmark execution failed: {result.stderr}")
                return None

            # Log the subprocess output for debugging
            logger.info("=== Subprocess STDOUT ===")
            logger.info(result.stdout)
            logger.info("=== Subprocess STDERR ===")
            logger.info(result.stderr)
            logger.info("=== End Subprocess Output ===")

            # Parse output for metrics (data-juicer logs to stderr, not stdout)
            return self._parse_benchmark_output(result.stdout)

        except Exception as e:
            logger.error(f"Error executing benchmark with core optimizer: {e}")
            return None

    def _cleanup_output_directory(self):
        """Clean up the output directory before running benchmark to prevent multiple outputs."""
        try:

            output_path = os.path.join(self.config.output_dir, "output.jsonl")

            # Remove existing output directory if it exists
            if os.path.exists(output_path):
                logger.info(f"Cleaning up existing output directory: {output_path}")
                if os.path.isdir(output_path):
                    shutil.rmtree(output_path)
                else:
                    os.remove(output_path)
                logger.info("Output directory cleaned up successfully")
            else:
                logger.info("No existing output directory to clean up")

        except Exception as e:
            logger.warning(f"Failed to clean up output directory: {e}")
            # Don't fail the benchmark if cleanup fails

    def _get_config_hash(self) -> str:
        """Generate hash of current configuration."""
        config_str = f"{self.config.dataset_path}_{self.config.config_path}_{self.config.strategy_name}"
        if self.config.strategy_config:
            config_str += str(sorted(self.config.strategy_config.items()))

        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _parse_benchmark_output(self, output: str) -> Dict[str, Any]:
        """Parse benchmark output to extract metrics."""

        # Try file-based metrics first (more reliable)
        file_metrics = self._get_file_based_metrics()
        if file_metrics:
            logger.info(f"=== File-Based Metrics ===")
            logger.info(f"Initial samples: {file_metrics.get('samples_processed', 'N/A')}")
            logger.info(f"Final samples: {file_metrics.get('samples_retained', 'N/A')}")
            logger.info(f"Retention rate: {file_metrics.get('retention_rate', 'N/A')}")
            logger.info(f"=== End File-Based Metrics ===")
            return file_metrics

        # Fallback to text parsing (less reliable)
        logger.info("File-based metrics not available, falling back to text parsing...")
        return self._parse_text_output(output)

    def _get_file_based_metrics(self) -> Optional[Dict[str, Any]]:
        """Get metrics by counting actual files (more reliable than text parsing)."""
        try:
            # Get initial dataset size
            initial_samples = self._count_input_records()

            # Get final dataset size from output files
            final_samples = self._count_output_records()

            if initial_samples is None or final_samples is None:
                logger.warning("Could not determine initial or final sample counts from files")
                return None

            retention_rate = final_samples / initial_samples if initial_samples > 0 else 0

            return {
                "samples_processed": initial_samples,
                "samples_retained": final_samples,
                "retention_rate": retention_rate,
            }

        except Exception as e:
            logger.error(f"Error getting file-based metrics: {e}")
            return None

    def _count_input_records(self) -> Optional[int]:
        """Count records in the input dataset."""
        try:
            import subprocess

            import yaml

            # Get dataset path - either from config or from config file
            dataset_path = self.config.dataset_path
            if dataset_path is None:
                # Try to extract dataset_path from config file
                try:
                    with open(self.config.config_path, "r") as f:
                        config_data = yaml.safe_load(f)
                        dataset_path = config_data.get("dataset_path")
                except Exception as e:
                    logger.warning(f"Could not read dataset_path from config file: {e}")
                    return None

            if dataset_path is None:
                logger.warning("No dataset path available for counting input records")
                return None

            result = subprocess.run(["wc", "-l", dataset_path], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return int(result.stdout.split()[0])
        except Exception as e:
            logger.error(f"Error counting input records: {e}")
        return None

    def _count_output_records(self) -> Optional[int]:
        """Count records in the output files."""
        try:
            import os
            import subprocess

            output_dir = os.path.join(self.config.output_dir, "output.jsonl")
            if not os.path.exists(output_dir):
                logger.warning(f"Output directory not found: {output_dir}")
                return None

            # Count all JSON files in the output directory
            result = subprocess.run(
                ["find", output_dir, "-name", "*.json", "-exec", "wc", "-l", "{}", "+"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                # Parse the output to get total count
                lines = result.stdout.strip().split("\n")

                # Check if we have a "total" line (from using + syntax)
                for line in lines:
                    if "total" in line.lower():
                        parts = line.split()
                        if parts and parts[0].isdigit():
                            return int(parts[0])

                # Fallback: sum individual counts (for \; syntax)
                total = 0
                for line in lines:
                    if line.strip() and "total" not in line.lower():
                        parts = line.split()
                        if parts and parts[0].isdigit():
                            total += int(parts[0])
                return total

        except Exception as e:
            logger.error(f"Error counting output records: {e}")
        return None

    def _parse_text_output(self, output: str) -> Dict[str, Any]:
        """Fallback text parsing method (less reliable)."""
        metrics = {}
        lines = output.split("\n")
        initial_samples = None
        final_samples = None

        for line in lines:
            # Look for initial sample count patterns
            if "samples left after filtering empty text" in line:
                try:
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            initial_samples = int(part)
                            break
                except Exception as e:
                    logger.error(f"Error parsing initial samples: {e}")

            # Look for final sample count patterns
            if "Left" in line and "samples" in line and "Done in" in line:
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Left" and i + 1 < len(parts):
                            if parts[i + 1].isdigit():
                                final_samples = int(parts[i + 1])
                                break
                except Exception as e:
                    logger.error(f"Error parsing final samples: {e}")

        if initial_samples is not None:
            metrics["samples_processed"] = initial_samples
        if final_samples is not None:
            metrics["samples_retained"] = final_samples

        logger.info(f"=== Text Parsing Results ===")
        logger.info(f"Initial samples found: {initial_samples}")
        logger.info(f"Final samples found: {final_samples}")
        logger.info(f"Parsed metrics: {metrics}")
        logger.info(f"=== End Text Parsing ===")
        return metrics
