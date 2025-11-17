#!/usr/bin/env python3
"""
Performance Test: Individual Pipeline vs Optimized Mode Comparison

This script runs performance benchmarks comparing individual pipeline execution
vs optimized pipeline execution using separate processes to ensure fair comparison.

Features:
- Separate process execution for isolation
- Support for recipe path and dataset path
- Comprehensive metrics collection
- Result validation and comparison
- Detailed reporting
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger

from data_juicer.config import init_configs
from data_juicer.core import DefaultExecutor
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.core.optimizer.filter_fusion_strategy import FilterFusionStrategy
from data_juicer.core.optimizer.mapper_fusion_strategy import MapperFusionStrategy
from data_juicer.core.optimizer.optimizer import PipelineOptimizer
from data_juicer.core.optimizer.performance_benchmark import PerformanceBenchmark
from data_juicer.core.pipeline_ast import PipelineAST
from data_juicer.ops import load_ops

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class PipelinePerformanceTester:
    """Performance tester for comparing individual vs optimized pipeline execution."""

    def __init__(self, output_dir: str = "./outputs/pipeline_perf_test"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self.output_dir / "perf_test.log"
        logger.add(log_file, rotation="10 MB", level="INFO")

        self.results = {"individual": {}, "optimized": {}, "comparison": {}, "metadata": {}}

    def load_dataset(self, dataset_path: str) -> Any:
        """Load dataset from path using DatasetBuilder."""
        logger.info(f"Loading dataset from: {dataset_path}")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        # Build a minimal config Namespace for DatasetBuilder
        cfg = Namespace(dataset_path=dataset_path)
        builder = DatasetBuilder(cfg)
        dataset = builder.load_dataset()
        dataset_length = len(dataset.to_list()) if dataset is not None else 0
        logger.info(f"Loaded dataset with {dataset_length} samples")
        return dataset

    def create_temp_config(self, recipe_path: str, dataset_path: str, mode: str) -> str:
        """Create a temporary config file for execution."""
        # Load the original recipe
        with open(recipe_path, "r") as f:
            recipe_config = yaml.safe_load(f)

        # Create temp config
        temp_config = {
            "project_name": f"perf-test-{mode}",
            "dataset_path": dataset_path,
            "export_path": str(self.output_dir / f"result_{mode}.jsonl"),
            "np": 1,  # Single process for fair comparison
            "use_cache": False,
            "op_fusion": mode == "optimized",  # Enable fusion only for optimized mode
            "process": recipe_config.get("process", []),
        }

        # Write temp config
        temp_config_path = self.output_dir / f"temp_config_{mode}.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config, f, default_flow_style=False)

        return str(temp_config_path)

    def run_individual_pipeline(self, recipe_path: str, dataset_path: str) -> Dict[str, Any]:
        """Run individual pipeline execution in separate process."""
        logger.info("Running individual pipeline execution...")

        temp_config_path = self.create_temp_config(recipe_path, dataset_path, "individual")

        # Run in separate process
        start_time = time.time()
        result = self._run_in_process(temp_config_path, "individual")
        end_time = time.time()

        result["wall_time"] = end_time - start_time
        result["config_path"] = temp_config_path

        return result

    def run_optimized_pipeline(self, recipe_path: str, dataset_path: str) -> Dict[str, Any]:
        """Run optimized pipeline execution in separate process."""
        logger.info("Running optimized pipeline execution...")

        temp_config_path = self.create_temp_config(recipe_path, dataset_path, "optimized")

        # Run in separate process
        start_time = time.time()
        result = self._run_in_process(temp_config_path, "optimized")
        end_time = time.time()

        result["wall_time"] = end_time - start_time
        result["config_path"] = temp_config_path

        return result

    def _run_in_process(self, config_path: str, mode: str) -> Dict[str, Any]:
        """Run pipeline execution in a separate process."""
        # Create process and run
        result_queue = mp.Queue()
        process = mp.Process(target=_worker_process, args=(config_path, mode, result_queue))

        process.start()
        process.join(timeout=3600)  # 1 hour timeout

        if process.is_alive():
            process.terminate()
            process.join()
            return {"execution_time": 0, "output_samples": 0, "success": False, "error": "Process timeout"}

        if not result_queue.empty():
            return result_queue.get()
        else:
            return {"execution_time": 0, "output_samples": 0, "success": False, "error": "No result from process"}


def _worker_process(config_path: str, mode: str, result_queue: mp.Queue):
    """Worker function for running pipeline execution in separate process."""
    try:
        # Add the project root to the path
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))

        # Initialize config
        args = ["--config", config_path]
        cfg = init_configs(args=args)

        # Create executor
        executor = DefaultExecutor(cfg)

        # Run and collect metrics
        start_time = time.time()
        dataset = executor.run()
        end_time = time.time()

        # Collect results
        dataset_length = len(dataset) if dataset is not None else 0
        result = {
            "execution_time": end_time - start_time,
            "output_samples": dataset_length,
            "success": True,
            "error": None,
        }

        result_queue.put(result)

    except Exception as e:
        result = {"execution_time": 0, "output_samples": 0, "success": False, "error": str(e)}
        result_queue.put(result)

    def run_benchmark_comparison(self, recipe_path: str, dataset_path: str) -> Dict[str, Any]:
        """Run comprehensive benchmark comparison using the performance benchmark framework."""
        logger.info("Running comprehensive benchmark comparison...")

        # Load dataset
        dataset = self.load_dataset(dataset_path)

        # Create benchmark instance
        benchmark = PerformanceBenchmark()

        # Load recipe and create AST
        ast = PipelineAST()
        ast.build_from_yaml(recipe_path)

        # Get analyzer insights
        analyzer_insights = benchmark.get_analyzer_insights(dataset)

        # Create optimizer
        optimizer = PipelineOptimizer(
            [FilterFusionStrategy(), MapperFusionStrategy()], analyzer_insights=analyzer_insights
        )

        # Optimize pipeline
        optimized_ast = optimizer.optimize(ast)

        # Convert to operations
        original_operations = benchmark._convert_ast_to_operations(ast)
        optimized_operations = benchmark._convert_ast_to_operations(optimized_ast)

        # Load operations
        loaded_original_ops = self._load_operations(original_operations)
        loaded_optimized_ops = self._load_operations(optimized_operations)

        # Run benchmark
        results = benchmark.run_mixed_operations_benchmark_with_original_ops(
            loaded_original_ops, loaded_optimized_ops, dataset, "recipe"
        )

        return results

    def _load_operations(self, operation_configs: List[Dict]) -> List:
        """Load operations from configs."""
        loaded_ops = []

        for op_config in operation_configs:
            op_name = list(op_config.keys())[0]
            op_args = op_config[op_name]

            if op_name == "fused_filter":
                # Handle fused filter
                fused_op_list = op_args.get("fused_op_list", [])
                individual_filters = []

                for filter_config in fused_op_list:
                    filter_name = list(filter_config.keys())[0]
                    filter_args = filter_config[filter_name]
                    loaded_filters = load_ops([{filter_name: filter_args}])
                    if loaded_filters:
                        individual_filters.append(loaded_filters[0])

                if individual_filters:
                    from data_juicer.core.optimizer.fused_op import FusedFilter

                    fused_filter = FusedFilter(name="fused_filter", fused_filters=individual_filters)
                    fused_filter.execution_strategy = "sequential"
                    loaded_ops.append(fused_filter)

            elif op_name == "fused_mapper":
                # Handle fused mapper
                from data_juicer.core.optimizer.fused_op import FusedMapper

                name = op_args.get("name", "fused_mapper")
                fused_mappers = op_args.get("fused_mappers", [])
                fused_mapper = FusedMapper(name=name, fused_mappers=fused_mappers)
                loaded_ops.append(fused_mapper)

            else:
                # Load regular operation
                loaded_ops_list = load_ops([op_config])
                if loaded_ops_list:
                    loaded_ops.append(loaded_ops_list[0])

        return loaded_ops

    def validate_results(self, individual_result: Dict, optimized_result: Dict) -> Dict[str, Any]:
        """Validate that both executions produced similar results."""
        logger.info("Validating results...")

        validation = {
            "samples_match": False,
            "individual_samples": individual_result.get("output_samples", 0),
            "optimized_samples": optimized_result.get("output_samples", 0),
            "sample_difference": 0,
            "validation_passed": False,
        }

        if individual_result.get("success") and optimized_result.get("success"):
            individual_samples = individual_result["output_samples"]
            optimized_samples = optimized_result["output_samples"]

            validation["samples_match"] = individual_samples == optimized_samples
            validation["sample_difference"] = abs(individual_samples - optimized_samples)
            validation["validation_passed"] = validation["samples_match"]

            if validation["validation_passed"]:
                logger.info("✅ Validation passed: Both executions produced the same number of samples")
            else:
                logger.warning(
                    f"❌ Validation failed: Sample count mismatch "
                    f"(individual: {individual_samples}, optimized: {optimized_samples})"
                )
        else:
            logger.error("❌ Validation failed: One or both executions failed")

        return validation

    def compare_performance(self, individual_result: Dict, optimized_result: Dict) -> Dict[str, Any]:
        """Compare performance metrics between individual and optimized execution."""
        logger.info("Comparing performance metrics...")

        comparison = {
            "individual_time": individual_result.get("wall_time", 0),
            "optimized_time": optimized_result.get("wall_time", 0),
            "speedup": 0,
            "improvement_percent": 0,
            "faster_mode": "none",
        }

        if individual_result.get("success") and optimized_result.get("success"):
            individual_time = individual_result["wall_time"]
            optimized_time = optimized_result["wall_time"]

            if individual_time > 0:
                comparison["speedup"] = individual_time / optimized_time
                comparison["improvement_percent"] = ((individual_time - optimized_time) / individual_time) * 100

            if optimized_time < individual_time:
                comparison["faster_mode"] = "optimized"
            elif individual_time < optimized_time:
                comparison["faster_mode"] = "individual"
            else:
                comparison["faster_mode"] = "equal"

        return comparison

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive performance report."""
        logger.info("Generating performance report...")

        report_path = self.output_dir / "performance_report.md"

        with open(report_path, "w") as f:
            f.write("# Pipeline Performance Test Report\n\n")

            # Summary
            f.write("## Summary\n\n")
            comparison = results["comparison"]
            validation = results["validation"]

            f.write(f"- **Test Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Recipe**: {results['metadata']['recipe_path']}\n")
            f.write(f"- **Dataset**: {results['metadata']['dataset_path']}\n")
            f.write(f"- **Validation**: {'✅ PASSED' if validation['validation_passed'] else '❌ FAILED'}\n")
            f.write(f"- **Faster Mode**: {comparison['faster_mode'].title()}\n")
            f.write(f"- **Speedup**: {comparison['speedup']:.2f}x\n")
            f.write(f"- **Improvement**: {comparison['improvement_percent']:.1f}%\n\n")

            # Detailed Results
            f.write("## Detailed Results\n\n")

            f.write("### Individual Pipeline\n")
            f.write(f"- Execution Time: {results['individual']['wall_time']:.2f}s\n")
            f.write(f"- Output Samples: {results['individual']['output_samples']:,}\n")
            f.write(f"- Success: {results['individual']['success']}\n")
            if not results["individual"]["success"]:
                f.write(f"- Error: {results['individual']['error']}\n")
            f.write("\n")

            f.write("### Optimized Pipeline\n")
            f.write(f"- Execution Time: {results['optimized']['wall_time']:.2f}s\n")
            f.write(f"- Output Samples: {results['optimized']['output_samples']:,}\n")
            f.write(f"- Success: {results['optimized']['success']}\n")
            if not results["optimized"]["success"]:
                f.write(f"- Error: {results['optimized']['error']}\n")
            f.write("\n")

            # Performance Comparison
            f.write("### Performance Comparison\n")
            f.write(f"- Individual Time: {comparison['individual_time']:.2f}s\n")
            f.write(f"- Optimized Time: {comparison['optimized_time']:.2f}s\n")
            f.write(f"- Speedup: {comparison['speedup']:.2f}x\n")
            f.write(f"- Improvement: {comparison['improvement_percent']:.1f}%\n")
            f.write(f"- Faster Mode: {comparison['faster_mode'].title()}\n\n")

            # Validation Results
            f.write("### Validation Results\n")
            f.write(f"- Samples Match: {validation['samples_match']}\n")
            f.write(f"- Individual Samples: {validation['individual_samples']:,}\n")
            f.write(f"- Optimized Samples: {validation['optimized_samples']:,}\n")
            f.write(f"- Sample Difference: {validation['sample_difference']}\n")
            f.write(f"- Validation Passed: {validation['validation_passed']}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            if validation["validation_passed"]:
                if comparison["faster_mode"] == "optimized":
                    f.write(
                        "✅ **Use Optimized Pipeline**: The optimized pipeline is faster and produces correct results.\n"
                    )
                elif comparison["faster_mode"] == "individual":
                    f.write(
                        "⚠️ **Consider Individual Pipeline**: The individual pipeline is faster, but optimization may still be beneficial for larger datasets.\n"
                    )
                else:
                    f.write(
                        "ℹ️ **Both Modes Similar**: Performance is similar between individual and optimized modes.\n"
                    )
            else:
                f.write("❌ **Investigation Required**: Results don't match between individual and optimized modes.\n")

        return str(report_path)

    def save_results(self, results: Dict[str, Any]) -> str:
        """Save results to JSON file."""
        results_path = self.output_dir / "results.json"

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        return str(results_path)

    def run_test(self, recipe_path: str, dataset_path: str, use_benchmark_framework: bool = False) -> Dict[str, Any]:
        """Run the complete performance test."""
        logger.info("Starting pipeline performance test...")
        logger.info(f"Recipe: {recipe_path}")
        logger.info(f"Dataset: {dataset_path}")

        # Store metadata
        self.results["metadata"] = {
            "recipe_path": recipe_path,
            "dataset_path": dataset_path,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "use_benchmark_framework": use_benchmark_framework,
        }

        if use_benchmark_framework:
            # Use the comprehensive benchmark framework
            logger.info("Using comprehensive benchmark framework...")
            benchmark_results = self.run_benchmark_comparison(recipe_path, dataset_path)
            self.results.update(benchmark_results)
        else:
            # Use separate process execution
            logger.info("Using separate process execution...")

            # Run individual pipeline
            individual_result = self.run_individual_pipeline(recipe_path, dataset_path)
            self.results["individual"] = individual_result

            # Run optimized pipeline
            optimized_result = self.run_optimized_pipeline(recipe_path, dataset_path)
            self.results["optimized"] = optimized_result

            # Validate results
            validation = self.validate_results(individual_result, optimized_result)
            self.results["validation"] = validation

            # Compare performance
            comparison = self.compare_performance(individual_result, optimized_result)
            self.results["comparison"] = comparison

        # Save results
        results_path = self.save_results(self.results)
        logger.info(f"Results saved to: {results_path}")

        # Generate report
        report_path = self.generate_report(self.results)
        logger.info(f"Report generated: {report_path}")

        # Print summary
        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print a summary of the test results."""
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("=" * 60)

        comparison = self.results.get("comparison", {})
        validation = self.results.get("validation", {})

        logger.info(f"Individual Time: {comparison.get('individual_time', 0):.2f}s")
        logger.info(f"Optimized Time:  {comparison.get('optimized_time', 0):.2f}s")
        logger.info(f"Speedup:        {comparison.get('speedup', 0):.2f}x")
        logger.info(f"Improvement:    {comparison.get('improvement_percent', 0):.1f}%")
        logger.info(f"Validation:     {'✅ PASSED' if validation.get('validation_passed') else '❌ FAILED'}")
        logger.info(f"Faster Mode:    {comparison.get('faster_mode', 'none').title()}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Pipeline Performance Test: Compare individual vs optimized execution")
    parser.add_argument("--recipe-path", type=str, required=True, help="Path to the recipe YAML file")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/pipeline_perf_test",
        help="Output directory for results and reports",
    )
    parser.add_argument(
        "--use-benchmark-framework",
        action="store_true",
        help="Use the comprehensive benchmark framework instead of separate processes",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)

    # Create tester and run test
    tester = PipelinePerformanceTester(args.output_dir)

    try:
        results = tester.run_test(args.recipe_path, args.dataset_path, args.use_benchmark_framework)

        # Exit with appropriate code
        validation = results.get("validation", {})
        if validation.get("validation_passed"):
            logger.info("✅ Test completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Test failed validation")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
