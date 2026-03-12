#!/usr/bin/env python3
"""
Report generation for benchmark results.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

from .metrics_collector import BenchmarkMetrics
from .result_analyzer import ComparisonResult


class ReportGenerator:
    """Generates comprehensive reports from benchmark results."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def generate_ab_test_report(
        self,
        results: Dict[str, List[BenchmarkMetrics]],
        comparisons: Dict[str, ComparisonResult],
        test_name: str = "A/B Test",
    ) -> str:
        """Generate a comprehensive A/B test report."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"ab_test_report_{timestamp}.html"

        # Generate HTML report
        html_content = self._generate_html_report(results, comparisons, test_name)

        with open(report_file, "w") as f:
            f.write(html_content)

        # Also generate JSON data
        json_file = self.output_dir / f"ab_test_data_{timestamp}.json"
        self._save_json_data(results, comparisons, json_file)

        logger.info(f"Generated A/B test report: {report_file}")
        return str(report_file)

    def generate_workload_report(
        self, workload_results: Dict[str, Dict[str, List[BenchmarkMetrics]]], test_name: str = "Workload Benchmark"
    ) -> str:
        """Generate a report for workload testing across different scenarios."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"workload_report_{timestamp}.html"

        html_content = self._generate_workload_html_report(workload_results, test_name)

        with open(report_file, "w") as f:
            f.write(html_content)

        logger.info(f"Generated workload report: {report_file}")
        return str(report_file)

    def _generate_html_report(
        self, results: Dict[str, List[BenchmarkMetrics]], comparisons: Dict[str, ComparisonResult], test_name: str
    ) -> str:
        """Generate HTML report content."""

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{test_name} - Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ background-color: #e8f4fd; padding: 15px; margin: 20px 0; border-radius: 5px; }}
        .comparison {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }}
        .improvement {{ border-left-color: #28a745; }}
        .regression {{ border-left-color: #dc3545; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background-color: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #007acc; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{test_name}</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="summary">
        <h2>Executive Summary</h2>
        {self._generate_summary_html(comparisons)}
    </div>

    <div class="metrics">
        {self._generate_metrics_cards(results)}
    </div>

    <h2>Detailed Comparisons</h2>
    {self._generate_comparisons_html(comparisons)}

    <h2>Raw Data</h2>
    {self._generate_raw_data_table(results)}

</body>
</html>
        """

        return html

    def _generate_summary_html(self, comparisons: Dict[str, ComparisonResult]) -> str:
        """Generate summary section HTML."""
        if not comparisons:
            return "<p>No comparisons available.</p>"

        summary_html = "<ul>"
        for strategy_name, comparison in comparisons.items():
            status_class = (
                "improvement" if comparison.is_improvement() else "regression" if comparison.is_regression() else ""
            )
            summary_html += f"""
            <li class="comparison {status_class}">
                <strong>{strategy_name}:</strong> {comparison.summary}
            </li>
            """
        summary_html += "</ul>"

        return summary_html

    def _generate_metrics_cards(self, results: Dict[str, List[BenchmarkMetrics]]) -> str:
        """Generate metrics cards HTML."""
        cards_html = ""

        for strategy_name, metrics_list in results.items():
            if not metrics_list:
                continue

            # Calculate aggregate metrics
            avg_time = sum(m.total_wall_time for m in metrics_list) / len(metrics_list)
            avg_throughput = sum(m.samples_per_second for m in metrics_list) / len(metrics_list)
            max_memory = max(m.peak_memory_mb for m in metrics_list)
            avg_retention = sum(m.data_retention_rate for m in metrics_list) / len(metrics_list)

            cards_html += f"""
            <div class="metric-card">
                <div class="metric-value">{avg_time:.2f}s</div>
                <div class="metric-label">Avg Time - {strategy_name}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_throughput:.1f}</div>
                <div class="metric-label">Samples/sec - {strategy_name}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{max_memory:.0f}MB</div>
                <div class="metric-label">Peak Memory - {strategy_name}</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{avg_retention:.1%}</div>
                <div class="metric-label">Retention Rate - {strategy_name}</div>
            </div>
            """

        return cards_html

    def _generate_comparisons_html(self, comparisons: Dict[str, ComparisonResult]) -> str:
        """Generate comparisons section HTML."""
        if not comparisons:
            return "<p>No comparisons available.</p>"

        comparisons_html = ""
        for strategy_name, comparison in comparisons.items():
            status_class = (
                "improvement" if comparison.is_improvement() else "regression" if comparison.is_regression() else ""
            )

            comparisons_html += f"""
            <div class="comparison {status_class}">
                <h3>{strategy_name} vs {comparison.baseline_name}</h3>
                <p><strong>Speedup:</strong> {comparison.speedup:.2f}x</p>
                <p><strong>Throughput Improvement:</strong> {comparison.throughput_improvement:.2f}x</p>
                <p><strong>Memory Efficiency:</strong> {comparison.memory_efficiency:.2f}x</p>
                <p><strong>Statistical Significance:</strong> {comparison.is_significant} (p={comparison.p_value:.4f})</p>
                <p><strong>Summary:</strong> {comparison.summary}</p>
            </div>
            """

        return comparisons_html

    def _generate_raw_data_table(self, results: Dict[str, List[BenchmarkMetrics]]) -> str:
        """Generate raw data table HTML."""
        table_html = "<table><tr><th>Strategy</th><th>Run</th><th>Time (s)</th><th>Throughput</th><th>Memory (MB)</th><th>Retention</th></tr>"

        for strategy_name, metrics_list in results.items():
            for i, metrics in enumerate(metrics_list):
                table_html += f"""
                <tr>
                    <td>{strategy_name}</td>
                    <td>{i+1}</td>
                    <td>{metrics.total_wall_time:.2f}</td>
                    <td>{metrics.samples_per_second:.1f}</td>
                    <td>{metrics.peak_memory_mb:.0f}</td>
                    <td>{metrics.data_retention_rate:.1%}</td>
                </tr>
                """

        table_html += "</table>"
        return table_html

    def _generate_workload_html_report(
        self, workload_results: Dict[str, Dict[str, List[BenchmarkMetrics]]], test_name: str
    ) -> str:
        """Generate HTML report for workload testing."""
        # Similar structure but organized by workload
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{test_name} - Workload Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .workload-section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .workload-title {{ background-color: #f0f0f0; padding: 10px; margin: -20px -20px 20px -20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{test_name}</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
"""

        for workload_name, workload_data in workload_results.items():
            html += f"""
            <div class="workload-section">
                <div class="workload-title">
                    <h2>{workload_name}</h2>
                </div>
                {self._generate_metrics_cards(workload_data)}
            </div>
            """

        html += "</body></html>"
        return html

    def _save_json_data(
        self, results: Dict[str, List[BenchmarkMetrics]], comparisons: Dict[str, ComparisonResult], json_file: Path
    ):
        """Save raw data as JSON."""
        data = {"timestamp": datetime.now().isoformat(), "results": {}, "comparisons": {}}

        # Convert results to serializable format
        for strategy_name, metrics_list in results.items():
            data["results"][strategy_name] = [
                {
                    "total_wall_time": m.total_wall_time,
                    "samples_per_second": m.samples_per_second,
                    "peak_memory_mb": m.peak_memory_mb,
                    "average_cpu_percent": m.average_cpu_percent,
                    "samples_processed": m.samples_processed,
                    "samples_retained": m.samples_retained,
                    "data_retention_rate": m.data_retention_rate,
                    "strategy_name": m.strategy_name,
                    "config_hash": m.config_hash,
                }
                for m in metrics_list
            ]

        # Convert comparisons to serializable format
        for strategy_name, comparison in comparisons.items():
            data["comparisons"][strategy_name] = {
                "baseline_name": comparison.baseline_name,
                "test_name": comparison.test_name,
                "speedup": comparison.speedup,
                "throughput_improvement": comparison.throughput_improvement,
                "memory_efficiency": comparison.memory_efficiency,
                "is_significant": comparison.is_significant,
                "confidence_level": comparison.confidence_level,
                "p_value": comparison.p_value,
                "summary": comparison.summary,
                # Note: baseline_metrics and test_metrics are excluded as they contain non-serializable BenchmarkMetrics objects
            }

        try:
            # Convert numpy types to Python native types
            data = self._convert_numpy_types(data)
            with open(json_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully saved JSON data to {json_file}")
        except Exception as e:
            logger.error(f"Failed to save JSON data to {json_file}: {e}")
            # Try to save a minimal version without problematic fields
            try:
                minimal_data = {
                    "timestamp": data["timestamp"],
                    "results": data["results"],
                    "comparisons": {
                        name: {
                            "baseline_name": comp.baseline_name,
                            "test_name": comp.test_name,
                            "speedup": comp.speedup,
                            "summary": comp.summary,
                        }
                        for name, comp in comparisons.items()
                    },
                }
                minimal_data = self._convert_numpy_types(minimal_data)
                with open(json_file, "w") as f:
                    json.dump(minimal_data, f, indent=2)
                logger.warning(f"Saved minimal JSON data to {json_file}")
            except Exception as e2:
                logger.error(f"Failed to save even minimal JSON data: {e2}")
                raise
