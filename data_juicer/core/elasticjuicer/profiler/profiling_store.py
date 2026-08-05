"""
Profiling Store

Persistent storage and query interface for:
- Resource-throughput curves
- OCS signatures
- Historical performance data

Supports online learning and model updating.
"""

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .metrics import OpExecutionStats, ResourceSnapshot
from .ocs_annotator import OpCostSignature

PROFILE_SCHEMA_VERSION = 1
PROFILE_FILENAME = "profiles.json"


def _fit_robust_line(x_values, y_values):
    """Return a Theil-Sen style slope/intercept robust to sparse outliers."""

    slopes = []
    for left in range(len(x_values)):
        for right in range(left + 1, len(x_values)):
            span = x_values[right] - x_values[left]
            if span != 0:
                slopes.append((y_values[right] - y_values[left]) / span)
    if not slopes:
        raise ValueError("at least two distinct x values are required")
    slope = float(np.median(slopes))
    intercept = float(np.median(y_values - slope * x_values))
    return slope, intercept


@dataclass
class ResourceThroughputCurve:
    """
    Resource-throughput relationship for an operator.

    Models T(r, b) where:
    - T = throughput (samples/sec)
    - r = resource allocation (memory, GPU)
    - b = batch size
    """

    op_name: str
    # Curve parameters (fitted from data)
    coefficients: Dict[str, Any]
    # Model type: 'linear', 'polynomial', 'power'
    model_type: str = "linear"
    # Goodness of fit
    r_squared: float = 0.0
    # Sample count used for fitting
    n_samples: int = 0

    def predict_throughput(self, batch_size: int, memory_mb: float) -> float:
        """Predict throughput given batch size and memory"""
        if self.model_type == "linear":
            # T = a * batch_size + b * memory + c
            a = self.coefficients.get("batch_coef", 0)
            b = self.coefficients.get("memory_coef", 0)
            c = self.coefficients.get("intercept", 0)
            return max(0, a * batch_size + b * memory_mb + c)

        elif self.model_type == "power":
            # T = a * batch_size^b
            a = self.coefficients.get("scale", 1)
            b = self.coefficients.get("power", 1)
            return a * (batch_size**b)

        elif self.model_type == "piecewise":
            batch_sizes = np.asarray(self.coefficients.get("batch_sizes", []), dtype=float)
            throughputs = np.asarray(self.coefficients.get("throughputs", []), dtype=float)
            if len(batch_sizes) < 2 or len(batch_sizes) != len(throughputs):
                return 0.0
            if batch_size <= batch_sizes[0]:
                left, right = 0, 1
            elif batch_size >= batch_sizes[-1]:
                # The empirical model represents a saturating throughput
                # curve. Avoid unstable upward extrapolation past evidence.
                return max(0.0, float(throughputs[-1]))
            else:
                return max(0.0, float(np.interp(batch_size, batch_sizes, throughputs)))
            span = batch_sizes[right] - batch_sizes[left]
            slope = 0.0 if span == 0 else (throughputs[right] - throughputs[left]) / span
            predicted = throughputs[left] + slope * (batch_size - batch_sizes[left])
            return max(0.0, float(predicted))

        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "ResourceThroughputCurve":
        """Create from dictionary"""
        return cls(**data)


class ProfilingStore:
    """
    Persistent store for operator profiling data.

    Provides:
    - Storage and retrieval of execution stats
    - Resource-throughput curve fitting
    - Online model updates
    - Query interface for schedulers
    """

    def __init__(self, storage_dir: str = "./elastic_juicer_profiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.load_errors = []
        self._lock = threading.RLock()

        # In-memory caches
        self.execution_stats: Dict[str, OpExecutionStats] = {}
        self.ocs_signatures: Dict[str, OpCostSignature] = {}
        self.throughput_curves: Dict[str, ResourceThroughputCurve] = {}

        # Load existing data
        self._load_all()

    def _load_all(self):
        profile_file = self.storage_dir / PROFILE_FILENAME
        if profile_file.exists():
            try:
                with profile_file.open("r", encoding="utf-8") as source:
                    data = json.load(source)
                self._load_profile_payload(data)
                return
            except Exception as error:
                self.load_errors.append(f"{PROFILE_FILENAME}: {error}")

        # Legacy JSON files are safe to import automatically.
        ocs_file = self.storage_dir / "ocs_signatures.json"
        if ocs_file.exists():
            try:
                with ocs_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.ocs_signatures = {name: OpCostSignature.from_dict(sig) for name, sig in data.items()}
            except Exception as error:
                self.load_errors.append(f"{ocs_file.name}: {error}")

        curves_file = self.storage_dir / "throughput_curves.json"
        if curves_file.exists():
            try:
                with curves_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.throughput_curves = {
                        name: ResourceThroughputCurve.from_dict(curve) for name, curve in data.items()
                    }
            except Exception as error:
                self.load_errors.append(f"{curves_file.name}: {error}")

    def save_all(self):
        """Persist all profiles as one versioned, atomically replaced JSON file."""

        with self._lock:
            payload = {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "execution_stats": {name: self._stats_to_dict(stats) for name, stats in self.execution_stats.items()},
                "ocs_signatures": {name: signature.to_dict() for name, signature in self.ocs_signatures.items()},
                "throughput_curves": {name: curve.to_dict() for name, curve in self.throughput_curves.items()},
            }
            output_path = self.storage_dir / PROFILE_FILENAME
            descriptor, temporary_name = tempfile.mkstemp(
                dir=self.storage_dir,
                prefix=f".{PROFILE_FILENAME}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(payload, output, allow_nan=False, indent=2, sort_keys=True)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary_name, output_path)
            except BaseException:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
                raise

    def _load_profile_payload(self, data: Dict):
        version = data.get("schema_version")
        if version != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"unsupported profile schema version: {version}")
        self.execution_stats = {
            name: self._stats_from_dict(stats) for name, stats in data.get("execution_stats", {}).items()
        }
        self.ocs_signatures = {
            name: OpCostSignature.from_dict(signature) for name, signature in data.get("ocs_signatures", {}).items()
        }
        self.throughput_curves = {
            name: ResourceThroughputCurve.from_dict(curve) for name, curve in data.get("throughput_curves", {}).items()
        }

    @staticmethod
    def _stats_to_dict(stats: OpExecutionStats) -> Dict:
        return stats.to_dict()

    @staticmethod
    def _stats_from_dict(data: Dict) -> OpExecutionStats:
        return OpExecutionStats.from_dict(data)

    def update_execution_stats(self, op_name: str, stats: OpExecutionStats):
        """Update execution statistics for an operator"""
        if stats.op_name != op_name:
            raise ValueError("op_name must match stats.op_name")
        with self._lock:
            self.execution_stats[op_name] = stats
            self._fit_throughput_curve(op_name, stats)

    def record_snapshot(self, op_name: str, snapshot: ResourceSnapshot):
        """Append one observation, preserving any existing operator history."""

        with self._lock:
            stats = self.execution_stats.get(op_name)
            if stats is None:
                stats = OpExecutionStats(op_name=op_name)
                self.execution_stats[op_name] = stats
            stats.update(snapshot)
            self._fit_throughput_curve(op_name, stats)

    def list_operator_names(self):
        """Return stored operator names in deterministic order."""

        return sorted(self.execution_stats)

    def update_ocs_signature(self, op_name: str, signature: OpCostSignature):
        """Update OCS signature for an operator"""
        self.ocs_signatures[op_name] = signature

    def get_execution_stats(self, op_name: str) -> Optional[OpExecutionStats]:
        """Get execution statistics for an operator"""
        return self.execution_stats.get(op_name)

    def get_ocs_signature(self, op_name: str) -> Optional[OpCostSignature]:
        """Get OCS signature for an operator"""
        return self.ocs_signatures.get(op_name)

    def get_throughput_curve(self, op_name: str) -> Optional[ResourceThroughputCurve]:
        """Get resource-throughput curve for an operator"""
        return self.throughput_curves.get(op_name)

    def _fit_throughput_curve(self, op_name: str, stats: OpExecutionStats):
        """
        Fit resource-throughput curve from execution statistics.

        Uses online learning approach (inspired by Autothrottle).
        """
        if len(stats.snapshots) < 5:
            # Not enough data points
            return

        # Extract features and target
        batch_sizes = np.array([s.batch_size for s in stats.snapshots])
        memories = np.array([s.memory_mb for s in stats.snapshots])
        throughputs = np.array([s.throughput for s in stats.snapshots])

        # Filter out invalid data
        valid_idx = throughputs > 0
        if valid_idx.sum() < 5:
            return

        batch_sizes = batch_sizes[valid_idx]
        memories = memories[valid_idx]
        throughputs = throughputs[valid_idx]

        try:
            # Try linear model first: T = a*batch + b*mem + c
            X = np.column_stack([batch_sizes, memories, np.ones_like(batch_sizes)])
            coeffs, _, _, _ = np.linalg.lstsq(X, throughputs, rcond=None)

            # Calculate R²
            predicted = X @ coeffs
            ss_res = np.sum((throughputs - predicted) ** 2)
            ss_tot = np.sum((throughputs - np.mean(throughputs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            if r_squared >= 0.98 or len(np.unique(batch_sizes)) < 3:
                curve = ResourceThroughputCurve(
                    op_name=op_name,
                    coefficients={
                        "batch_coef": float(coeffs[0]),
                        "memory_coef": float(coeffs[1]),
                        "intercept": float(coeffs[2]),
                    },
                    model_type="linear",
                    r_squared=float(r_squared),
                    n_samples=len(batch_sizes),
                )
            else:
                unique_batches = sorted(set(int(value) for value in batch_sizes))
                median_throughputs = [float(np.median(throughputs[batch_sizes == value])) for value in unique_batches]
                fitted = np.interp(batch_sizes, unique_batches, median_throughputs)
                piecewise_residual = np.sum((throughputs - fitted) ** 2)
                piecewise_r_squared = 1 - (piecewise_residual / ss_tot) if ss_tot > 0 else 0
                curve = ResourceThroughputCurve(
                    op_name=op_name,
                    coefficients={
                        "batch_sizes": unique_batches,
                        "throughputs": median_throughputs,
                    },
                    model_type="piecewise",
                    r_squared=float(piecewise_r_squared),
                    n_samples=len(batch_sizes),
                )

            self.throughput_curves[op_name] = curve

        except (ValueError, np.linalg.LinAlgError) as error:
            self.load_errors.append(f"curve fit for {op_name}: {error}")

    def predict_memory_for_batch(self, op_name: str, batch_size: int) -> Optional[float]:
        """
        Predict memory usage for a given batch size.

        Based on historical data with online learning.
        """
        stats = self.execution_stats.get(op_name)
        if not stats or len(stats.snapshots) < 3:
            return None

        # Simple linear regression: memory = a * batch_size + b
        valid_snapshots = [
            snapshot for snapshot in stats.snapshots if snapshot.batch_size > 0 and snapshot.memory_mb >= 0
        ]
        if len(valid_snapshots) < 3:
            return None
        batch_sizes = np.array([s.batch_size for s in valid_snapshots])
        memories = np.array([s.memory_mb for s in valid_snapshots])

        try:
            slope, intercept = _fit_robust_line(batch_sizes, memories)
            predicted = slope * batch_size + intercept
            return float(predicted)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            # Fall back to average
            return float(np.mean(memories))

    def get_safe_batch_size(self, op_name: str, available_memory_mb: float, safety_margin: float = 0.9) -> int:
        """
        Recommend safe batch size given available memory.

        Args:
            op_name: Operator name
            available_memory_mb: Available memory in MB
            safety_margin: Use only this fraction of available memory (default 90%)

        Returns:
            Recommended batch size
        """
        stats = self.execution_stats.get(op_name)
        if not stats or len(stats.snapshots) < 3:
            return 1  # Conservative default

        # Fit memory = slope * batch + intercept so fixed operator/context
        # memory is not incorrectly charged once per sample.
        valid = [snapshot for snapshot in stats.snapshots if snapshot.batch_size > 0 and snapshot.memory_mb > 0]
        if len(valid) < 3:
            return 1
        batch_sizes = np.array([s.batch_size for s in valid])
        memories = np.array([s.memory_mb for s in valid])

        target_memory = available_memory_mb * safety_margin
        try:
            slope, intercept = _fit_robust_line(batch_sizes, memories)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return 1
        if not np.isfinite(slope) or not np.isfinite(intercept):
            return 1
        if slope <= 0:
            safe_observations = batch_sizes[memories <= target_memory]
            return max(1, int(np.max(safe_observations))) if len(safe_observations) else 1
        safe_batch = int((target_memory - intercept) / slope)

        return max(1, safe_batch)

    def export_report(self, output_file: str):
        """Export profiling report as markdown"""
        lines = ["# ElasticJuicer Profiling Report\n"]

        lines.append("## Operator Execution Statistics\n")
        for op_name, stats in sorted(self.execution_stats.items()):
            lines.append(f"### {op_name}\n")
            lines.append(f"- Total Samples: {stats.total_samples}")
            lines.append(f"- Total Batches: {stats.total_batches}")
            lines.append(f"- Avg Latency: {stats.avg_latency_ms:.2f} ms")
            lines.append(f"- P95 Latency: {stats.p95_latency_ms:.2f} ms")
            lines.append(f"- Avg Throughput: {stats.avg_throughput:.2f} samples/s")
            lines.append(f"- Peak Memory: {stats.peak_memory_mb:.2f} MB")
            if stats.peak_gpu_memory_mb:
                lines.append(f"- Peak GPU Memory: {stats.peak_gpu_memory_mb:.2f} MB")
            lines.append("")

        lines.append("\n## OCS Signatures\n")
        for op_name, sig in sorted(self.ocs_signatures.items()):
            lines.append(f"### {op_name}")
            lines.append(f"- Type: {sig.op_type}")
            lines.append(f"- Memory Locality: {sig.memory_locality.value}")
            lines.append(f"- Transfer Cost: {sig.transfer_cost.value}")
            lines.append(f"- Failure Cost: {sig.failure_cost.value}")
            lines.append(f"- State Free: {sig.state_free}")
            lines.append("")

        with open(output_file, "w") as f:
            f.writelines(line + "\n" for line in lines)
