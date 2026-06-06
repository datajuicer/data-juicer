"""
Profiling Store

Persistent storage and query interface for:
- Resource-throughput curves
- OCS signatures
- Historical performance data

Supports online learning and model updating.
"""

import json
import logging
import pickle
import time as _time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import curve_fit

from .resource_monitor import OpExecutionStats, ResourceSnapshot
from .ocs_annotator import OpCostSignature

logger = logging.getLogger(__name__)


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
    coefficients: Dict[str, float]
    # Model type: 'linear', 'polynomial', 'power'
    model_type: str = 'linear'
    # Goodness of fit
    r_squared: float = 0.0
    # Sample count used for fitting
    n_samples: int = 0
    
    def predict_throughput(self, batch_size: int, memory_mb: float) -> float:
        """Predict throughput given batch size and memory"""
        if self.model_type == 'linear':
            # T = a * batch_size + b * memory + c
            a = self.coefficients.get('batch_coef', 0)
            b = self.coefficients.get('memory_coef', 0)
            c = self.coefficients.get('intercept', 0)
            return max(0, a * batch_size + b * memory_mb + c)
        
        elif self.model_type == 'power':
            # T = a * batch_size^b
            a = self.coefficients.get('scale', 1)
            b = self.coefficients.get('power', 1)
            return a * (batch_size ** b)
        
        return 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResourceThroughputCurve':
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

    SCHEMA_VERSION = "1.0"

    def __init__(self, storage_dir: str = "./elastic_juicer_profiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.execution_stats: Dict[str, OpExecutionStats] = {}
        self.ocs_signatures: Dict[str, OpCostSignature] = {}
        self.throughput_curves: Dict[str, ResourceThroughputCurve] = {}
        
        # Load existing data
        self._load_all()
    
    def _load_all(self):
        """Load all stored profiles, checking schema version."""
        # Load execution stats (versioned pickle wrapper)
        stats_file = self.storage_dir / "execution_stats.pkl"
        if stats_file.exists():
            try:
                with open(stats_file, 'rb') as f:
                    raw = pickle.load(f)
                if isinstance(raw, dict) and "meta" in raw and "stats" in raw:
                    file_version = raw["meta"].get("schema_version")
                    if file_version != self.SCHEMA_VERSION:
                        logger.warning(
                            "ProfilingStore schema version mismatch: "
                            "file=%s, code=%s. Skipping load.",
                            file_version, self.SCHEMA_VERSION,
                        )
                    else:
                        self.execution_stats = raw["stats"]
                elif isinstance(raw, dict):
                    # Legacy format (pre-versioning): dict[str, OpExecutionStats]
                    logger.warning(
                        "ProfilingStore data has no schema version (legacy). "
                        "Loading anyway; will be re-saved with version on next save_all()."
                    )
                    self.execution_stats = raw
                else:
                    logger.warning(
                        "ProfilingStore data has unexpected type %s. Skipping.",
                        type(raw).__name__,
                    )
            except Exception as e:
                logger.warning("Failed to load execution_stats.pkl: %s", e)
        
        # Load OCS signatures
        ocs_file = self.storage_dir / "ocs_signatures.json"
        if ocs_file.exists():
            with open(ocs_file, 'r') as f:
                data = json.load(f)
                self.ocs_signatures = {
                    name: OpCostSignature.from_dict(sig)
                    for name, sig in data.items()
                }
        
        # Load throughput curves
        curves_file = self.storage_dir / "throughput_curves.json"
        if curves_file.exists():
            with open(curves_file, 'r') as f:
                data = json.load(f)
                self.throughput_curves = {
                    name: ResourceThroughputCurve.from_dict(curve)
                    for name, curve in data.items()
                }
    
    def save_all(self):
        """Persist all profiles to disk (versioned wrapper)."""
        # Save execution stats with schema version
        stats_file = self.storage_dir / "execution_stats.pkl"
        wrapped = {
            "meta": {
                "schema_version": self.SCHEMA_VERSION,
                "saved_at": _time.time(),
            },
            "stats": self.execution_stats,
        }
        with open(stats_file, 'wb') as f:
            pickle.dump(wrapped, f)
        logger.info(
            "ProfilingStore saved %d entries (schema=%s)",
            len(self.execution_stats), self.SCHEMA_VERSION,
        )
        
        # Save OCS signatures
        ocs_file = self.storage_dir / "ocs_signatures.json"
        with open(ocs_file, 'w') as f:
            data = {
                name: sig.to_dict()
                for name, sig in self.ocs_signatures.items()
            }
            json.dump(data, f, indent=2)
        
        # Save throughput curves
        curves_file = self.storage_dir / "throughput_curves.json"
        with open(curves_file, 'w') as f:
            data = {
                name: curve.to_dict()
                for name, curve in self.throughput_curves.items()
            }
            json.dump(data, f, indent=2)
    
    def update_execution_stats(self, op_name: str, stats: OpExecutionStats):
        """Update execution statistics for an operator"""
        self.execution_stats[op_name] = stats
        self._fit_throughput_curve(op_name, stats)
    
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
            coeffs, residuals, _, _ = np.linalg.lstsq(X, throughputs, rcond=None)
            
            # Calculate R²
            ss_res = residuals[0] if len(residuals) > 0 else 0
            ss_tot = np.sum((throughputs - np.mean(throughputs)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            curve = ResourceThroughputCurve(
                op_name=op_name,
                coefficients={
                    'batch_coef': float(coeffs[0]),
                    'memory_coef': float(coeffs[1]),
                    'intercept': float(coeffs[2]),
                },
                model_type='linear',
                r_squared=float(r_squared),
                n_samples=len(batch_sizes),
            )
            
            self.throughput_curves[op_name] = curve
            
        except Exception as e:
            # Fitting failed, use simple mean
            pass
    
    def predict_memory_for_batch(self, op_name: str, batch_size: int) -> Optional[float]:
        """
        Predict memory usage for a given batch size.
        
        Based on historical data with online learning.
        """
        stats = self.execution_stats.get(op_name)
        if not stats or len(stats.snapshots) < 3:
            return None
        
        # Simple linear regression: memory = a * batch_size + b
        batch_sizes = np.array([s.batch_size for s in stats.snapshots])
        memories = np.array([s.memory_mb for s in stats.snapshots])
        
        try:
            # Fit linear model
            coeffs = np.polyfit(batch_sizes, memories, deg=1)
            predicted = coeffs[0] * batch_size + coeffs[1]
            return float(predicted)
        except Exception:
            # Fall back to average
            return float(np.mean(memories))
    
    def get_safe_batch_size(self, op_name: str, available_memory_mb: float, 
                           safety_margin: float = 0.9) -> int:
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
        
        # Find batch sizes and their memory usage
        batch_sizes = np.array([s.batch_size for s in stats.snapshots])
        memories = np.array([s.memory_mb for s in stats.snapshots])
        
        # Calculate memory per sample
        mem_per_sample = memories / batch_sizes
        avg_mem_per_sample = np.median(mem_per_sample)  # Use median for robustness
        
        # Calculate safe batch size
        target_memory = available_memory_mb * safety_margin
        safe_batch = int(target_memory / avg_mem_per_sample)
        
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
        
        with open(output_file, 'w') as f:
            f.writelines(line + '\n' for line in lines)
