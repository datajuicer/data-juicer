"""
Scheduler Configuration

Centralized configuration for micro and macro schedulers.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class SchedulerConfig:
    """Configuration for ElasticJuicer schedulers"""
    
    # Batch size control
    initial_batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 1000
    
    # Memory management
    target_memory_utilization: float = 0.85  # 85% utilization target
    safety_buffer_mb: float = 1000.0  # 1GB safety buffer
    use_gpu_memory: bool = False
    
    # PID tuning
    pid_kp: float = 0.5   # Proportional gain
    pid_ki: float = 0.05  # Integral gain
    pid_kd: float = 0.1   # Derivative gain
    
    # Auto-adjustment
    enable_auto_adjust: bool = True
    enable_prediction: bool = True
    
    # Predictor settings
    predictor_window_size: int = 100
    predictor_min_samples: int = 5
    predictor_confidence_level: float = 0.95
    
    # Safety settings
    max_batch_change_ratio: float = 0.5  # Max 50% change per adjustment
    oom_backoff_ratio: float = 0.5  # Reduce to 50% on OOM
    
    # Tower macro-scheduler settings (PBT output)
    rebalance_interval_sec: float = 5.0  # Tower macro-scheduler rebalance loop interval in seconds
    tower_allocation_weights: Optional[Dict[str, float]] = field(default=None)  # Per-stage resource allocation weights from PBT tuning
    backpressure_threshold: float = 0.9  # Memory utilization threshold above which backpressure is applied
    backpressure_slowdown_ratio: float = 0.5  # Factor to reduce throughput when backpressure is active
    
    @classmethod
    def conservative(cls) -> 'SchedulerConfig':
        """Conservative configuration (prioritizes safety)"""
        return cls(
            target_memory_utilization=0.70,
            safety_buffer_mb=2000.0,
            max_batch_change_ratio=0.25,
            rebalance_interval_sec=10.0,
            backpressure_threshold=0.8,
        )
    
    @classmethod
    def aggressive(cls) -> 'SchedulerConfig':
        """Aggressive configuration (prioritizes throughput)"""
        return cls(
            target_memory_utilization=0.95,
            safety_buffer_mb=500.0,
            max_batch_change_ratio=0.75,
            rebalance_interval_sec=2.0,
            backpressure_threshold=0.95,
        )
    
    @classmethod
    def gpu(cls) -> 'SchedulerConfig':
        """GPU-optimized configuration"""
        return cls(
            use_gpu_memory=True,
            target_memory_utilization=0.90,
            safety_buffer_mb=1024.0,  # 1GB buffer for GPU
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> 'SchedulerConfig':
        """Load config from a YAML file (the output of PBT tuning).
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            SchedulerConfig instance with values from YAML, using defaults for missing fields.
            
        Raises:
            ImportError: If PyYAML is not installed.
            FileNotFoundError: If the YAML file does not exist.
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install it with: pip install pyyaml"
            )
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        # Filter to only include valid fields for SchedulerConfig
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    def to_yaml(self, path: str) -> None:
        """Export config to YAML file.
        
        Args:
            path: Path to write the YAML configuration file.
            
        Raises:
            ImportError: If PyYAML is not installed.
        """
        if not YAML_AVAILABLE:
            raise ImportError(
                "PyYAML is required for YAML support. "
                "Install it with: pip install pyyaml"
            )
        
        data = asdict(self)
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_stage_weight(self, stage_name: str) -> float:
        """Return the allocation weight for a given stage.
        
        Args:
            stage_name: Name of the stage to get weight for.
            
        Returns:
            The allocation weight for the stage. Returns 1.0 if tower_allocation_weights
            is None or the stage is not found (equal weight).
        """
        if self.tower_allocation_weights is None:
            return 1.0
        return self.tower_allocation_weights.get(stage_name, 1.0)
