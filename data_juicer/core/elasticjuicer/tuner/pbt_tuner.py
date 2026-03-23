"""
OFFLINE Phase: Ray Tune Population Based Training (PBT)
for hyperparameter optimization of ElasticJuicer scheduling parameters.

Tunes:
  - PID controller params (kp, ki, kd)
  - Safety buffers (safety_buffer_mb, target_memory_utilization)
  - Predictor params (predictor_window_size, predictor_confidence_level)
  - Tower allocation weights (per-stage resource proportions)
Output: base_config.yaml (a SchedulerConfig serialized to YAML)

Usage:
    from data_juicer.core.elasticjuicer.tuner import PBTTuner
    
    config = PBTTunerConfig(
        stage_names=["filter", "mapper", "deduplicator"],
        num_samples=8,
        max_iterations=50,
    )
    tuner = PBTTuner(config)
    best_config = tuner.tune()
    tuner.export_config(best_config, "base_config.yaml")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import random
import numpy as np

# Graceful handling of optional Ray dependency
try:
    import ray
    from ray import tune
    from ray.tune.schedulers import PopulationBasedTraining
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None
    tune = None
    PopulationBasedTraining = None

from ..scheduler.scheduler_config import SchedulerConfig
from ..scheduler.micro_scheduler import MicroScheduler, BatchSizeController


@dataclass
class PBTTunerConfig:
    """Configuration for PBT-based hyperparameter tuning.
    
    Attributes:
        num_samples: Number of PBT population members (parallel trials).
        max_iterations: Maximum training iterations per trial.
        perturbation_interval: How often PBT perturbs hyperparameters.
        metric: Metric to optimize (e.g., "throughput", "score").
        mode: Optimization mode - "max" to maximize, "min" to minimize.
        stage_names: Operator stage names to tune allocation weights for.
        resources_per_trial: Resources allocated per trial (cpu, gpu).
        grace_period: Minimum iterations before stopping poor trials.
    """
    num_samples: int = 8
    max_iterations: int = 50
    perturbation_interval: int = 5
    metric: str = "throughput"
    mode: str = "max"
    stage_names: List[str] = field(default_factory=list)
    resources_per_trial: Dict[str, float] = field(
        default_factory=lambda: {"cpu": 2, "gpu": 0}
    )
    grace_period: int = 5


class PBTTuner:
    """
    Population Based Training (PBT) tuner for ElasticJuicer scheduling parameters.
    
    This class implements OFFLINE hyperparameter optimization using Ray Tune's PBT
    scheduler. It tunes PID controller parameters, memory safety buffers, predictor
    settings, and per-stage resource allocation weights.
    
    The tuning process simulates batch processing with the given configuration and
    measures throughput and OOM rates to find optimal parameters.
    
    Attributes:
        config: PBTTunerConfig instance with tuning settings.
        simulation_fn: Callable that simulates execution and returns metrics.
    
    Example:
        >>> tuner_config = PBTTunerConfig(
        ...     stage_names=["filter", "mapper"],
        ...     num_samples=4,
        ...     max_iterations=20,
        ... )
        >>> tuner = PBTTuner(tuner_config)
        >>> best_config = tuner.tune()
        >>> tuner.export_config(best_config, "base_config.yaml")
    """
    
    def __init__(
        self,
        config: PBTTunerConfig,
        simulation_fn: Optional[Callable[[SchedulerConfig], Dict[str, float]]] = None,
    ):
        """
        Initialize the PBT tuner.
        
        Args:
            config: PBTTunerConfig with tuning parameters.
            simulation_fn: Optional callable that takes a SchedulerConfig and returns
                a dict with "throughput" and "oom_rate" keys. If None, uses default
                simulation that creates a MicroScheduler and simulates batch processing.
        
        Raises:
            ImportError: If Ray is not installed and tune() is called.
        """
        self.config = config
        self.simulation_fn = simulation_fn or self._default_simulation
    
    def _get_search_space(self) -> Dict[str, Any]:
        """
        Get the Ray Tune search space for hyperparameters.
        
        Returns:
            Dictionary mapping parameter names to Ray Tune search distributions.
            
        Raises:
            ImportError: If Ray Tune is not available.
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray Tune is required for PBT tuning. "
                "Install it with: pip install 'ray[tune]'"
            )
        
        search_space = {
            # PID controller parameters
            "pid_kp": tune.uniform(0.1, 2.0),
            "pid_ki": tune.uniform(0.01, 0.2),
            "pid_kd": tune.uniform(0.01, 0.5),
            
            # Safety and memory parameters
            "safety_buffer_mb": tune.uniform(256, 4096),
            "target_memory_utilization": tune.uniform(0.6, 0.95),
            
            # Predictor parameters
            "predictor_window_size": tune.choice([50, 100, 200, 500]),
            "predictor_confidence_level": tune.uniform(0.9, 0.99),
        }
        
        # Add per-stage allocation weights
        for stage_name in self.config.stage_names:
            search_space[f"weight_{stage_name}"] = tune.uniform(0.1, 5.0)
        
        return search_space
    
    def _default_simulation(self, scheduler_config: SchedulerConfig) -> Dict[str, float]:
        """
        Default simulation function that tests a SchedulerConfig.
        
        Creates a MicroScheduler with the given PID parameters and simulates
        N iterations of batch processing with random memory fluctuations.
        Measures simulated throughput and OOM events.
        
        Args:
            scheduler_config: Configuration to evaluate.
            
        Returns:
            Dictionary with "throughput" (samples/sec) and "oom_rate" (0.0-1.0).
        """
        # Create MicroScheduler with config parameters
        micro_scheduler = MicroScheduler(
            memory_predictor=None,
            initial_batch_size=scheduler_config.initial_batch_size,
            min_batch_size=scheduler_config.min_batch_size,
            max_batch_size=scheduler_config.max_batch_size,
            target_memory_utilization=scheduler_config.target_memory_utilization,
            safety_buffer_mb=scheduler_config.safety_buffer_mb,
            use_gpu=scheduler_config.use_gpu_memory,
            enable_auto_adjust=scheduler_config.enable_auto_adjust,
        )
        
        # Override PID parameters in the controller
        micro_scheduler.controller.pid.kp = scheduler_config.pid_kp
        micro_scheduler.controller.pid.ki = scheduler_config.pid_ki
        micro_scheduler.controller.pid.kd = scheduler_config.pid_kd
        
        # Simulation parameters
        num_iterations = 100
        total_samples_processed = 0
        oom_events = 0
        
        # Simulated available memory (starts high, fluctuates)
        base_memory_mb = 8000.0  # 8GB base
        
        for i in range(num_iterations):
            # Get current batch size from scheduler
            batch_size = micro_scheduler.controller.current_batch_size
            
            # Simulate memory usage per sample (varies randomly)
            memory_per_sample = np.random.uniform(5.0, 20.0)  # 5-20 MB per sample
            
            # Add random memory fluctuation (simulates other processes)
            memory_fluctuation = np.random.uniform(-500, 500)
            
            # Calculate simulated memory state
            simulated_used_memory = batch_size * memory_per_sample + memory_fluctuation
            simulated_available = base_memory_mb - simulated_used_memory
            
            # Check for simulated OOM
            if simulated_available < scheduler_config.safety_buffer_mb * 0.5:
                oom_events += 1
                # Report OOM to scheduler
                micro_scheduler.controller.report_oom(batch_size, simulated_used_memory)
                # Penalize throughput for OOM
                total_samples_processed += batch_size // 4
            else:
                # Successful batch
                total_samples_processed += batch_size
                
                # Update scheduler (simulates feedback loop)
                micro_scheduler.controller.update_batch_size(
                    predicted_memory_per_sample=memory_per_sample
                )
        
        # Calculate metrics
        throughput = total_samples_processed / num_iterations  # samples per iteration
        oom_rate = oom_events / num_iterations
        
        return {
            "throughput": throughput,
            "oom_rate": oom_rate,
        }
    
    def _trial_config_to_scheduler_config(self, trial_config: Dict) -> SchedulerConfig:
        """
        Convert Ray Tune trial config dict to SchedulerConfig.
        
        Args:
            trial_config: Dictionary of hyperparameters from Ray Tune.
            
        Returns:
            SchedulerConfig instance with the trial's hyperparameters.
        """
        # Extract tower allocation weights from weight_{stage} keys
        tower_weights = {}
        for key, value in trial_config.items():
            if key.startswith("weight_"):
                stage_name = key[7:]  # Remove "weight_" prefix
                tower_weights[stage_name] = value
        
        return SchedulerConfig(
            # PID parameters
            pid_kp=trial_config.get("pid_kp", 0.5),
            pid_ki=trial_config.get("pid_ki", 0.05),
            pid_kd=trial_config.get("pid_kd", 0.1),
            
            # Safety parameters
            safety_buffer_mb=trial_config.get("safety_buffer_mb", 1000.0),
            target_memory_utilization=trial_config.get("target_memory_utilization", 0.85),
            
            # Predictor parameters
            predictor_window_size=int(trial_config.get("predictor_window_size", 100)),
            predictor_confidence_level=trial_config.get("predictor_confidence_level", 0.95),
            
            # Tower allocation weights
            tower_allocation_weights=tower_weights if tower_weights else None,
        )
    
    def _trainable(self, trial_config: Dict) -> None:
        """
        Ray Tune trainable function.
        
        Converts trial config to SchedulerConfig, runs simulation,
        and reports metrics to Ray Tune.
        
        Args:
            trial_config: Dictionary of hyperparameters from Ray Tune.
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray Tune is required for PBT tuning. "
                "Install it with: pip install 'ray[tune]'"
            )
        
        # Convert trial config to SchedulerConfig
        scheduler_config = self._trial_config_to_scheduler_config(trial_config)
        
        # Run simulation
        results = self.simulation_fn(scheduler_config)
        
        throughput = results.get("throughput", 0.0)
        oom_rate = results.get("oom_rate", 1.0)
        
        # Calculate composite score (higher is better)
        # Penalize OOM events heavily
        score = throughput * (1.0 - oom_rate)
        
        # Report metrics to Ray Tune
        tune.report(
            throughput=throughput,
            oom_rate=oom_rate,
            score=score,
        )
    
    def tune(self) -> SchedulerConfig:
        """
        Run PBT hyperparameter tuning.
        
        Sets up Ray Tune with PBT scheduler, runs the tuning process,
        and returns the best configuration found.
        
        Returns:
            SchedulerConfig with the best hyperparameters found.
            
        Raises:
            ImportError: If Ray Tune is not installed.
            RuntimeError: If tuning fails or no results are found.
        """
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray Tune is required for PBT tuning. "
                "Install it with: pip install 'ray[tune]'"
            )
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        # Get search space
        search_space = self._get_search_space()
        
        # Define perturbation bounds for PBT
        hyperparam_mutations = {
            "pid_kp": tune.uniform(0.1, 2.0),
            "pid_ki": tune.uniform(0.01, 0.2),
            "pid_kd": tune.uniform(0.01, 0.5),
            "safety_buffer_mb": tune.uniform(256, 4096),
            "target_memory_utilization": tune.uniform(0.6, 0.95),
            "predictor_window_size": [50, 100, 200, 500],
            "predictor_confidence_level": tune.uniform(0.9, 0.99),
        }
        
        # Add stage weight mutations
        for stage_name in self.config.stage_names:
            hyperparam_mutations[f"weight_{stage_name}"] = tune.uniform(0.1, 5.0)
        
        # Create PBT scheduler
        pbt_scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=self.config.perturbation_interval,
            hyperparam_mutations=hyperparam_mutations,
            quantile_fraction=0.25,  # Top 25% survive
            resample_probability=0.25,  # 25% chance to resample instead of perturb
        )
        
        # Run tuning
        analysis = tune.run(
            self._trainable,
            config=search_space,
            metric=self.config.metric,
            mode=self.config.mode,
            num_samples=self.config.num_samples,
            scheduler=pbt_scheduler,
            resources_per_trial=self.config.resources_per_trial,
            stop={"training_iteration": self.config.max_iterations},
            verbose=1,
            raise_on_failed_trial=False,
        )
        
        # Get best trial
        best_trial = analysis.get_best_trial(
            metric=self.config.metric,
            mode=self.config.mode,
        )
        
        if best_trial is None:
            raise RuntimeError(
                "PBT tuning failed: no successful trials found. "
                "Check simulation function and resource availability."
            )
        
        # Convert best config to SchedulerConfig
        best_config = self._trial_config_to_scheduler_config(best_trial.config)
        
        return best_config
    
    def export_config(self, config: SchedulerConfig, path: str = "base_config.yaml") -> None:
        """
        Export a SchedulerConfig to a YAML file.
        
        Args:
            config: SchedulerConfig to export.
            path: Output file path (default: "base_config.yaml").
        """
        config.to_yaml(path)
    
    @staticmethod
    def load_config(path: str) -> SchedulerConfig:
        """
        Load a SchedulerConfig from a YAML file.
        
        Args:
            path: Path to the YAML configuration file.
            
        Returns:
            SchedulerConfig instance loaded from the file.
        """
        return SchedulerConfig.from_yaml(path)
