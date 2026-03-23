"""
ElasticJuicer: Top-level orchestrator for the complete ElasticJuicer flow.

The complete flow consists of two phases:
  OFFLINE: Ray Tune PBT to find optimal scheduling parameters
           -> Outputs base_config.yaml
  ONLINE:  Adaptive Tower (Macro) + Captain (Micro) scheduling
           -> Tower runs rebalance loop, Captains run PID+Pred micro-scheduling
           -> Metrics feedback from Captains to Tower

Usage:
    # Complete flow
    ej = ElasticJuicer()
    ej.run(operators=[op1, op2, op3])

    # Or step by step
    ej = ElasticJuicer()
    config = ej.run_offline_tuning(stage_names=["filter", "mapper"])
    ej.run_online(config=config, operators=[op1, op2, op3])
    
    # Or skip offline, use existing config
    ej = ElasticJuicer()
    config = SchedulerConfig.from_yaml("base_config.yaml")
    ej.run_online(config=config, operators=[op1, op2])
    
    # Graceful shutdown
    ej.stop()
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from .scheduler.scheduler_config import SchedulerConfig
from .scheduler.tower import Tower, ClusterState
from .scheduler.captain import Captain, CaptainConfig, CaptainPool

logger = logging.getLogger(__name__)


def _get_default_cluster_state() -> ClusterState:
    """Create a default ClusterState based on current system resources.
    
    Returns:
        ClusterState with detected or default resource values.
    """
    try:
        import psutil
        
        cpu_count = psutil.cpu_count(logical=True) or 4
        memory_info = psutil.virtual_memory()
        total_memory_mb = memory_info.total / (1024 * 1024)
        available_memory_mb = memory_info.available / (1024 * 1024)
        
        return ClusterState(
            total_cpu_cores=cpu_count,
            total_memory_mb=total_memory_mb,
            total_gpu_count=0,  # GPU detection requires additional libraries
            available_cpu_cores=float(cpu_count),
            available_memory_mb=available_memory_mb,
            available_gpus=0.0,
        )
    except ImportError:
        # Fallback to sensible defaults if psutil not available
        return ClusterState(
            total_cpu_cores=4,
            total_memory_mb=8192.0,
            total_gpu_count=0,
            available_cpu_cores=4.0,
            available_memory_mb=6144.0,
            available_gpus=0.0,
        )


class ElasticJuicer:
    """
    Top-level orchestrator for the complete ElasticJuicer flow.
    
    ElasticJuicer combines OFFLINE (PBT hyperparameter tuning) and ONLINE 
    (adaptive bi-level scheduling) phases to provide automatic, 
    resource-efficient data processing.
    
    The bi-level scheduling architecture:
    - Tower (macro-scheduler): Global resource allocation and rebalancing
    - Captains (micro-schedulers): Per-operator batch size control with PID
    
    Attributes:
        config: Current SchedulerConfig instance.
        tower: Tower macro-scheduler instance (created during run_online).
        captain_pool: CaptainPool managing per-operator Captains.
        is_running: Whether the online phase is currently active.
    
    Example:
        >>> # Complete flow with automatic tuning
        >>> with ElasticJuicer() as ej:
        ...     ej.run(operators=[filter_op, mapper_op, dedup_op])
        
        >>> # Skip tuning, use existing config
        >>> ej = ElasticJuicer(config_path="base_config.yaml")
        >>> ej.run_online(operators=[filter_op, mapper_op])
        >>> ej.stop()
    """
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        config_path: Optional[str] = None,
        cluster_state: Optional[ClusterState] = None,
    ):
        """Initialize ElasticJuicer.
        
        Args:
            config: Pre-existing SchedulerConfig. If None, default config used.
            config_path: Path to load config from YAML. Takes precedence over config.
            cluster_state: Optional ClusterState for Tower. If None, auto-detected.
        """
        # Load config from YAML if path provided, otherwise use provided config or default
        if config_path is not None:
            self._config = SchedulerConfig.from_yaml(config_path)
            logger.info(f"Loaded config from {config_path}")
        elif config is not None:
            self._config = config
        else:
            self._config = SchedulerConfig()
        
        # Cluster state for Tower initialization
        self._cluster_state = cluster_state
        
        # Runtime components (created during run_online)
        self._tower: Optional[Tower] = None
        self._captain_pool: Optional[CaptainPool] = None
        self._captain_ids: Dict[str, str] = {}  # stage_name -> captain_id mapping
        
        # State tracking
        self._is_running: bool = False
    
    @property
    def config(self) -> SchedulerConfig:
        """Get the current SchedulerConfig."""
        return self._config
    
    @property
    def tower(self) -> Optional[Tower]:
        """Get the Tower macro-scheduler instance (None if not started)."""
        return self._tower
    
    @property
    def captain_pool(self) -> Optional[CaptainPool]:
        """Get the CaptainPool managing all Captains (None if not started)."""
        return self._captain_pool
    
    @property
    def is_running(self) -> bool:
        """Check if the online phase is currently running."""
        return self._is_running
    
    def run_offline_tuning(
        self,
        stage_names: Optional[List[str]] = None,
        simulation_fn: Optional[Callable[[SchedulerConfig], Dict[str, float]]] = None,
        num_samples: int = 8,
        max_iterations: int = 50,
        export_path: str = "base_config.yaml",
    ) -> SchedulerConfig:
        """OFFLINE Phase: Run Ray Tune PBT to find optimal params.
        
        Uses Population Based Training to optimize:
        - PID controller parameters (kp, ki, kd)
        - Memory safety buffers
        - Predictor settings
        - Per-stage resource allocation weights
        
        Args:
            stage_names: Operator stage names to tune allocation weights for.
                If None, an empty list is used (no per-stage weights).
            simulation_fn: Custom simulation function. If None, uses default
                simulation that tests batch processing with random memory.
            num_samples: PBT population size (number of parallel trials).
            max_iterations: Maximum training iterations per trial.
            export_path: Path to save the tuned config YAML.
        
        Returns:
            Optimized SchedulerConfig.
        
        Raises:
            ImportError: If Ray Tune is not installed.
            RuntimeError: If tuning fails.
        """
        # Import locally to avoid errors when Ray is not installed
        from .tuner.pbt_tuner import PBTTuner, PBTTunerConfig
        
        tuner_config = PBTTunerConfig(
            num_samples=num_samples,
            max_iterations=max_iterations,
            stage_names=stage_names or [],
        )
        
        tuner = PBTTuner(config=tuner_config, simulation_fn=simulation_fn)
        
        logger.info(
            f"Starting offline PBT tuning with {num_samples} samples, "
            f"{max_iterations} iterations"
        )
        
        self._config = tuner.tune()
        tuner.export_config(self._config, export_path)
        
        logger.info(f"Offline tuning complete. Config saved to {export_path}")
        return self._config
    
    def run_online(
        self,
        operators: List[Any],
        config: Optional[SchedulerConfig] = None,
    ) -> None:
        """ONLINE Phase: Start Adaptive Tower + Captains.
        
        Sets up the bi-level scheduling hierarchy:
        1. Creates Tower (macro-scheduler) with rebalance loop
        2. Creates Captains (micro-schedulers) for each operator
        3. Registers Captains with Tower
        4. Starts Tower rebalance loop
        
        Args:
            operators: List of operator objects. Each should have a 'name' 
                attribute (or str representation will be used).
            config: Optional config override. If None, uses self._config.
        
        Raises:
            ValueError: If operators list is empty.
            RuntimeError: If already running.
        """
        # Validate inputs
        if not operators:
            raise ValueError("operators list cannot be empty")
        
        if self._is_running:
            raise RuntimeError(
                "ElasticJuicer is already running. Call stop() first."
            )
        
        # Update config if provided
        if config is not None:
            self._config = config
        
        # Get or create cluster state
        cluster_state = self._cluster_state or _get_default_cluster_state()
        
        # Create Tower macro-scheduler
        self._tower = Tower(
            cluster_state=cluster_state,
            target_queue_depth=100,
            sla_latency_ms=5000.0,
            update_interval_sec=self._config.rebalance_interval_sec,
            config=self._config,
        )
        
        # Create CaptainPool
        self._captain_pool = CaptainPool()
        self._captain_ids.clear()
        
        # Register each operator as a stage with its own Captain
        for op in operators:
            op_name = getattr(op, 'name', None)
            if op_name is None:
                op_name = str(op)
            
            # Register stage in Tower (returns captain_id)
            captain_id = self._tower.register_stage(
                stage_name=op_name,
                initial_parallelism=1,
            )
            self._captain_ids[op_name] = captain_id
            
            # Create Captain config
            captain_config = CaptainConfig(
                stage_name=op_name,
                initial_batch_size=self._config.initial_batch_size,
                enable_micro_scheduler=self._config.enable_auto_adjust,
                enable_prediction=self._config.enable_prediction,
            )
            
            # Create Captain via pool (adds to pool automatically)
            captain = self._captain_pool.add_captain(captain_config)
            
            # Register Captain with Tower for metrics collection + quota broadcast
            self._tower.register_captain(captain_id, captain)
        
        # Start Tower rebalance loop
        self._tower.start()
        self._is_running = True
        
        logger.info(
            f"Online phase started with {len(operators)} operators: "
            f"{[getattr(op, 'name', str(op)) for op in operators]}"
        )
    
    def run(
        self,
        operators: List[Any],
        skip_offline: bool = False,
        config_path: Optional[str] = None,
        **offline_kwargs,
    ) -> None:
        """Complete flow: OFFLINE -> ONLINE.
        
        Runs both phases in sequence:
        1. OFFLINE: PBT tuning (unless skip_offline=True)
        2. ONLINE: Start adaptive scheduling
        
        Args:
            operators: List of operator objects.
            skip_offline: If True, skip PBT tuning and use existing config.
            config_path: Path to load/save config. If skip_offline is False,
                this is used as export_path. If skip_offline is True, this is
                used to load an existing config.
            **offline_kwargs: Additional kwargs passed to run_offline_tuning().
                Supported kwargs: simulation_fn, num_samples, max_iterations.
        
        Raises:
            ValueError: If operators list is empty.
            ImportError: If Ray not installed and skip_offline is False.
        """
        if not operators:
            raise ValueError("operators list cannot be empty")
        
        if not skip_offline:
            # Extract stage names from operators
            stage_names = [
                getattr(op, 'name', str(op)) for op in operators
            ]
            export_path = config_path or "base_config.yaml"
            
            self.run_offline_tuning(
                stage_names=stage_names,
                export_path=export_path,
                **offline_kwargs,
            )
        elif config_path:
            # Load existing config
            self._config = SchedulerConfig.from_yaml(config_path)
            logger.info(f"Loaded existing config from {config_path}")
        
        # Start online phase
        self.run_online(operators=operators)
    
    def stop(self) -> None:
        """Graceful shutdown of all components.
        
        Stops the Tower rebalance loop and cleans up resources.
        Safe to call even if not running.
        """
        if self._tower is not None:
            self._tower.stop()
        
        self._is_running = False
        self._tower = None
        self._captain_pool = None
        self._captain_ids.clear()
        
        logger.info("ElasticJuicer stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status.
        
        Returns:
            Dictionary containing:
            - is_running: Whether online phase is active
            - config: String representation of current config
            - tower_stats: Global stats from Tower (if running)
            - captain_stats: Stats from all Captains (if running)
        """
        status: Dict[str, Any] = {
            "is_running": self._is_running,
            "config": str(self._config),
        }
        
        if self._tower is not None:
            status["tower_stats"] = self._tower.get_global_stats()
        
        if self._captain_pool is not None:
            status["captain_stats"] = self._captain_pool.get_all_stats()
        
        return status
    
    def get_captain(self, stage_name: str) -> Optional[Captain]:
        """Get the Captain for a specific stage.
        
        Args:
            stage_name: Name of the operator stage.
        
        Returns:
            Captain instance for the stage, or None if not found.
        """
        if self._captain_pool is None:
            return None
        return self._captain_pool.get_captain(stage_name)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters dynamically.
        
        Only updates the config object. For runtime changes to take effect,
        the Tower and Captains may need to be restarted.
        
        Args:
            **kwargs: Configuration parameters to update. See SchedulerConfig
                for available parameters.
        
        Example:
            >>> ej.update_config(target_memory_utilization=0.9, pid_kp=0.8)
        """
        from dataclasses import fields
        
        valid_fields = {f.name for f in fields(SchedulerConfig)}
        
        for key, value in kwargs.items():
            if key not in valid_fields:
                logger.warning(f"Unknown config parameter ignored: {key}")
                continue
            setattr(self._config, key, value)
        
        logger.info(f"Config updated with: {kwargs}")
    
    def save_config(self, path: str) -> None:
        """Save current configuration to a YAML file.
        
        Args:
            path: Output file path for the YAML config.
        """
        self._config.to_yaml(path)
        logger.info(f"Config saved to {path}")
    
    # Context manager support
    def __enter__(self) -> 'ElasticJuicer':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit - ensures graceful shutdown."""
        self.stop()
        return False
    
    def __repr__(self) -> str:
        """String representation of ElasticJuicer instance."""
        return (
            f"ElasticJuicer(is_running={self._is_running}, "
            f"stages={list(self._captain_ids.keys()) if self._captain_ids else []})"
        )
