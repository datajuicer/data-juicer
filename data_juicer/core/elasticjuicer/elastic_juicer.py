"""
ElasticJuicer: Scheduling Facade for Bi-Level Adaptive Scheduling.

This module provides ElasticJuicer, a facade that manages ALL scheduling 
infrastructure: Tower (macro-scheduler), Captains (per-stage micro-schedulers), 
MetricsBridge, and Ray named actors for metrics collection and quota distribution.

Architecture:
    ElasticRayExecutor → ElasticJuicer (facade) → Tower, Captain, MetricsBridge, etc.

The facade lives on the driver (not serializable to Ray actors). Ray workers 
(AdaptiveOperator) communicate via Ray named actors (PipelineMetricsCollector, 
SharedQuotaStore) created and managed by this facade.

Usage:
    elastic = ElasticJuicer(config=scheduler_config)
    elastic.register_stages(stage_configs)
    elastic.start()
    try:
        # ... run pipeline with AdaptiveOperator ...
    finally:
        elastic.stop()
"""

import logging
import threading
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from .scheduler.scheduler_config import SchedulerConfig
from .scheduler.tower import Tower, ClusterState
from .scheduler.captain import Captain, CaptainConfig, CaptainPool

logger = logging.getLogger(__name__)


def _get_default_cluster_state() -> ClusterState:
    """Create a default ClusterState based on current system resources.
    
    Detects CPU, memory, and GPU resources available on the system.
    
    Returns:
        ClusterState with detected or default resource values.
    """
    try:
        import psutil
        
        cpu_count = psutil.cpu_count(logical=True) or 4
        memory_info = psutil.virtual_memory()
        total_memory_mb = memory_info.total / (1024 * 1024)
        available_memory_mb = memory_info.available / (1024 * 1024)
        
        # Try to detect GPUs
        gpu_count = 0
        try:
            import torch
            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            pass
        
        return ClusterState(
            total_cpu_cores=cpu_count,
            total_memory_mb=total_memory_mb,
            total_gpu_count=gpu_count,
            available_cpu_cores=float(cpu_count),
            available_memory_mb=available_memory_mb,
            available_gpus=float(gpu_count),
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


def _create_pipeline_metrics_collector():
    """Create the PipelineMetricsCollector Ray actor class.
    
    This actor aggregates metrics from all AdaptiveOperator instances,
    providing a centralized view of pipeline performance.
    """
    import ray

    @ray.remote
    class PipelineMetricsCollector:
        """Shared actor to aggregate metrics from all stages."""

        def __init__(self):
            self.stage_metrics: Dict[str, Dict[str, Any]] = {}

        def report(
            self, stage_name: str, batch_size: int, latency_ms: float, memory_mb: float
        ):
            """Report metrics for a batch processed by a stage."""
            import time
            if stage_name not in self.stage_metrics:
                self.stage_metrics[stage_name] = {
                    "batch_sizes": [],
                    "latencies": [],
                    "memories": [],
                    "total_samples": 0,
                    "total_batches": 0,
                    "start_time": time.time(),
                }
            m = self.stage_metrics[stage_name]
            m["batch_sizes"].append(batch_size)
            m["latencies"].append(latency_ms)
            m["memories"].append(memory_mb)
            m["total_samples"] += batch_size
            m["total_batches"] += 1
            m["last_time"] = time.time()

        def get_summary(self) -> Dict[str, Any]:
            """Get summary metrics for all stages."""
            result = {}
            for stage, data in self.stage_metrics.items():
                bs = data["batch_sizes"]
                lat = data["latencies"]
                mem = data["memories"]
                wall_clock_elapsed = max(
                    data.get("last_time", 0) - data.get("start_time", 0), 0.001
                )
                result[stage] = {
                    "total_samples": data["total_samples"],
                    "total_batches": data["total_batches"],
                    "min_bs": min(bs) if bs else 0,
                    "max_bs": max(bs) if bs else 0,
                    "avg_bs": sum(bs) / len(bs) if bs else 0,
                    "avg_latency_ms": sum(lat) / len(lat) if lat else 0,
                    "avg_memory_mb": sum(mem) / len(mem) if mem else 0,
                    "peak_memory_mb": max(mem) if mem else 0,
                    "wall_clock_throughput": data["total_samples"] / wall_clock_elapsed,
                }
            return result

        def reset(self):
            """Reset all metrics."""
            self.stage_metrics = {}

    return PipelineMetricsCollector


def _create_shared_quota_store():
    """Create the SharedQuotaStore Ray actor class.
    
    This actor serves as a bridge between driver-side Tower/Captains and 
    Ray actor-side AdaptiveOperators. The MetricsBridge thread updates 
    quotas here, and AdaptiveOperators read them periodically.
    """
    import ray

    @ray.remote
    class SharedQuotaStore:
        """Shared store for Tower quotas, readable by AdaptiveOperator actors."""

        def __init__(self):
            self.quotas = {}  # {stage_name: {'batch_size': int, 'backpressure': bool, ...}}

        def update_quota(self, stage_name: str, quota_dict: Dict):
            """Update quota for a stage."""
            self.quotas[stage_name] = quota_dict

        def get_quota(self, stage_name: str):
            """Get quota for a stage."""
            return self.quotas.get(stage_name, None)

        def get_all_quotas(self):
            """Get all quotas."""
            return dict(self.quotas)

    return SharedQuotaStore


class MetricsBridge(threading.Thread):
    """
    Bridge between Ray actor metrics and driver-side Tower/Captains.
    
    This thread runs on the driver and periodically:
    1. Polls PipelineMetricsCollector for per-stage metrics from actors
    2. Feeds metrics to corresponding Captains (updating their internal state)
    3. Tower's rebalance loop collects from Captains and computes quotas
    4. Reads Captain quotas and pushes to SharedQuotaStore for actors to read
    
    This bridges the gap between:
    - Actor-side: AdaptiveOperator with MicroScheduler reporting to PipelineMetricsCollector
    - Driver-side: Tower collecting from Captains and broadcasting quotas
    """

    def __init__(
        self,
        tower,
        captains: Dict[str, Any],
        metrics_collector,
        quota_store,
        interval: float = 2.0,
    ):
        """
        Initialize MetricsBridge.
        
        Args:
            tower: Tower macro-scheduler instance
            captains: Dict mapping stage_name to Captain instance
            metrics_collector: PipelineMetricsCollector Ray actor handle
            quota_store: SharedQuotaStore Ray actor handle
            interval: Bridge cycle interval in seconds
        """
        super().__init__(daemon=True, name="MetricsBridge")
        self.tower = tower
        self.captains = captains
        self.metrics_collector = metrics_collector
        self.quota_store = quota_store
        self.interval = interval
        self._running = False

    def run(self):
        """Main bridge loop."""
        self._running = True
        while self._running:
            try:
                self._bridge_cycle()
            except Exception as e:
                logger.debug(f"MetricsBridge cycle error: {e}")
            time.sleep(self.interval)

    def stop(self):
        """Stop the bridge thread."""
        self._running = False

    def _bridge_cycle(self):
        """Execute one bridge cycle."""
        import ray

        # 1. Get metrics from PipelineMetricsCollector (actor-side metrics)
        try:
            summary = ray.get(self.metrics_collector.get_summary.remote())
        except Exception:
            return

        # 2. Feed actor metrics to Captains to update their internal state
        for stage_name, captain in self.captains.items():
            stage_data = summary.get(stage_name, {})
            if stage_data:
                # Update captain's internal metrics tracking fields
                total_batches = stage_data.get("total_batches", 1)
                avg_latency_ms = stage_data.get("avg_latency_ms", 0)
                total_samples = stage_data.get("total_samples", 0)
                
                # Calculate throughput: use wall-clock throughput if available
                throughput = stage_data.get("wall_clock_throughput", 0)
                if throughput <= 0:
                    # Fallback to old formula
                    time_sec = (avg_latency_ms * total_batches) / 1000.0 if avg_latency_ms > 0 else 1.0
                    throughput = total_samples / max(time_sec, 0.001)
                
                # Update captain's internal metrics for Tower to collect
                captain._recent_throughput = throughput
                captain._recent_latency_ms = avg_latency_ms
                captain.metrics.throughput = throughput
                captain.metrics.avg_latency_ms = avg_latency_ms

                # Update memory utilization on captain
                avg_memory_mb = stage_data.get('avg_memory_mb', 0.0)
                peak_memory_mb = stage_data.get('peak_memory_mb', 0.0)
                if hasattr(captain, '_current_memory_util'):
                    try:
                        import psutil
                        total_mem_mb = psutil.virtual_memory().total / (1024 * 1024)
                        captain._current_memory_util = (peak_memory_mb / total_mem_mb * 100.0) if total_mem_mb > 0 else 0.0
                    except Exception:
                        pass
                if hasattr(captain, 'metrics') and hasattr(captain.metrics, 'memory_utilization'):
                    captain.metrics.memory_utilization = captain._current_memory_util

        # 3. Push quota updates from Captains to SharedQuotaStore
        for stage_name, captain in self.captains.items():
            try:
                # Get captain's current state (batch size, backpressure)
                current_batch_size = (
                    captain.micro_scheduler.controller.current_batch_size
                    if captain.micro_scheduler
                    else captain.config.initial_batch_size
                )
                
                quota_dict = {
                    "batch_size": current_batch_size,
                    "backpressure": captain._backpressure_active,
                    "memory_quota_mb": (
                        captain.quota.memory_quota_mb if captain.quota else 0
                    ),
                }
                ray.get(self.quota_store.update_quota.remote(stage_name, quota_dict))
            except Exception:
                pass


class ElasticJuicer:
    """
    Scheduling facade for bi-level adaptive scheduling.
    
    Manages Tower (macro-scheduler), Captains (per-stage), MetricsBridge,
    and Ray named actors for metrics collection and quota distribution.
    
    The facade lives on the driver (not serializable). Ray workers connect
    to the named actors by name to receive quotas and report metrics.
    
    Usage:
        elastic = ElasticJuicer(config=scheduler_config, cluster_state=cluster_state)
        elastic.register_stages(stage_configs)
        elastic.start()
        try:
            # ... run pipeline with AdaptiveOperator ...
        finally:
            elastic.stop()
    
    Attributes:
        config: SchedulerConfig instance.
        tower: Tower macro-scheduler instance (None until register_stages called).
        is_running: Whether the scheduling system is currently active.
    """
    
    def __init__(
        self,
        config: Optional[SchedulerConfig] = None,
        config_path: Optional[str] = None,
        cluster_state: Optional[ClusterState] = None,
        preset: str = 'gpu',
    ):
        """
        Initialize with config from preset, YAML file, or SchedulerConfig object.
        
        Args:
            config: Pre-existing SchedulerConfig. Takes precedence over other options.
            config_path: Path to load config from YAML. Used if config is None.
            cluster_state: Optional ClusterState for Tower. If None, auto-detected.
            preset: Preset name if no config or config_path provided. 
                    Options: 'conservative', 'gpu', 'aggressive'. Default: 'gpu'.
        """
        # Determine SchedulerConfig
        if config is not None:
            self._config = config
        elif config_path is not None:
            self._config = SchedulerConfig.from_yaml(config_path)
            logger.info(f"Loaded config from {config_path}")
        else:
            # Use preset
            presets = {
                'conservative': SchedulerConfig.conservative,
                'gpu': SchedulerConfig.gpu,
                'aggressive': SchedulerConfig.aggressive,
            }
            factory = presets.get(preset, SchedulerConfig.gpu)
            self._config = factory()
            logger.info(f"Using ElasticJuicer {preset} preset config")
        
        # Detect cluster state
        self._cluster_state = cluster_state or _get_default_cluster_state()
        
        # Components (initialized in register_stages / start)
        self._tower: Optional[Tower] = None
        self._captains: Dict[str, Captain] = {}
        self._captain_pool: Optional[CaptainPool] = None
        self._captain_ids: Dict[str, str] = {}  # stage_name -> captain_id mapping
        self._metrics_bridge: Optional[MetricsBridge] = None
        self._metrics_collector = None  # Ray named actor
        self._quota_store = None  # Ray named actor
        self._is_running = False
        self._stage_names: List[str] = []
    
    def register_stages(
        self,
        stage_configs: List[Dict[str, Any]],
    ) -> None:
        """
        Register operator stages for scheduling.
        
        Creates Tower, Captains, and registers all stages. Must be called
        before start().
        
        Args:
            stage_configs: List of dicts with keys:
                - 'name': str (stage identifier, e.g. 'stage_0_video_aesthetics_filter')
                - 'batch_size': int (initial batch size, optional)
                - 'num_gpus': float (GPU requirement, 0 for CPU-only, optional)
                - 'num_actors': int (actor pool size, optional)
        """
        # Create Tower macro-scheduler
        self._tower = Tower(
            cluster_state=self._cluster_state,
            target_queue_depth=100,
            sla_latency_ms=5000.0,
            update_interval_sec=self._config.rebalance_interval_sec,
            config=self._config,
        )
        
        self._stage_names = []
        self._captains = {}
        self._captain_ids = {}
        
        for sc in stage_configs:
            stage_name = sc['name']
            batch_size = sc.get('batch_size', self._config.initial_batch_size)
            num_actors = sc.get('num_actors', 1)
            
            # Register stage with Tower (returns captain_id)
            captain_id = self._tower.register_stage(
                stage_name=stage_name, initial_parallelism=num_actors
            )
            self._captain_ids[stage_name] = captain_id
            
            # Create Captain config
            captain_config = CaptainConfig(
                stage_name=stage_name,
                initial_batch_size=batch_size,
                enable_micro_scheduler=self._config.enable_auto_adjust,
                enable_prediction=self._config.enable_prediction,
            )
            
            # Create Captain instance
            captain = Captain(config=captain_config)
            self._captains[stage_name] = captain
            
            # Register captain with Tower for metrics collection and quota broadcast
            self._tower.register_captain(captain_id, captain)
            
            self._stage_names.append(stage_name)
        
        logger.info(f"ElasticJuicer: Registered {len(stage_configs)} stages with Tower")
    
    def start(self) -> None:
        """
        Start the scheduling system.
        
        Creates Ray named actors (PipelineMetricsCollector, SharedQuotaStore),
        starts Tower rebalance loop, and starts MetricsBridge thread.
        
        Must call register_stages() before this method.
        """
        if self._is_running:
            logger.warning("ElasticJuicer is already running")
            return
        
        if self._tower is None:
            raise RuntimeError(
                "Must call register_stages() before start(). "
                "No stages have been registered."
            )
        
        import ray
        
        # Create PipelineMetricsCollector named actor
        try:
            collector_cls = _create_pipeline_metrics_collector()
            self._metrics_collector = collector_cls.options(
                name="elastic_pipeline_metrics", get_if_exists=True
            ).remote()
            logger.info("PipelineMetricsCollector named actor created")
        except Exception as e:
            logger.warning(f"Failed to create PipelineMetricsCollector: {e}")
            self._metrics_collector = None
        
        # Create SharedQuotaStore named actor
        try:
            quota_cls = _create_shared_quota_store()
            self._quota_store = quota_cls.options(
                name="elastic_quota_store", get_if_exists=True
            ).remote()
            logger.info("SharedQuotaStore named actor created")
        except Exception as e:
            logger.warning(f"Failed to create SharedQuotaStore: {e}")
            self._quota_store = None
        
        # Start Tower rebalance loop
        self._tower.start()
        logger.info("Tower rebalance loop started")
        
        # Start MetricsBridge
        if self._captains and self._metrics_collector is not None and self._quota_store is not None:
            self._metrics_bridge = MetricsBridge(
                tower=self._tower,
                captains=self._captains,
                metrics_collector=self._metrics_collector,
                quota_store=self._quota_store,
                interval=self._config.rebalance_interval_sec,
            )
            self._metrics_bridge.start()
            logger.info("MetricsBridge started - connecting actors ↔ Captains ↔ Tower")
        
        self._is_running = True
        logger.info("ElasticJuicer: Started (Tower + MetricsBridge + Ray actors)")
    
    def stop(self) -> None:
        """
        Stop all scheduling components.
        
        Stops MetricsBridge thread, Tower rebalance loop, and cleans up 
        Ray named actors. Safe to call even if not running.
        """
        if not self._is_running:
            return
        
        # Stop MetricsBridge
        if self._metrics_bridge is not None:
            try:
                self._metrics_bridge.stop()
                self._metrics_bridge.join(timeout=5)
                logger.info("MetricsBridge stopped")
            except Exception:
                pass
            self._metrics_bridge = None
        
        # Stop Tower
        if self._tower is not None:
            try:
                self._tower.stop()
                logger.info("Tower rebalance loop stopped")
            except Exception:
                pass
        
        # Cleanup Ray actors
        import ray
        for actor in [self._metrics_collector, self._quota_store]:
            if actor is not None:
                try:
                    ray.kill(actor)
                except Exception:
                    pass
        self._metrics_collector = None
        self._quota_store = None
        
        self._is_running = False
        logger.info("ElasticJuicer: Stopped")
    
    def get_adaptive_op_config(self, stage_name: str) -> Dict[str, Any]:
        """
        Get configuration dict for AdaptiveOperator constructor.
        
        This is what gets passed to fn_constructor_kwargs in map_batches().
        AdaptiveOperator uses this to connect to the named actors.
        
        Args:
            stage_name: Name of the stage to get config for.
        
        Returns:
            Dict with: stage_name, initial_batch_size, scheduler_config_dict
        """
        captain = self._captains.get(stage_name)
        batch_size = captain.config.initial_batch_size if captain else self._config.initial_batch_size
        
        return {
            'stage_name': stage_name,
            'initial_batch_size': batch_size,
            'scheduler_config_dict': asdict(self._config),
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get per-stage metrics from PipelineMetricsCollector.
        
        Returns:
            Dict mapping stage_name to metrics dict with:
            total_samples, total_batches, min_bs, max_bs, avg_bs,
            avg_latency_ms, avg_memory_mb, peak_memory_mb
        """
        if self._metrics_collector is None:
            return {}
        try:
            import ray
            return ray.get(self._metrics_collector.get_summary.remote())
        except Exception:
            return {}
    
    def get_captain_stats(self) -> Dict[str, Any]:
        """
        Get per-stage Captain statistics.
        
        Returns:
            Dict mapping stage_name to stats dict with:
            throughput, latency_ms, batch_size, backpressure, oom_count
        """
        stats = {}
        for name, captain in self._captains.items():
            try:
                metrics = captain.collect_metrics()
                stats[name] = {
                    'throughput': getattr(metrics, 'throughput', 0),
                    'latency_ms': getattr(metrics, 'avg_latency_ms', 0),
                    'batch_size': (
                        captain.micro_scheduler.controller.current_batch_size
                        if captain.micro_scheduler
                        else captain.config.initial_batch_size
                    ),
                    'backpressure': captain._backpressure_active if hasattr(captain, '_backpressure_active') else False,
                    'oom_count': captain._total_oom_count if hasattr(captain, '_total_oom_count') else 0,
                }
            except Exception:
                stats[name] = {}
        return stats
    
    def get_tower_stats(self) -> Dict[str, Any]:
        """
        Get Tower global statistics.
        
        Returns:
            Dict with global stats: total_stages, total_parallelism,
            sla_compliance_rate, total_requests, sla_violations, etc.
        """
        if self._tower is None:
            return {}
        try:
            return self._tower.get_global_stats()
        except Exception:
            return {}
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get complete system status.
        
        Returns:
            Dict containing: is_running, stages, config_preset,
            rebalance_interval, metrics, captains, tower
        """
        return {
            'is_running': self._is_running,
            'stages': self._stage_names,
            'rebalance_interval': self._config.rebalance_interval_sec,
            'metrics': self.get_metrics_summary(),
            'captains': self.get_captain_stats(),
            'tower': self.get_tower_stats(),
        }
    
    # ---- Offline tuning ----
    
    def run_offline_tuning(
        self,
        stage_names: Optional[List[str]] = None,
        simulation_fn: Optional[Callable[[SchedulerConfig], Dict[str, float]]] = None,
        num_samples: int = 8,
        max_iterations: int = 50,
        export_path: str = "base_config.yaml",
    ) -> SchedulerConfig:
        """
        Run PBT offline tuning to find optimal SchedulerConfig.
        
        Uses Population Based Training to optimize PID controller parameters,
        memory safety buffers, predictor settings, and per-stage resource 
        allocation weights.
        
        Args:
            stage_names: Operator stage names to tune allocation weights for.
                If None, uses self._stage_names or an empty list.
            simulation_fn: Custom simulation function. If None, uses default.
            num_samples: PBT population size (number of parallel trials).
            max_iterations: Maximum training iterations per trial.
            export_path: Path to save the tuned config YAML.
        
        Returns:
            Optimized SchedulerConfig.
        
        Raises:
            ImportError: If Ray Tune is not installed.
        """
        from .tuner.pbt_tuner import PBTTuner, PBTTunerConfig
        
        tuner_config = PBTTunerConfig(
            num_samples=num_samples,
            max_iterations=max_iterations,
            stage_names=stage_names or self._stage_names or [],
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
    
    # ---- Properties ----
    
    @property
    def config(self) -> SchedulerConfig:
        """Get the current SchedulerConfig."""
        return self._config
    
    @property
    def tower(self) -> Optional[Tower]:
        """Get the Tower macro-scheduler instance (None if not registered)."""
        return self._tower
    
    @property
    def captains(self) -> Dict[str, Captain]:
        """Get all Captain instances mapped by stage name."""
        return self._captains
    
    @property
    def is_running(self) -> bool:
        """Check if the scheduling system is currently running."""
        return self._is_running
    
    @property
    def stage_names(self) -> List[str]:
        """Get list of registered stage names."""
        return self._stage_names.copy()
    
    # ---- Captain access ----
    
    def get_captain(self, stage_name: str) -> Optional[Captain]:
        """
        Get the Captain for a specific stage.
        
        Args:
            stage_name: Name of the operator stage.
        
        Returns:
            Captain instance for the stage, or None if not found.
        """
        return self._captains.get(stage_name)
    
    # ---- Config management ----
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters dynamically.
        
        Note: For runtime changes to take effect on Tower and Captains,
        the system may need to be restarted.
        
        Args:
            **kwargs: Configuration parameters to update. See SchedulerConfig.
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
        """
        Save current configuration to a YAML file.
        
        Args:
            path: Output file path for the YAML config.
        """
        self._config.to_yaml(path)
        logger.info(f"Config saved to {path}")
    
    # ---- Context manager support ----
    
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
            f"stages={self._stage_names})"
        )
