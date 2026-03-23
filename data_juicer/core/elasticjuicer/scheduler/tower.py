"""
Tower: Global Macro-Scheduler for ElasticJuicer

Based on Autothrottle's bi-level architecture, Tower is the global resource allocator
that sets performance targets and resource quotas for local Captains.

Key responsibilities:
1. Monitor global queue depth and cluster resource utilization
2. Set target parallelism for each operator stage
3. Allocate resource budgets to Captains
4. Make topology decisions (co-location vs distributed)
5. Handle global SLA guarantees

References:
- Autothrottle (NSDI 2024): Bi-level control for SLO-targeted microservices
- Report Section 5.1: Tower/Captain architecture
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from collections import deque

if TYPE_CHECKING:
    from .scheduler_config import SchedulerConfig


class TopologyMode(Enum):
    """Topology execution mode based on transfer cost and resource availability"""
    CO_LOCATION = "co_location"  # Operators on same node (high transfer cost)
    DISTRIBUTED = "distributed"   # Operators on different nodes (high parallelism)
    ADAPTIVE = "adaptive"         # Let Tower decide based on current state


@dataclass
class StageMetrics:
    """Performance metrics for an operator stage"""
    stage_name: str
    queue_depth: int = 0              # Number of pending samples
    current_parallelism: int = 1      # Current number of actors
    throughput: float = 0.0           # Samples/sec
    avg_latency_ms: float = 0.0       # Average processing latency
    cpu_utilization: float = 0.0      # % CPU used
    memory_utilization: float = 0.0   # % Memory used
    gpu_utilization: float = 0.0      # % GPU used (if applicable)
    oom_count: int = 0                # Number of OOM events
    last_update: float = field(default_factory=time.time)


@dataclass
class ResourceQuota:
    """Resource allocation quota for a Captain"""
    captain_id: str
    target_parallelism: int           # Target number of actors
    cpu_quota: float                  # CPU cores allocated
    memory_quota_mb: float            # Memory budget in MB
    gpu_quota: float = 0.0            # GPU cores allocated (0-1)
    target_throughput: float = 0.0    # Target samples/sec (SLO)
    topology_mode: TopologyMode = TopologyMode.ADAPTIVE
    backpressure: bool = False        # Whether upstream backpressure is active


@dataclass
class ClusterState:
    """Global cluster resource state"""
    total_cpu_cores: int
    total_memory_mb: float
    total_gpu_count: int
    available_cpu_cores: float
    available_memory_mb: float
    available_gpus: float
    timestamp: float = field(default_factory=time.time)


class Tower:
    """
    Global Macro-Scheduler (Tower from Autothrottle architecture)
    
    Tower doesn't directly control individual actors' behavior. Instead, it:
    1. Monitors global system state (queue depths, resource utilization)
    2. Sets performance targets and resource quotas for Captains
    3. Makes high-level topology decisions
    4. Ensures cluster-wide SLA guarantees
    
    The bi-level design (Tower + Captain) solves the single-point bottleneck
    problem of centralized schedulers, enabling high-frequency local decisions
    under global constraints.
    """
    
    def __init__(
        self,
        cluster_state: ClusterState,
        target_queue_depth: int = 100,
        sla_latency_ms: float = 5000.0,
        update_interval_sec: float = 5.0,
        history_window: int = 20,
        config: Optional['SchedulerConfig'] = None
    ):
        """
        Initialize Tower global scheduler
        
        Args:
            cluster_state: Initial cluster resource state
            target_queue_depth: Target queue depth to maintain
            sla_latency_ms: SLA latency target (max allowed latency)
            update_interval_sec: How often to recompute resource allocation
            history_window: Window size for tracking metrics history
            config: Optional SchedulerConfig for rebalance settings
        """
        self.cluster = cluster_state
        self.target_queue_depth = target_queue_depth
        self.sla_latency_ms = sla_latency_ms
        self.update_interval = update_interval_sec
        self.config = config
        
        # Track all stages and their metrics
        self.stages: Dict[str, StageMetrics] = {}
        
        # Track resource quotas allocated to each Captain
        self.quotas: Dict[str, ResourceQuota] = {}
        
        # Metrics history for trend analysis
        self.metrics_history: Dict[str, deque] = {}
        self.history_window = history_window
        
        # Last allocation time
        self.last_allocation_time = time.time()
        
        # SLA violation tracking
        self.sla_violations = 0
        self.total_requests = 0
        
        # Captain registry for direct metric collection and quota broadcast
        self._captains: Dict[str, Any] = {}
        
        # Rebalance loop control
        self._running: bool = False
        self._rebalance_thread: Optional[threading.Thread] = None
        
        # Rebalance interval from config or fallback to update_interval
        self.rebalance_interval: float = (
            config.rebalance_interval_sec if config else update_interval_sec
        )
        
        # Backpressure threshold from config
        self._backpressure_threshold: float = (
            config.backpressure_threshold if config else 0.9
        )
        
        # Per-stage backpressure state tracking
        self._backpressure_states: Dict[str, bool] = {}
        
        # Track registration order for upstream detection
        self._stage_order: List[str] = []
    
    def register_stage(self, stage_name: str, initial_parallelism: int = 1) -> str:
        """
        Register a new operator stage with Tower
        
        Args:
            stage_name: Name of the operator stage
            initial_parallelism: Initial number of actors
            
        Returns:
            captain_id: Unique ID for the Captain managing this stage
        """
        captain_id = f"captain_{stage_name}_{int(time.time())}"
        
        # Initialize stage metrics
        self.stages[stage_name] = StageMetrics(
            stage_name=stage_name,
            current_parallelism=initial_parallelism
        )
        
        # Initialize metrics history
        self.metrics_history[stage_name] = deque(maxlen=self.history_window)
        
        # Track stage registration order for upstream detection
        if stage_name not in self._stage_order:
            self._stage_order.append(stage_name)
        
        # Initialize backpressure state
        self._backpressure_states[stage_name] = False
        
        # Allocate initial quota
        initial_quota = self._compute_initial_quota(stage_name, initial_parallelism)
        self.quotas[captain_id] = initial_quota
        
        return captain_id
    
    def update_stage_metrics(self, stage_name: str, metrics: StageMetrics):
        """
        Update metrics for a stage (called by Captain)
        
        Args:
            stage_name: Name of the stage
            metrics: Latest metrics from Captain
        """
        if stage_name not in self.stages:
            raise ValueError(f"Stage {stage_name} not registered")
        
        # Update current metrics
        self.stages[stage_name] = metrics
        
        # Add to history
        self.metrics_history[stage_name].append({
            'timestamp': metrics.last_update,
            'queue_depth': metrics.queue_depth,
            'throughput': metrics.throughput,
            'latency_ms': metrics.avg_latency_ms,
            'cpu_util': metrics.cpu_utilization,
            'memory_util': metrics.memory_utilization
        })
        
        # Track SLA violations
        self.total_requests += 1
        if metrics.avg_latency_ms > self.sla_latency_ms:
            self.sla_violations += 1
    
    def allocate_resources(self) -> Dict[str, ResourceQuota]:
        """
        Compute and allocate resource quotas to all Captains
        
        This is the core global decision-making function. It:
        1. Analyzes global bottlenecks (queue depths, latencies)
        2. Computes target parallelism for each stage
        3. Allocates CPU/GPU/memory budgets
        4. Returns updated quotas for Captains to enforce
        
        Returns:
            Updated resource quotas for all Captains
        """
        current_time = time.time()
        
        # Rate limit allocation updates (avoid thrashing)
        if current_time - self.last_allocation_time < self.update_interval:
            return self.quotas
        
        self.last_allocation_time = current_time
        
        # Identify bottleneck stages
        bottlenecks = self._identify_bottlenecks()
        
        # Compute resource allocation strategy
        for captain_id, quota in self.quotas.items():
            stage_name = self._get_stage_from_captain(captain_id)
            if stage_name not in self.stages:
                continue
            
            metrics = self.stages[stage_name]
            
            # Decide target parallelism based on queue depth and throughput
            target_parallelism = self._compute_target_parallelism(
                metrics, 
                is_bottleneck=(stage_name in bottlenecks)
            )
            
            # Allocate resources proportionally
            resource_allocation = self._allocate_stage_resources(
                stage_name, 
                target_parallelism
            )
            
            # Update quota
            quota.target_parallelism = target_parallelism
            quota.cpu_quota = resource_allocation['cpu']
            quota.memory_quota_mb = resource_allocation['memory_mb']
            quota.gpu_quota = resource_allocation['gpu']
            quota.target_throughput = self._compute_target_throughput(metrics)
            quota.topology_mode = self._decide_topology(stage_name, metrics)
        
        return self.quotas
    
    def _identify_bottlenecks(self) -> List[str]:
        """
        Identify bottleneck stages based on queue depth and latency
        
        A stage is a bottleneck if:
        1. Queue depth > target_queue_depth
        2. Latency approaching SLA limit
        3. Throughput declining over time
        
        Returns:
            List of bottleneck stage names
        """
        bottlenecks = []
        
        for stage_name, metrics in self.stages.items():
            # Check queue depth
            queue_pressure = metrics.queue_depth > self.target_queue_depth
            
            # Check latency
            latency_pressure = metrics.avg_latency_ms > (self.sla_latency_ms * 0.8)
            
            # Check throughput trend
            throughput_declining = False
            if stage_name in self.metrics_history and len(self.metrics_history[stage_name]) >= 3:
                recent = list(self.metrics_history[stage_name])[-3:]
                throughputs = [m['throughput'] for m in recent]
                if len(throughputs) >= 2:
                    throughput_declining = throughputs[-1] < throughputs[0] * 0.9
            
            if queue_pressure or latency_pressure or throughput_declining:
                bottlenecks.append(stage_name)
        
        return bottlenecks
    
    def _compute_target_parallelism(
        self, 
        metrics: StageMetrics, 
        is_bottleneck: bool
    ) -> int:
        """
        Compute target parallelism for a stage
        
        Strategy:
        - If bottleneck: Increase parallelism to drain queue
        - If underutilized: Decrease parallelism to free resources
        - Consider resource availability
        
        Args:
            metrics: Current stage metrics
            is_bottleneck: Whether this stage is a bottleneck
            
        Returns:
            Target parallelism (number of actors)
        """
        current = metrics.current_parallelism
        
        if is_bottleneck:
            # Estimate needed parallelism to drain queue
            if metrics.throughput > 0:
                # Time to process queue at current throughput
                queue_drain_time = metrics.queue_depth / metrics.throughput
                
                # If drain time > SLA, scale up
                if queue_drain_time > (self.sla_latency_ms / 1000.0):
                    scale_factor = min(2.0, queue_drain_time / (self.sla_latency_ms / 1000.0))
                    target = int(current * scale_factor)
                else:
                    target = current + 1  # Conservative increase
            else:
                target = current + 1  # No throughput data, try increasing
        else:
            # Check if we can scale down (free resources)
            if metrics.queue_depth < self.target_queue_depth * 0.5 and current > 1:
                target = max(1, current - 1)
            else:
                target = current  # Keep current level
        
        # Clamp to available resources
        max_possible = self._estimate_max_parallelism()
        target = min(target, max_possible)
        
        return max(1, target)  # At least 1 actor
    
    def _allocate_stage_resources(
        self, 
        stage_name: str, 
        target_parallelism: int
    ) -> Dict[str, float]:
        """
        Allocate CPU/GPU/memory to a stage based on target parallelism
        
        Args:
            stage_name: Name of the stage
            target_parallelism: Target number of actors
            
        Returns:
            Resource allocation dict with 'cpu', 'memory_mb', 'gpu'
        """
        # Simple proportional allocation (can be enhanced with OCS annotations)
        total_stages = len(self.stages)
        
        if total_stages == 0:
            cpu_share = self.cluster.available_cpu_cores
            memory_share = self.cluster.available_memory_mb
            gpu_share = self.cluster.available_gpus
        else:
            # Equal share for now (TODO: weight by OCS cost)
            cpu_share = self.cluster.available_cpu_cores / total_stages
            memory_share = self.cluster.available_memory_mb / total_stages
            gpu_share = self.cluster.available_gpus / total_stages
        
        return {
            'cpu': cpu_share * target_parallelism,
            'memory_mb': memory_share * target_parallelism,
            'gpu': gpu_share * target_parallelism
        }
    
    def _compute_target_throughput(self, metrics: StageMetrics) -> float:
        """
        Compute target throughput to meet SLA
        
        Args:
            metrics: Current stage metrics
            
        Returns:
            Target throughput in samples/sec
        """
        # To meet SLA, we need throughput >= queue_depth / (SLA_time - current_latency)
        sla_time_sec = self.sla_latency_ms / 1000.0
        current_latency_sec = metrics.avg_latency_ms / 1000.0
        
        remaining_time = max(0.1, sla_time_sec - current_latency_sec)
        
        if metrics.queue_depth > 0:
            target = metrics.queue_depth / remaining_time
        else:
            target = metrics.throughput  # Maintain current
        
        return max(1.0, target)
    
    def _decide_topology(
        self, 
        stage_name: str, 
        metrics: StageMetrics
    ) -> TopologyMode:
        """
        Decide topology mode for operator placement
        
        Based on Report Section 5.4:
        - CO_LOCATION: High transfer cost, sufficient local resources
        - DISTRIBUTED: Different resource bottlenecks, ample bandwidth
        
        Args:
            stage_name: Name of the stage
            metrics: Current metrics
            
        Returns:
            Topology mode decision
        """
        # Check resource pressure
        high_cpu = metrics.cpu_utilization > 80
        high_memory = metrics.memory_utilization > 80
        high_gpu = metrics.gpu_utilization > 80
        
        # If single resource bottleneck, distribute to specialize
        bottleneck_count = sum([high_cpu, high_memory, high_gpu])
        
        if bottleneck_count >= 2:
            # Multiple bottlenecks on same node -> distribute
            return TopologyMode.DISTRIBUTED
        elif bottleneck_count == 0:
            # No pressure -> co-locate for efficiency
            return TopologyMode.CO_LOCATION
        else:
            # Single bottleneck -> adaptive
            return TopologyMode.ADAPTIVE
    
    def _estimate_max_parallelism(self) -> int:
        """
        Estimate maximum parallelism given available resources
        
        Returns:
            Maximum number of actors cluster can support
        """
        # Conservative estimate: assume each actor needs 1 CPU + 1GB memory
        cpu_limit = int(self.cluster.available_cpu_cores)
        memory_limit = int(self.cluster.available_memory_mb / 1024)  # 1GB per actor
        
        return max(1, min(cpu_limit, memory_limit))
    
    def _get_stage_from_captain(self, captain_id: str) -> str:
        """Extract stage name from captain ID"""
        # captain_video_decoder_1234567890 -> video_decoder
        parts = captain_id.split('_')
        if len(parts) >= 3:
            return '_'.join(parts[1:-1])
        return captain_id
    
    def _compute_initial_quota(
        self, 
        stage_name: str, 
        parallelism: int
    ) -> ResourceQuota:
        """Compute initial resource quota for a new stage"""
        captain_id = f"captain_{stage_name}_{int(time.time())}"
        
        # Equal share allocation initially
        total_stages = max(1, len(self.stages))
        
        return ResourceQuota(
            captain_id=captain_id,
            target_parallelism=parallelism,
            cpu_quota=self.cluster.available_cpu_cores / total_stages,
            memory_quota_mb=self.cluster.available_memory_mb / total_stages,
            gpu_quota=self.cluster.available_gpus / total_stages,
            target_throughput=10.0,  # Default
            topology_mode=TopologyMode.ADAPTIVE
        )
    
    def get_sla_compliance_rate(self) -> float:
        """
        Calculate SLA compliance rate
        
        Returns:
            Percentage of requests meeting SLA (0-100)
        """
        if self.total_requests == 0:
            return 100.0
        
        return ((self.total_requests - self.sla_violations) / self.total_requests) * 100.0
    
    def get_global_stats(self) -> Dict:
        """Get global system statistics"""
        return {
            'total_stages': len(self.stages),
            'total_parallelism': sum(q.target_parallelism for q in self.quotas.values()),
            'sla_compliance_rate': self.get_sla_compliance_rate(),
            'total_requests': self.total_requests,
            'sla_violations': self.sla_violations,
            'cluster_cpu_util': (
                (self.cluster.total_cpu_cores - self.cluster.available_cpu_cores) /
                self.cluster.total_cpu_cores * 100
            ) if self.cluster.total_cpu_cores > 0 else 0,
            'cluster_memory_util': (
                (self.cluster.total_memory_mb - self.cluster.available_memory_mb) /
                self.cluster.total_memory_mb * 100
            ) if self.cluster.total_memory_mb > 0 else 0
        }
    
    # ========== Captain Registry Methods ==========
    
    def register_captain(self, captain_id: str, captain: Any) -> None:
        """Register a Captain instance for direct metric collection and quota broadcast.
        
        Args:
            captain_id: Unique identifier for the captain
            captain: Captain instance to register
        """
        self._captains[captain_id] = captain
    
    def unregister_captain(self, captain_id: str) -> None:
        """Remove a Captain from the registry.
        
        Args:
            captain_id: Unique identifier of the captain to remove
        """
        self._captains.pop(captain_id, None)
    
    # ========== Rebalance Loop Methods ==========
    
    def collect_all_metrics(self) -> Dict[str, StageMetrics]:
        """Step 1: Collect metrics from all registered Captains.
        
        For each registered captain, call captain.collect_metrics() if available,
        or use captain.metrics directly. Also update internal stage_metrics dict.
        
        Returns:
            Dict mapping stage_name to StageMetrics
        """
        collected_metrics: Dict[str, StageMetrics] = {}
        
        for captain_id, captain in self._captains.items():
            try:
                # Try collect_metrics() method first, fall back to metrics attribute
                if hasattr(captain, 'collect_metrics'):
                    metrics = captain.collect_metrics()
                elif hasattr(captain, 'metrics'):
                    metrics = captain.metrics
                else:
                    continue
                
                if metrics and hasattr(metrics, 'stage_name'):
                    stage_name = metrics.stage_name
                    collected_metrics[stage_name] = metrics
                    
                    # Update internal stage metrics
                    if stage_name in self.stages:
                        self.stages[stage_name] = metrics
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to collect metrics from captain {captain_id}: {e}")
        
        # Also include stages that have metrics but no registered captain
        for stage_name, metrics in self.stages.items():
            if stage_name not in collected_metrics:
                collected_metrics[stage_name] = metrics
        
        return collected_metrics
    
    def identify_bottleneck(self, metrics: Dict[str, StageMetrics]) -> Optional[str]:
        """Step 2: Identify the bottleneck stage (highest queue depth / lowest throughput).
        
        Find the single worst bottleneck based on queue depth and throughput.
        Uses existing _identify_bottlenecks() logic but returns only the worst one.
        
        Args:
            metrics: Dict of stage metrics
            
        Returns:
            stage_name of worst bottleneck, or None if no bottleneck
        """
        if not metrics:
            return None
        
        # Get all bottleneck candidates
        bottlenecks = self._identify_bottlenecks()
        
        if not bottlenecks:
            return None
        
        # Find the worst bottleneck based on score
        # Score = (queue_depth / target) + (1 - throughput_ratio)
        worst_bottleneck = None
        worst_score = -1.0
        
        for stage_name in bottlenecks:
            if stage_name not in metrics:
                continue
            
            stage_metrics = metrics[stage_name]
            
            # Calculate bottleneck severity score
            queue_ratio = (
                stage_metrics.queue_depth / self.target_queue_depth
                if self.target_queue_depth > 0 else 0
            )
            
            # Consider throughput relative to what's needed
            # Higher queue with lower throughput = worse bottleneck
            throughput_factor = 1.0
            if stage_metrics.throughput > 0 and stage_metrics.queue_depth > 0:
                # Time to drain queue at current throughput
                drain_time = stage_metrics.queue_depth / stage_metrics.throughput
                sla_time = self.sla_latency_ms / 1000.0
                throughput_factor = drain_time / sla_time if sla_time > 0 else 1.0
            
            score = queue_ratio + throughput_factor
            
            if score > worst_score:
                worst_score = score
                worst_bottleneck = stage_name
        
        return worst_bottleneck
    
    def reallocate_resources(
        self,
        bottleneck: Optional[str],
        metrics: Dict[str, StageMetrics]
    ) -> Dict[str, ResourceQuota]:
        """Step 3: Reallocate resources, increasing quota for bottleneck.
        
        If bottleneck exists, shift resources toward it using tower_allocation_weights
        from config (via get_stage_weight).
        
        Args:
            bottleneck: Name of bottleneck stage, or None
            metrics: Current stage metrics
            
        Returns:
            Dict mapping captain_id to new ResourceQuota
        """
        # Compute weights for each stage
        stage_weights: Dict[str, float] = {}
        for stage_name in self.stages:
            # Base weight from config
            if self.config:
                base_weight = self.config.get_stage_weight(stage_name)
            else:
                base_weight = 1.0
            
            # Boost weight for bottleneck stage
            if stage_name == bottleneck:
                base_weight *= 1.5  # 50% boost for bottleneck
            
            stage_weights[stage_name] = base_weight
        
        # Normalize weights
        total_weight = sum(stage_weights.values())
        if total_weight > 0:
            for stage_name in stage_weights:
                stage_weights[stage_name] /= total_weight
        
        # Allocate resources based on weights
        for captain_id, quota in self.quotas.items():
            stage_name = self._get_stage_from_captain(captain_id)
            if stage_name not in self.stages:
                continue
            
            stage_metrics = metrics.get(stage_name, self.stages[stage_name])
            weight = stage_weights.get(stage_name, 1.0 / max(1, len(self.stages)))
            
            # Compute target parallelism
            target_parallelism = self._compute_target_parallelism(
                stage_metrics,
                is_bottleneck=(stage_name == bottleneck)
            )
            
            # Allocate resources proportionally to weight
            resource_allocation = {
                'cpu': self.cluster.available_cpu_cores * weight * target_parallelism,
                'memory_mb': self.cluster.available_memory_mb * weight * target_parallelism,
                'gpu': self.cluster.available_gpus * weight * target_parallelism
            }
            
            # Update quota
            quota.target_parallelism = target_parallelism
            quota.cpu_quota = resource_allocation['cpu']
            quota.memory_quota_mb = resource_allocation['memory_mb']
            quota.gpu_quota = resource_allocation['gpu']
            quota.target_throughput = self._compute_target_throughput(stage_metrics)
            quota.topology_mode = self._decide_topology(stage_name, stage_metrics)
        
        return self.quotas
    
    def apply_backpressure(
        self,
        bottleneck: Optional[str],
        metrics: Dict[str, StageMetrics]
    ) -> None:
        """Step 3b: Apply backpressure to upstream stages if needed.
        
        If bottleneck stage memory_utilization > backpressure_threshold:
            Find upstream stages (stages registered before bottleneck)
            Set backpressure=True flag on their quotas
        
        Args:
            bottleneck: Name of bottleneck stage, or None
            metrics: Current stage metrics
        """
        # Reset all backpressure states first
        for stage_name in self._backpressure_states:
            self._backpressure_states[stage_name] = False
        
        if not bottleneck or bottleneck not in metrics:
            # No bottleneck, clear all backpressure
            for quota in self.quotas.values():
                quota.backpressure = False
            return
        
        bottleneck_metrics = metrics[bottleneck]
        
        # Check if memory utilization exceeds threshold
        # memory_utilization is in percentage (0-100), threshold is ratio (0-1)
        memory_util_ratio = bottleneck_metrics.memory_utilization / 100.0
        
        if memory_util_ratio <= self._backpressure_threshold:
            # Below threshold, no backpressure needed
            for quota in self.quotas.values():
                quota.backpressure = False
            return
        
        # Find bottleneck position in stage order
        try:
            bottleneck_idx = self._stage_order.index(bottleneck)
        except ValueError:
            return
        
        # Apply backpressure to all upstream stages (before bottleneck)
        upstream_stages = set(self._stage_order[:bottleneck_idx])
        
        for captain_id, quota in self.quotas.items():
            stage_name = self._get_stage_from_captain(captain_id)
            if stage_name in upstream_stages:
                quota.backpressure = True
                self._backpressure_states[stage_name] = True
            else:
                quota.backpressure = False
    
    def broadcast_quotas(self, quotas: Dict[str, ResourceQuota]) -> None:
        """Step 4: Send updated quotas to all Captains.
        
        For each captain in registry, call captain.set_quota(quota).
        
        Args:
            quotas: Dict mapping captain_id to ResourceQuota
        """
        logger = logging.getLogger(__name__)
        
        for captain_id, captain in self._captains.items():
            # Find matching quota
            quota = quotas.get(captain_id)
            
            if quota is None:
                # Try to find quota by stage name
                for qid, q in quotas.items():
                    if self._get_stage_from_captain(qid) == self._get_stage_from_captain(captain_id):
                        quota = q
                        break
            
            if quota is None:
                continue
            
            try:
                if hasattr(captain, 'set_quota'):
                    captain.set_quota(quota)
            except Exception as e:
                logger.warning(f"Failed to broadcast quota to captain {captain_id}: {e}")
    
    # ========== Rebalance Loop Lifecycle ==========
    
    def start(self) -> None:
        """Start the rebalance loop in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._rebalance_thread = threading.Thread(
            target=self._rebalance_loop,
            daemon=True,
            name="Tower-Rebalance-Loop"
        )
        self._rebalance_thread.start()
    
    def stop(self) -> None:
        """Stop the rebalance loop."""
        self._running = False
        if self._rebalance_thread and self._rebalance_thread.is_alive():
            self._rebalance_thread.join(timeout=self.rebalance_interval * 2)
            self._rebalance_thread = None
    
    def _rebalance_loop(self) -> None:
        """Main rebalance loop - runs periodically.
        
        Implements the adaptive tower macro-scheduler:
        for each rebalance_interval:
            1. Collect metrics from all Captains
            2. Identify bottleneck stage (highest queue / lowest throughput)
            3. Reallocate resources (increase quota for bottleneck, apply backpressure)
            4. Broadcast new quotas to Captains
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Tower rebalance loop started (interval={self.rebalance_interval}s)")
        
        while self._running:
            try:
                # Step 1: Collect metrics from all Captains
                metrics = self.collect_all_metrics()
                
                if metrics:
                    # Step 2: Identify bottleneck stage
                    bottleneck = self.identify_bottleneck(metrics)
                    
                    # Step 3: Reallocate resources
                    new_quotas = self.reallocate_resources(bottleneck, metrics)
                    
                    # Step 3b: Apply backpressure if needed
                    self.apply_backpressure(bottleneck, metrics)
                    
                    # Step 4: Broadcast new quotas to Captains
                    self.broadcast_quotas(new_quotas)
                    
                    if bottleneck:
                        logger.debug(f"Rebalance: bottleneck={bottleneck}, applying adjustments")
                        
            except Exception as e:
                logger.error(f"Rebalance loop error: {e}")
            
            # Wait for next interval
            time.sleep(self.rebalance_interval)
        
        logger.info("Tower rebalance loop stopped")
    
    # ========== Context Manager Support ==========
    
    def __enter__(self) -> 'Tower':
        """Context manager entry - start the rebalance loop."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop the rebalance loop."""
        self.stop()
