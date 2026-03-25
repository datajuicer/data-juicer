"""
Micro-Scheduler with JABAS-style PID Control

Implements dynamic batch size adjustment based on memory feedback.
Prevents OOM by continuously monitoring memory and adjusting batch sizes.

Based on:
- JABAS (EuroSys 2025): Adaptive batching for heterogeneous GPUs
- Report Section 4.1: PID Control for Batch Size

Key Features:
- PID controller for smooth batch size adjustment
- Memory pressure monitoring
- Safety thresholds and fallback strategies
- Integration with MemoryPredictor
"""

import time
import psutil
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from collections import deque
import numpy as np

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


@dataclass
class SampleFeatures:
    """Sample features for MicroScheduler prediction."""
    batch_size: int = 1
    estimated_memory_mb: float = 0.0


@dataclass
class MemoryState:
    """Current memory state"""
    timestamp: float
    # CPU memory
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    memory_percent: float
    # GPU memory (if available)
    gpu_total_mb: Optional[float] = None
    gpu_used_mb: Optional[float] = None
    gpu_available_mb: Optional[float] = None
    gpu_percent: Optional[float] = None
    
    def get_available_memory(self, use_gpu: bool = False) -> float:
        """Get available memory in MB"""
        if use_gpu and self.gpu_available_mb is not None:
            return self.gpu_available_mb
        return self.available_memory_mb


class PIDController:
    """
    PID (Proportional-Integral-Derivative) Controller.
    
    Classic control theory algorithm used in JABAS for smooth adjustments.
    
    Formula:
        output(t) = Kp * error(t) + Ki * Σerror + Kd * Δerror
    
    Where:
        - error = setpoint - current_value
        - Kp, Ki, Kd are tuning parameters
    """
    
    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.05,
        setpoint: float = 1000.0,
        output_limits: tuple = (1, 1000),
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            setpoint: Target value (e.g., target available memory in MB)
            output_limits: (min, max) bounds for output
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        # State
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
    def update(self, current_value: float, dt: Optional[float] = None) -> float:
        """
        Update PID controller with new measurement.
        
        Args:
            current_value: Current measured value
            dt: Time delta since last update (optional)
            
        Returns:
            Control output
        """
        # Calculate error
        error = self.setpoint - current_value
        
        # Calculate time delta
        current_time = time.time()
        if self.last_time is None or dt is not None:
            dt = dt or 0.1  # Default dt
        else:
            dt = current_time - self.last_time
        self.last_time = current_time
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term (with anti-windup)
        self.integral += error * dt
        # Clamp integral to prevent windup
        max_integral = self.output_limits[1] / (self.ki + 1e-6)
        self.integral = np.clip(self.integral, -max_integral, max_integral)
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.last_error) / (dt + 1e-6)
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update state
        self.last_error = error
        
        return output
    
    def reset(self):
        """Reset controller state"""
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
    
    def set_setpoint(self, setpoint: float):
        """Update setpoint"""
        self.setpoint = setpoint


class BatchSizeController:
    """
    Controls batch size using PID feedback based on memory pressure.
    
    This is the core of the micro-scheduler - it continuously monitors
    memory and adjusts batch size to maximize throughput while preventing OOM.
    
    Strategy (from Report Section 4.1):
        B_next = B_curr × (M_target / M_curr)
        
    Enhanced with PID for smoothness.
    """
    
    def __init__(
        self,
        initial_batch_size: int = 1,
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
        target_memory_utilization: float = 0.85,
        safety_buffer_mb: float = 1000.0,
        use_gpu: bool = False,
        enable_prediction: bool = True,
        memory_predictor = None,
    ):
        """
        Initialize batch size controller.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            target_memory_utilization: Target memory usage (0.0-1.0)
            safety_buffer_mb: Safety buffer to keep free (MB)
            use_gpu: Monitor GPU memory instead of CPU
            enable_prediction: Use MemoryPredictor for proactive adjustment
            memory_predictor: MemoryPredictor instance
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_utilization = target_memory_utilization
        self.safety_buffer_mb = safety_buffer_mb
        self.use_gpu = use_gpu
        self.enable_prediction = enable_prediction
        self.memory_predictor = memory_predictor
        
        # PID controller for smooth adjustments
        # Setpoint will be dynamically updated based on available memory
        self.pid = PIDController(
            kp=0.5,      # Moderate proportional gain
            ki=0.05,     # Small integral gain
            kd=0.1,      # Small derivative gain
            setpoint=safety_buffer_mb,
            output_limits=(min_batch_size, max_batch_size),
        )
        
        # History
        self.batch_size_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        self.oom_events = []
        
        # Statistics
        self.total_adjustments = 0
        self.increase_count = 0
        self.decrease_count = 0
        
    def get_memory_state(self) -> MemoryState:
        """Get current memory state"""
        # CPU memory
        mem = psutil.virtual_memory()
        total_mb = mem.total / (1024 * 1024)
        used_mb = mem.used / (1024 * 1024)
        available_mb = mem.available / (1024 * 1024)
        percent = mem.percent
        
        # GPU memory
        gpu_total = None
        gpu_used = None
        gpu_available = None
        gpu_percent = None
        
        if self.use_gpu and GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_total = gpu.memoryTotal
                    gpu_used = gpu.memoryUsed
                    gpu_available = gpu.memoryFree
                    gpu_percent = (gpu_used / gpu_total * 100) if gpu_total > 0 else 0
            except Exception:
                pass
        
        return MemoryState(
            timestamp=time.time(),
            total_memory_mb=total_mb,
            used_memory_mb=used_mb,
            available_memory_mb=available_mb,
            memory_percent=percent,
            gpu_total_mb=gpu_total,
            gpu_used_mb=gpu_used,
            gpu_available_mb=gpu_available,
            gpu_percent=gpu_percent,
        )
    
    def calculate_next_batch_size(
        self,
        memory_state: MemoryState,
        predicted_memory_per_sample: Optional[float] = None,
    ) -> int:
        """
        Calculate next batch size using PID control and predictions.
        
        Args:
            memory_state: Current memory state
            predicted_memory_per_sample: Predicted memory per sample (optional)
            
        Returns:
            Recommended batch size
        """
        available_mb = memory_state.get_available_memory(self.use_gpu)
        
        # Method 1: Direct calculation based on available memory
        # More memory available -> larger batch size
        if available_mb > self.safety_buffer_mb:
            usable_memory = available_mb - self.safety_buffer_mb
            # Scale batch size proportionally to usable memory
            # Normalize to a reasonable range
            memory_based_batch = int((usable_memory / 1000.0) * self.max_batch_size)
            memory_based_batch = np.clip(memory_based_batch, self.min_batch_size, self.max_batch_size)
        else:
            # Below safety buffer, use minimum
            memory_based_batch = self.min_batch_size
        
        # Method 2: Ratio-based adjustment (from JABAS paper)
        # B_next = B_curr × (M_target / M_curr)
        total_mb = memory_state.total_memory_mb if not self.use_gpu else (memory_state.gpu_total_mb or 1000)
        target_used = total_mb * self.target_utilization
        current_used = total_mb - available_mb
        
        if current_used > 0:
            ratio = target_used / current_used
            # Clamp ratio to prevent extreme changes
            ratio = np.clip(ratio, 0.5, 2.0)
            ratio_batch_size = int(self.current_batch_size * ratio)
        else:
            ratio_batch_size = self.current_batch_size
        
        # Method 3: Prediction-based adjustment (if predictor available)
        prediction_batch_size = None
        if self.enable_prediction and predicted_memory_per_sample:
            # Calculate how many samples can fit
            usable_memory = available_mb - self.safety_buffer_mb
            if usable_memory > 0 and predicted_memory_per_sample > 0:
                prediction_batch_size = int(usable_memory / predicted_memory_per_sample)
        
        # Combine methods (weighted average)
        candidates = []
        weights = []
        
        candidates.append(memory_based_batch)
        weights.append(0.4)  # 40% weight on memory-based
        
        candidates.append(ratio_batch_size)
        weights.append(0.3)  # 30% weight on ratio
        
        if prediction_batch_size is not None:
            candidates.append(prediction_batch_size)
            weights.append(0.3)  # 30% weight on prediction
        
        # Weighted average
        next_batch_size = int(np.average(candidates, weights=weights))
        
        # Apply bounds
        next_batch_size = np.clip(next_batch_size, self.min_batch_size, self.max_batch_size)
        
        # Smooth changes (avoid drastic jumps)
        max_change = max(1, int(self.current_batch_size * 0.5))  # Max 50% change per step
        if abs(next_batch_size - self.current_batch_size) > max_change:
            if next_batch_size > self.current_batch_size:
                next_batch_size = self.current_batch_size + max_change
            else:
                next_batch_size = self.current_batch_size - max_change
        
        return int(next_batch_size)
    
    def update_batch_size(
        self,
        actual_memory_used: Optional[float] = None,
        predicted_memory_per_sample: Optional[float] = None,
    ) -> int:
        """
        Update batch size based on current memory state.
        
        Args:
            actual_memory_used: Actual memory used by last batch (for feedback)
            predicted_memory_per_sample: Predicted memory per sample
            
        Returns:
            New batch size
        """
        # Get current memory state
        memory_state = self.get_memory_state()
        
        # Calculate next batch size
        next_batch_size = self.calculate_next_batch_size(
            memory_state,
            predicted_memory_per_sample,
        )
        
        # Update statistics
        if next_batch_size > self.current_batch_size:
            self.increase_count += 1
        elif next_batch_size < self.current_batch_size:
            self.decrease_count += 1
        
        if next_batch_size != self.current_batch_size:
            self.total_adjustments += 1
        
        # Update current batch size
        old_batch_size = self.current_batch_size
        self.current_batch_size = next_batch_size
        
        # Record history
        self.batch_size_history.append({
            'timestamp': time.time(),
            'old_batch': old_batch_size,
            'new_batch': next_batch_size,
            'available_mb': memory_state.get_available_memory(self.use_gpu),
            'memory_percent': memory_state.memory_percent,
        })
        
        self.memory_history.append(memory_state)
        
        return next_batch_size
    
    def report_oom(self, batch_size: int, memory_mb: float):
        """Report an OOM event to adjust strategy"""
        self.oom_events.append({
            'timestamp': time.time(),
            'batch_size': batch_size,
            'memory_mb': memory_mb,
        })
        
        # Emergency reduction
        self.current_batch_size = max(1, batch_size // 2)
        self.max_batch_size = batch_size  # Don't go higher than OOM point
        
        # Reset PID to avoid windup
        self.pid.reset()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics"""
        return {
            'current_batch_size': self.current_batch_size,
            'min_batch_size': self.min_batch_size,
            'max_batch_size': self.max_batch_size,
            'total_adjustments': self.total_adjustments,
            'increase_count': self.increase_count,
            'decrease_count': self.decrease_count,
            'oom_events': len(self.oom_events),
            'avg_batch_size': np.mean([h['new_batch'] for h in self.batch_size_history]) if self.batch_size_history else 0,
        }


class MicroScheduler:
    """
    Micro-Scheduler with JABAS-style adaptive batching.
    
    Orchestrates:
    - Memory monitoring
    - Batch size control via PID
    - Memory prediction integration
    - OOM prevention
    
    Usage:
        scheduler = MicroScheduler(memory_predictor=predictor)
        
        for batch in data_loader:
            # Get recommended batch size
            batch_size = scheduler.get_batch_size()
            
            # Process batch
            result = process(batch[:batch_size])
            
            # Update scheduler with feedback
            scheduler.update(actual_memory_used=memory_mb)
    """
    
    def __init__(
        self,
        memory_predictor=None,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
        target_memory_utilization: float = 0.85,
        safety_buffer_mb: float = 1000.0,
        use_gpu: bool = False,
        enable_auto_adjust: bool = True,
    ):
        """
        Initialize micro-scheduler.
        
        Args:
            memory_predictor: MemoryPredictor instance
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_memory_utilization: Target memory usage (0.0-1.0)
            safety_buffer_mb: Safety buffer in MB
            use_gpu: Monitor GPU memory
            enable_auto_adjust: Enable automatic batch size adjustment
        """
        self.memory_predictor = memory_predictor
        self.enable_auto_adjust = enable_auto_adjust
        
        # Batch size controller
        self.controller = BatchSizeController(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            target_memory_utilization=target_memory_utilization,
            safety_buffer_mb=safety_buffer_mb,
            use_gpu=use_gpu,
            enable_prediction=memory_predictor is not None,
            memory_predictor=memory_predictor,
        )
        
        # State
        self.iteration = 0
        self.last_prediction = None
        
    def get_batch_size(self, sample_features=None) -> int:
        """
        Get recommended batch size for next iteration.
        
        Args:
            sample_features: Optional sample features for prediction
            
        Returns:
            Recommended batch size
        """
        if not self.enable_auto_adjust:
            return self.controller.current_batch_size
        
        # Get memory prediction if available
        predicted_per_sample = None
        if self.memory_predictor and sample_features:
            prediction = self.memory_predictor.predict(sample_features)
            if prediction:
                self.last_prediction = prediction
                # Estimate per-sample memory
                predicted_per_sample = prediction.predicted_memory_mb / sample_features.batch_size
        
        # Update batch size
        new_batch_size = self.controller.update_batch_size(
            predicted_memory_per_sample=predicted_per_sample,
        )
        
        self.iteration += 1
        return new_batch_size
    
    def update(self, actual_memory_used: float, sample_features=None):
        """
        Update scheduler with feedback from actual execution.
        
        Args:
            actual_memory_used: Actual memory used in MB
            sample_features: Sample features (for predictor update)
        """
        # Update memory predictor if available
        if self.memory_predictor and sample_features:
            self.memory_predictor.observe(sample_features, actual_memory_used)
    
    def report_oom(self, batch_size: int, memory_mb: float):
        """Report OOM event"""
        self.controller.report_oom(batch_size, memory_mb)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        stats = self.controller.get_stats()
        stats['iteration'] = self.iteration
        if self.last_prediction:
            stats['last_prediction'] = {
                'predicted_mb': self.last_prediction.predicted_memory_mb,
                'confidence_lower': self.last_prediction.confidence_lower,
                'confidence_upper': self.last_prediction.confidence_upper,
            }
        return stats
