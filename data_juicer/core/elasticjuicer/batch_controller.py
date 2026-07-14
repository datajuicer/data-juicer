"""Deterministic actor-local batch-size control.

The controller is the sole authority for an actor's next batch size. A remote
coordinator may lower ``hard_limit``, but it does not blend or overwrite local
success/OOM state. The module deliberately has no runtime, Ray, GPU, or memory
sampler dependency.
"""

from dataclasses import dataclass
from typing import Optional


class MinimumBatchSizeOOM(RuntimeError):
    """Raised when even the configured minimum batch size cannot execute."""


@dataclass(frozen=True)
class BatchControllerState:
    """Immutable diagnostic snapshot of controller state."""

    current_batch_size: int
    min_batch_size: int
    max_batch_size: int
    hard_limit: int
    success_lower_bound: Optional[int]
    oom_upper_bound: Optional[int]
    consecutive_successes: int
    cooldown_remaining: int
    success_events: int
    oom_events: int


class AdaptiveBatchController:
    """A bounded success/OOM state machine with deterministic probing."""

    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
        successes_before_growth: int = 3,
        growth_step: int = 1,
        cooldown_successes: int = 2,
    ):
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be at least 1")
        if max_batch_size < min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        if not min_batch_size <= initial_batch_size <= max_batch_size:
            raise ValueError("initial_batch_size must be within configured bounds")
        if successes_before_growth < 1:
            raise ValueError("successes_before_growth must be at least 1")
        if growth_step < 1:
            raise ValueError("growth_step must be at least 1")
        if cooldown_successes < 0:
            raise ValueError("cooldown_successes must be non-negative")

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.successes_before_growth = successes_before_growth
        self.growth_step = growth_step
        self.cooldown_successes = cooldown_successes

        self.current_batch_size = initial_batch_size
        self.hard_limit = max_batch_size
        self.success_lower_bound: Optional[int] = None
        self.oom_upper_bound: Optional[int] = None
        self.consecutive_successes = 0
        self.cooldown_remaining = 0
        self.success_events = 0
        self.oom_events = 0

    @property
    def state(self) -> BatchControllerState:
        return BatchControllerState(
            current_batch_size=self.current_batch_size,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            hard_limit=self.hard_limit,
            success_lower_bound=self.success_lower_bound,
            oom_upper_bound=self.oom_upper_bound,
            consecutive_successes=self.consecutive_successes,
            cooldown_remaining=self.cooldown_remaining,
            success_events=self.success_events,
            oom_events=self.oom_events,
        )

    def next_batch_size(self, remaining_samples: int) -> int:
        """Return a bounded size; a final partial batch may be below the minimum."""

        if remaining_samples < 0:
            raise ValueError("remaining_samples must be non-negative")
        if remaining_samples == 0:
            return 0
        return min(self.current_batch_size, self._effective_maximum(), remaining_samples)

    def set_hard_limit(self, hard_limit: int) -> int:
        """Apply a driver-provided upper cap without replacing local state."""

        if not self.min_batch_size <= hard_limit <= self.max_batch_size:
            raise ValueError("hard_limit must be within static batch-size bounds")
        self.hard_limit = hard_limit
        self.current_batch_size = self._clamp(self.current_batch_size)
        return self.current_batch_size

    def observe_success(self, batch_size: int) -> int:
        """Record a successful batch and cautiously probe after a stable streak."""

        self._validate_observation(batch_size)
        self.success_events += 1

        # A final slice can be smaller than the controller's requested size.
        # It is successful work, but it provides no evidence that the current
        # size is safe and therefore must not advance cooldown or growth.
        if batch_size < self.current_batch_size:
            return self.current_batch_size

        if self.oom_upper_bound is None or batch_size < self.oom_upper_bound:
            if self.success_lower_bound is None:
                self.success_lower_bound = batch_size
            else:
                self.success_lower_bound = max(self.success_lower_bound, batch_size)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            self.consecutive_successes = 0
            self.current_batch_size = self._clamp(self.current_batch_size)
            return self.current_batch_size

        self.consecutive_successes += 1
        if self.consecutive_successes < self.successes_before_growth:
            return self.current_batch_size

        self.consecutive_successes = 0
        self.current_batch_size = self._next_probe()
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        """Make the failed size an exclusive upper bound and back off safely."""

        self._validate_observation(batch_size)
        self.oom_events += 1
        if self.oom_upper_bound is None:
            self.oom_upper_bound = batch_size
        else:
            self.oom_upper_bound = min(self.oom_upper_bound, batch_size)

        self.consecutive_successes = 0
        self.cooldown_remaining = self.cooldown_successes

        if batch_size <= self.min_batch_size:
            self.current_batch_size = self.min_batch_size
            raise MinimumBatchSizeOOM(f"OOM at minimum batch size {self.min_batch_size}; no smaller retry is allowed")

        if self.success_lower_bound is not None and self.success_lower_bound >= self.oom_upper_bound:
            self.success_lower_bound = None

        backed_off = max(self.min_batch_size, batch_size // 2)
        if self.success_lower_bound is not None:
            backed_off = max(backed_off, self.success_lower_bound)
        self.current_batch_size = self._clamp(backed_off)
        return self.current_batch_size

    def _next_probe(self) -> int:
        effective_maximum = self._effective_maximum()
        if self.oom_upper_bound is not None and self.success_lower_bound is not None:
            midpoint = (self.success_lower_bound + self.oom_upper_bound) // 2
            return self._clamp(max(self.success_lower_bound, midpoint))
        return self._clamp(min(effective_maximum, self.current_batch_size + self.growth_step))

    def _effective_maximum(self) -> int:
        maximum = min(self.max_batch_size, self.hard_limit)
        if self.oom_upper_bound is not None:
            maximum = min(maximum, self.oom_upper_bound - 1)
        return max(self.min_batch_size, maximum)

    def _clamp(self, batch_size: int) -> int:
        return max(self.min_batch_size, min(batch_size, self._effective_maximum()))

    def _validate_observation(self, batch_size: int):
        if not 1 <= batch_size <= self.max_batch_size:
            raise ValueError("observed batch_size must be within static batch-size bounds")
