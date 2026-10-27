"""Fixed microbatch policy for the opt-in resident DAG executor."""

from dataclasses import dataclass
from typing import Optional

from .batch_controller import MinimumBatchSizeOOM


@dataclass
class FixedControllerState:
    """Diagnostic snapshot with the fields the harness observability reads."""

    current_batch_size: int
    min_batch_size: int
    max_batch_size: int
    hard_limit: int
    success_lower_bound: Optional[int]
    oom_upper_bound: Optional[int]
    consecutive_successes: int = 0
    cooldown_remaining: int = 0
    success_events: int = 0
    oom_events: int = 0
    probe_oom_events: int = 0
    oom_reprobe_events: int = 0
    successes_since_oom: int = 0
    capacity_recovery_hint_pending: bool = False
    capacity_recovery_hint_expires_at_ms: Optional[int] = None
    capacity_recovery_hints: int = 0


class _FixedBatchControllerBase:
    """Controller state and bounds retained from the resident DAG fixed policy."""

    def __init__(
        self,
        initial_batch_size: int,
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
    ):
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be at least 1")
        if max_batch_size < min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        if not min_batch_size <= initial_batch_size <= max_batch_size:
            raise ValueError("initial_batch_size must be within configured bounds")
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.hard_limit = max_batch_size
        self.success_lower_bound: Optional[int] = None
        self.oom_upper_bound: Optional[int] = None
        self.consecutive_successes = 0
        self.success_events = 0
        self.oom_events = 0

    # -- harness surface ------------------------------------------------------

    @property
    def state(self) -> FixedControllerState:
        return FixedControllerState(
            current_batch_size=self.current_batch_size,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            hard_limit=self.hard_limit,
            success_lower_bound=self.success_lower_bound,
            oom_upper_bound=self.oom_upper_bound,
            consecutive_successes=self.consecutive_successes,
            success_events=self.success_events,
            oom_events=self.oom_events,
        )

    def next_batch_size(self, remaining_samples: int) -> int:
        if remaining_samples <= 0:
            return 0
        return min(self.current_batch_size, self._effective_maximum(), remaining_samples)

    def set_hard_limit(self, hard_limit: int) -> int:
        if not self.min_batch_size <= hard_limit <= self.max_batch_size:
            raise ValueError("hard_limit must be within static batch-size bounds")
        self.hard_limit = hard_limit
        self.current_batch_size = self._clamp(self.current_batch_size)
        return self.current_batch_size

    def seed_bounds(
        self,
        safe_batch_size: Optional[int] = None,
        oom_upper_bound: Optional[int] = None,
    ) -> int:
        """Adopt prior learning as an initial prior (same semantics as EJ)."""
        if oom_upper_bound is not None and oom_upper_bound > self.min_batch_size:
            self.oom_upper_bound = (
                oom_upper_bound if self.oom_upper_bound is None else min(self.oom_upper_bound, oom_upper_bound)
            )
        if safe_batch_size is not None and safe_batch_size >= self.min_batch_size:
            if self.success_lower_bound is None:
                self.success_lower_bound = safe_batch_size
            else:
                self.success_lower_bound = max(self.success_lower_bound, safe_batch_size)
        self.current_batch_size = self._clamp(self._initial_from_seed(safe_batch_size))
        return self.current_batch_size

    def _initial_from_seed(self, safe_batch_size: Optional[int]) -> int:
        if safe_batch_size is not None and safe_batch_size >= self.min_batch_size:
            return safe_batch_size
        return self.current_batch_size

    def record_capacity_recovery_hint(self, expires_at_ms: int, now_ms: Optional[int] = None) -> bool:
        # Fixed batching does not consume recovery hints.
        return False

    def reset_oom_bound(self) -> int:
        self.oom_upper_bound = None
        return self.current_batch_size

    # -- shared helpers -------------------------------------------------------

    def _effective_maximum(self) -> int:
        return min(self.hard_limit, self.max_batch_size)

    def _clamp(self, size: int) -> int:
        upper = self._effective_maximum()
        if self.oom_upper_bound is not None:
            upper = min(upper, max(self.min_batch_size, self.oom_upper_bound - 1))
        return max(self.min_batch_size, min(size, upper))

    def _record_success(self, batch_size: int) -> None:
        self.success_events += 1
        self.consecutive_successes += 1
        if self.success_lower_bound is None:
            self.success_lower_bound = batch_size
        else:
            self.success_lower_bound = max(self.success_lower_bound, batch_size)

    def _record_oom(self, batch_size: int) -> None:
        self.oom_events += 1
        self.consecutive_successes = 0
        self.oom_upper_bound = batch_size if self.oom_upper_bound is None else min(self.oom_upper_bound, batch_size)
        if self.oom_upper_bound <= self.min_batch_size:
            raise MinimumBatchSizeOOM(f"OOM at minimum batch size {self.min_batch_size}")


class FixedBatchController(_FixedBatchControllerBase):
    """Static configuration arm: constant size, OOM propagates as misconfig."""

    def observe_success(self, batch_size: int) -> int:
        self._record_success(batch_size)
        return self.current_batch_size

    def observe_oom(self, batch_size: int) -> int:
        self._record_oom(batch_size)
        # Fixed configs have no fallback: surface the failure like a static
        # config that picked an infeasible batch.
        raise MinimumBatchSizeOOM(f"fixed batch {self.current_batch_size} hit OOM")
