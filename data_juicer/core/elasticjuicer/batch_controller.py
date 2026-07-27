"""Deterministic actor-local batch-size control.

The controller is the sole authority for an actor's next batch size. A remote
coordinator may constrain or relax ``hard_limit``, but it does not blend or
overwrite local success/OOM state. The module deliberately has no runtime,
Ray, GPU, or memory sampler dependency.
"""

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional


def _current_time_ms() -> int:
    return time.time_ns() // 1_000_000


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
    probe_oom_events: int
    oom_reprobe_events: int
    successes_since_oom: int
    capacity_recovery_hint_pending: bool
    capacity_recovery_hint_expires_at_ms: Optional[int]
    capacity_recovery_hints: int


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
        max_probe_ooms: int = 3,
        minimum_growth_fraction: float = 0.125,
        oom_reprobe_successes: int = 32,
        max_oom_reprobes: int = 0,
        recovery_requires_hint: bool = False,
        clock_ms: Callable[[], int] = _current_time_ms,
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
        if max_probe_ooms < 1:
            raise ValueError("max_probe_ooms must be at least 1")
        if not 0 < minimum_growth_fraction <= 1:
            raise ValueError("minimum_growth_fraction must be in (0, 1]")
        if oom_reprobe_successes < 1:
            raise ValueError("oom_reprobe_successes must be at least 1")
        if max_oom_reprobes < 0:
            raise ValueError("max_oom_reprobes must be non-negative")
        if not isinstance(recovery_requires_hint, bool):
            raise TypeError("recovery_requires_hint must be a boolean")
        if not callable(clock_ms):
            raise TypeError("clock_ms must be callable")

        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.successes_before_growth = successes_before_growth
        self.growth_step = growth_step
        self.cooldown_successes = cooldown_successes
        self.max_probe_ooms = max_probe_ooms
        self.minimum_growth_fraction = minimum_growth_fraction
        self.oom_reprobe_successes = oom_reprobe_successes
        self.max_oom_reprobes = max_oom_reprobes
        self.recovery_requires_hint = recovery_requires_hint
        self._clock_ms = clock_ms

        self.current_batch_size = initial_batch_size
        self.hard_limit = max_batch_size
        self.success_lower_bound: Optional[int] = None
        self.oom_upper_bound: Optional[int] = None
        self.consecutive_successes = 0
        self.cooldown_remaining = 0
        self.success_events = 0
        self.oom_events = 0
        self.probe_oom_events = 0
        self.oom_reprobe_events = 0
        self.successes_since_oom = 0
        self._probe_ooms_for_bound = 0
        self._next_reprobe_successes = oom_reprobe_successes
        self._reprobe_origin: Optional[int] = None
        self._capacity_recovery_hint_pending = False
        self._capacity_recovery_hint_expires_at_ms: Optional[int] = None
        self.capacity_recovery_hints = 0

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
            probe_oom_events=self.probe_oom_events,
            oom_reprobe_events=self.oom_reprobe_events,
            successes_since_oom=self.successes_since_oom,
            capacity_recovery_hint_pending=self._capacity_recovery_hint_pending,
            capacity_recovery_hint_expires_at_ms=self._capacity_recovery_hint_expires_at_ms,
            capacity_recovery_hints=self.capacity_recovery_hints,
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

    def seed_bounds(
        self,
        safe_batch_size: Optional[int] = None,
        oom_upper_bound: Optional[int] = None,
    ) -> int:
        """Adopt prior-incarnation learning as this controller's initial prior.

        This is an actor-local administrative API for incarnation start (for
        example cross-partition profile seeding). It is not part of the remote
        actuator surface and must run before the first observation so the
        controller stays the only authority afterwards. A seed can never relax
        static bounds: values are clamped into the configured range, and an
        advisory OOM bound at or below the minimum is ignored rather than
        bricking the incarnation.
        """

        if self.success_events or self.oom_events:
            raise RuntimeError("seed_bounds is only valid before the first observation")
        if oom_upper_bound is not None:
            if isinstance(oom_upper_bound, bool) or not isinstance(oom_upper_bound, int):
                raise TypeError("oom_upper_bound must be an integer")
            if self.min_batch_size < oom_upper_bound <= self.max_batch_size:
                self.oom_upper_bound = oom_upper_bound
        if safe_batch_size is not None:
            if isinstance(safe_batch_size, bool) or not isinstance(safe_batch_size, int):
                raise TypeError("safe_batch_size must be an integer")
            self.current_batch_size = self._clamp(safe_batch_size)
        else:
            self.current_batch_size = self._clamp(self.current_batch_size)
        return self.current_batch_size

    def reset_oom_bound(self) -> int:
        """Forget local OOM evidence through an actor-local administrative API.

        This method is not part of the remote quota actuator surface. Raising a
        quota alone is not proof that the actor's capacity has increased.
        """

        previous_upper_bound = self.oom_upper_bound
        self.oom_upper_bound = None
        self.successes_since_oom = 0
        self._probe_ooms_for_bound = 0
        self._next_reprobe_successes = self.oom_reprobe_successes
        self._reprobe_origin = previous_upper_bound
        self._capacity_recovery_hint_pending = False
        self._capacity_recovery_hint_expires_at_ms = None
        return self.current_batch_size

    def record_capacity_recovery_hint(self, expires_at_ms: int, now_ms: Optional[int] = None) -> bool:
        """Record advisory recovery evidence without deleting local OOM state."""

        self.capacity_recovery_hints += 1
        resolved_now = self._clock_ms() if now_ms is None else now_ms
        if expires_at_ms <= resolved_now:
            return False
        if (
            self.oom_upper_bound is None
            or self.oom_reprobe_events >= self.max_oom_reprobes
            or self.hard_limit < self.oom_upper_bound
        ):
            return False
        self._capacity_recovery_hint_pending = True
        self._capacity_recovery_hint_expires_at_ms = expires_at_ms
        return True

    def _expire_capacity_recovery_hint(self, now_ms: Optional[int] = None) -> None:
        if self._capacity_recovery_hint_expires_at_ms is None:
            return
        resolved_now = self._clock_ms() if now_ms is None else now_ms
        if resolved_now >= self._capacity_recovery_hint_expires_at_ms:
            self._capacity_recovery_hint_pending = False
            self._capacity_recovery_hint_expires_at_ms = None

    def observe_success(self, batch_size: int) -> int:
        """Record a successful batch and cautiously probe after a stable streak."""

        self._validate_observation(batch_size)
        self._expire_capacity_recovery_hint()
        self.success_events += 1

        # A final slice can be smaller than the controller's requested size.
        # It is successful work, but it provides no evidence that the current
        # size is safe and therefore must not advance cooldown or growth.
        if batch_size < self.current_batch_size:
            return self.current_batch_size

        if self.oom_upper_bound is not None:
            self.successes_since_oom += 1
            if (
                self.oom_reprobe_events < self.max_oom_reprobes
                and self.successes_since_oom >= self._next_reprobe_successes
                and (not self.recovery_requires_hint or self._capacity_recovery_hint_pending)
            ):
                previous_upper_bound = self.oom_upper_bound
                self.oom_upper_bound = None
                self.successes_since_oom = 0
                self._reprobe_origin = previous_upper_bound
                self._capacity_recovery_hint_pending = False
                self._capacity_recovery_hint_expires_at_ms = None

        if self._reprobe_origin is not None and batch_size >= self._reprobe_origin:
            # A previously failing size now succeeds, so capacity recovery is
            # proven and future recovery checks may use the base interval again.
            self._reprobe_origin = None
            self.oom_reprobe_events = 0
            self._next_reprobe_successes = self.oom_reprobe_successes
            self._probe_ooms_for_bound = 0
            self._capacity_recovery_hint_pending = False
            self._capacity_recovery_hint_expires_at_ms = None

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
        if self.success_lower_bound is not None and batch_size > self.success_lower_bound:
            self.probe_oom_events += 1
            self._probe_ooms_for_bound += 1
        if self._reprobe_origin is not None:
            self.oom_reprobe_events += 1
            self._next_reprobe_successes *= 2
            self._reprobe_origin = None
        if self.oom_upper_bound is None:
            self.oom_upper_bound = batch_size
        else:
            self.oom_upper_bound = min(self.oom_upper_bound, batch_size)

        self.consecutive_successes = 0
        self.cooldown_remaining = self.cooldown_successes
        self.successes_since_oom = 0
        # A new local failure supersedes an unused external hint.
        self._capacity_recovery_hint_pending = False
        self._capacity_recovery_hint_expires_at_ms = None

        if batch_size <= self.min_batch_size:
            self.current_batch_size = self.min_batch_size
            raise MinimumBatchSizeOOM(f"OOM at minimum batch size {self.min_batch_size}; no smaller retry is allowed")

        if self.success_lower_bound is not None and self.success_lower_bound >= self.oom_upper_bound:
            self.success_lower_bound = None
            self._probe_ooms_for_bound = 0

        if self.success_lower_bound is not None and self._probe_ooms_for_bound >= self.max_probe_ooms:
            # Once the probe budget is exhausted, freeze at the best proven
            # size. A later bounded re-probe or explicit reset can reopen it.
            self.oom_upper_bound = self.success_lower_bound + 1

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
        growth_increment = max(
            self.growth_step,
            int(math.ceil(self.current_batch_size * self.minimum_growth_fraction)),
        )
        return self._clamp(min(effective_maximum, self.current_batch_size + growth_increment))

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
