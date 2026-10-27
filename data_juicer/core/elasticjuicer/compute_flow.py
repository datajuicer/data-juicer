"""Queue/rate driven leases over an already-created resident worker pool.

No Ray calls or batch mutations live here. A lease ends when its task completes;
changing a target never preempts a worker or unloads a resident model.
"""

import math
from dataclasses import asdict, dataclass

from .dag_plan import DAGContractError


@dataclass(frozen=True)
class ComputeFlowConfig:
    mode: str = "dynamic"
    pools: tuple = ()  # Empty selects every GPU pool; CPU pools are opt-in.
    interval_seconds: float = 0.5
    ewma_alpha: float = 0.25
    starvation_seconds: float = 5.0
    hysteresis_workers: int = 1
    initial_slots: tuple = ()  # Optional (group_name, workers) pairs.

    def validate(self):
        if self.mode not in ("dynamic", "static"):
            raise DAGContractError("compute flow mode must be dynamic or static")
        for value in (self.interval_seconds, self.starvation_seconds):
            if not math.isfinite(value) or value <= 0:
                raise DAGContractError("compute flow intervals must be positive and finite")
        if not math.isfinite(self.ewma_alpha) or not 0 < self.ewma_alpha <= 1:
            raise DAGContractError("compute flow ewma_alpha must be in (0, 1]")
        if type(self.hysteresis_workers) is not int or self.hysteresis_workers < 1:
            raise DAGContractError("hysteresis_workers must be a positive integer")
        if len(set(self.pools)) != len(self.pools):
            raise DAGContractError("duplicate compute flow pool")
        try:
            slots = dict(self.initial_slots)
        except (TypeError, ValueError) as error:
            raise DAGContractError("initial_slots requires (group, workers) pairs") from error
        if len(slots) != len(self.initial_slots) or any(type(n) is not int or n < 0 for n in slots.values()):
            raise DAGContractError("initial_slots requires unique groups and nonnegative integer counts")


@dataclass(frozen=True)
class ReadyWork:
    unit: int
    rows: int
    since: float


class ComputeFlowController:
    """Single driver authority for group concurrency, separate from local batch.

    Dynamic mode allocates soft targets by estimated remaining worker-seconds.
    Busy leases drain, idle capacity can be borrowed, and an age override prevents
    starvation even when there are more groups than workers. Static mode pins
    slots to groups, providing a fixed-allocation control with identical models.
    """

    def __init__(self, compiled, config):
        config.validate()
        self.config = config
        self.compiled = compiled
        self.pools = config.pools or tuple(p.name for p in compiled["pools"].values() if p.num_gpus)
        if not self.pools or not set(self.pools) <= compiled["pools"].keys():
            raise DAGContractError("compute flow requires known selected pools")
        self.groups = {
            p: tuple(g for g in compiled["group_order"] if compiled["groups"][g].pool == p) for p in self.pools
        }
        selected = {g for groups in self.groups.values() for g in groups}
        specified = dict(config.initial_slots)
        if not set(specified) <= selected:
            raise DAGContractError("initial_slots references an unselected group")
        self.targets, self.static_owner = {}, {}
        for pool, groups in self.groups.items():
            workers = compiled["pools"][pool].workers
            if set(groups) & specified.keys():
                if not set(groups) <= specified.keys() or sum(specified[g] for g in groups) != workers:
                    raise DAGContractError("initial_slots must cover a pool exactly and sum to its workers")
                allocation = {g: specified[g] for g in groups}
            else:
                allocation = {g: workers // len(groups) + int(i < workers % len(groups)) for i, g in enumerate(groups)}
            if config.mode == "static" and any(n == 0 for n in allocation.values()):
                raise DAGContractError("static allocation needs at least one worker per selected group")
            self.targets.update(allocation)
            owners = [g for g in groups for _ in range(allocation[g])]
            self.static_owner.update({(pool, i): g for i, g in enumerate(owners)})
        self.cost = {}  # EWMA seconds/input-row; a partial final batch is not a full batch.
        self.observations = {}
        self.last_group = {}
        self.last_plan_at = {p: -math.inf for p in self.pools}
        self.pool_epochs = {p: 0 for p in self.pools}
        self.epoch, self.reassignments, self.borrowed, self.dispatches = 0, 0, 0, 0
        self.rate_samples, self.cold_samples = 0, 0

    def observe(self, group, key, rows, seconds):
        if group not in self.targets:
            return
        identity = (key, group)
        count = self.observations.get(identity, 0)
        # Empty filtered units must neither poison rates nor consume warmup.
        if rows <= 0 or not math.isfinite(seconds) or seconds <= 0:
            return
        self.observations[identity] = count + 1
        if count == 0:
            self.cold_samples += 1  # Lazy model load/first CUDA use is not steady-state throughput.
            return
        sample = seconds / rows
        previous = self.cost.get(group, sample)
        self.cost[group] = previous + self.config.ewma_alpha * (sample - previous)
        self.rate_samples += 1

    def _cost(self, group):
        # An unknown group uses the mean observed cost, rather than an arbitrary
        # absolute unit that overwhelms groups whose throughput is already known.
        return self.cost.get(group, sum(self.cost.values()) / len(self.cost) if self.cost else 1.0)

    def replan(self, ready, running, now):
        """Return bounded diagnostic snapshots; caller persists them if desired."""
        snapshots = []
        for pool, groups in self.groups.items():
            if now - self.last_plan_at[pool] < self.config.interval_seconds:
                continue
            self.last_plan_at[pool] = now
            backlog = {g: sum(max(1, w.rows) for w in ready.get(g, ())) for g in groups}
            counts = {g: len(ready.get(g, ())) + len(running.get(g, ())) for g in groups}
            work = {g: (backlog[g] + sum(max(1, rows) for rows in running.get(g, ()))) * self._cost(g) for g in groups}
            if self.config.mode == "dynamic":
                allocation = {g: 0 for g in groups}
                active = [g for g in groups if counts[g]]
                workers = self.compiled["pools"][pool].workers
                # Give each active group a floor when the pool can afford it.
                if len(active) <= workers:
                    for g in active:
                        allocation[g] = 1
                for _ in range(workers - sum(allocation.values())):
                    eligible = [g for g in active if allocation[g] < counts[g]]
                    if not eligible:
                        break
                    g = max(eligible, key=lambda name: work[name] / (allocation[name] + 1))
                    allocation[g] += 1
                changed_active = any(bool(allocation[g]) != bool(self.targets[g]) for g in groups)
                distance = max(abs(allocation[g] - self.targets[g]) for g in groups)
                if changed_active or distance >= self.config.hysteresis_workers:
                    self.targets.update(allocation)
            self.epoch += 1
            self.pool_epochs[pool] = self.epoch
            snapshots.append(
                dict(
                    kind="plan",
                    epoch=self.epoch,
                    at=now,
                    pool=pool,
                    mode=self.config.mode,
                    targets={g: self.targets[g] for g in groups},
                    ready_rows=backlog,
                    running={g: len(running.get(g, ())) for g in groups},
                    estimated_worker_seconds=work,
                    seconds_per_row={g: self.cost.get(g) for g in groups},
                )
            )
        return snapshots

    def choose(self, pool, idle_keys, ready, running_counts, now):
        """Select one idle actor and one ready group without mutating queues."""
        groups = [g for g in self.groups[pool] if ready.get(g)]
        if not groups or not idle_keys:
            return None
        if self.config.mode == "static":
            for key in idle_keys:
                group = self.static_owner[key]
                if group in groups:
                    return key, group, "static"
            return None
        overdue = [g for g in groups if now - ready[g][0].since >= self.config.starvation_seconds]
        under = [g for g in groups if running_counts.get(g, 0) < self.targets[g]]
        if overdue:
            group = min(overdue, key=lambda g: ready[g][0].since)
            reason = "age"
        else:
            candidates = under or groups
            # Prefer large estimated queue work per occupied slot; downstream
            # wins ties to release live payloads earlier.
            group = max(
                candidates,
                key=lambda g: (
                    sum(max(1, w.rows) for w in ready[g]) * self._cost(g) / (running_counts.get(g, 0) + 1),
                    self.compiled["group_order"].index(g),
                ),
            )
            reason = "target" if under else "borrow"
        return idle_keys[0], group, reason

    def dispatched(self, key, group, reason):
        previous = self.last_group.get(key)
        moved = previous is not None and previous != group
        self.reassignments += int(moved)
        self.borrowed += int(reason == "borrow")
        self.dispatches += 1
        self.last_group[key] = group
        return dict(
            kind="lease",
            epoch=self.pool_epochs[key[0]],
            pool=key[0],
            slot=key[1],
            group=group,
            previous_group=previous,
            reassigned=moved,
            reason=reason,
        )

    def summary(self):
        return dict(
            config=asdict(self.config),
            epochs=self.epoch,
            targets=dict(self.targets),
            seconds_per_row=dict(self.cost),
            rate_samples=self.rate_samples,
            cold_samples_excluded=self.cold_samples,
            dispatches=self.dispatches,
            reassignments=self.reassignments,
            borrowed=self.borrowed,
            authority="idle actor group leases; actor-local batch state is untouched",
            live_actor_resize=False,
        )
