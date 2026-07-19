"""Ray actor integration for lossless adaptive Mapper micro-batching."""

import traceback
from collections.abc import Mapping
from typing import Callable, Optional, Sequence
from uuid import uuid4

from loguru import logger

from .actor_resource_sampler import ActorResourceSampler
from .adaptive_mapper import OOMSafeAdaptiveMapper
from .async_metrics_sink import AsyncMetricsReporter
from .batch_controller import AdaptiveBatchController
from .oom import is_oom_error
from .quota import ActorQuotaState, BatchSizeQuota, apply_batch_size_quota


class RayAdaptiveMapperActor:
    """Own one operator, controller, and sampler for the actor lifetime."""

    def __init__(
        self,
        operator_class,
        operator_args: Optional[Sequence] = None,
        operator_kwargs: Optional[Mapping] = None,
        initial_batch_size: int = 1,
        min_batch_size: int = 1,
        max_batch_size: Optional[int] = None,
        max_retries_per_slice: int = 16,
        sample_interval_sec: float = 0.01,
        sampler_factory: Callable = ActorResourceSampler,
        metrics_sink=None,
        metrics_max_in_flight: int = 64,
        job_id: Optional[str] = None,
        op_name: Optional[str] = None,
        actor_id: Optional[str] = None,
    ):
        operator_args = tuple(operator_args or ())
        operator_kwargs = dict(operator_kwargs or {})
        max_batch_size = initial_batch_size if max_batch_size is None else max_batch_size

        self.operator = operator_class(*operator_args, **operator_kwargs)
        self.controller = AdaptiveBatchController(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
        )
        self.job_id = job_id
        self.actor_id = actor_id or uuid4().hex
        self._last_quota_revision = 0
        self.sampler = sampler_factory(sample_interval_sec=sample_interval_sec)
        self.metrics_reporter = None
        if metrics_sink is not None:
            resolved_op_name = op_name or getattr(self.operator, "_name", None) or operator_class.__name__
            self.metrics_reporter = AsyncMetricsReporter(
                sink_handle=metrics_sink,
                job_id=self.job_id,
                actor_id=self.actor_id,
                op_name=resolved_op_name,
                max_in_flight=metrics_max_in_flight,
            )
            set_callback = getattr(self.sampler, "set_snapshot_callback", None)
            if not callable(set_callback):
                raise TypeError("sampler must support set_snapshot_callback when metrics_sink is configured")
            set_callback(self.metrics_reporter.report)
        self.mapper = OOMSafeAdaptiveMapper(
            mapper=self._process_strict,
            controller=self.controller,
            sampler=self.sampler,
            max_retries_per_slice=max_retries_per_slice,
        )

    def __call__(self, batch):
        try:
            return self.mapper(batch)
        except Exception as error:
            if is_oom_error(error) or not getattr(self.operator, "skip_op_error", False):
                raise
            logger.error(
                f"An error occurred in {getattr(self.operator, '_name', '')}: " f"{error} -- {traceback.format_exc()}"
            )
            return self._empty_batch(batch)

    def apply_quota(self, quota: BatchSizeQuota):
        """Apply a newer driver cap without replacing actor-local learning."""

        if not self.job_id:
            raise RuntimeError("actor must have a job_id before a quota can be applied")
        application = apply_batch_size_quota(
            self.controller,
            quota,
            expected_job_id=self.job_id,
            expected_actor_id=self.actor_id,
            last_revision=self._last_quota_revision,
        )
        if application.applied:
            self._last_quota_revision = quota.revision
        return application

    def get_quota_state(self) -> ActorQuotaState:
        """Return immutable quota and local-bound diagnostics."""

        state = self.controller.state
        return ActorQuotaState(
            job_id=self.job_id,
            actor_id=self.actor_id,
            last_revision=self._last_quota_revision,
            min_batch_size=state.min_batch_size,
            static_max_batch_size=state.max_batch_size,
            hard_limit=state.hard_limit,
            current_batch_size=state.current_batch_size,
            local_success_lower_bound=state.success_lower_bound,
            local_oom_upper_bound=state.oom_upper_bound,
        )

    def reset_oom_bound(self):
        """Allow a coordinator to explicitly reopen local capacity probing."""

        self.controller.reset_oom_bound()
        return self.get_quota_state()

    def get_metrics_state(self):
        """Return actor-local producer pressure diagnostics."""

        if self.metrics_reporter is None:
            return {
                "enabled": False,
                "submitted_events": 0,
                "dropped_events": 0,
                "pending_events": 0,
                "max_in_flight": 0,
                "last_sequence": 0,
            }
        return {"enabled": True, **self.metrics_reporter.snapshot()}

    def _process_strict(self, batch):
        """Run below Mapper's broad skip wrapper so OOM reaches the controller."""

        process_batched = getattr(self.operator, "process_batched", None)
        if process_batched is None:
            return self.operator.process(batch)
        if hasattr(batch, "to_pydict"):
            batch = batch.to_pydict()
        return process_batched(batch)

    @staticmethod
    def _empty_batch(batch):
        if hasattr(batch, "to_pydict"):
            batch = batch.to_pydict()
        if not isinstance(batch, Mapping):
            raise TypeError("a skipped batched Mapper input must be mapping-like")

        from data_juicer.utils.constant import Fields

        result = {key: [] for key in batch}
        result[Fields.stats] = []
        result[Fields.source_file] = []
        return result
