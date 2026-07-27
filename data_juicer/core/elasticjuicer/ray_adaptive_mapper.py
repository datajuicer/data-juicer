"""Ray actor integration for lossless adaptive Mapper micro-batching."""

import traceback
from collections.abc import Mapping
from typing import Callable, Optional, Sequence
from uuid import uuid4

from loguru import logger

from .actor_resource_sampler import ActorResourceSampler
from .adaptive_mapper import OOMSafeAdaptiveMapper
from .async_metrics_sink import ActorControlMetrics, AsyncMetricsReporter
from .batch_controller import AdaptiveBatchController
from .control_service import ActorControlPoller
from .oom import is_oom_error
from .quota import (
    ActorQuotaState,
    ActorRegistration,
    QuotaEnvelope,
    apply_batch_size_quota,
)
from .stage_identity import operator_fingerprint


def _detect_resource_class() -> str:
    """Best-effort device class so profiles never cross incompatible hardware."""

    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "cuda:" + "-".join(name.lower().split())
    except Exception:
        pass
    return "cpu"


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
        actor_incarnation_id: Optional[str] = None,
        stage_id: str = "stage-0",
        control_service=None,
        control_poll_interval_sec: float = 0.1,
        control_poller_factory: Callable = ActorControlPoller,
        oom_reprobe_successes: int = 32,
        max_oom_reprobes: int = 1,
        profile_seed_enabled: bool = False,
        profile_seed_timeout_sec: float = 2.0,
        partition_id: Optional[int] = None,
        resource_class: Optional[str] = None,
        profile_report_max_attempts: int = 5,
        profile_ack_fn: Optional[Callable] = None,
    ):
        operator_args = tuple(operator_args or ())
        operator_kwargs = dict(operator_kwargs or {})
        max_batch_size = initial_batch_size if max_batch_size is None else max_batch_size

        self.operator = operator_class(*operator_args, **operator_kwargs)
        self.op_name = op_name or getattr(self.operator, "_name", None) or operator_class.__name__
        self.controller = AdaptiveBatchController(
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            oom_reprobe_successes=oom_reprobe_successes,
            max_oom_reprobes=max_oom_reprobes,
            recovery_requires_hint=True,
        )
        self.job_id = job_id
        self.actor_id = actor_id or uuid4().hex
        self.actor_incarnation_id = actor_incarnation_id or uuid4().hex
        self.stage_id = stage_id
        if partition_id is not None and (
            isinstance(partition_id, bool) or not isinstance(partition_id, int) or partition_id < 0
        ):
            raise ValueError("partition_id must be a non-negative integer or None")
        self.partition_id = partition_id
        self.op_fingerprint = operator_fingerprint(self.operator)
        self.resource_class = resource_class or _detect_resource_class()
        self._last_quota_revision = 0
        self._control_service_handle = control_service
        self.profile_seed_enabled = bool(profile_seed_enabled)
        self.profile_seed_timeout_sec = profile_seed_timeout_sec
        self._profile_control_service = control_service if self.profile_seed_enabled else None
        self._last_reported_profile = None
        self._pending_profile_learned = None
        if profile_report_max_attempts < 1:
            raise ValueError("profile_report_max_attempts must be at least 1")
        self._profile_report_max_attempts = profile_report_max_attempts
        self._profile_ack_fn = profile_ack_fn
        self._profile_report_attempts_for_current = 0
        self._profile_reports_attempted = 0
        self._profile_reports_delivered = 0
        self._profile_reports_failed = 0
        self._profile_reports_dropped = 0
        if self._profile_control_service is not None and self.job_id:
            self._seed_from_stage_profile(self._profile_control_service)
        self.sampler = sampler_factory(sample_interval_sec=sample_interval_sec)
        self.metrics_reporter = None
        if metrics_sink is not None:
            self.metrics_reporter = AsyncMetricsReporter(
                sink_handle=metrics_sink,
                job_id=self.job_id,
                actor_id=self.actor_id,
                actor_incarnation_id=self.actor_incarnation_id,
                stage_id=self.stage_id,
                op_name=self.op_name,
                max_in_flight=metrics_max_in_flight,
                control_state_provider=self._metrics_control_state,
                partition_id=self.partition_id,
            )
        self.mapper = OOMSafeAdaptiveMapper(
            mapper=self._process_strict,
            controller=self.controller,
            sampler=self.sampler,
            max_retries_per_slice=max_retries_per_slice,
            before_slice=self._apply_pending_quota,
            snapshot_callback=None if self.metrics_reporter is None else self.metrics_reporter.report,
        )
        self.control_poller = None
        if control_service is not None:
            if not self.job_id:
                raise ValueError("control_service requires a job_id")
            registration = ActorRegistration(
                job_id=self.job_id,
                stage_id=self.stage_id,
                op_name=self.op_name,
                actor_id=self.actor_id,
                actor_incarnation_id=self.actor_incarnation_id,
                static_min_batch_size=min_batch_size,
                static_max_batch_size=max_batch_size,
                partition_id=self.partition_id,
            )
            self.control_poller = control_poller_factory(
                control_handle=control_service,
                registration=registration,
                poll_interval_sec=control_poll_interval_sec,
            )
            self.control_poller.start()

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
        finally:
            self._report_stage_profile()

    def _seed_from_stage_profile(self, control_service) -> None:
        """Adopt a prior incarnation's learned bounds once, before any slice.

        The fetch is bounded and best-effort; on any failure the incarnation
        starts unseeded. Afterwards the actor-local controller remains the
        only authority over the executable micro-batch (RFC section 4.5).
        """

        try:
            import ray

            profile = ray.get(
                self._remote_method(
                    control_service.get_stage_profile,
                    self.job_id,
                    self.stage_id,
                    self.op_fingerprint,
                    self.resource_class,
                ),
                timeout=self.profile_seed_timeout_sec,
            )
        except Exception as error:
            logger.warning(f"ElasticJuicer stage-profile seed fetch failed; starting unseeded: {error}")
            return
        if profile is None:
            return
        seeded_size = self.controller.seed_bounds(
            safe_batch_size=profile.safe_batch_size,
            oom_upper_bound=profile.oom_upper_bound,
        )
        logger.info(
            f"ElasticJuicer seeded actor {self.actor_id} on stage {self.stage_id} from a prior "
            f"incarnation profile: batch_size={seeded_size}, oom_upper_bound={self.controller.oom_upper_bound}"
        )

    def _report_stage_profile(self) -> None:
        """Publish learned bounds with a bounded ack wait at batch boundaries.

        Delivery is only recorded after the control service acknowledges the
        merge, there is never more than one report in flight (the ack wait is
        synchronous and bounded), and an undelivered change is retried at the
        next boundary up to a bounded number of attempts before it is dropped.
        """

        if self._profile_control_service is None or not self.job_id:
            return
        state = self.controller.state
        learned = (state.success_lower_bound, state.oom_upper_bound)
        if learned == (None, None) or learned == self._last_reported_profile:
            return
        if learned != self._pending_profile_learned:
            self._pending_profile_learned = learned
            self._profile_report_attempts_for_current = 0
        if self._profile_report_attempts_for_current >= self._profile_report_max_attempts:
            return
        from .control_service import StageProfile
        from .quota import current_time_ms

        self._profile_reports_attempted += 1
        self._profile_report_attempts_for_current += 1
        try:
            profile = StageProfile(
                job_id=self.job_id,
                stage_id=self.stage_id,
                op_name=self.op_name,
                safe_batch_size=state.success_lower_bound,
                oom_upper_bound=state.oom_upper_bound,
                observed_at_ms=current_time_ms(),
                op_fingerprint=self.op_fingerprint,
                resource_class=self.resource_class,
                partition_id=self.partition_id,
            )
            reference = self._remote_method(self._profile_control_service.report_stage_profile, profile)
            self._await_profile_ack(reference)
        except Exception as error:
            self._profile_reports_failed += 1
            if self._profile_report_attempts_for_current >= self._profile_report_max_attempts:
                self._profile_reports_dropped += 1
                logger.warning(
                    f"ElasticJuicer stage-profile report dropped after "
                    f"{self._profile_report_attempts_for_current} attempts: {error}"
                )
            else:
                logger.warning(f"ElasticJuicer stage-profile report failed; will retry at the next boundary: {error}")
            return
        self._profile_reports_delivered += 1
        self._last_reported_profile = learned

    def _await_profile_ack(self, reference) -> None:
        """Block briefly until the control service confirms the merge."""

        if self._profile_ack_fn is not None:
            self._profile_ack_fn(reference)
            return
        import ray

        ray.get(reference, timeout=self.profile_seed_timeout_sec)

    def get_profile_report_state(self):
        """Return delivery counters for the advisory stage-profile channel."""

        return {
            "enabled": self._profile_control_service is not None,
            "attempted": self._profile_reports_attempted,
            "delivered": self._profile_reports_delivered,
            "failed": self._profile_reports_failed,
            "dropped": self._profile_reports_dropped,
            "last_delivered_bounds": self._last_reported_profile,
        }

    @staticmethod
    def _remote_method(method, *args):
        remote = getattr(method, "remote", None)
        if not callable(remote):
            raise TypeError("control service methods must expose .remote")
        return remote(*args)

    def _apply_quota(self, quota: QuotaEnvelope):
        """Apply one locally cached cap without replacing actor-local learning."""

        if not self.job_id:
            raise RuntimeError("actor must have a job_id before a quota can be applied")
        application = apply_batch_size_quota(
            self.controller,
            quota,
            expected_job_id=self.job_id,
            expected_actor_id=self.actor_id,
            expected_actor_incarnation_id=self.actor_incarnation_id,
            last_revision=self._last_quota_revision,
        )
        if application.applied:
            self._last_quota_revision = quota.revision
        return application

    def _apply_pending_quota(self):
        if self.control_poller is None:
            return None
        try:
            self.control_poller.poll_once()
            quota = self.control_poller.take_pending()
        except Exception as error:
            logger.warning(f"ElasticJuicer control poll failed at actor slice boundary: {error}")
            return None
        if quota is None:
            return None
        try:
            return self._apply_quota(quota)
        except Exception as error:
            logger.warning(f"Ignoring invalid ElasticJuicer quota at actor batch boundary: {error}")
            return None

    def get_quota_state(self) -> ActorQuotaState:
        """Return immutable quota and local-bound diagnostics."""

        state = self.controller.state
        return ActorQuotaState(
            job_id=self.job_id,
            actor_id=self.actor_id,
            actor_incarnation_id=self.actor_incarnation_id,
            last_revision=self._last_quota_revision,
            min_batch_size=state.min_batch_size,
            static_max_batch_size=state.max_batch_size,
            hard_limit=state.hard_limit,
            current_batch_size=state.current_batch_size,
            local_success_lower_bound=state.success_lower_bound,
            local_oom_upper_bound=state.oom_upper_bound,
        )

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

    def _metrics_control_state(self):
        state = self.controller.state
        return ActorControlMetrics(
            quota_revision=self._last_quota_revision,
            current_batch_size=state.current_batch_size,
            hard_limit=state.hard_limit,
            static_min_batch_size=state.min_batch_size,
            static_max_batch_size=state.max_batch_size,
            local_success_lower_bound=state.success_lower_bound,
            local_oom_upper_bound=state.oom_upper_bound,
        )

    def get_control_state(self):
        if self.control_poller is None:
            return {
                "enabled": False,
                "registered": False,
                "last_seen_revision": 0,
                "pending_revision": None,
                "poll_errors": 0,
                "last_error": None,
            }
        return self.control_poller.snapshot()

    def close(self):
        # Flush any undelivered learned-bound change before the actor retires
        # so a later incarnation can still be seeded from this partition.
        try:
            self._report_stage_profile()
        except Exception as error:
            logger.warning(f"ElasticJuicer stage-profile flush on close failed: {error}")
        self._deregister_from_control()
        if self.control_poller is not None:
            self.control_poller.close()
        close_sampler = getattr(self.sampler, "close", None)
        if callable(close_sampler):
            close_sampler()

    def _deregister_from_control(self) -> None:
        """Best-effort lease release so the Captain never waits on this actor."""

        if self._control_service_handle is None or not self.job_id:
            return
        try:
            self._remote_method(
                self._control_service_handle.deregister,
                self.job_id,
                self.actor_id,
                self.actor_incarnation_id,
            )
        except Exception as error:
            logger.warning(f"ElasticJuicer control deregister failed; lease TTL will expire it: {error}")

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
