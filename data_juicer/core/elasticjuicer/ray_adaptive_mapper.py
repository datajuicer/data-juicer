"""Optional microbatch execution inside PR1054's existing Ray actor pools."""

import gc
import sys
import time
import uuid
from collections import Counter, OrderedDict

import pyarrow as pa
from loguru import logger

from data_juicer.ops import Mapper
from data_juicer.utils.constant import Fields

from .adaptive_mapper import AdaptiveBatchContractError, OOMSafeAdaptiveMapper
from .batch_controller import AdaptiveBatchController
from .oom import is_oom_error
from .profile_context import make_profile_context
from .stage_identity import STAGE_IDENTITY_ATTR
from .stage_profile import (
    MAX_OBSERVED_SIZES,
    PROFILE_RPC_TIMEOUT_SECONDS,
    validate_prior,
)

# PR1054 execution-group routing column; avoid importing the executor here.
_PARTITION_COLUMN = "__data_juicer_logical_partition_id__"


def adaptive_batching_enabled(op, enabled=False):
    """Validate an opted-in, row-preserving, slice-independent GPU Mapper."""
    if not enabled:
        return False
    requested = getattr(op, "adaptive_batching", None)
    if requested is None:
        requested = getattr(op, "_supports_adaptive_batching", False)
    if not requested:
        return False
    name = getattr(op, "_name", None) or type(op).__name__
    if not isinstance(op, Mapper) or not op.is_batched_op():
        raise ValueError(f"{name}: adaptive batching requires a batched Mapper")
    if op.accelerator != "cuda" or op.ray_execution_mode == "task":
        raise ValueError(f"{name}: adaptive batching requires a CUDA Ray actor")
    if op.num_gpus is not None and not 0 < op.num_gpus <= 1:
        raise ValueError(f"{name}: adaptive batching supports at most one GPU per actor")
    if isinstance(op.batch_size, bool) or not isinstance(op.batch_size, int) or op.batch_size < 1:
        raise ValueError(f"{name}: adaptive batching requires a positive integer batch_size")
    return True


def _cleanup_cuda():
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


class RayAdaptiveMapperActor:
    """Keep one model and local batch-size state across Ray outer batches.

    Ray resource reservations and outer batch size stay fixed. Opt-in asserts
    row/order preservation, slice independence and retry safety. External side
    effects and operator-owned OOM swallowing are not compatible.
    """

    def __init__(
        self,
        op_class,
        op_args,
        op_kwargs,
        max_batch_size,
        stage_id=None,
        profile_store=None,
        profile_request=None,
        profile_resource_envelope=None,
    ):
        self.op = op_class(*op_args, **op_kwargs)
        if stage_id:
            setattr(self.op, STAGE_IDENTITY_ATTR, stage_id)
        self.stage_id = stage_id or self.op._name or op_class.__name__
        self.controller = AdaptiveBatchController(
            initial_batch_size=max_batch_size,
            max_batch_size=max_batch_size,
            max_oom_reprobes=4,
            minimum_growth_fraction=0.25,
        )
        self.actor_id = uuid.uuid4().hex
        self.profile_diagnostics = Counter()
        self._profile_store = profile_store
        self._profile_request = profile_request
        self._profile_resource_envelope = profile_resource_envelope
        self._seed_attempted = False
        self._seeded_context = None
        self._observations = None
        # This bounded history belongs to this incarnation, including when
        # StageProfile is disabled. Only fully validated outer calls add proof.
        self._recovery_history = OrderedDict()
        self._recovery_context = None
        self._last_recovery_context = None
        self._recovery_successes = set()
        self.mapper = OOMSafeAdaptiveMapper(
            self._process_slice,
            self.controller,
            oom_cleanup=_cleanup_cuda,
            label=self.stage_id,
            observation_callback=self._observe,
        )

    def _profile_rpc(self, method, *args, **kwargs):
        import ray

        try:
            return ray.get(
                getattr(self._profile_store, method).remote(*args, **kwargs), timeout=PROFILE_RPC_TIMEOUT_SECONDS
            )
        except Exception as error:
            self.profile_diagnostics["rpc_errors"] += 1
            # A service outage costs at most one bounded wait per incarnation.
            self._profile_store = None
            logger.warning(f"StageProfile[{self.stage_id}] unavailable; using local batching: {type(error).__name__}")
            return None

    def _observe(self, kind, size):
        context = self._recovery_context
        if context is not None:
            if kind == "successes":
                self._recovery_successes.add(size)
                if len(self._recovery_successes) > MAX_OBSERVED_SIZES:
                    self._recovery_successes.remove(min(self._recovery_successes))
            else:
                self._recovery_successes = {value for value in self._recovery_successes if value < size}
                proof = self._recovery_history.get(context)
                if proof is not None and proof[0] >= size:
                    del self._recovery_history[context]
        if self._observations is not None:
            sizes = self._observations[kind]
            sizes.add(size)
            if len(sizes) > MAX_OBSERVED_SIZES:
                # Always retain a genuinely observed small success below OOMs.
                ordered = sorted(sizes)
                keep = (
                    ordered[:MAX_OBSERVED_SIZES]
                    if kind == "ooms"
                    else ordered[:1] + ordered[-(MAX_OBSERVED_SIZES - 1) :]
                )
                self._observations[kind] = set(keep)

    def _context(self, batch):
        try:
            return make_profile_context(batch, self.op, self.controller, self._profile_resource_envelope)[0]
        except Exception:
            return None

    def _prepare_recovery(self, batch):
        self._recovery_context = self._context(batch)
        if self._recovery_context != self._last_recovery_context:
            self.controller.clear_context_recovery_hint()
        self._last_recovery_context = self._recovery_context
        self._recovery_successes = set()
        now = time.time_ns() // 1_000_000
        for context, proof in list(self._recovery_history.items()):
            if proof[1] <= now:
                del self._recovery_history[context]
        proof = self._recovery_history.get(self._recovery_context)
        upper = self.controller.oom_upper_bound
        if proof is not None and not proof[2] and upper is not None and proof[0] >= upper:
            if self.controller.record_context_recovery_hint(proof[1]):
                # The same old proof cannot repeatedly rearm a failing probe.
                self._recovery_history[self._recovery_context] = (proof[0], proof[1], True)

    def _publish_recovery(self, batch):
        context = self._recovery_context
        if context is None or not self._recovery_successes or context != self._context(batch):
            return
        size = max(self._recovery_successes)
        proof = self._recovery_history.get(context)
        if proof is None or size >= proof[0]:
            # Smaller successes must not renew larger, older capacity evidence.
            self._recovery_history[context] = (size, time.time_ns() // 1_000_000 + 3_600_000, False)
        self._recovery_history.move_to_end(context)
        while len(self._recovery_history) > 64:
            self._recovery_history.popitem(last=False)

    def _prepare_profile(self, batch):
        if self._profile_store is None or not self._profile_request:
            return None
        try:
            context, reason = make_profile_context(batch, self.op, self.controller, self._profile_resource_envelope)
        except Exception:
            context, reason = None, "invalid_context_contract"
        if context is None:
            self.profile_diagnostics[reason] += 1
            if self.profile_diagnostics[reason] == 1:
                logger.debug(f"StageProfile[{self.stage_id}] cold start: {reason}")
            return None
        request = {**self._profile_request, "context": context}
        state = self.controller.state
        if not self._seed_attempted and not state.success_events and not state.oom_events:
            self._seed_attempted = True
            prior = self._profile_rpc("read", request)
            if prior is not None:
                try:
                    safe, upper = validate_prior(prior, request)
                    self.controller.seed_bounds(safe, upper)
                    self._seeded_context = context
                    self.profile_diagnostics["seeded"] += 1
                    logger.info(
                        f"StageProfile[{self.stage_id}] seeded actor {self.actor_id}: batch={safe}, OOM={upper}"
                    )
                except (ValueError, TypeError, KeyError, RuntimeError):
                    self.profile_diagnostics["rejected_prior"] += 1
                    logger.warning(f"StageProfile[{self.stage_id}] rejected invalid prior; using local batching")
        if self._profile_store is not None:
            self._observations = {"successes": set(), "ooms": set()}
            return request
        return None

    def _publish_profile(self, batch, request):
        if request is None or self._profile_store is None or self._observations is None:
            return
        try:
            context, _ = make_profile_context(batch, self.op, self.controller, self._profile_resource_envelope)
        except Exception:
            context = None
        if context != request["context"]:
            self.profile_diagnostics["context_changed_during_call"] += 1
            return
        if not any(self._observations.values()):
            return
        observations = {kind: sorted(sizes) for kind, sizes in self._observations.items()}
        if self._profile_rpc("publish", request, observations, self.actor_id, self._seeded_context == context):
            self.profile_diagnostics["published"] += 1

    def _process_slice(self, batch):
        if isinstance(batch, pa.Table):
            batch = batch.to_pydict()
        tags = list(batch[_PARTITION_COLUMN]) if _PARTITION_COLUMN in batch else None
        # Bypass Mapper's generic skip wrapper so OOM reaches the controller.
        output = self.op.process_batched(batch)
        if isinstance(output, pa.Table):
            output = output.to_pydict()
        if tags is not None:
            if (
                not isinstance(output, dict)
                or _PARTITION_COLUMN not in output
                or list(output[_PARTITION_COLUMN]) != tags
            ):
                raise AdaptiveBatchContractError(f"{self.stage_id}: mapper changed logical partition tags")
        return output

    def __call__(self, batch):
        self._observations = None
        request = self._prepare_profile(batch)
        self._prepare_recovery(batch)
        try:
            result = self.mapper(batch)
        except Exception as error:
            if is_oom_error(error) or isinstance(error, AdaptiveBatchContractError) or not self.op.skip_op_error:
                raise
            # Match the original policy: an ordinary error skips the entire
            # outer batch, including any previously successful microbatches.
            logger.exception(f"An error occurred in {self.stage_id}; skipping the outer batch")
            keys = batch.column_names if isinstance(batch, pa.Table) else batch.keys()
            result = {key: [] for key in keys}
            result[Fields.stats] = []
            result[Fields.source_file] = []
            return result
        else:
            # No partially processed/skipped/contract-invalid outer batch can
            # contribute a successful profile, even if early slices succeeded.
            self._publish_profile(batch, request)
            self._publish_recovery(batch)
            return result
        finally:
            self._observations = None
            self._recovery_context = None
            self._recovery_successes = set()
