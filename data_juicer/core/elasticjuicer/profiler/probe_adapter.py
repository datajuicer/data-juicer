"""Bridge between Stock Data-Juicer Adapter and ElasticJuicer ProfilingStore.

Translates the system-wide resource measurements produced by
``Adapter.adapt_workloads()`` (in ``data_juicer/core/adapter.py``) into the
per-process schema expected by ElasticJuicer's ProfilingStore
(``ResourceSnapshot`` / ``OpExecutionStats``).

Background
----------
DJ Adapter probes a small batch and uses ``psutil`` / ``GPUtil`` to measure
SYSTEM-WIDE resources, then uses a 90% utilization heuristic to size per-OP
batches.  ElasticJuicer's ProfilingStore expects PER-PROCESS RSS metrics.

Several translation steps are lossy:

==========================  ================================  ===================
DJ Adapter field            EJ field                          Translation rule
==========================  ================================  ===================
``"CPU util."``  [0,1]      ``cpu_percent``  [0,100]          ``x * 100``
``"Used mem."``  system MB  ``memory_mb``  process MB         direct, confidence=0.5
``"GPU used mem."``  list   ``gpu_memory_mb``  scalar         ``[0]`` (warn if len>1)
``"GPU util."``  list[0,1]  ``gpu_utilization``  [0,100]      ``[0] * 100``
``"speed"``                 ``throughput``                    direct
``"time"`` total seconds    ``latency_ms`` per-batch          ``1000 * bs / speed``
``"timestamp"``             ``timestamp``                     direct
(absent)                    ``batch_size``                    external injection
(absent)                    ``sample_features``               None (no content data)
==========================  ================================  ===================

Per-op confidence scores are tracked in ``self.confidence_by_op`` out-of-band so
we don't have to extend the ``OpExecutionStats`` dataclass schema in this PR
(that change is owned by PR-2 schema versioning).

Usage
-----
    from data_juicer.core.elasticjuicer.profiler.probe_adapter import ProbeAdapter
    from data_juicer.core.elasticjuicer.profiler.profiling_store import ProfilingStore

    store = ProfilingStore('./elastic_juicer_profiles')
    bridge = ProbeAdapter(store)

    # In PR-4, `adapter._last_analysis` is added by a small change to adapter.py
    # that stashes the probe-results list before `adapt_workloads` returns.
    bridge.ingest_probe_results(
        probe_results=adapter._last_analysis,
        op_names=[op._name for op in ops],
        probe_batch_sizes=bs_per_op,
    )

This module is pure library code in PR-1: no integration with DefaultExecutor
or ElasticRayExecutor.  Those wirings are PR-4 and PR-5 respectively.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .ocs_annotator import MemoryLocality, OCSAnnotator, OpCostSignature
from .profiling_store import ProfilingStore
from .resource_monitor import OpExecutionStats, ResourceSnapshot

logger = logging.getLogger(__name__)


class ProbeAdapter:
    """Translate DJ Adapter probe results to ElasticJuicer ProfilingStore schema."""

    SCHEMA_VERSION = "1.0"

    # Confidence markers in [0, 1].  1.0 means "trust this number"; lower values
    # mean "translation was lossy, downstream consumers should treat as an upper
    # bound or otherwise discount".
    CONFIDENCE_FULL: float = 1.0
    CONFIDENCE_SYSTEM_MEMORY_AS_PROCESS: float = 0.5  # system MB used as process RSS proxy
    CONFIDENCE_FIRST_GPU_ONLY: float = 0.3            # multi-GPU collapsed to GPU[0]

    # Heuristic thresholds for memory locality inference (on GPU util max).
    GPU_STRONG_UTIL_THRESHOLD: float = 0.5
    GPU_BALANCED_UTIL_THRESHOLD: float = 0.1

    def __init__(
        self,
        store: ProfilingStore,
        annotator: Optional[OCSAnnotator] = None,
    ) -> None:
        self.store = store
        self.annotator = annotator or OCSAnnotator()
        # Out-of-band confidence dict: {op_name: {field_name: confidence}}.
        # Stored on the bridge instance, not in OpExecutionStats, so PR-1 does
        # not need to coordinate with PR-2's schema versioning changes.
        self.confidence_by_op: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_probe_results(
        self,
        probe_results: List[Dict[str, Any]],
        op_names: List[str],
        probe_batch_sizes: List[int],
    ) -> Dict[str, OpExecutionStats]:
        """Translate DJ Adapter probe outputs and persist to ProfilingStore.

        Parameters
        ----------
        probe_results
            List of probe dicts as produced by ``Adapter.adapt_workloads``
            after ``Monitor.analyze_resource_util_list`` has been applied.
            Each dict is expected to contain keys ``"time"``, ``"speed"``,
            ``"resource"`` (list of per-tick snapshots) and
            ``"resource_analysis"`` (max/min/avg per DYNAMIC_FIELD).
        op_names
            Parallel list of op names (one per probe dict).
        probe_batch_sizes
            Parallel list of batch sizes used during probing.

        Returns
        -------
        Dict mapping ``op_name`` to the ``OpExecutionStats`` that was written.
        Ops that fail to translate are logged but skipped (other ops still
        proceed).

        Raises
        ------
        ValueError
            If the three input lists have mismatched lengths.
        """
        if not (len(probe_results) == len(op_names) == len(probe_batch_sizes)):
            raise ValueError(
                "Length mismatch: "
                f"probe_results={len(probe_results)}, "
                f"op_names={len(op_names)}, "
                f"probe_batch_sizes={len(probe_batch_sizes)}"
            )

        written: Dict[str, OpExecutionStats] = {}
        for probe_dict, op_name, bs in zip(probe_results, op_names, probe_batch_sizes):
            try:
                stats, confidence = self._translate_one(probe_dict, op_name, bs)
                self.store.update_execution_stats(op_name, stats)
                written[op_name] = stats
                self.confidence_by_op[op_name] = confidence

                sig = self._derive_signature(probe_dict, op_name, stats)
                if sig is not None:
                    self.store.update_ocs_signature(op_name, sig)
            except Exception as e:
                # One bad op should not block the rest.
                logger.error(
                    "Failed to translate probe for op '%s' (batch_size=%s): %s",
                    op_name, bs, e, exc_info=True,
                )

        try:
            self.store.save_all()
            logger.info(
                "Ingested %d/%d probe results to ProfilingStore "
                "(schema_version=%s)",
                len(written), len(op_names), self.SCHEMA_VERSION,
            )
        except Exception as e:
            logger.error("ProfilingStore.save_all() failed after ingest: %s", e, exc_info=True)

        return written

    def get_confidence(self, op_name: str, field: str) -> float:
        """Return tracked confidence for ``(op_name, field)``.

        Returns ``CONFIDENCE_FULL`` if the op or field is unknown (the caller
        is implicitly trusting the value).
        """
        return self.confidence_by_op.get(op_name, {}).get(field, self.CONFIDENCE_FULL)

    # ------------------------------------------------------------------
    # Translation primitives
    # ------------------------------------------------------------------

    def _translate_one(
        self,
        probe: Dict[str, Any],
        op_name: str,
        batch_size: int,
    ) -> Tuple[OpExecutionStats, Dict[str, float]]:
        """Translate one probe dict into ``(OpExecutionStats, confidence map)``."""
        speed = float(probe.get("speed", 0.0) or 0.0)  # samples / sec
        if speed > 0:
            latency_ms_per_batch = (1000.0 / speed) * float(batch_size)
        else:
            latency_ms_per_batch = 0.0

        # Per-op confidence dict.  GPU-related fields may be downgraded below
        # if any sample shows multi-GPU.
        confidence: Dict[str, float] = {
            "cpu_percent": self.CONFIDENCE_FULL,
            "memory_mb": self.CONFIDENCE_SYSTEM_MEMORY_AS_PROCESS,
            "gpu_memory_mb": self.CONFIDENCE_FULL,
            "gpu_utilization": self.CONFIDENCE_FULL,
            "throughput": self.CONFIDENCE_FULL,
            "latency_ms": self.CONFIDENCE_FULL if speed > 0 else 0.0,
        }

        stats = OpExecutionStats(op_name=op_name)

        raw_snapshots = probe.get("resource") or []
        for raw in raw_snapshots:
            gpu_mem, gpu_mem_conf = self._extract_gpu_mem(raw)
            gpu_util, gpu_util_conf = self._extract_gpu_util(raw)

            # Op-level confidence drops to the worst per-snapshot confidence.
            if gpu_mem_conf < confidence["gpu_memory_mb"]:
                confidence["gpu_memory_mb"] = gpu_mem_conf
            if gpu_util_conf < confidence["gpu_utilization"]:
                confidence["gpu_utilization"] = gpu_util_conf

            snap = ResourceSnapshot(
                timestamp=float(raw.get("timestamp", 0.0) or 0.0),
                batch_size=int(batch_size),
                cpu_percent=float(raw.get("CPU util.", 0.0) or 0.0) * 100.0,
                memory_mb=float(raw.get("Used mem.", 0.0) or 0.0),
                gpu_memory_mb=gpu_mem,
                gpu_utilization=gpu_util,
                latency_ms=latency_ms_per_batch,
                throughput=speed,
            )
            # OpExecutionStats.update appends to snapshots and recomputes
            # the aggregate fields (avg/p95/p99/peak).
            stats.update(snap)

        return stats, confidence

    @staticmethod
    def _extract_gpu_mem(raw: Dict[str, Any]) -> Tuple[Optional[float], float]:
        """Extract scalar GPU memory (MB) from a probe snapshot.

        Returns ``(value_mb, confidence)``.  ``value_mb`` is ``None`` if the
        snapshot has no GPU data.  Confidence drops to
        ``CONFIDENCE_FIRST_GPU_ONLY`` and a warning is logged if multi-GPU.
        """
        gpu_used = raw.get("GPU used mem.")
        if gpu_used is None or len(gpu_used) == 0:
            return None, ProbeAdapter.CONFIDENCE_FULL

        if len(gpu_used) > 1:
            logger.warning(
                "Multi-GPU probe detected (%d GPUs); using gpus[0]=%.1f MB. "
                "ElasticJuicer currently assumes single-GPU.",
                len(gpu_used), float(gpu_used[0]),
            )
            return float(gpu_used[0]), ProbeAdapter.CONFIDENCE_FIRST_GPU_ONLY

        return float(gpu_used[0]), ProbeAdapter.CONFIDENCE_FULL

    @staticmethod
    def _extract_gpu_util(raw: Dict[str, Any]) -> Tuple[Optional[float], float]:
        """Extract scalar GPU utilization (percent) from a probe snapshot.

        DJ Adapter reports ratio in ``[0, 1]``; EJ expects percent ``[0, 100]``.
        Multi-GPU is collapsed to ``gpus[0]`` with reduced confidence
        (warning is emitted by ``_extract_gpu_mem``; not duplicated here).
        """
        gpu_util = raw.get("GPU util.")
        if gpu_util is None or len(gpu_util) == 0:
            return None, ProbeAdapter.CONFIDENCE_FULL

        confidence = (
            ProbeAdapter.CONFIDENCE_FIRST_GPU_ONLY
            if len(gpu_util) > 1
            else ProbeAdapter.CONFIDENCE_FULL
        )
        return float(gpu_util[0]) * 100.0, confidence

    # ------------------------------------------------------------------
    # Signature derivation
    # ------------------------------------------------------------------

    def _derive_signature(
        self,
        probe: Dict[str, Any],
        op_name: str,
        stats: OpExecutionStats,
    ) -> Optional[OpCostSignature]:
        """Derive an ``OpCostSignature`` from probe + stats + heuristics.

        Auto-derived (approximately 6 of the 14 signature fields):

        * ``preferred_batch_size`` -- from the probe batch size
        * ``min_memory_mb`` / ``max_memory_mb`` -- from stats peaks
        * ``memory_locality`` -- from GPU util max in ``resource_analysis``
        * ``handles_{text,image,video,audio}`` -- inherited from the
          existing ``OCSAnnotator`` substring heuristics

        The remaining fields (``op_type``, ``transfer_cost``, ``failure_cost``,
        ``state_free``, ``deterministic``) default to ``OCSAnnotator``'s
        registered or inferred defaults.
        """
        existing = self.annotator.get_signature(op_name)
        if existing is not None:
            sig = existing
        else:
            sig = self.annotator.infer_signature(op_name, op_type="unknown")

        # Memory bounds from probe stats.
        if stats.peak_memory_mb > 0:
            sig.max_memory_mb = stats.peak_memory_mb
        if stats.avg_memory_mb > 0:
            # Conservative lower bound: half the average.
            sig.min_memory_mb = stats.avg_memory_mb * 0.5

        if stats.snapshots:
            sig.preferred_batch_size = int(stats.snapshots[0].batch_size)

        # Memory locality from GPU activity during the probe window.
        analysis = probe.get("resource_analysis") or {}
        gpu_util_stats = analysis.get("GPU util.") or {}
        gpu_util_max = 0.0
        if isinstance(gpu_util_stats, dict):
            gpu_util_max = float(gpu_util_stats.get("max", 0.0) or 0.0)

        if gpu_util_max > self.GPU_STRONG_UTIL_THRESHOLD:
            sig.memory_locality = MemoryLocality.GPU_STRONG
        elif gpu_util_max > self.GPU_BALANCED_UTIL_THRESHOLD:
            sig.memory_locality = MemoryLocality.BALANCED
        # Otherwise leave whatever default OCSAnnotator gave us.

        return sig

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @staticmethod
    def attach_to_default_executor(
        executor: Any,
        storage_dir: str = "./elastic_juicer_profiles",
    ) -> "ProbeAdapter":
        """Wire a ProbeAdapter onto a ``DefaultExecutor`` instance.

        Call AFTER ``executor.adapter.adapt_workloads(...)`` returns.  PR-4
        adds the call site in ``default_executor.py``; this helper is provided
        so unit / integration tests can do the same wiring without touching
        the executor source.

        Returns
        -------
        The constructed ``ProbeAdapter``, also stashed on
        ``executor._probe_bridge`` for later inspection.
        """
        store = ProfilingStore(storage_dir=storage_dir)
        bridge = ProbeAdapter(store)
        executor._probe_bridge = bridge
        return bridge
