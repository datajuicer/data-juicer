"""Bridge the existing Adapter probe into the profiling metric schema."""

from pathlib import Path
from time import time
from typing import Any, Mapping, Optional, Sequence

from .metrics import MetricScope, ResourceSnapshot
from .profiling_store import ProfilingStore


class ProbeAdapter:
    """Convert lossy system-level probe results and stash them safely."""

    SOURCE = "adapter_probe"
    SYSTEM_PROBE_CONFIDENCE = 0.5

    def __init__(self, store: Optional[ProfilingStore] = None):
        self.store = store

    @classmethod
    def from_config(cls, cfg) -> "ProbeAdapter":
        """Create a job-scoped sink, or a no-op sink without a work dir."""

        work_dir = cls._config_value(cfg, "work_dir")
        if not work_dir:
            return cls()
        storage_dir = cls._config_value(cfg, "profiling_store_dir")
        if not storage_dir:
            storage_dir = Path(work_dir) / "elastic_juicer_profiles"
        return cls(ProfilingStore(storage_dir=str(storage_dir)))

    @staticmethod
    def _config_value(cfg, name: str):
        if isinstance(cfg, Mapping):
            return cfg.get(name)
        return getattr(cfg, name, None)

    def stash(self, operators: Sequence, probe_results: Sequence[Mapping[str, Any]], batch_size: int):
        """Convert one result per operator and persist all observations."""

        if len(operators) != len(probe_results):
            raise ValueError("operators and probe results must have the same length")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.store is None:
            return

        for operator, probe_result in zip(operators, probe_results):
            op_name = getattr(operator, "_name", None) or operator.__class__.__name__
            self.store.record_snapshot(op_name, self._to_snapshot(probe_result, batch_size))
        self.store.save_all()

    @classmethod
    def _to_snapshot(cls, probe_result: Mapping[str, Any], batch_size: int) -> ResourceSnapshot:
        records = probe_result.get("resource") or []
        duration_seconds = float(probe_result.get("time") or 0.0)
        throughput = probe_result.get("speed")
        if throughput is None:
            throughput = batch_size / duration_seconds if duration_seconds > 0 else 0.0

        return ResourceSnapshot(
            timestamp=cls._latest_timestamp(records),
            batch_size=batch_size,
            cpu_percent=cls._peak(records, "CPU util.", scale=100.0),
            memory_mb=cls._peak(records, "Used mem."),
            gpu_memory_mb=cls._optional_peak(records, "GPU used mem."),
            gpu_utilization=cls._optional_peak(records, "GPU util.", scale=100.0),
            latency_ms=duration_seconds * 1000.0,
            throughput=float(throughput),
            source=cls.SOURCE,
            scope=MetricScope.SYSTEM,
            confidence=cls.SYSTEM_PROBE_CONFIDENCE,
        )

    @staticmethod
    def _latest_timestamp(records: Sequence[Mapping[str, Any]]) -> float:
        timestamps = [float(record["timestamp"]) for record in records if record.get("timestamp") is not None]
        return max(timestamps, default=time())

    @classmethod
    def _peak(cls, records: Sequence[Mapping[str, Any]], key: str, scale: float = 1.0) -> float:
        value = cls._optional_peak(records, key, scale)
        return 0.0 if value is None else value

    @staticmethod
    def _optional_peak(records: Sequence[Mapping[str, Any]], key: str, scale: float = 1.0) -> Optional[float]:
        values = []
        for record in records:
            raw_value = record.get(key)
            if raw_value is None:
                continue
            if isinstance(raw_value, (list, tuple)):
                values.extend(float(value) for value in raw_value if value is not None)
            else:
                values.append(float(raw_value))
        if not values:
            return None
        return max(values) * scale
