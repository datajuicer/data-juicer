"""Unit tests for ``ProbeAdapter``.

Tests the schema translation between DJ Adapter probe outputs and
ElasticJuicer ProfilingStore schema.  Pure unit tests -- no real DJ Adapter
or pipeline required.  Uses a real ``ProfilingStore`` backed by a temp dir
(so we exercise the actual store contract rather than mocks).

Run from repo root::

    python -m unittest tests/core/elasticjuicer/test_probe_adapter.py -v

or::

    pytest tests/core/elasticjuicer/test_probe_adapter.py -v
"""

import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_juicer.core.elasticjuicer.profiler.ocs_annotator import (
    MemoryLocality,
)
from data_juicer.core.elasticjuicer.profiler.probe_adapter import ProbeAdapter
from data_juicer.core.elasticjuicer.profiler.profiling_store import ProfilingStore


# ----------------------------------------------------------------------
# Shared probe-dict factory
# ----------------------------------------------------------------------

def _make_probe(
    *,
    cpu_util_ratio: float = 0.85,
    used_mem_mb: float = 12000.0,
    gpu_used_mb: Optional[List[float]] = None,
    gpu_util_ratios: Optional[List[float]] = None,
    speed: float = 200.0,
    total_time: float = 5.0,
    n_snapshots: int = 1,
    timestamp: float = 1.0,
) -> Dict[str, Any]:
    """Build a DJ-Adapter-shaped probe dict for testing.

    Mirrors the dict shape produced by ``Adapter.execute_and_probe`` +
    ``Monitor.analyze_resource_util_list`` (per ``data_juicer/core/monitor.py``).
    """
    resource = []
    for i in range(n_snapshots):
        snap = {
            "timestamp": timestamp + i,
            "CPU util.": cpu_util_ratio,
            "Used mem.": used_mem_mb,
            "GPU used mem.": gpu_used_mb,
            "GPU util.": gpu_util_ratios,
        }
        resource.append(snap)

    analysis: Dict[str, Dict[str, float]] = {}
    if cpu_util_ratio is not None:
        analysis["CPU util."] = {
            "max": cpu_util_ratio,
            "min": cpu_util_ratio,
            "avg": cpu_util_ratio,
        }
    if used_mem_mb is not None:
        analysis["Used mem."] = {
            "max": used_mem_mb,
            "min": used_mem_mb * 0.9,
            "avg": used_mem_mb * 0.95,
        }
    if gpu_util_ratios:
        gpu_max = max(gpu_util_ratios)
        analysis["GPU util."] = {
            "max": gpu_max,
            "min": gpu_max * 0.5,
            "avg": gpu_max * 0.75,
        }

    return {
        "time": total_time,
        "sampling interval": 0.5,
        "speed": speed,
        "resource": resource,
        "resource_analysis": analysis,
    }


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

class TestProbeAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="ej_probe_adapter_test_"))
        self.store = ProfilingStore(storage_dir=str(self.tmpdir))
        self.bridge = ProbeAdapter(self.store)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # -- Translation primitives -------------------------------------------------

    def test_translate_one_basic(self) -> None:
        """Field-by-field translation for a typical CPU-only probe."""
        probe = _make_probe(
            cpu_util_ratio=0.85,
            used_mem_mb=12000.0,
            speed=200.0,
            n_snapshots=1,
            timestamp=1.0,
        )

        stats, _ = self.bridge._translate_one(probe, "myop", batch_size=1000)

        self.assertEqual(stats.op_name, "myop")
        self.assertEqual(len(stats.snapshots), 1)

        snap = stats.snapshots[0]
        # Ratio -> percent
        self.assertAlmostEqual(snap.cpu_percent, 85.0, places=5)
        # System MB copied directly (lossy: marked low-confidence elsewhere)
        self.assertAlmostEqual(snap.memory_mb, 12000.0, places=5)
        # Externally-injected batch size
        self.assertEqual(snap.batch_size, 1000)
        # speed -> throughput
        self.assertAlmostEqual(snap.throughput, 200.0, places=5)
        # 1000 samples / 200 samples-per-sec = 5 s = 5000 ms per batch
        self.assertAlmostEqual(snap.latency_ms, 5000.0, places=2)
        self.assertEqual(snap.timestamp, 1.0)
        # No GPU data
        self.assertIsNone(snap.gpu_memory_mb)
        self.assertIsNone(snap.gpu_utilization)

    def test_extract_gpu_mem_single_gpu(self) -> None:
        """Single-GPU list returns the value with full confidence."""
        raw = {"GPU used mem.": [8192.0]}
        val, conf = ProbeAdapter._extract_gpu_mem(raw)
        self.assertEqual(val, 8192.0)
        self.assertEqual(conf, ProbeAdapter.CONFIDENCE_FULL)

    def test_extract_gpu_mem_no_gpu(self) -> None:
        """Missing / None / empty GPU list returns (None, full confidence)."""
        for raw in ({}, {"GPU used mem.": None}, {"GPU used mem.": []}):
            val, conf = ProbeAdapter._extract_gpu_mem(raw)
            self.assertIsNone(val)
            self.assertEqual(conf, ProbeAdapter.CONFIDENCE_FULL)

    def test_extract_gpu_mem_multi_gpu_warns(self) -> None:
        """Multi-GPU collapses to gpus[0] with reduced confidence + WARNING log."""
        raw = {"GPU used mem.": [8192.0, 4096.0, 2048.0]}

        with self.assertLogs(
            logger="data_juicer.core.elasticjuicer.profiler.probe_adapter",
            level=logging.WARNING,
        ) as captured:
            val, conf = ProbeAdapter._extract_gpu_mem(raw)

        self.assertEqual(val, 8192.0)
        self.assertEqual(conf, ProbeAdapter.CONFIDENCE_FIRST_GPU_ONLY)
        self.assertTrue(
            any("Multi-GPU" in line for line in captured.output),
            f"expected 'Multi-GPU' substring in log lines, got: {captured.output}",
        )

    def test_extract_gpu_util_percent_conversion(self) -> None:
        """GPU util ratio in [0, 1] -> percent in [0, 100]."""
        raw = {"GPU util.": [0.7]}
        val, conf = ProbeAdapter._extract_gpu_util(raw)
        self.assertAlmostEqual(val, 70.0, places=5)
        self.assertEqual(conf, ProbeAdapter.CONFIDENCE_FULL)

    def test_speed_zero_safe(self) -> None:
        """speed=0 must not raise (no divide-by-zero); latency reported as 0."""
        probe = _make_probe(speed=0.0)
        stats, conf = self.bridge._translate_one(probe, "myop", batch_size=100)
        self.assertEqual(len(stats.snapshots), 1)
        self.assertEqual(stats.snapshots[0].latency_ms, 0.0)
        # latency confidence should reflect that we couldn't compute it
        self.assertEqual(conf["latency_ms"], 0.0)

    # -- Signature derivation ---------------------------------------------------

    def test_derive_signature_gpu_strong(self) -> None:
        """GPU util max > 0.5 -> memory_locality = GPU_STRONG."""
        probe = _make_probe(gpu_used_mb=[4096.0], gpu_util_ratios=[0.85])
        stats, _ = self.bridge._translate_one(probe, "image_op", batch_size=16)

        sig = self.bridge._derive_signature(probe, "image_op", stats)

        self.assertIsNotNone(sig)
        self.assertEqual(sig.memory_locality, MemoryLocality.GPU_STRONG)

    def test_derive_signature_max_memory_from_stats(self) -> None:
        """max_memory_mb in signature comes from stats.peak_memory_mb."""
        probe = _make_probe(used_mem_mb=8000.0)
        stats, _ = self.bridge._translate_one(probe, "myop", batch_size=100)

        sig = self.bridge._derive_signature(probe, "myop", stats)

        self.assertIsNotNone(sig)
        self.assertAlmostEqual(sig.max_memory_mb, stats.peak_memory_mb)
        # preferred_batch_size set from first snapshot's batch_size
        self.assertEqual(sig.preferred_batch_size, 100)

    # -- End-to-end ingest ------------------------------------------------------

    def test_ingest_writes_to_store(self) -> None:
        """Full ingest: probe results -> ProfilingStore has both stats and sig."""
        probes = [_make_probe(speed=100.0), _make_probe(speed=200.0)]

        written = self.bridge.ingest_probe_results(
            probe_results=probes,
            op_names=["op_a", "op_b"],
            probe_batch_sizes=[500, 1000],
        )

        self.assertEqual(set(written.keys()), {"op_a", "op_b"})
        # Stats persisted
        self.assertIsNotNone(self.store.get_execution_stats("op_a"))
        self.assertIsNotNone(self.store.get_execution_stats("op_b"))
        # Signatures persisted
        self.assertIsNotNone(self.store.get_ocs_signature("op_a"))
        self.assertIsNotNone(self.store.get_ocs_signature("op_b"))

    def test_confidence_marker_system_memory(self) -> None:
        """memory_mb is always tracked with reduced confidence (system->process)."""
        probe = _make_probe(used_mem_mb=12000.0)
        self.bridge.ingest_probe_results([probe], ["op_a"], [100])

        # System-memory translation is always lossy
        self.assertEqual(
            self.bridge.get_confidence("op_a", "memory_mb"),
            ProbeAdapter.CONFIDENCE_SYSTEM_MEMORY_AS_PROCESS,
        )
        # cpu_percent is a direct math operation, full confidence
        self.assertEqual(
            self.bridge.get_confidence("op_a", "cpu_percent"),
            ProbeAdapter.CONFIDENCE_FULL,
        )

    def test_confidence_marker_multi_gpu_downgrade(self) -> None:
        """If any snapshot in the probe is multi-GPU, op-level GPU confidence drops."""
        probe = _make_probe(
            gpu_used_mb=[8192.0, 4096.0],  # two GPUs
            gpu_util_ratios=[0.8, 0.6],
        )
        # Suppress warning noise emitted by the bridge during ingest.
        with self.assertLogs(
            logger="data_juicer.core.elasticjuicer.profiler.probe_adapter",
            level=logging.WARNING,
        ):
            self.bridge.ingest_probe_results([probe], ["op_a"], [16])

        self.assertEqual(
            self.bridge.get_confidence("op_a", "gpu_memory_mb"),
            ProbeAdapter.CONFIDENCE_FIRST_GPU_ONLY,
        )
        self.assertEqual(
            self.bridge.get_confidence("op_a", "gpu_utilization"),
            ProbeAdapter.CONFIDENCE_FIRST_GPU_ONLY,
        )

    def test_ingest_length_mismatch_raises(self) -> None:
        """Mismatched input list lengths must raise ValueError."""
        with self.assertRaises(ValueError):
            self.bridge.ingest_probe_results(
                probe_results=[_make_probe()],
                op_names=["a", "b"],
                probe_batch_sizes=[100],
            )

    def test_empty_resource_list_yields_empty_stats(self) -> None:
        """Probe with no resource snapshots -> OpExecutionStats with no snapshots."""
        probe = {
            "time": 0.0,
            "speed": 0.0,
            "resource": [],
            "resource_analysis": {},
        }
        stats, _ = self.bridge._translate_one(probe, "empty_op", batch_size=1)
        self.assertEqual(stats.op_name, "empty_op")
        self.assertEqual(len(stats.snapshots), 0)

    def test_ingest_continues_after_per_op_failure(self) -> None:
        """If one probe dict is malformed, ingest still writes the good ones."""
        bad_probe: Dict[str, Any] = {"resource": object()}  # not iterable
        good_probe = _make_probe()

        with self.assertLogs(
            logger="data_juicer.core.elasticjuicer.profiler.probe_adapter",
            level=logging.ERROR,
        ):
            written = self.bridge.ingest_probe_results(
                probe_results=[bad_probe, good_probe],
                op_names=["bad", "good"],
                probe_batch_sizes=[10, 100],
            )

        self.assertIn("good", written)
        self.assertNotIn("bad", written)
        self.assertIsNotNone(self.store.get_execution_stats("good"))
        self.assertIsNone(self.store.get_execution_stats("bad"))


if __name__ == "__main__":
    unittest.main()
