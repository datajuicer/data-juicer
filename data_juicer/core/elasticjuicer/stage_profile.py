"""Bounded, job-scoped advisory profiles; no data/checkpoint ownership.

The pure store is hosted in one unnamed Ray actor. Each item of success/OOM
evidence has its own receipt-time TTL, so fresh successes cannot rejuvenate an
old OOM bound. Only observations from delivered outer batches are published.
"""

import json
import os
import tempfile
import time
import uuid
from collections import Counter, OrderedDict

from loguru import logger

from .profile_context import validate_context

PROFILE_SCHEMA_VERSION = 1
PROFILE_TTL_SECONDS = 3600
PROFILE_RPC_TIMEOUT_SECONDS = 2
MAX_PROFILE_ENTRIES = 256
MAX_OBSERVED_SIZES = 32


def _positive_integer(value):
    return type(value) is int and value > 0


def validate_prior(prior, request):
    """Validate again at the consumer before touching the local controller."""
    if not isinstance(prior, dict) or any(prior.get(key) != value for key, value in request.items()):
        raise ValueError("profile identity mismatch")
    if type(prior.get("schema_version")) is not int or prior["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise ValueError("profile schema mismatch")
    minimum, _, maximum = validate_context(request["context"])["limits"]
    safe, upper = prior.get("safe_batch_size"), prior.get("oom_upper_bound")
    if safe is not None and (not _positive_integer(safe) or not minimum <= safe <= maximum):
        raise ValueError("invalid safe batch size")
    if upper is not None and (not _positive_integer(upper) or not minimum < upper <= maximum):
        raise ValueError("invalid OOM upper bound")
    if safe is None and upper is None:
        raise ValueError("empty profile")
    if safe is not None and upper is not None and safe >= upper:
        raise ValueError("conflicting profile bounds")
    ttl = prior.get("remaining_ttl_seconds")
    if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not 0 < ttl <= PROFILE_TTL_SECONDS:
        raise ValueError("expired or invalid profile TTL")
    return safe, upper


class StageProfileStore:
    def __init__(
        self, job_id, manifest, ttl_seconds=PROFILE_TTL_SECONDS, max_entries=MAX_PROFILE_ENTRIES, clock=time.monotonic
    ):
        if not 0 < ttl_seconds <= PROFILE_TTL_SECONDS or not _positive_integer(max_entries):
            raise ValueError("invalid profile store bounds")
        self.job_id = job_id
        self.pipeline_fingerprint = manifest["pipeline_fingerprint"]
        self.stages = {entry["stage_id"]: entry["op_fingerprint"] for entry in manifest["stages"]}
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self.records = OrderedDict()
        self.metrics = Counter()

    def ready(self):
        return True

    def _validate_request(self, request):
        if not isinstance(request, dict) or set(request) != {
            "schema_version",
            "job_id",
            "pipeline_fingerprint",
            "stage_id",
            "op_fingerprint",
            "context",
        }:
            raise ValueError("invalid_request")
        if type(request["schema_version"]) is not int or request["schema_version"] != PROFILE_SCHEMA_VERSION:
            raise ValueError("schema_mismatch")
        if request["job_id"] != self.job_id:
            raise ValueError("job_mismatch")
        if request["pipeline_fingerprint"] != self.pipeline_fingerprint:
            raise ValueError("pipeline_mismatch")
        stage_id = request["stage_id"]
        if not isinstance(stage_id, str) or stage_id not in self.stages:
            raise ValueError("stage_mismatch")
        if request["op_fingerprint"] != self.stages[stage_id]:
            raise ValueError("operator_mismatch")
        try:
            validate_context(request["context"])
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError("invalid_context") from error
        return stage_id, request["context"]

    def _expire(self, now):
        for key, record in list(self.records.items()):
            for kind in ("successes", "ooms"):
                record[kind] = {size: at for size, at in record[kind].items() if now - at < self.ttl_seconds}
            if not record["successes"] and not record["ooms"]:
                del self.records[key]
                self.metrics["expired"] += 1

    def _view(self, record, now):
        minimum = validate_context(record["request"]["context"])["limits"][0]
        upper = min(record["ooms"], default=None)
        # An OOM at the floor cannot seed a useful prior. Local bounded retries
        # remain responsible for transient floor starvation.
        if upper is not None and upper <= minimum:
            return None
        candidates = [size for size in record["successes"] if upper is None or size < upper]
        safe = max(candidates, default=None)
        evidence = []
        if safe is not None:
            evidence.append(record["successes"][safe])
        if upper is not None:
            evidence.append(record["ooms"][upper])
        if not evidence:
            return None
        return {
            **record["request"],
            "safe_batch_size": safe,
            "oom_upper_bound": upper,
            "remaining_ttl_seconds": min(self.ttl_seconds - (now - at) for at in evidence),
            "source_actor_ids": list(record["sources"]),
            "seeded_actor_ids": list(record["seeded"]),
        }

    def read(self, request):
        self.metrics["reads"] += 1
        try:
            key = self._validate_request(request)
        except (ValueError, TypeError, KeyError) as error:
            self.metrics[f"rejected:{error}"] += 1
            return None
        now = self.clock()
        self._expire(now)
        record = self.records.get(key)
        prior = self._view(record, now) if record is not None else None
        if prior is None:
            self.metrics["misses"] += 1
        else:
            self.metrics["hits"] += 1
            self.records.move_to_end(key)
        return prior

    def publish(self, request, observations, actor_id, seeded=False):
        try:
            key = self._validate_request(request)
            minimum, _, maximum = validate_context(request["context"])["limits"]
            if not isinstance(actor_id, str) or not actor_id or len(actor_id) > 128 or type(seeded) is not bool:
                raise ValueError("invalid_actor")
            if not isinstance(observations, dict) or set(observations) != {"successes", "ooms"}:
                raise ValueError("invalid_observations")
            for kind in ("successes", "ooms"):
                sizes = observations[kind]
                if (
                    not isinstance(sizes, list)
                    or len(sizes) > MAX_OBSERVED_SIZES
                    or any(not _positive_integer(size) or not minimum <= size <= maximum for size in sizes)
                ):
                    raise ValueError("invalid_observations")
            if not observations["successes"] and not observations["ooms"]:
                raise ValueError("empty_observations")
        except (ValueError, TypeError, KeyError) as error:
            self.metrics[f"rejected:{error}"] += 1
            return False
        now = self.clock()
        self._expire(now)
        record = self.records.setdefault(
            key, {"request": dict(request), "successes": {}, "ooms": {}, "sources": [], "seeded": []}
        )
        for kind in ("successes", "ooms"):
            record[kind].update({size: now for size in observations[kind]})
            # Retain conservative OOMs and the largest proven successes.
            ordered = sorted(record[kind])
            sizes = (
                ordered[:MAX_OBSERVED_SIZES] if kind == "ooms" else ordered[:1] + ordered[-(MAX_OBSERVED_SIZES - 1) :]
            )
            record[kind] = {size: record[kind][size] for size in sizes}
        for field, include in (("sources", True), ("seeded", seeded)):
            if include and actor_id not in record[field]:
                record[field].append(actor_id)
                del record[field][:-16]
        self.records.move_to_end(key)
        while len(self.records) > self.max_entries:
            self.records.popitem(last=False)
            self.metrics["evicted"] += 1
        self.metrics["publishes"] += 1
        return True

    def snapshot(self):
        now = self.clock()
        self._expire(now)
        profiles = [self._view(record, now) for record in self.records.values()]
        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "job_id": self.job_id,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "scope": "driver_run",
            "ttl_seconds": self.ttl_seconds,
            "metrics": dict(self.metrics),
            "profiles": [profile for profile in profiles if profile is not None],
        }


class StageProfileSession:
    """Driver-owned service shared by execution groups and actor incarnations."""

    def __init__(self, manifest, work_dir):
        self.manifest = manifest
        self.work_dir = work_dir
        self.job_id = uuid.uuid4().hex  # Isolate even consecutive runs in one Ray job/work_dir.
        self.store = None

    def start(self):
        import ray

        try:
            if not ray.is_initialized():
                raise RuntimeError("Ray must already be initialized")
            self.store = ray.remote(num_cpus=0, max_restarts=0)(StageProfileStore).remote(self.job_id, self.manifest)
            ray.get(self.store.ready.remote(), timeout=10)
        except Exception as error:
            logger.warning(f"StageProfile service unavailable; using local batching: {type(error).__name__}")
            self._kill()
        return self.store is not None

    def actor_arguments(self, op, stage_id):
        entry = next((entry for entry in self.manifest["stages"] if entry["stage_id"] == stage_id), None)
        if self.store is None or entry is None:
            return {}
        return {
            "profile_store": self.store,
            "profile_request": {
                "schema_version": PROFILE_SCHEMA_VERSION,
                "job_id": self.job_id,
                "pipeline_fingerprint": self.manifest["pipeline_fingerprint"],
                "stage_id": stage_id,
                "op_fingerprint": entry["op_fingerprint"],
            },
            "profile_resource_envelope": {
                "num_cpus": op.num_cpus,
                "num_gpus": op.num_gpus,
                "num_proc": op.num_proc,
                "memory": getattr(op, "memory", None),
                "runtime_env": op.runtime_env,
            },
        }

    def finish(self):
        import ray

        try:
            if self.store is None:
                return None
            snapshot = ray.get(self.store.snapshot.remote(), timeout=PROFILE_RPC_TIMEOUT_SECONDS)
            snapshot["saved_at_unix_seconds"] = time.time()
            os.makedirs(self.work_dir, exist_ok=True)
            path = os.path.join(self.work_dir, "elastic_juicer_stage_profiles.json")
            descriptor, temporary = tempfile.mkstemp(prefix=".stage-profile-", dir=self.work_dir)
            try:
                with os.fdopen(descriptor, "w") as stream:
                    json.dump(snapshot, stream, sort_keys=True, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return snapshot
        except Exception as error:
            logger.warning(f"Could not save StageProfile audit: {type(error).__name__}")
            return None
        finally:
            self._kill()

    def _kill(self):
        import ray

        if self.store is not None:
            try:
                ray.kill(self.store, no_restart=True)
            except Exception:
                pass
            self.store = None
