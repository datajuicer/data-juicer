"""Detection of worker-broadcast multi-node launch contexts."""

from __future__ import annotations

import hashlib
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ELASTIC_SHARD_CHILD_ENV = "DJ_ELASTIC_SHARD_CHILD"

_LAUNCH_ENVIRONMENTS = (
    ("WORLD_SIZE", "RANK", "LOCAL_RANK", "torch"),
    ("OMPI_COMM_WORLD_SIZE", "OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "openmpi"),
    ("SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID", "slurm"),
)
_RUN_ID_ENVIRONMENTS = (
    "DJ_ELASTIC_RUN_ID",
    "PAI_JOB_ID",
    "DLC_JOB_ID",
    "JOB_ID",
    "SLURM_JOB_ID",
)


@dataclass(frozen=True)
class LaunchContext:
    """A validated distributed process identity.

    ``world_size`` describes launched processes, not physical nodes. Physical
    multi-node detection is completed later by rendezvous and hostname
    de-duplication.
    """

    world_size: int
    rank: int
    local_rank: int | None
    run_id: str | None
    hostname: str
    source: str

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def has_stable_run_id(self) -> bool:
        return bool(self.run_id and self.run_id.strip())


def _parse_int(values: Mapping[str, str], name: str, *, required: bool) -> int | None:
    raw = values.get(name)
    if raw is None or not str(raw).strip():
        if required:
            raise ValueError(f"Distributed launch variable {name} is missing")
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Distributed launch variable {name} must be an integer; got {raw!r}") from exc


def _configured_run_id(cfg: Any) -> str | None:
    elastic_cfg = getattr(cfg, "elastic_sharding", None)
    explicit = getattr(elastic_cfg, "run_id", None) if elastic_cfg is not None else None
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    if getattr(cfg, "_user_provided_job_id", False):
        job_id = getattr(cfg, "job_id", None)
        if job_id is not None and str(job_id).strip():
            return str(job_id).strip()
    return None


def detect_launch_context(
    environ: Mapping[str, str] | None = None,
    *,
    hostname: str | None = None,
    run_id: str | None = None,
) -> LaunchContext | None:
    """Return a validated distributed launch context when one is advertised.

    A world size alone never proves physical multi-node execution. Callers
    must rendezvous and count distinct hostnames before enabling sharding.
    """

    values = os.environ if environ is None else environ
    selected = None
    for world_name, rank_name, local_rank_name, source in _LAUNCH_ENVIRONMENTS:
        if world_name in values or rank_name in values:
            selected = (world_name, rank_name, local_rank_name, source)
            break
    if selected is None:
        return None

    world_name, rank_name, local_rank_name, source = selected
    world_size = _parse_int(values, world_name, required=True)
    rank = _parse_int(values, rank_name, required=True)
    local_rank = _parse_int(values, local_rank_name, required=False)
    assert world_size is not None and rank is not None
    if world_size < 1:
        raise ValueError(f"{world_name} must be positive; got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"{rank_name} must be in [0, {world_size}); got {rank}")
    if world_size == 1:
        return None

    resolved_run_id = str(run_id).strip() if run_id is not None and str(run_id).strip() else None
    if resolved_run_id is None:
        for name in _RUN_ID_ENVIRONMENTS:
            value = values.get(name)
            if value is not None and str(value).strip():
                resolved_run_id = str(value).strip()
                break

    return LaunchContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        run_id=resolved_run_id,
        hostname=hostname or socket.gethostname(),
        source=source,
    )


def launch_context_for_config(
    cfg: Any,
    environ: Mapping[str, str] | None = None,
    *,
    hostname: str | None = None,
) -> LaunchContext | None:
    """Resolve launch context with config-provided identity as a fallback."""

    return detect_launch_context(
        environ,
        hostname=hostname,
        run_id=_configured_run_id(cfg),
    )


def elastic_mode(cfg: Any) -> str:
    elastic_cfg = getattr(cfg, "elastic_sharding", None)
    return str(getattr(elastic_cfg, "mode", "auto")).lower()


def is_elastic_child(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return str(values.get(ELASTIC_SHARD_CHILD_ENV, "")).strip().lower() in {"1", "true", "yes"}


def should_wrap_executor(cfg: Any, environ: Mapping[str, str] | None = None) -> bool:
    """Cheap preflight used by the executor factory.

    Hostname rendezvous and pipeline checks deliberately happen in the outer
    executor so ordinary single-process construction stays side-effect free.
    """

    mode = elastic_mode(cfg)
    if mode == "off" or is_elastic_child(environ):
        return False
    if mode == "on":
        return True
    try:
        context = launch_context_for_config(cfg, environ)
    except ValueError:
        return False
    return context is not None and context.has_stable_run_id


def automatic_job_id(environ: Mapping[str, str] | None = None) -> str | None:
    """Return a shared job id for a recognized distributed submission."""

    try:
        context = detect_launch_context(environ)
    except ValueError:
        return None
    if context is None or not context.has_stable_run_id:
        return None
    digest = hashlib.sha256(context.run_id.encode("utf-8")).hexdigest()[:16]
    return f"elastic_{digest}"
