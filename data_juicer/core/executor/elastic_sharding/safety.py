"""Conservative eligibility checks for independent shard execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data_juicer.ops import OPERATORS, Deduplicator, Filter, Mapper
from data_juicer.utils.file_utils import is_remote_path


@dataclass(frozen=True)
class ShardabilityReport:
    eligible: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    operator_names: tuple[str, ...] = field(default_factory=tuple)


def _cfg_value(cfg: Any, name: str, default=None):
    if hasattr(cfg, "get"):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _input_reason(cfg: Any) -> str | None:
    if _cfg_value(cfg, "dataset", None):
        return "dataset configurations are not supported by elastic JSONL sharding"
    if _cfg_value(cfg, "generated_dataset_config", None):
        return "generated datasets are not supported by elastic JSONL sharding"
    dataset_path = _cfg_value(cfg, "dataset_path", "")
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        return "dataset_path must name one local JSONL file or directory"
    if is_remote_path(dataset_path):
        return "remote dataset_path is not supported by the POSIX shard coordinator"
    path = Path(dataset_path).expanduser()
    if not path.exists():
        return f"dataset_path does not exist: {path}"
    if path.is_file() and path.suffix.lower() != ".jsonl":
        return f"dataset file must use the .jsonl suffix: {path}"
    if path.is_dir() and not any(candidate.is_file() for candidate in path.rglob("*.jsonl")):
        return f"dataset directory contains no .jsonl files: {path}"
    return None


def _output_reason(cfg: Any) -> str | None:
    export_path = _cfg_value(cfg, "export_path", "")
    export_type = _cfg_value(cfg, "export_type", None)
    if not isinstance(export_path, str) or not export_path:
        return "export_path must be configured"
    if is_remote_path(export_path):
        return "remote export_path is not supported by atomic POSIX publication"
    resolved_type = str(export_type or Path(export_path).suffix.lstrip(".")).lower()
    if resolved_type != "jsonl":
        return f"elastic sharding currently supports JSONL export, not {resolved_type!r}"
    return None


def analyze_shardability(cfg: Any) -> ShardabilityReport:
    """Prove that a configured pipeline can execute independently per shard."""

    reasons: list[str] = []
    warnings: list[str] = []
    config_paths = _cfg_value(cfg, "config", None)
    if not config_paths:
        reasons.append("elastic sharding requires a file-backed recipe")
    executor_type = str(_cfg_value(cfg, "executor_type", "default"))
    if executor_type not in {"default", "ray"}:
        reasons.append(f"executor_type={executor_type!r} is not supported")
    if executor_type == "ray" and str(_cfg_value(cfg, "ray_address", "auto")) != "local":
        reasons.append("ray_address must be 'local'; a non-local Ray cluster already coordinates nodes")
    if _cfg_value(cfg, "decrypt_after_reading", False) or _cfg_value(cfg, "encrypt_before_export", False):
        reasons.append("dataset encryption is not supported by the JSONL shard coordinator")
    if int(_cfg_value(cfg, "export_shard_size", 0) or 0) != 0:
        reasons.append("export_shard_size must be 0 so the coordinator can publish one deterministic result")
    if bool(_cfg_value(cfg, "export_in_parallel", False)):
        reasons.append("export_in_parallel is incompatible with coordinator-side merge")
    if float(_cfg_value(cfg, "data_probe_ratio", 1.0) or 0.0) != 1.0:
        reasons.append("data_probe_ratio must be 1.0 because per-shard sampling changes global semantics")

    input_reason = _input_reason(cfg)
    if input_reason:
        reasons.append(input_reason)
    output_reason = _output_reason(cfg)
    if output_reason:
        reasons.append(output_reason)

    process = _cfg_value(cfg, "process", None)
    if not isinstance(process, list):
        reasons.append("process must be an explicit list")
        return ShardabilityReport(False, tuple(reasons))

    names: list[str] = []
    for index, raw in enumerate(process):
        if not isinstance(raw, dict) or len(raw) != 1:
            reasons.append(f"process[{index}] must contain exactly one operator")
            continue
        name, args = next(iter(raw.items()))
        names.append(name)
        operator_class = OPERATORS.modules.get(name)
        if operator_class is None:
            reasons.append(f"process[{index}] references unknown operator {name!r}")
            continue
        explicitly_global = bool(getattr(operator_class, "is_global_operation", False))
        global_name = any(token in name.lower() for token in ("deduplicator", "global_", "full_dataset_"))
        if explicitly_global or issubclass(operator_class, Deduplicator) or global_name:
            reasons.append(f"process[{index}] {name!r} is a global operation")
            continue
        if not issubclass(operator_class, (Mapper, Filter)):
            reasons.append(f"process[{index}] {name!r} is not a record-local Mapper or Filter")
        if getattr(operator_class, "converge_after", False):
            reasons.append(f"process[{index}] {name!r} requires convergence")

        module = operator_class.__module__
        if not module.startswith("data_juicer.ops.") and not bool(getattr(operator_class, "partition_safe", False)):
            reasons.append(f"custom operator {name!r} must declare partition_safe = True")

        if isinstance(args, dict):
            if args.get("stats_export_path"):
                reasons.append(f"operator {name!r} writes one shared stats_export_path")
            if args.get("save_dir"):
                reasons.append(f"operator {name!r} uses a fixed save_dir that may collide across shards")

    if not process:
        warnings.append("the process list is empty; sharding only redistributes export work")
    return ShardabilityReport(not reasons, tuple(reasons), tuple(warnings), tuple(names))
