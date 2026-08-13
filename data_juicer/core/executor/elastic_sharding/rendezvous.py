"""Shared-POSIX rendezvous for worker-broadcast launches."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context import LaunchContext
from .job import ShardJobError, _atomic_write_json


@dataclass(frozen=True)
class RendezvousResult:
    members: tuple[dict[str, Any], ...]
    coordinator_rank: int
    host_leader_ranks: tuple[int, ...]

    @property
    def hostnames(self) -> tuple[str, ...]:
        return tuple(sorted({str(member["hostname"]) for member in self.members}))

    def is_host_leader(self, rank: int) -> bool:
        return rank in self.host_leader_ranks


class RendezvousTimeout(ShardJobError):
    pass


class SharedRendezvous:
    """Discover all launched ranks and prove that the directory is shared."""

    def __init__(
        self,
        root: Path,
        context: LaunchContext,
        *,
        fingerprint: str,
        timeout_secs: int,
        poll_interval_secs: float,
    ):
        self.root = root.expanduser().resolve()
        self.context = context
        self.fingerprint = fingerprint
        self.timeout_secs = timeout_secs
        self.poll_interval_secs = poll_interval_secs

    def _write_member(self) -> None:
        member = {
            "schema_version": 1,
            "run_id": self.context.run_id,
            "world_size": self.context.world_size,
            "rank": self.context.rank,
            "local_rank": self.context.local_rank,
            "hostname": self.context.hostname,
            "source": self.context.source,
            "pid": os.getpid(),
            "fingerprint": self.fingerprint,
            "registered_at_epoch": time.time(),
        }
        _atomic_write_json(self.root / "members" / f"rank-{self.context.rank:08d}.json", member)

    def _publish_posix_probe(self) -> None:
        if self.context.rank != 0:
            return
        probe_dir = self.root / "probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        source = probe_dir / "source"
        linked = probe_dir / "hardlink"
        token = uuid.uuid4().hex
        try:
            source.write_text(token, encoding="utf-8")
            linked.unlink(missing_ok=True)
            os.link(source, linked)
            source_stat = source.stat()
            linked_stat = linked.stat()
        except OSError as exc:
            raise ShardJobError(f"Coordination path does not support required POSIX hard links: {exc}") from exc
        if (source_stat.st_dev, source_stat.st_ino) != (linked_stat.st_dev, linked_stat.st_ino):
            raise ShardJobError("Coordination path failed the shared POSIX hard-link probe")
        _atomic_write_json(
            probe_dir / "result.json",
            {
                "token": token,
                "device": source_stat.st_dev,
                "inode": source_stat.st_ino,
                "fingerprint": self.fingerprint,
            },
        )

    def _validate_posix_probe(self) -> bool:
        probe_dir = self.root / "probe"
        result_path = probe_dir / "result.json"
        if not result_path.exists():
            return False
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            source = probe_dir / "source"
            linked = probe_dir / "hardlink"
            source_stat = source.stat()
            linked_stat = linked.stat()
            source_token = source.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError):
            return False
        return (
            result.get("fingerprint") == self.fingerprint
            and source_token == result.get("token")
            and (source_stat.st_dev, source_stat.st_ino) == (linked_stat.st_dev, linked_stat.st_ino)
            and source_stat.st_ino == result.get("inode")
        )

    def _read_members(self) -> tuple[dict[str, Any], ...] | None:
        members: list[dict[str, Any]] = []
        for rank in range(self.context.world_size):
            path = self.root / "members" / f"rank-{rank:08d}.json"
            try:
                member = json.loads(path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, json.JSONDecodeError):
                return None
            if (
                member.get("rank") != rank
                or member.get("world_size") != self.context.world_size
                or member.get("run_id") != self.context.run_id
                or member.get("fingerprint") != self.fingerprint
                or not member.get("hostname")
            ):
                raise ShardJobError(f"Inconsistent elastic-sharding rendezvous member: {path}")
            members.append(member)
        return tuple(members)

    def wait(self) -> RendezvousResult:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "members").mkdir(parents=True, exist_ok=True)
        self._write_member()
        self._publish_posix_probe()

        deadline = time.monotonic() + self.timeout_secs
        while time.monotonic() < deadline:
            members = self._read_members()
            if members is not None and self._validate_posix_probe():
                by_host: dict[str, list[int]] = {}
                for member in members:
                    by_host.setdefault(str(member["hostname"]), []).append(int(member["rank"]))
                return RendezvousResult(
                    members=members,
                    coordinator_rank=0,
                    host_leader_ranks=tuple(sorted(min(ranks) for ranks in by_host.values())),
                )
            time.sleep(self.poll_interval_secs)
        raise RendezvousTimeout(
            f"Timed out after {self.timeout_secs}s waiting for {self.context.world_size} ranks "
            f"on shared coordination path {self.root}"
        )
