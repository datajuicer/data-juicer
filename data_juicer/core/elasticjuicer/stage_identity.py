"""Deterministic, restart-stable stage identities for ElasticJuicer.

The identity of a topology stage must survive process restarts, checkpoint
resume (where already-completed operators are skipped), and repeated
operators of the same class. It therefore cannot depend on ``id(op)`` or on
first-seen allocation order. Identities are derived once from the complete
prepared operator list (before checkpoint grouping) and stamped onto each
operator object; every later consumer reads the stamp.

An identity is composed of:

- the original operator index in the full pipeline,
- the occurrence counter among operators sharing the same fingerprint,
- the operator/config fingerprint (class, name, and normalized init kwargs),
- the operator name (kept as a readable ``:{op_name}`` suffix).

The pipeline fingerprint over the ordered operator fingerprints is recorded
alongside so a resumed job can verify it is talking about the same pipeline.
"""

import hashlib
import json
from typing import Dict, List, Optional

STAGE_IDENTITY_ATTR = "_elastic_juicer_stage_identity"
STAGE_IDENTITY_SCHEMA_VERSION = 1


def _normalize(value):
    """Make init kwargs JSON-stable regardless of dict ordering or types."""

    if isinstance(value, dict):
        return {str(key): _normalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def operator_fingerprint(op) -> str:
    """Hash the operator class, registered name, and normalized config."""

    payload = {
        "class": f"{op.__class__.__module__}.{op.__class__.__qualname__}",
        "name": getattr(op, "_name", None) or op.__class__.__name__,
        "kwargs": _normalize(getattr(op, "_init_kwargs", None) or {}),
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_stage_id(op_index: int, occurrence: int, fingerprint: str, op_name: str) -> str:
    if op_index < 0 or occurrence < 0:
        raise ValueError("op_index and occurrence must be non-negative")
    if not fingerprint or not op_name:
        raise ValueError("fingerprint and op_name must be non-empty")
    return f"stage-{op_index:04d}-occ{occurrence}-{fingerprint[:8]}:{op_name}"


def assign_stage_identities(ops) -> Dict:
    """Stamp every operator and return a persistable identity manifest.

    Must be called with the complete prepared operator list (after fusion,
    before checkpoint grouping) so indices and occurrences are stable across
    fresh runs and checkpoint resumes of the same configuration.
    """

    occurrences: Dict[str, int] = {}
    stages: List[Dict] = []
    fingerprints: List[str] = []
    for op_index, op in enumerate(ops):
        fingerprint = operator_fingerprint(op)
        fingerprints.append(fingerprint)
        occurrence = occurrences.get(fingerprint, 0)
        occurrences[fingerprint] = occurrence + 1
        op_name = getattr(op, "_name", None) or op.__class__.__name__
        stage_id = build_stage_id(op_index, occurrence, fingerprint, op_name)
        setattr(op, STAGE_IDENTITY_ATTR, stage_id)
        stages.append(
            {
                "stage_id": stage_id,
                "op_index": op_index,
                "occurrence": occurrence,
                "op_name": op_name,
                "op_fingerprint": fingerprint,
            }
        )
    pipeline_fingerprint = hashlib.sha256("".join(fingerprints).encode("utf-8")).hexdigest()
    return {
        "schema_version": STAGE_IDENTITY_SCHEMA_VERSION,
        "pipeline_fingerprint": pipeline_fingerprint,
        "stages": stages,
    }


def stamped_stage_identity(op) -> Optional[str]:
    """Return the executor-assigned identity, or None when never stamped."""

    stamped = getattr(op, STAGE_IDENTITY_ATTR, None)
    if isinstance(stamped, str) and stamped:
        return stamped
    return None
