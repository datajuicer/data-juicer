"""Exact, explicitly declared compatibility keys for advisory batch profiles."""

import hashlib
import json
import math

from .adaptive_mapper import _batch_length


def _canonical(value):
    # Reject opaque objects, non-string dictionary keys and non-finite values.
    # Never use repr/default=str: process-local objects are not compatibility evidence.
    def validate(item, depth=0):
        if depth > 16:
            raise ValueError("profile context is too deeply nested")
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float) and math.isfinite(item):
            return
        if isinstance(item, (tuple, list)):
            for child in item:
                validate(child, depth + 1)
            return
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            for child in item.values():
                validate(child, depth + 1)
            return
        raise ValueError("profile context must contain finite JSON values")

    validate(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded) > 8192:
        raise ValueError("profile context is too large")
    return encoded


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def validate_context(context):
    if not isinstance(context, str) or len(context) > 1024:
        raise ValueError("invalid profile context")
    data = json.loads(context)
    expected = {"schema_version", "cost_sha256", "resources_sha256", "envelope_sha256", "limits"}
    if not isinstance(data, dict) or set(data) != expected or type(data["schema_version"]) is not int:
        raise ValueError("invalid profile context schema")
    if data["schema_version"] != 1 or _canonical(data) != context:
        raise ValueError("invalid profile context schema or encoding")
    for key in ("cost_sha256", "resources_sha256", "envelope_sha256"):
        value = data[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid profile context digest")
    limits = data["limits"]
    if not isinstance(limits, list) or len(limits) != 3 or any(type(size) is not int for size in limits):
        raise ValueError("invalid profile context limits")
    minimum, maximum, hard_limit = limits
    if not 1 <= minimum <= hard_limit <= maximum:
        raise ValueError("invalid profile context limits")
    return data


def make_profile_context(batch, operator, controller, resource_envelope):
    """Fail closed unless every row has the same declared input cost.

    The operator owns the meaning/completeness of its cost and resource contract.
    Raw input values, paths and resource declarations never enter the profile artifact.
    """
    columns = getattr(operator, "elastic_juicer_cost_columns", ())
    if (
        not isinstance(columns, (tuple, list))
        or not columns
        or any(not isinstance(column, str) or not column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        return None, "missing_input_cost_contract"
    try:
        rows = _batch_length(batch)
        if rows == 0:
            return None, "empty_batch"
        if hasattr(batch, "column_names"):
            if any(column not in batch.column_names for column in columns):
                return None, "missing_input_cost"
            values = batch.select(columns).to_pydict()
        else:
            values = {column: batch[column] for column in columns}
        cost = None
        for index in range(rows):
            row = {column: values[column][index] for column in columns}
            if any(value is None for value in row.values()):
                return None, "missing_input_cost"
            encoded = _canonical(row)
            if cost is not None and encoded != cost:
                return None, "mixed_input_cost"
            cost = encoded
    except Exception:
        return None, "invalid_input_cost"
    provider = getattr(operator, "elastic_juicer_profile_resource_context", None)
    if not callable(provider):
        return None, "missing_resource_contract"
    try:
        resources = provider()
        if (
            not isinstance(resources, dict)
            or not isinstance(resources.get("resource_class"), str)
            or not resources["resource_class"]
            or type(resources.get("memory_budget_bytes")) is not int
            or resources["memory_budget_bytes"] < 1
        ):
            return None, "invalid_resource_contract"
        context = _canonical(
            {
                "schema_version": 1,
                "cost_sha256": hashlib.sha256(cost.encode()).hexdigest(),
                "resources_sha256": _digest(resources),
                "envelope_sha256": _digest(resource_envelope),
                "limits": [controller.min_batch_size, controller.max_batch_size, controller.hard_limit],
            }
        )
        validate_context(context)
        return context, "eligible"
    except Exception:
        return None, "invalid_resource_contract"
