"""
Preflight checks for Data-Juicer pipelines.

Two-stage validation that catches configuration errors before expensive
computation begins:

1. pre_instantiation_check: validates op names, param names, param types
   (runs before load_ops)
2. post_instantiation_check: validates schema compatibility, environment
   readiness (runs after load_ops, before dataset.process)
"""

import inspect
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from data_juicer.ops.base_op import OP, OPERATORS

if TYPE_CHECKING:
    from data_juicer.core.data.schema import Schema

# ---------------------------------------------------------------------------
# Error classes
# ---------------------------------------------------------------------------


class PreflightError:
    """Single preflight error entry."""

    def __init__(self, op_name: str, message: str, suggestions: Optional[List[str]] = None):
        self.op_name = op_name
        self.message = message
        self.suggestions = suggestions or []

    def __str__(self):
        s = f"[{self.op_name}] {self.message}"
        if self.suggestions:
            s += f" (did you mean: {', '.join(self.suggestions)}?)"
        return s


class PipelineConfigError(Exception):
    """Raised when pre-instantiation checks find errors."""

    def __init__(self, errors: List[PreflightError]):
        self.errors = errors
        lines = ["Pipeline configuration errors detected:"]
        for e in errors:
            lines.append(f"  ✗ {e}")
        lines.append(f"\n{len(errors)} error(s) found. Fix the config and retry.")
        lines.append("Set 'strict_preflight: false' in config to skip preflight checks.")
        super().__init__("\n".join(lines))


class PipelineRuntimeError(Exception):
    """Raised when post-instantiation checks find errors."""

    def __init__(self, errors: List[PreflightError]):
        self.errors = errors
        lines = ["Pipeline runtime preflight errors detected:"]
        for e in errors:
            lines.append(f"  ✗ {e}")
        lines.append(f"\n{len(errors)} error(s) found. Fix the config and retry.")
        lines.append("Set 'strict_preflight: false' in config to skip preflight checks.")
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# Pre-instantiation check
# ---------------------------------------------------------------------------


def _get_valid_params(cls: Type[OP]) -> Dict[str, Any]:
    """
    Get the set of valid parameter names and their type annotations
    for an OP class. Merges:
    1. _BASE_PARAMS from each class in the MRO (OP, Filter, etc.)
    2. Explicit __init__ signature params from each class in the MRO

    Returns: {param_name: expected_type_or_None}
    """
    params = {}

    # Walk the MRO to collect _BASE_PARAMS and __init__ signatures
    for klass in inspect.getmro(cls):
        if klass is object:
            continue

        # Collect _BASE_PARAMS declared on this class
        if "_BASE_PARAMS" in klass.__dict__:
            for param_name, (param_type, _default) in klass._BASE_PARAMS.items():
                if param_name not in params:
                    params[param_name] = param_type

        # Collect explicit __init__ signature params
        if "__init__" in klass.__dict__:
            try:
                sig = inspect.signature(klass.__init__)
            except (ValueError, TypeError):
                continue
            for name, p in sig.parameters.items():
                if name == "self":
                    continue
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                annotation = p.annotation if p.annotation != inspect.Parameter.empty else None
                if name not in params:
                    params[name] = annotation

    return params


def _type_compatible(value: Any, expected_type: Any) -> bool:
    """
    Check if value is compatible with expected_type.
    Lenient for numerics: int annotation accepts int; float annotation accepts int or float.
    None type means no check.
    """
    if expected_type is None:
        return True
    if value is None:
        return True

    # Handle typing module types (Optional, Union, etc.)
    origin = getattr(expected_type, "__origin__", None)
    if origin is not None:
        # For complex types (Union, Optional, List, etc.), skip checking
        return True

    # Numeric leniency: int annotation accepts int only;
    # float annotation accepts int or float
    if expected_type is int:
        if isinstance(value, float) and value == int(value):
            return True
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type is bool:
        return isinstance(value, bool)
    if expected_type is str:
        return isinstance(value, str)

    # For other types, use isinstance
    try:
        return isinstance(value, expected_type)
    except TypeError:
        # If isinstance fails (e.g., with some typing constructs), skip
        return True


def pre_instantiation_check(process_list: List[Dict]) -> None:
    """
    Validate pipeline config before operator instantiation.

    Checks:
    - Operator names exist in the registry (with fuzzy match suggestions)
    - Parameter names are valid for the operator (with fuzzy match suggestions)
    - Parameter types match annotations

    Args:
        process_list: cfg.process - list of {op_name: args_dict} dicts

    Raises:
        PipelineConfigError: if any errors are found, with all errors collected
    """
    if not process_list:
        return

    errors: List[PreflightError] = []
    all_op_names = list(OPERATORS.modules.keys())

    for item in process_list:
        if not isinstance(item, dict) or len(item) != 1:
            errors.append(PreflightError("__config__", f"Malformed process entry: {item!r}"))
            continue
        op_name, op_args = list(item.items())[0]

        # Check 1: operator name exists
        if op_name not in OPERATORS.modules:
            suggestions = get_close_matches(op_name, all_op_names, n=3, cutoff=0.6)
            errors.append(
                PreflightError(
                    op_name,
                    f"Unknown operator '{op_name}' not found in registry",
                    suggestions,
                )
            )
            continue

        # No args to check
        if not op_args:
            continue

        cls = OPERATORS.modules[op_name]
        valid_params = _get_valid_params(cls)
        valid_param_names = list(valid_params.keys())

        for param_name, param_value in op_args.items():
            # Check 2: parameter name is valid
            if param_name not in valid_params:
                suggestions = get_close_matches(param_name, valid_param_names, n=2, cutoff=0.6)
                errors.append(
                    PreflightError(
                        op_name,
                        f"Unknown parameter '{param_name}'",
                        suggestions,
                    )
                )
                continue

            # Check 3: parameter type matches
            expected_type = valid_params[param_name]
            if expected_type is not None and param_value is not None:
                if not _type_compatible(param_value, expected_type):
                    errors.append(
                        PreflightError(
                            op_name,
                            f"Parameter '{param_name}' expects type "
                            f"'{expected_type.__name__}', got "
                            f"'{type(param_value).__name__}' (value: {param_value!r})",
                        )
                    )

    if errors:
        raise PipelineConfigError(errors)


# ---------------------------------------------------------------------------
# Post-instantiation check
# ---------------------------------------------------------------------------


def post_instantiation_check(
    ops: List[OP],
    dataset_schema: "Schema",
    cfg: Optional[Any] = None,
) -> None:
    """
    Validate pipeline state after operator instantiation.

    Checks:
    - Each op's text_key exists in the dataset schema (non-default only)
    - Op type is supported by the configured executor (Ray mode restrictions)
    - Export path accessibility (if cfg provided)

    Args:
        ops: list of instantiated OP objects
        dataset_schema: Schema object from the dataset
        cfg: optional global config namespace

    Raises:
        PipelineRuntimeError: if any errors are found
    """
    errors: List[PreflightError] = []
    schema_columns = set(dataset_schema.columns) if dataset_schema else set()

    # Determine executor type for compatibility checks
    executor_type = getattr(cfg, "executor_type", "default") if cfg else "default"

    for op in ops:
        op_display_name = op._name or type(op).__name__

        # Check: text_key exists in schema — only when user explicitly set
        # a non-default text_key (meaning they expect that column to exist).
        # Default text_key may not be needed by multimedia ops.
        op_base_params = type(op)._BASE_PARAMS if hasattr(type(op), "_BASE_PARAMS") else OP._BASE_PARAMS
        default_text_key = op_base_params.get("text_key", (str, "text"))[1]
        if schema_columns and op.text_key != default_text_key and op.text_key not in schema_columns:
            errors.append(
                PreflightError(
                    op_display_name,
                    f"text_key '{op.text_key}' not found in dataset " f"columns: {sorted(schema_columns)}",
                )
            )

        # Check: op type is supported by the current executor mode
        if executor_type and executor_type not in op._supported_exec_modes:
            op_type_name = next(
                (b.__name__ for b in type(op).__mro__[1:] if issubclass(b, OP) and b is not OP),
                type(op).__name__,
            )
            errors.append(
                PreflightError(
                    op_display_name,
                    f"Operator type '{op_type_name}' does not support "
                    f"executor mode '{executor_type}'. "
                    f"Supported modes: {op._supported_exec_modes}",
                )
            )

    # Check config-level constraints
    if cfg:
        errors.extend(_check_export_path(cfg))

    if errors:
        raise PipelineRuntimeError(errors)


def _check_export_path(cfg) -> List[PreflightError]:
    """Check export path accessibility (local paths only)."""
    import os
    from urllib.parse import urlparse

    errors = []
    export_path = getattr(cfg, "export_path", None)
    if not export_path:
        return errors

    scheme = urlparse(export_path).scheme.lower()
    if scheme in ("s3", "hdfs", "oss"):
        # Remote paths: rely on the SDK's credential provider chain at export
        # time. Preflight cannot reliably detect all valid auth mechanisms
        # (IAM roles, Web Identity, shared credential files, etc.)
        pass
    else:
        # Local path: check parent directory is writable
        export_dir = os.path.dirname(os.path.abspath(export_path))
        if os.path.exists(export_dir) and not os.access(export_dir, os.W_OK):
            errors.append(
                PreflightError(
                    "__config__",
                    f"Export directory '{export_dir}' is not writable",
                )
            )

    return errors
