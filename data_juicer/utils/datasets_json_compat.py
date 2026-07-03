"""
HuggingFace ``datasets`` parses JSON/JSON lines via pandas, which may call
``ujson``. UltraJSON rejects some values that CPython's ``json`` accepts,
notably very large integers, raising ``ValueError: Value is too big!``.

Set environment variable ``DATA_JUICER_USE_STDLIB_JSON=1`` (or ``true`` /
``yes`` / ``on``) before running ``dj-process`` (or any code path that calls
``init_configs``) to force the datasets stack to use ``json.loads`` instead.
"""

from __future__ import annotations

import importlib
import json
import os
from typing import Any, Union

from loguru import logger

_ENV_FLAG = "DATA_JUICER_USE_STDLIB_JSON"
_PATCHED = False


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def apply_stdlib_json_patch_for_datasets() -> bool:
    """
    If ``DATA_JUICER_USE_STDLIB_JSON`` is enabled, replace
    ``datasets.utils.json.ujson_loads`` with ``json.loads`` (bytes-safe).

    .. note::
        We patch both ``datasets.utils.json`` and
        ``datasets.packaged_modules.json.json`` because the latter imports
        ``ujson_loads`` at module load time (``from ... import ujson_loads``),
        which binds the function object directly. Modifying the source module's
        attribute does not affect already-bound references.

    :return: whether the patch was applied in this process.
    """
    global _PATCHED
    if _PATCHED:
        return True
    if not _truthy_env(_ENV_FLAG):
        return False
    try:
        import datasets.utils.json as ds_json
    except ImportError:
        logger.warning(f"{_ENV_FLAG} is set but `datasets` is not installed; skipping JSON patch.")
        return False
    if not hasattr(ds_json, "ujson_loads"):
        logger.warning(
            f"{_ENV_FLAG} is set but `datasets.utils.json` has no ujson_loads; "
            "skipping JSON patch (your datasets version may differ)."
        )
        return False

    def _stdlib_loads(data: Union[str, bytes, bytearray], *_args: Any, **kwargs: Any) -> Any:
        # ``json.loads`` does not accept ujson-only kwargs; ignore extras for compatibility.
        kwargs.pop("precise_float", None)
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        return json.loads(data)

    # Patch the source module
    ds_json.ujson_loads = _stdlib_loads  # type: ignore[assignment]

    # Also patch the module that actually uses ujson_loads in load_dataset().
    # datasets.packaged_modules.json.json imports ujson_loads at module load time:
    #     from datasets.utils.json import ujson_loads
    # This binds the function object directly, so modifying ds_json.ujson_loads
    # does not affect the already-bound reference in that module.
    # Note: we use importlib because the module name 'json' conflicts with stdlib.
    try:
        ds_json_loader = importlib.import_module("datasets.packaged_modules.json.json")
        if hasattr(ds_json_loader, "ujson_loads"):
            ds_json_loader.ujson_loads = _stdlib_loads
    except ImportError:
        pass  # Older datasets versions may not have this module structure

    _PATCHED = True
    logger.info(
        f"Applied datasets JSON workaround: {_ENV_FLAG}=1 " "(using stdlib json instead of ujson for JSONL parsing)."
    )
    return True
