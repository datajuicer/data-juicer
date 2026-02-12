import json
import sys
import time
from ast import literal_eval
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_juicer.core.data import NestedDataset as Dataset

import importlib
import signal
from contextlib import contextmanager

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


EXAMPLES_PATH = Path(__file__).parent / "examples.json"
BACKUP_ENABLED = True
MAX_ROWS_PER_METHOD = 50  # Maximum number of dataset rows to process per method, None for no limit
PRINT_EVERY = 50  # Print progress every N processed methods

# Specify which operator modules to import
OPERATOR_MODULES = [
    "data_juicer.ops.filter",
    "data_juicer.ops.mapper",
    "data_juicer.ops.deduplicator",
    "data_juicer.ops.selector",
    "data_juicer.ops.aggregator",
    "data_juicer.ops.grouper",
    "data_juicer.ops.pipeline"
]


def _build_eval_env() -> Dict[str, Any]:
    """
    Build restricted eval environment with specified operator modules.
    Only imports public symbols from predefined operator modules.
    """
    env: Dict[str, Any] = {"__builtins__": {}}

    for modname in OPERATOR_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            print(f"[FATAL] cannot import {modname}: {e}", file=sys.stderr)
            continue

        for k, v in mod.__dict__.items():
            if not k.startswith("_"):
                env[k] = v

    env["True"] = True
    env["False"] = False
    env["None"] = None
    return env


_EVAL_ENV = _build_eval_env()


def _safe_eval_op(op_code: str):
    return eval(op_code, _EVAL_ENV, {})


def _result_to_list(result) -> List[Dict[str, Any]]:
    try:
        return str(result.to_list())
    except Exception as e:
        raise RuntimeError(f"result.to_list() failed: {e}") from e


def _iter_methods_dict(maybe_nested: Dict[str, Any], op_name: str) -> Optional[Dict[str, Any]]:
    """
    Handle two data structures:
    1) {op_name: {method: payload}}
    2) {method: payload}
    Returns {method: payload} or None
    """
    if not isinstance(maybe_nested, dict):
        return None
    if op_name in maybe_nested and isinstance(maybe_nested[op_name], dict):
        return maybe_nested[op_name]
    return maybe_nested


def fill_tgt_inplace() -> None:
    """
    Scan examples.json, execute op.run() for methods with op_code+ds but missing tgt,
    and write back the results to tgt field.
    """
    path = EXAMPLES_PATH.resolve()
    if not path.exists():
        print(f"[ERROR] not found: {path}", file=sys.stderr)
        return

    # Create backup
    if BACKUP_ENABLED:
        bak = path.with_suffix(path.suffix + f".{int(time.time())}.bak")
        try:
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[Info] backup saved -> {bak}")
        except Exception as e:
            print(f"[WARN] backup failed: {e}", file=sys.stderr)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[FATAL] invalid JSON: {e}", file=sys.stderr)
        return

    total_methods = 0
    updated = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    # Iterate through each operator
    for top_op_name, nested in list(data.items()):
        timeout_flag = False
        methods = _iter_methods_dict(nested, top_op_name)
        if not isinstance(methods, dict):
            continue

        for method, payload in list(methods.items()):
            if timeout_flag:
                continue
            total_methods += 1
            if total_methods % PRINT_EVERY == 0:
                dt = time.time() - t0
                print(
                    f"[Progress] processed={total_methods}, updated={updated}, skipped={skipped}, failed={failed}, elapsed={dt:.1f}s"
                )

            try:
                op_code = payload.get("op_code")
                ds_obj = payload.get("ds")
                tgt = payload.get("tgt", None)

                # Condition: has op_code + ds, but missing tgt
                if not op_code or ds_obj is None:
                    skipped += 1
                    continue
                if tgt is not None:
                    skipped += 1
                    continue

                ds_obj = literal_eval(ds_obj)
                # Expect ds to be list[dict]; wrap single dict in list
                if not isinstance(ds_obj, list):
                    ds_list = [ds_obj]
                else:
                    ds_list = ds_obj

                # Create Dataset
                try:
                    dataset = Dataset.from_list(ds_list)
                except Exception as e:
                    failed += 1
                    print(f"[WARN] {top_op_name}::{method} Dataset.from_list failed: {e}", file=sys.stderr)
                    continue

                # Instantiate and run operator
                try:
                    op = _safe_eval_op(op_code)
                except Exception as e:
                    failed += 1
                    print(f"[WARN] {top_op_name}::{method} eval op_code failed: {e}", file=sys.stderr)
                    continue

                try:
                    with timeout(180):  # 3 minutes = 180 seconds
                        result = op.run(dataset)
                except TimeoutError as e:
                    failed += 1
                    timeout_flag = True
                    print(f"[WARN] {top_op_name}::{method} op.run timeout: {e}", file=sys.stderr)
                    continue
                except Exception as e:
                    failed += 1
                    tb = traceback.format_exc(limit=2)
                    print(f"[WARN] {top_op_name}::{method} op.run failed: {e} | {tb}", file=sys.stderr)
                    continue

                # Extract results and write back to JSON (store as Python list for JSON serialization)
                try:
                    tgt_list = _result_to_list(result)
                    payload["tgt"] = tgt_list
                    updated += 1
                except Exception as e:
                    failed += 1
                    print(f"[WARN] {top_op_name}::{method} result.to_list failed: {e}", file=sys.stderr)
                    continue

                # Write back to file
                try:
                    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception as e:
                    print(f"[ERROR] write output failed: {e}", file=sys.stderr)
                    return
            except Exception as e:
                failed += 1
                print(f"[WARN] {top_op_name}::{method} unexpected error: {e}", file=sys.stderr)
                continue
        
        timeout_flag = False

    dt = time.time() - t0
    print(f"[Done] methods={total_methods} updated={updated} skipped={skipped} failed={failed} elapsed={dt:.1f}s")


if __name__ == "__main__":
    fill_tgt_inplace()
