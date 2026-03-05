#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runtime Hook for capturing operator examples from unit tests.

Instead of statically parsing test files (which is fragile due to diverse
coding styles), this module monkey-patches operator base classes so that
every call to process_single / compute_stats_single / process etc.
automatically records (input, output, op_params) tuples.

Usage:
    1. As a pytest plugin (automatic):
       Place the companion conftest.py in the test root and run pytest.

    2. Programmatic:
       >>> from capture_examples import ExampleCapture
       >>> with ExampleCapture() as cap:
       ...     # run any test code that exercises operators
       ...     pass
       >>> cap.save("captured_examples.json")
"""

import copy
import inspect
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


MAX_STR_PREVIEW_LEN = 500
MAX_EXAMPLES_PER_OP = 10

# Project root directory — used to convert absolute paths to relative ones
# so that captured JSON never leaks machine-specific paths.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


@dataclass
class CapturedExample:
    """One captured input/output pair for an operator."""

    op_class: str
    op_name: str
    op_type: str
    op_params: Dict[str, Any]
    test_method: str
    test_file: str
    input_data: Any
    output_data: Any


def _truncate_long_strings(obj, max_length=MAX_STR_PREVIEW_LEN):
    """Recursively truncate long string values for readability."""
    if isinstance(obj, str):
        if len(obj) > max_length:
            return obj[:max_length] + f"... [truncated, total {len(obj)} chars]"
        return obj
    elif isinstance(obj, dict):
        return {k: _truncate_long_strings(v, max_length) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        truncated = [_truncate_long_strings(item, max_length) for item in obj]
        return type(obj)(truncated)
    return obj


def _convert_to_plain_python(obj):
    """
    Convert HuggingFace LazyBatch/LazyDict and other special types
    to plain Python dicts/lists before deep copying.

    The HuggingFace datasets library wraps data in LazyBatch (for
    batched map) and LazyDict (for single map) objects that cannot
    be deep-copied. We convert them to plain dicts first.
    """
    try:
        from datasets.formatting.formatting import LazyBatch, LazyDict
        if isinstance(obj, (LazyBatch, LazyDict)):
            return {k: _convert_to_plain_python(obj[k]) for k in obj.keys()}
    except ImportError:
        pass

    try:
        import pyarrow as pa
        if isinstance(obj, pa.Table):
            return obj.to_pydict()
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _convert_to_plain_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_plain_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_plain_python(item) for item in obj)

    return obj


def _safe_deepcopy(obj):
    """Deep copy with fallback for non-copyable objects."""
    # First convert special types (LazyBatch, LazyDict, etc.) to plain Python
    obj = _convert_to_plain_python(obj)
    try:
        return copy.deepcopy(obj)
    except Exception:
        try:
            return copy.copy(obj)
        except Exception:
            return repr(obj)


def _get_base_op_defaults() -> Dict[str, Any]:
    """Extract the default values for OP base class parameters.

    Since ``OP.__init__`` uses ``*args, **kwargs`` (no named parameters
    in the signature), we cannot rely on ``inspect.signature`` to get
    defaults.  Instead we maintain a mapping that mirrors the
    ``kwargs.get(name, default)`` calls in ``OP.__init__``.

    Parameters whose default depends on runtime state (e.g.
    ``batch_size`` depends on ``accelerator``) use a sentinel so they
    are always kept when the user passes them explicitly.
    """
    return {
        # Data key parameters
        "text_key": "text",
        "image_key": "images",
        "audio_key": "audios",
        "video_key": "videos",
        "image_bytes_key": "image_bytes",
        "system_key": "system",
        "instruction_key": "instruction",
        "prompt_key": "prompt",
        "query_key": "query",
        "response_key": "response",
        "history_key": "history",
        "index_key": None,
        "work_dir": None,
        # Execution parameters
        "skip_op_error": False,
        "auto_op_parallelism": True,
        "batch_mode": None,
        "accelerator": None,
        "batch_size": None,  # depends on accelerator, use None as sentinel
        "num_proc": None,    # depends on auto_op_parallelism, use None
        # Resource parameters
        "cpu_required": None,
        "gpu_required": None,
        "mem_required": None,
        "num_cpus": None,
        "num_gpus": None,
        "memory": None,
        "runtime_env": None,
        "ray_execution_mode": None,
        "turbo": False,
    }


# Cache the base defaults so we don't rebuild the dict on every call
_BASE_OP_DEFAULTS = _get_base_op_defaults()


def _extract_op_params(op_instance) -> Dict[str, Any]:
    """
    Extract the user-specified constructor parameters of an operator.

    The OPMetaClass already stores _init_args and _init_kwargs on every
    OP instance, so we can inspect the __init__ signature to map
    positional args to parameter names and merge with kwargs.

    Instead of using a hard-coded skip list, we compare each base-class
    parameter against its default value.  If the user explicitly passed
    a value that differs from the default, it is kept — this matches
    the behaviour of AST-based extraction where the literal source text
    is preserved.
    """
    params = {}

    init_kwargs = getattr(op_instance, "_init_kwargs", {})
    init_args = getattr(op_instance, "_init_args", ())

    # Get the __init__ signature to map positional args
    try:
        sig = inspect.signature(op_instance.__class__.__init__)
        param_names = [
            name for name, param in sig.parameters.items()
            if name != "self"
            and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ]

        # Map positional args to parameter names
        for idx, arg_value in enumerate(init_args):
            if idx < len(param_names):
                params[param_names[idx]] = arg_value

    except (ValueError, TypeError):
        pass

    # Merge keyword arguments (these take precedence)
    params.update(init_kwargs)

    # Filter out base-class parameters whose value equals the default.
    # If the user explicitly overrode a base param (e.g. batch_size=32
    # when the default is 1000), it is preserved.
    _sentinel = object()
    filtered_params = {}
    for key, value in params.items():
        if key in ("args", "kwargs"):
            continue
        default = _BASE_OP_DEFAULTS.get(key, _sentinel)
        if default is not _sentinel and default == value:
            # Base-class param with unchanged default → skip
            continue
        filtered_params[key] = _make_json_serializable(value)

    return filtered_params


def _make_json_serializable(obj):
    """Convert an object to a JSON-serializable form."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return sorted(_make_json_serializable(item) for item in obj)
    elif isinstance(obj, bytes):
        return f"<bytes, len={len(obj)}>"
    elif hasattr(obj, "__dict__"):
        return f"<{type(obj).__name__}>"
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return repr(obj)


def _get_op_type_name(op_instance) -> str:
    """Determine the operator type (mapper, filter, etc.)."""
    from data_juicer.ops.base_op import (
        Aggregator,
        Deduplicator,
        Filter,
        Grouper,
        Mapper,
        Pipeline,
        Selector,
    )

    type_map = [
        (Mapper, "mapper"),
        (Filter, "filter"),
        (Deduplicator, "deduplicator"),
        (Selector, "selector"),
        (Grouper, "grouper"),
        (Aggregator, "aggregator"),
        (Pipeline, "pipeline"),
    ]
    for cls, name in type_map:
        if isinstance(op_instance, cls):
            return name
    return "unknown"


def _get_caller_test_info() -> tuple:
    """Walk the call stack to find the test method and file that triggered this call."""
    for frame_info in inspect.stack():
        func_name = frame_info.function
        filename = frame_info.filename
        if func_name.startswith("test_") and "tests/" in filename:
            return func_name, filename
    return "unknown", "unknown"


class ExampleCapture:
    """
    Context manager that monkey-patches operator classes to capture
    input/output examples at runtime.

    Supports incremental persistence (each captured example is appended
    to a JSONL file immediately) and resume from a previous capture
    session (skipping operators that already have enough examples).
    """

    # Media field keys whose values are file paths.  When the output
    # contains paths that were *not* present in the input, the operator
    # has generated new files (artifacts) that only exist on the local
    # disk and will be broken links in CI-generated documentation.
    _MEDIA_KEYS = ("images", "videos", "audios")

    def __init__(self, max_examples_per_op=MAX_EXAMPLES_PER_OP,
                 truncate_strings=False, max_str_len=MAX_STR_PREVIEW_LEN,
                 streaming_output: Optional[str] = None,
                 resume_from: Optional[str] = None,
                 skip_multimodal_mappers: bool = True):
        self.max_examples_per_op = max_examples_per_op
        self.truncate_strings = truncate_strings
        self.max_str_len = max_str_len
        self.skip_multimodal_mappers = skip_multimodal_mappers

        self.captured: List[CapturedExample] = []
        self._example_counts: Dict[str, int] = defaultdict(int)
        self._original_methods: Dict[str, Any] = {}
        self._patched = False

        # Streaming output: append each example to this JSONL file
        self._streaming_path: Optional[Path] = None
        self._streaming_file = None

        # Resume support: load existing examples and skip already-captured ops
        if resume_from and Path(resume_from).exists():
            self._load_existing(resume_from)
            # When resuming, default streaming output to the same file
            if streaming_output is None:
                streaming_output = resume_from

        if streaming_output:
            self._streaming_path = Path(streaming_output)
            self._streaming_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_existing(self, filepath: str):
        """Load previously captured examples from a JSONL or JSON file for resume.

        Handles two formats transparently:

        1. **JSONL** – each line is one captured record (produced by
           streaming mode).
        2. **JSON array** – a ``[{...}, ...]`` array of captured records
           (produced by :meth:`save`).

        Each record's ``input_data`` / ``output_data`` are kept as-is
        (dict-of-lists when captured at the ``process`` level).
        """
        path = Path(filepath)

        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return

        raw_records: List[Dict] = []
        if content.startswith("["):
            raw_records = json.loads(content)
        else:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        for record in raw_records:
            example = CapturedExample(**record)
            self.captured.append(example)
            self._example_counts[example.op_name] += 1

        print(f"Resumed from {filepath}: loaded {len(raw_records)} "
              f"records for {len(self._example_counts)} operators")

    def _open_streaming_file(self):
        """Open the streaming output file for appending."""
        if self._streaming_path and self._streaming_file is None:
            self._streaming_file = open(
                self._streaming_path, "a", encoding="utf-8"
            )

    def _close_streaming_file(self):
        """Flush and close the streaming output file."""
        if self._streaming_file is not None:
            self._streaming_file.flush()
            self._streaming_file.close()
            self._streaming_file = None

    def _flush_example(self, example: CapturedExample):
        """Immediately append one example to the streaming JSONL file."""
        if self._streaming_file is None:
            return
        line = json.dumps(asdict(example), ensure_ascii=False)
        self._streaming_file.write(line + "\n")
        self._streaming_file.flush()

    def start(self):
        """Install monkey-patches on operator base classes."""
        if self._patched:
            return
        self._open_streaming_file()
        self._patch_all_ops()
        self._patched = True

    def stop(self):
        """Remove monkey-patches and restore original methods."""
        if not self._patched:
            return
        self._restore_all()
        self._close_streaming_file()
        self._patched = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    @staticmethod
    def _compact_binary_fields(data):
        """Replace bulky binary-placeholder lists with a short summary.

        Fields like ``__dj__minhash`` contain lists of serialized bytes
        (rendered as ``"<bytes, len=80>"`` after JSON conversion).
        These add noise to documentation examples.  We keep the field
        to show it *exists*, but replace the value with a concise
        English note like ``"<25 hashes omitted>"`` so readers know
        the data is present but abbreviated.

        Other ``__dj__*`` metadata fields (``__dj__stats``,
        ``__dj__uid``, etc.) are kept intact — they are useful for
        understanding what the operator actually produces.
        """
        if not isinstance(data, dict):
            return data

        compacted = {}
        for key, value in data.items():
            if (
                "minhash" in key
                and isinstance(value, list)
                and len(value) > 1
                and all(
                    isinstance(item, str) and item.startswith("<bytes")
                    for item in value
                )
            ):
                compacted[key] = f"<{len(value)} hashes omitted>"
            else:
                compacted[key] = value
        return compacted

    @staticmethod
    def _to_relative_path(filepath: str) -> str:
        """Convert an absolute file path to a project-root-relative path.

        If the path starts with the project root, strip it and return
        the relative portion.  Otherwise return the path unchanged.
        """
        if not filepath or filepath == "unknown":
            return filepath
        # Normalize both to handle trailing slashes etc.
        root = _PROJECT_ROOT.rstrip("/") + "/"
        if filepath.startswith(root):
            return filepath[len(root):]
        return filepath

    @staticmethod
    def _relativize_paths(data):
        """Recursively replace absolute paths with project-relative paths.

        Walks dicts, lists, and strings.  Any string that contains the
        project root directory is shortened to a relative path.
        """
        root_prefix = _PROJECT_ROOT.rstrip("/") + "/"
        if isinstance(data, str):
            return data.replace(root_prefix, "")
        elif isinstance(data, dict):
            return {
                k: ExampleCapture._relativize_paths(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [ExampleCapture._relativize_paths(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(ExampleCapture._relativize_paths(item) for item in data)
        return data

    def _clean_sample_list(self, samples):
        """Clean a list-of-dicts: make JSON-serializable, compact binary
        fields, relativize paths, and optionally truncate strings."""
        cleaned = []
        for sample in samples:
            s = _safe_deepcopy(sample)
            if self.truncate_strings:
                s = _truncate_long_strings(s, self.max_str_len)
            s = _make_json_serializable(s)
            s = self._compact_binary_fields(s)
            s = self._relativize_paths(s)
            cleaned.append(s)
        return cleaned

    def _is_multimodal_mapper(self, op_instance) -> bool:
        """Check whether the operator is a multimodal mapper.

        A multimodal mapper is a Mapper whose name contains image,
        video, or audio prefixes, indicating it processes media files
        that may only exist on the local disk.
        """
        from data_juicer.ops.base_op import Mapper

        if not isinstance(op_instance, Mapper):
            return False

        op_name = getattr(op_instance, "_name", op_instance.__class__.__name__)
        multimodal_prefixes = ("image_", "video_", "audio_")
        return any(op_name.startswith(prefix) for prefix in multimodal_prefixes)

    def _record(self, op_instance, input_samples, output_samples):
        """Record a captured example from an operator's ``run`` method.

        ``input_samples`` and ``output_samples`` are list-of-dicts
        obtained from ``dataset.to_list()`` before and after ``run``.
        """
        op_class_name = op_instance.__class__.__name__
        op_name = getattr(op_instance, "_name", op_class_name)
        op_type = _get_op_type_name(op_instance)

        # Rate-limit per operator
        if self._example_counts[op_name] >= self.max_examples_per_op:
            return

        # Skip multimodal mappers entirely — their media files only
        # exist locally and would be broken links in CI documentation.
        if self.skip_multimodal_mappers and self._is_multimodal_mapper(op_instance):
            return

        test_method, test_file = _get_caller_test_info()
        op_params = _extract_op_params(op_instance)

        # Clean samples
        input_clean = self._clean_sample_list(input_samples)
        output_clean = self._clean_sample_list(output_samples)

        # Convert absolute paths to relative paths
        test_file = self._to_relative_path(test_file)
        op_params = self._relativize_paths(op_params)

        example = CapturedExample(
            op_class=op_class_name,
            op_name=op_name,
            op_type=op_type,
            op_params=op_params,
            test_method=test_method,
            test_file=test_file,
            input_data=input_clean,
            output_data=output_clean,
        )
        self.captured.append(example)
        self._example_counts[op_name] += 1

        # Incremental persistence: flush to disk immediately
        self._flush_example(example)

    # ---- Patching strategy per operator type ----
    #
    # Each operator type has a different ``run`` workflow, so we
    # customise the capture points accordingly:
    #
    # Mapper / Aggregator:
    #   run → dataset.map(self.process)
    #   ⇒ capture input & output on the map call.
    #
    # Filter:
    #   run → dataset.map(self.compute_stats) → dataset.filter(self.process)
    #   ⇒ capture *input* on map(compute_stats) (before stats are added),
    #     capture *output* on filter(process) (after rows are removed).
    #   The two halves are stitched together via ``_pending_filter_inputs``.
    #
    # Deduplicator:
    #   run → dataset.map(self.compute_hash) → self.process(dataset, …)
    #   ⇒ capture *input* on map(compute_hash),
    #     capture *output* by hooking self.process (dataset-level).
    #
    # Selector / Grouper:
    #   run → self.process(dataset)
    #   ⇒ hook self.process to capture input & output (dataset-level).

    @staticmethod
    def _extract_op_from_func(func):
        """Try to extract the OP instance from a function passed to
        ``dataset.map`` or ``dataset.filter``.

        Returns ``(op_instance, method_name)`` or ``(None, None)``.
        """
        from data_juicer.ops.base_op import OP

        # Unwrap decorators (wrap_func_with_nested_access, etc.)
        original = func
        while not inspect.ismethod(original) and hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        if inspect.ismethod(original) and isinstance(original.__self__, OP):
            return original.__self__, original.__func__.__name__

        return None, None

    def _patch_all_ops(self):
        from data_juicer.core.data import NestedDataset
        from data_juicer.ops.base_op import (
            Deduplicator,
            Filter,
            Grouper,
            Selector,
        )

        original_map = NestedDataset.map
        original_filter = NestedDataset.filter
        capture = self

        # Pending input samples for Filter ops (keyed by id(op_instance)).
        # When we see map(compute_stats) we stash the *pre-stats* input;
        # when we later see filter(process) for the same op we pop it.
        self._pending_filter_inputs: Dict[int, list] = {}

        # Pending input samples for Deduplicator ops (keyed by id(op)).
        # Stashed on map(compute_hash), consumed when process() is called.
        self._pending_dedup_inputs: Dict[int, list] = {}

        # ----------------------------------------------------------
        # Patched NestedDataset.map
        # ----------------------------------------------------------
        def patched_map(dataset_self, *args, **kargs):
            func = args[0] if args else kargs.get("function")
            op_instance, method_name = (
                capture._extract_op_from_func(func) if func else (None, None)
            )

            if op_instance is None:
                return original_map(dataset_self, *args, **kargs)

            # --- Filter: map(compute_stats) is the INPUT capture point ---
            if isinstance(op_instance, Filter) and method_name in (
                "compute_stats",
                "compute_stats_single",
                "compute_stats_batched",
            ):
                try:
                    capture._pending_filter_inputs[id(op_instance)] = (
                        dataset_self.to_list()
                    )
                except Exception:
                    capture._pending_filter_inputs[id(op_instance)] = []
                # Run the real map but do NOT record yet — wait for filter()
                return original_map(dataset_self, *args, **kargs)

            # --- Deduplicator: map(compute_hash) is the INPUT capture point ---
            if isinstance(op_instance, Deduplicator) and method_name in (
                "compute_hash",
            ):
                try:
                    capture._pending_dedup_inputs[id(op_instance)] = (
                        dataset_self.to_list()
                    )
                except Exception:
                    capture._pending_dedup_inputs[id(op_instance)] = []
                return original_map(dataset_self, *args, **kargs)

            # --- Mapper / Aggregator: map(process) captures both sides ---
            try:
                input_samples = dataset_self.to_list()
            except Exception:
                input_samples = []

            result = original_map(dataset_self, *args, **kargs)

            try:
                output_samples = result.to_list()
            except Exception:
                output_samples = []

            capture._record(op_instance, input_samples, output_samples)
            return result

        # ----------------------------------------------------------
        # Patched NestedDataset.filter
        # ----------------------------------------------------------
        def patched_filter(dataset_self, *args, **kargs):
            func = args[0] if args else kargs.get("function")
            op_instance, method_name = (
                capture._extract_op_from_func(func) if func else (None, None)
            )

            if op_instance is None:
                return original_filter(dataset_self, *args, **kargs)

            # --- Filter: filter(process) is the OUTPUT capture point ---
            # Use the stashed pre-stats input if available; otherwise
            # fall back to the current dataset (which has stats columns).
            input_samples = capture._pending_filter_inputs.pop(
                id(op_instance), None
            )
            if input_samples is None:
                try:
                    input_samples = dataset_self.to_list()
                except Exception:
                    input_samples = []

            result = original_filter(dataset_self, *args, **kargs)

            try:
                output_samples = result.to_list()
            except Exception:
                output_samples = []

            capture._record(op_instance, input_samples, output_samples)
            return result

        NestedDataset.map = patched_map
        NestedDataset.filter = patched_filter
        self._original_methods["NestedDataset.map"] = original_map
        self._original_methods["NestedDataset.filter"] = original_filter

        # ----------------------------------------------------------
        # Patch Deduplicator.process (dataset-level)
        # ----------------------------------------------------------
        original_dedup_process = Deduplicator.process

        def patched_dedup_process(op_self, dataset, show_num=0):
            input_samples = capture._pending_dedup_inputs.pop(
                id(op_self), None
            )
            if input_samples is None:
                try:
                    input_samples = dataset.to_list()
                except Exception:
                    input_samples = []

            result = original_dedup_process(op_self, dataset, show_num)

            # Deduplicator.process returns (new_dataset, dup_pairs)
            new_dataset = result[0] if isinstance(result, tuple) else result
            try:
                output_samples = new_dataset.to_list()
            except Exception:
                output_samples = []

            capture._record(op_self, input_samples, output_samples)
            return result

        Deduplicator.process = patched_dedup_process
        self._original_methods["Deduplicator.process"] = original_dedup_process

        # ----------------------------------------------------------
        # Patch Selector.process (dataset-level)
        # ----------------------------------------------------------
        original_selector_process = Selector.process

        def patched_selector_process(op_self, dataset):
            try:
                input_samples = dataset.to_list()
            except Exception:
                input_samples = []

            result = original_selector_process(op_self, dataset)

            try:
                output_samples = result.to_list()
            except Exception:
                output_samples = []

            capture._record(op_self, input_samples, output_samples)
            return result

        Selector.process = patched_selector_process
        self._original_methods["Selector.process"] = original_selector_process

        # ----------------------------------------------------------
        # Patch Grouper.process (dataset-level)
        # ----------------------------------------------------------
        original_grouper_process = Grouper.process

        def patched_grouper_process(op_self, dataset):
            try:
                input_samples = dataset.to_list()
            except Exception:
                input_samples = []

            result = original_grouper_process(op_self, dataset)

            # Grouper.process returns a list of batched samples,
            # not a Dataset, so we use it directly.
            try:
                if isinstance(result, list):
                    output_samples = _safe_deepcopy(result)
                else:
                    output_samples = result.to_list()
            except Exception:
                output_samples = []

            capture._record(op_self, input_samples, output_samples)
            return result

        Grouper.process = patched_grouper_process
        self._original_methods["Grouper.process"] = original_grouper_process

    def _restore_all(self):
        """Restore all monkey-patched methods."""
        from data_juicer.core.data import NestedDataset
        from data_juicer.ops.base_op import (
            Deduplicator,
            Grouper,
            Selector,
        )

        if "NestedDataset.map" in self._original_methods:
            NestedDataset.map = self._original_methods["NestedDataset.map"]
        if "NestedDataset.filter" in self._original_methods:
            NestedDataset.filter = self._original_methods["NestedDataset.filter"]
        if "Deduplicator.process" in self._original_methods:
            Deduplicator.process = self._original_methods["Deduplicator.process"]
        if "Selector.process" in self._original_methods:
            Selector.process = self._original_methods["Selector.process"]
        if "Grouper.process" in self._original_methods:
            Grouper.process = self._original_methods["Grouper.process"]
        self._original_methods.clear()

    # ---- Resume support ----

    def get_covered_test_keys(self) -> set:
        """Return the set of (test_file, test_method) pairs already captured.

        Used by the pytest plugin to deselect tests that have already
        been captured in a previous session, so they are not run again.

        Records with ``test_file == "unknown"`` or
        ``test_method == "unknown"`` are excluded because they cannot
        be reliably matched to pytest items.
        """
        covered = set()
        for example in self.captured:
            test_file = example.test_file
            test_method = example.test_method
            if test_file == "unknown" or test_method == "unknown":
                continue
            covered.add((test_file, test_method))
        return covered

    # ---- Output ----

    def get_examples_by_op(self) -> Dict[str, List[CapturedExample]]:
        """Group captured examples by operator name."""
        grouped = defaultdict(list)
        for example in self.captured:
            grouped[example.op_name].append(example)
        return dict(grouped)

    def to_dict_list(self) -> List[Dict]:
        """Convert all captured examples to a list of dicts.

        Since we capture at the ``run`` level, each record already
        contains the complete input/output as list-of-dicts.
        No further aggregation is needed.
        """
        return [asdict(ex) for ex in self.captured]

    # TODO: delete this?
    def save(self, output_path):
        """Save all captured examples to a JSONL file.

        Each line is a single JSON object — one captured example.
        This is consistent with the streaming output format used
        during capture, so the file can be resumed transparently.

        :param output_path: File path (str or Path) to write the JSONL to.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict_list()
        with open(output_path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"✅ Saved {len(data)} captured examples to {output_path}")
        return output_path

    def summary(self) -> str:
        """Return a human-readable summary of captured examples."""
        grouped = self.get_examples_by_op()
        lines = [f"📊 Captured examples summary: {len(self.captured)} total"]
        for op_name, examples in sorted(grouped.items()):
            op_type = examples[0].op_type
            test_methods = {ex.test_method for ex in examples}
            lines.append(
                f"  - {op_name} ({op_type}): "
                f"{len(examples)} examples from "
                f"{len(test_methods)} test methods"
            )
        return "\n".join(lines)
