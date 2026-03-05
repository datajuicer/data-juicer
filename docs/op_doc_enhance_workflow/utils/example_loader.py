"""
Loader and processor for captured operator examples.

Responsible for:
- Loading captured examples from JSONL/JSON files
- Normalizing raw captured data into ExampleIR
- Selecting, deduplicating, and applying fallback rules for examples
- Converting ExampleIR to rendered HTML via view_model

This module is a pure data layer — it does NOT read or write markdown files.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .example_ir import ExampleIR, make_asset
from .view_model import to_legacy_view

# Operators excluded from example generation due to special requirements
NO_EXPLAIN_OPS = [
    "llm_task_relevance_filter",
    "in_context_influence_filter",
    "text_embd_similarity_filter",
    "audio_add_gaussian_noise_mapper",
    "image_blur_mapper",
    "image_captioning_from_gpt4v_mapper",
]

# Field aliases for normalizing captured data
FIELD_ALIASES: Dict[str, tuple] = {
    "text": ("text",),
    "images": ("images",),
    "videos": ("videos", "video"),
    "audios": ("audios", "audio"),
    "answer": ("answer",),
}
ALIAS_TO_CANON: Dict[str, str] = {
    alias: canon for canon, aliases in FIELD_ALIASES.items() for alias in aliases
}

# Internal fields to skip when normalizing captured data
SKIP_FIELDS = {}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_captured_examples(input_path: str) -> List[Dict]:
    """Load captured examples from a JSON array file or JSONL file."""
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    if content.startswith("["):
        return json.loads(content)

    records = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_examples_by_op(
    examples_path: Optional[str] = None,
    default_path: Optional[Path] = None,
) -> Dict[str, List[Dict]]:
    """Load captured examples and group them by operator name.

    Returns a dict mapping ``op_name`` to a list of raw captured dicts.
    """
    if examples_path is None:
        if default_path is None:
            return {}
        examples_path = str(default_path)

    path = Path(examples_path)
    if not path.exists():
        return {}

    raw = load_captured_examples(str(path))
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for example in raw:
        grouped[example["op_name"]].append(example)
    return dict(grouped)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _as_list_str(value: Any) -> List[str]:
    """Convert value to list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
        return list(value)
    return []


def _normalize_captured_sample(
    raw_data: Any, md_dir: Path
) -> List[Dict[str, Any]]:
    """Normalize a captured input/output dict into the sample list format
    expected by ExampleIR and view_model.

    Handles three shapes of captured data:
    - Single dict (one sample): {'text': '...', 'images': [...]}
    - List of dicts (multiple samples): [{'text': '...'}, ...]
    - Dict-of-lists (batched format): {'text': ['a', 'b'], 'images': [[], []]}
    """
    if raw_data is None:
        return []

    # Dict-of-lists (batched format) -> unbatch into list of dicts
    if isinstance(raw_data, dict):
        first_val = next(iter(raw_data.values()), None)
        is_batched = (
            isinstance(first_val, list)
            and all(isinstance(v, list) for v in raw_data.values())
        )
        if is_batched and first_val:
            num_rows = len(first_val)
            items = [
                {k: v[i] for k, v in raw_data.items() if i < len(v)}
                for i in range(num_rows)
            ]
        else:
            items = [raw_data]
    elif isinstance(raw_data, list) and raw_data and isinstance(raw_data[0], dict):
        items = raw_data
    else:
        return [{"text": str(raw_data)}]

    samples = []
    for item in items:
        sample: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}

        for key, value in item.items():
            if key in SKIP_FIELDS:
                continue

            canon = ALIAS_TO_CANON.get(key)
            if canon is None:
                meta[key] = value
                continue

            if canon in ("text", "answer"):
                sample[canon] = str(value) if value is not None else ""
            elif canon in ("images", "videos", "audios"):
                kind = canon.rstrip("s")  # image, video, audio
                paths = _as_list_str(value)
                if paths:
                    sample[canon] = [
                        make_asset(p, md_dir, kind=kind) for p in paths
                    ]
                else:
                    meta[key] = value
            else:
                meta[key] = value

        # Drop internal fields (__ prefix) whose values are empty
        filtered_meta = {
            k: v for k, v in meta.items()
            if not (
                k.startswith("__")
                and (not v)
            )
        }
        if filtered_meta:
            sample["meta"] = filtered_meta
        samples.append(sample)

    return samples


# ---------------------------------------------------------------------------
# ExampleIR construction
# ---------------------------------------------------------------------------

def _get_value_to_constant_name() -> Dict[str, str]:
    """Build a mapping from runtime constant values to their symbolic names.

    Used to render ``DATA_JUICER_ASSETS_CACHE`` as a bare variable name
    in op_code snippets instead of the expanded path string.
    """
    mapping = {}
    try:
        from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE
        mapping[DATA_JUICER_ASSETS_CACHE] = "DATA_JUICER_ASSETS_CACHE"
    except ImportError:
        pass
    return mapping

_VALUE_TO_CONST = _get_value_to_constant_name()


def format_op_code(op_class: str, op_params: Dict) -> str:
    """Generate a single-line Python snippet showing how to instantiate
    the operator.

    If a string parameter value matches a known internal constant
    (e.g. ``DATA_JUICER_ASSETS_CACHE``), it is rendered as the bare
    variable name without quotes.

    Example output:
        ExtractEntityAttributeMapper(api_model='qwen2.5-72b-instruct',
                                     query_entities=['李莲花', '方多病'])
    """
    if not op_params:
        return f"{op_class}()"

    param_parts = []
    for key, value in op_params.items():
        if isinstance(value, str):
            const_name = _VALUE_TO_CONST.get(value)
            if const_name:
                param_parts.append(f"{key}={const_name}")
            else:
                # Use repr() to preserve escape sequences like \n, \t
                param_parts.append(f"{key}={value!r}")
        else:
            param_parts.append(f"{key}={value!r}")

    params_str = ", ".join(param_parts)
    return f"{op_class}({params_str})"


def _captured_to_example_ir(
    captured: Dict, md_dir: Path
) -> ExampleIR:
    """Convert a single captured example dict into an ExampleIR instance."""
    input_samples = _normalize_captured_sample(captured.get("input_data"), md_dir)
    output_samples = _normalize_captured_sample(captured.get("output_data"), md_dir)

    return ExampleIR(
        method=captured.get("test_method", "unknown"),
        op_code=format_op_code(
            captured.get("op_class", ""),
            captured.get("op_params", {}),
        ),
        input={"samples": input_samples},
        output={"samples": output_samples},
    )


# ---------------------------------------------------------------------------
# Example selection & building
# ---------------------------------------------------------------------------

def build_examples_for_op(
    op_name: str,
    op_type: str,
    captured_by_op: Dict[str, List[Dict]],
    existing_examples: Optional[Dict] = None,
    md_dir: Optional[Path] = None,
) -> List[Dict]:
    """Build the examples list for a single operator from captured data.

    Always uses captured (JSONL) data as the source of truth for
    examples.  The only thing reused from existing markdown is the
    ``explanation`` field of examples whose ``method`` name matches a
    captured case — these explanations are typically human-reviewed.

    If there are no captured examples for this operator, an empty list
    is returned (the caller should delete the stale markdown file).

    Returns a list of example dicts with keys:
    ``method``, ``op_code``, ``input``, ``output``, ``explanation``.
    """
    if op_name in NO_EXPLAIN_OPS:
        return []

    if md_dir is None:
        return []

    captured_list = captured_by_op.get(op_name, [])
    if not captured_list:
        return []

    existing_by_method = (
        existing_examples if isinstance(existing_examples, dict) else {}
    )

    md_dir_abs = md_dir / op_type

    # Deduplicate by test_method and convert to ExampleIR -> HTML
    seen_methods = set()
    usable = []
    for captured in captured_list:
        method_key = captured.get("test_method", "unknown")
        if method_key in seen_methods:
            continue
        # Skip parallel / numpy variants
        if any(tag in method_key for tag in ["parallel", "np"]):
            continue
        seen_methods.add(method_key)

        example_ir = _captured_to_example_ir(captured, md_dir_abs)
        rendered = to_legacy_view(example_ir)
        usable.append({
            "method": method_key,
            "op_code": example_ir.op_code or "",
            "input": rendered["input"],
            "output": rendered["output"],
        })

    if not usable:
        return []

    # Attach existing explanations to matching method names
    for example in usable:
        existing = existing_by_method.get(example["method"])
        example["explanation"] = (
            existing.get("explanation", "")
            if isinstance(existing, dict) else ""
        ).strip()

    # Prioritise examples that already have a human-reviewed explanation,
    # then keep original order.  Limit to 2 examples per operator.
    indexed = [(i, e) for i, e in enumerate(usable)]
    indexed.sort(key=lambda pair: (not pair[1]["explanation"], pair[0]))
    return [e for _, e in indexed[:2]]


def build_template_examples(
    captured_examples: List[Dict],
    md_dir: Optional[Path] = None,
) -> Dict[str, List[Dict]]:
    """Convert raw captured examples into the format expected by
    op_doc.md.j2.

    Uses the same multi-modal rendering pipeline:
    captured dict -> ExampleIR -> view_model.to_legacy_view() -> HTML.

    Returns a dict mapping op_name -> list of example dicts, where each
    example dict has keys: method, op_code, input, output, explanation.
    """
    if md_dir is None:
        return {}

    grouped = defaultdict(list)
    for example in captured_examples:
        grouped[example["op_name"]].append(example)

    result = {}
    for op_name, examples in grouped.items():
        op_type = examples[0].get("op_type", "unknown")
        result[op_name] = build_examples_for_op(
            op_name=op_name,
            op_type=op_type,
            captured_by_op=dict(grouped),
            existing_examples=None,
            md_dir=md_dir,
        )

    return result
