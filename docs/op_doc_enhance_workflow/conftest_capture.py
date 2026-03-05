#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest plugin for automatic operator example capture.

This file is NOT a conftest.py itself — it provides a reusable pytest
plugin that can be activated in two ways:

1. Copy/symlink as ``tests/conftest.py`` (or import from there).
2. Run pytest with ``-p docs.op_doc_enhance_workflow.conftest_capture``.

When activated with ``--capture-op-examples``, it installs the runtime
hooks before any test runs and saves the captured examples to a JSON
file after the session finishes.

Usage:
    pytest tests/ops/ --capture-op-examples
    pytest tests/ops/ --capture-op-examples --capture-output captured.json
    pytest tests/ops/mapper/test_expand_macro_mapper.py --capture-op-examples

    # Resume from a previous (interrupted) capture session:
    pytest tests/ops/ --capture-op-examples --capture-resume captured_examples.jsonl
"""

import sys
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Register custom CLI options."""
    parser.addoption(
        "--capture-op-examples",
        action="store_true",
        default=False,
        help="Enable runtime capture of operator input/output examples.",
    )
    parser.addoption(
        "--capture-output",
        type=str,
        default=None,
        help="Output path for captured examples JSONL. "
             "Defaults to docs/op_doc_enhance_workflow/examples.jsonl",
    )
    parser.addoption(
        "--capture-resume",
        type=str,
        default=None,
        help="Path to an existing JSONL/JSON file to resume from. "
             "Defaults to docs/op_doc_enhance_workflow/examples.jsonl "
             "if it exists. Already-captured operators will be skipped. "
             "New examples are appended to this file incrementally.",
    )
    parser.addoption(
        "--capture-max-per-op",
        type=int,
        default=10,
        help="Maximum number of examples to capture per operator.",
    )
    parser.addoption(
        "--capture-max-str-len",
        type=int,
        default=500,
        help="Maximum string length before truncation in captured data.",
    )
    parser.addoption(
        "--capture-include-multimodal-mappers",
        action="store_true",
        default=False,
        help="Include multimodal mapper tests (image_*, video_*, audio_*) "
             "in capture. By default, these are skipped because their media "
             "files only exist locally and would be broken links in "
             "CI-generated documentation.",
    )


# Global reference so fixtures can access it
_capture_instance = None


def pytest_configure(config):
    """Install hooks at the very beginning of the test session."""
    global _capture_instance

    if not config.getoption("--capture-op-examples", default=False):
        return

    # Add project root to sys.path so imports work
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from .capture_examples import ExampleCapture

    max_per_op = config.getoption("--capture-max-per-op", default=10)
    max_str_len = config.getoption("--capture-max-str-len", default=500)
    include_multimodal = config.getoption(
        "--capture-include-multimodal-mappers", default=False
    )
    resume_from = config.getoption("--capture-resume", default=None)

    # Determine streaming output path
    output_path = config.getoption("--capture-output", default=None)
    if output_path is None:
        output_path = str(
            Path(__file__).resolve().parent / "examples.jsonl"
        )

    # Auto-resume from the output file if it exists and no explicit
    # resume path was given
    if resume_from is None:
        default_resume = Path(output_path)
        if default_resume.exists() and default_resume.stat().st_size > 0:
            resume_from = str(default_resume)

    _capture_instance = ExampleCapture(
        max_examples_per_op=max_per_op,
        truncate_strings=False,
        max_str_len=max_str_len,
        streaming_output=output_path,
        resume_from=resume_from,
        skip_multimodal_mappers=not include_multimodal,
    )
    _capture_instance.start()

    resume_msg = f", resuming from {resume_from}" if resume_from else ""
    multimodal_msg = ", multimodal mappers included" if include_multimodal else ""
    print(f"\n🎣 Operator example capture enabled "
          f"(max {max_per_op} per op"
          f"{resume_msg}{multimodal_msg})")


def _is_multimodal_mapper_test(relative_path: str) -> bool:
    """Check if a test file is for a multimodal mapper.

    Multimodal mapper tests are files under ``tests/ops/mapper/`` whose
    names start with ``test_image_``, ``test_video_``, or ``test_audio_``.
    """
    parts = Path(relative_path).parts
    if "mapper" not in parts:
        return False
    filename = Path(relative_path).name
    return any(
        filename.startswith(prefix)
        for prefix in ("test_image_", "test_video_", "test_audio_")
    )

def pytest_collection_modifyitems(config, items):
    """Deselect tests that should be skipped during capture.

    Skips:
    1. Tests whose examples have already been captured (resume mode).
    2. Multimodal mapper tests (unless ``--capture-include-multimodal-mappers``
       is set), because their media files only exist locally.
    """
    global _capture_instance

    if _capture_instance is None:
        return

    covered_keys = _capture_instance.get_covered_test_keys()
    skip_multimodal = _capture_instance.skip_multimodal_mappers

    project_root = Path(__file__).resolve().parents[2]

    remaining = []
    deselected = []

    for item in items:
        # Convert the item's absolute file path to a project-relative path
        item_path = Path(str(item.fspath))
        try:
            relative_path = str(item_path.relative_to(project_root))
        except ValueError:
            remaining.append(item)
            continue

        # Skip multimodal mapper tests when configured
        if skip_multimodal and _is_multimodal_mapper_test(relative_path):
            deselected.append(item)
            continue

        # Extract the test method name.
        # For unittest.TestCase methods collected by pytest, the
        # originalname attribute holds the bare method name (e.g.
        # "test_case_default").  For plain pytest functions it is
        # the function name.
        test_method = getattr(item, "originalname", None) or item.name

        if (relative_path, test_method) in covered_keys:
            deselected.append(item)
        else:
            remaining.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = remaining
        print(f"\n⏭️  Skipped {len(deselected)} tests (already captured), "
              f"running {len(remaining)} tests")


def pytest_unconfigure(config):
    """Save captured examples when the test session ends."""
    global _capture_instance

    if _capture_instance is None:
        return

    _capture_instance.stop()

    # Save a consolidated JSONL file (all examples including resumed ones)
    output_path = config.getoption("--capture-output", default=None)
    if output_path is None:
        output_path = str(
            Path(__file__).resolve().parent / "examples.jsonl"
        )

    _capture_instance.save(output_path)
    print(f"\n{_capture_instance.summary()}")

    _capture_instance = None


@pytest.fixture
def op_capture():
    """Fixture that provides access to the capture instance within tests."""
    return _capture_instance
