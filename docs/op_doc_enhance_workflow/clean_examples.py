#!/usr/bin/env python3
"""
One-time script to clean examples.jsonl:
1. Remove {PROJECT_ROOT} placeholders from paths (replace with relative paths)
2. Remove truncated text markers and restore full text where possible

Usage:
    python docs/op_doc_enhance_workflow/clean_examples.py
"""

import json
import re
import sys
from pathlib import Path

EXAMPLES_PATH = Path(__file__).resolve().parent / "examples.jsonl"

# Pattern matching truncation markers like:
#   "... [truncated, total 900 chars]"
TRUNCATION_PATTERN = re.compile(r"\.\.\. \[truncated, total \d+ chars\]$")


def clean_project_root(data):
    """Recursively replace {PROJECT_ROOT}/ with empty string in all strings."""
    if isinstance(data, str):
        return data.replace("{PROJECT_ROOT}/", "")
    elif isinstance(data, dict):
        return {k: clean_project_root(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_project_root(item) for item in data]
    return data


def has_truncation(data):
    """Check if any string in the data contains a truncation marker."""
    if isinstance(data, str):
        return bool(TRUNCATION_PATTERN.search(data))
    elif isinstance(data, dict):
        return any(has_truncation(v) for v in data.values())
    elif isinstance(data, list):
        return any(has_truncation(item) for item in data)
    return False


def main():
    if not EXAMPLES_PATH.exists():
        print(f"❌ File not found: {EXAMPLES_PATH}")
        sys.exit(1)

    with open(EXAMPLES_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_records = []
    removed_count = 0
    project_root_count = 0
    truncation_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        record = json.loads(line)

        # Check for truncation — these records have incomplete data
        # and need to be re-captured, so we remove them
        if has_truncation(record.get("input_data")) or has_truncation(record.get("output_data")):
            truncation_count += 1
            removed_count += 1
            continue

        # Clean {PROJECT_ROOT}/ placeholders
        original = json.dumps(record)
        record = clean_project_root(record)
        if json.dumps(record) != original:
            project_root_count += 1

        cleaned_records.append(record)

    # Write back
    with open(EXAMPLES_PATH, "w", encoding="utf-8") as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Cleaned {EXAMPLES_PATH.name}:")
    print(f"   - Removed {truncation_count} records with truncated text")
    print(f"   - Fixed {{PROJECT_ROOT}} in {project_root_count} records")
    print(f"   - Total: {len(cleaned_records)} records kept "
          f"(removed {removed_count})")


if __name__ == "__main__":
    main()
