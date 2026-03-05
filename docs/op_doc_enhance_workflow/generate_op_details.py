#!/usr/bin/env python3
"""
Script to auto-generate operator documentation.
"""

import os
import re
import fire
from pathlib import Path
from typing import Dict, List

from jinja2 import Environment, FileSystemLoader
from utils.md_parser import load_existing_op_md
from utils.llm_service import get_bilingual_descs

from data_juicer.tools.op_search import OPSearcher
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE

from utils.example_loader import load_examples_by_op, build_examples_for_op

# -----------------------------------------------------------------------------
# Constants & Paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OPS_DOCS_DIR = ROOT / "docs" / "operators"
OPS_DOCS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR = Path(__file__).parent / "templates"
DEFAULT_EXAMPLES_PATH = Path(__file__).resolve().parent / "examples.jsonl"

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def camel_to_snake(camel_str):
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", camel_str).lower()


def optimize_text(text):
    if not text:
        return ""
    text = "\n".join([line.strip() for line in text.split("\n")])
    lines = text.split("\n")
    result, i = [], 0
    while i < len(lines):
        curr = lines[i].strip()
        if not curr:
            result.append("")
            i += 1
            continue
        merged = curr
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith("- "):
                break
            merged += " " + nxt
            i += 1
        result.append(merged)
    return "\n".join(result)


def split_bilingual_text(text):
    def contains_chinese(s):
        return bool(re.search(r"[\u4e00-\u9fff]", s))

    lines = text.split("\n")
    idx = -1
    for i in range(len(lines)):
        curr = lines[i].strip()
        if not curr and i + 1 < len(lines):
            if contains_chinese(lines[i + 1][:15]):
                idx = i + 1
                break
    if idx == -1:
        return text.strip(), ""
    en = "\n".join(lines[:idx]).strip()
    zh = "\n".join(lines[idx:]).strip()
    return en, zh


def param_signature_to_list(sig, param_docs):
    params_info = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        typ = str(param.annotation) if param.annotation != param.empty else ""
        def_val = param.default if param.default != param.empty else ""
        if isinstance(def_val, str):
            def_val = f"'{def_val}'"
        if def_val == f"'{DATA_JUICER_ASSETS_CACHE}'":
            def_val = "DATA_JUICER_ASSETS_CACHE"
        params_info.append({"name": name, "type": typ, "default": def_val, "desc": param_docs.get(name, "")})
    return params_info




# -----------------------------------------------------------------------------
# Core Processing
# -----------------------------------------------------------------------------


class DocGenerator:
    def __init__(self):
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.captured_by_op: Dict[str, List[Dict]] = {}

    def rewrite_op_doc(self, op_name):
        """Placeholder for actual docstring rewrite logic."""
        from rewrite_op_docstrings import update_op_docstrings_with_names

        results = update_op_docstrings_with_names([op_name])
        if results:
            assert len(results[0]) == 1
        else:
            return None
        for result_info in results[0]:
            if result_info.get("new_docstring"):
                return optimize_text(result_info["new_docstring"])
        return None

    def _de_absolutize(self, data):
        """Turn absolute path strings in data into placeholders"""
        root_str = str(ROOT)
        if isinstance(data, dict):
            return {k: self._de_absolutize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._de_absolutize(i) for i in data]
        elif isinstance(data, str):
            # Replace the absolute path contained in the string
            return data.replace(root_str, "{PROJECT_ROOT}")
        return data

    def _absolutize(self, data):
        """Absolute path to return placeholder to current environment"""
        root_str = str(ROOT)
        if isinstance(data, dict):
            return {k: self._absolutize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._absolutize(i) for i in data]
        elif isinstance(data, str):
            return data.replace("{PROJECT_ROOT}", root_str)
        return data

    def handle_one(self, op_info, existing_md):
        params = param_signature_to_list(op_info["sig"], op_info["param_desc_map"])

        # Build examples via the decoupled example processing layer
        examples_list = build_examples_for_op(
            op_name=op_info["name"],
            op_type=op_info["type"],
            captured_by_op=self.captured_by_op,
            existing_examples=(
                existing_md.get("examples") if existing_md else None
            ),
            md_dir=OPS_DOCS_DIR,
        )

        if not (op_info["test_path"] and Path(op_info["test_path"]).exists()):
            op_info["test_path"] = None

        # Template Data
        op_dir = (OPS_DOCS_DIR / op_info["type"]).relative_to(ROOT)
        op_info_tmpl = {
            "name": op_info["name"],
            "type": op_info["type"],
            "tags": op_info["tags"],
            "params": params,
            "code_links": {
                "source": os.path.relpath(op_info["source_path"], op_dir),
                "test": os.path.relpath(op_info["test_path"], op_dir) if op_info["test_path"] else "",
            },
        }
        return op_info_tmpl, examples_list

    def gen(self, rewrite_docstring=False, explain_examples=False,
            captured_examples_path=None):
        """
        Generate documentation for operators.

        :param rewrite_docstring: Whether to rewrite docstrings using LLM.
        :param explain_examples: Whether to generate explanations for
            examples using LLM.
        :param captured_examples_path: Path to captured examples JSONL/JSON
            file. Defaults to ``examples.jsonl`` in the workflow directory.
        """
        # Load captured examples via the decoupled example layer
        examples_path = captured_examples_path or str(DEFAULT_EXAMPLES_PATH)
        self.captured_by_op = load_examples_by_op(examples_path)
        if self.captured_by_op:
            total = sum(len(v) for v in self.captured_by_op.values())
            print(f"[Captured] Loaded {total} examples for "
                  f"{len(self.captured_by_op)} operators from {examples_path}")
        else:
            print(f"[Warning] No captured examples found at: {examples_path}")
            print("          Run tests with --capture-op-examples first.")

        searcher = OPSearcher(include_formatter=True)
        all_ops = searcher.all_ops
        op_detail_list, original_descs = [], []

        for op_name, op_info in all_ops.items():
            if "Formatter" in op_name:
                op_name = camel_to_snake(op_name)
                op_info["name"] = op_name

            md_path = OPS_DOCS_DIR / op_info["type"] / f"{op_name}.md"
            existing_md = load_existing_op_md(md_path)
            op_tmpl, ex_list = self.handle_one(op_info, existing_md)

            cleaned_desc = optimize_text(op_info["desc"])
            if existing_md and existing_md.get("desc"):
                en, zh = split_bilingual_text(existing_md["desc"])
                if cleaned_desc.strip() != en.strip() or not zh:
                    original_descs.append(cleaned_desc)
                else:
                    op_tmpl["desc"] = f"{en}\n\n{zh}"
            else:
                if rewrite_docstring:
                    new_desc = self.rewrite_op_doc(op_name)
                    if new_desc:
                        cleaned_desc = new_desc
                original_descs.append(cleaned_desc)

            op_detail_list.append((op_info["name"], op_tmpl, ex_list))

        # Bilingual Batch Processing
        bilingual_descs = get_bilingual_descs(original_descs)
        desc_iter = iter(bilingual_descs)

        # Collect all valid op names (snake_case) for stale-file cleanup
        valid_op_names = set()
        for op_name_key, op_info in all_ops.items():
            name = camel_to_snake(op_name_key) if "Formatter" in op_name_key else op_name_key
            valid_op_names.add((op_info["type"], name))

        template = self.env.get_template("op_doc.md.j2")
        for name, tmpl, ex_list in op_detail_list:
            if not tmpl.get("desc"):
                tmpl["desc"] = next(desc_iter)

            content = template.render(**tmpl, examples=ex_list)
            out_path = OPS_DOCS_DIR / tmpl["type"] / f"{name}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"[Generated] {out_path}")

        # Clean up stale md files for operators no longer in the codebase
        removed_count = 0
        for md_file in OPS_DOCS_DIR.rglob("*.md"):
            op_type_dir = md_file.parent.name
            op_name_stem = md_file.stem
            if (op_type_dir, op_name_stem) not in valid_op_names:
                md_file.unlink()
                removed_count += 1
                print(f"[Deleted] {md_file} (operator no longer exists)")
        if removed_count:
            print(f"[Cleanup] Removed {removed_count} stale doc(s)")

if __name__ == "__main__":
    fire.Fire(DocGenerator)