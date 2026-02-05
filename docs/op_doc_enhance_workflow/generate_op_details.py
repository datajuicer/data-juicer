#!/usr/bin/env python3
"""
Script to auto-generate operator documentation.
"""

import json
import os
import re
import fire
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from utils.parse_class import extract_class_attr_paths
from utils.extractor import extract_test_info_from_path
from utils.router import route
from utils.view_model import to_legacy_view
from utils.md_parser import load_existing_op_md
from utils.llm_service import get_bilingual_descs, select_and_explain_examples

from data_juicer.tools.op_search import OPSearcher
from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE

# -----------------------------------------------------------------------------
# Constants & Paths
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
OPS_DOCS_DIR = ROOT / "docs" / "operators"
OPS_DOCS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR = Path(__file__).parent / "templates"
CACHE_PATH = Path(__file__).parent / "examples.json"

NO_EXPLAIN_OPS = [
    "llm_task_relevance_filter",
    "in_context_influence_filter",
    "text_embd_similarity_filter",
    "audio_add_gaussian_noise_mapper",
    "image_blur_mapper",
    "image_captioning_from_gpt4v_mapper",
]

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


def should_use_cache(new_info, cached_info):
    if not cached_info:
        return False
    return (new_info.get("op_code") == cached_info.get("op_code") and new_info.get("ds") == cached_info.get("ds")) or (
        new_info.get("ds") is None and cached_info.get("ds") is not None
    )


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
        self.cache = {}
        if CACHE_PATH.exists():
            raw_cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            self.cache = self._absolutize(raw_cache)

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

    def process_example_list(self, examples, attr_map, op_info, test_file_full, existing_examples, explain_examples):
        if op_info["name"] in NO_EXPLAIN_OPS:
            return []

        usable = {}
        md_dir_abs = OPS_DOCS_DIR / op_info["type"]
        for m, vals in examples.items():
            if (vals["ds"] and vals["tgt"]) or vals["samples"]:
                res = route(vals, attr_map, md_dir_abs, m)
                if res:
                    usable[m] = res

        if not usable:
            return []

        if existing_examples:
            select_methods = [m for m in existing_examples.keys() if m in usable]
            explanations = {m: existing_examples[m]["explanation"] for m in select_methods}
        elif not explain_examples:
            select_methods = list(usable.keys())[:2]
            explanations = {m: "" for m in select_methods}
        else:
            select_methods, explanations = select_and_explain_examples(usable, op_info, test_file_full)

        return [
            {
                "method": m,
                "op_code": usable[m].op_code or "",
                "explanation": (explanations.get(m, "") or "").strip(),
                **to_legacy_view(usable[m]),
            }
            for m in select_methods
        ]

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

    def handle_one(self, op_info, existing_md, explain_examples):
        params = param_signature_to_list(op_info["sig"], op_info["param_desc_map"])
        examples_list = []

        if op_info["test_path"] and Path(op_info["test_path"]).exists():
            test_path = ROOT / Path(op_info["test_path"])
            test_content = test_path.read_text(encoding="utf-8")[:5000]
            new_ex = extract_test_info_from_path(test_path)
            new_ex = {k: v for k, v in new_ex.items() if not any(x in k for x in ["parallel", "np"])}

            # Cache merge
            cached_op = self.cache.get(op_info["name"], {})
            final_ex = {}
            for m, info in new_ex.items():
                if should_use_cache(info, cached_op.get(m)):
                    final_ex[m] = cached_op[m].copy()
                    if info.get("tgt") is not None:
                        final_ex[m]["tgt"] = info["tgt"]
                else:
                    final_ex[m] = info

            self.cache[op_info["name"]] = self._de_absolutize(final_ex)
            examples_list = self.process_example_list(
                final_ex,
                extract_class_attr_paths(test_path),
                op_info,
                test_content,
                existing_md.get("examples") if existing_md else None,
                explain_examples,
            )
        else:
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

    def gen(self, rewrite_docstring=False, explain_examples=False):
        """
        Generate documentation for operators.

        :param rewrite_docstring: Whether to rewrite docstrings using LLM.
        :param explain_examples: Whether to generate explanations for examples using LLM.
        """
        searcher = OPSearcher(include_formatter=True)
        all_ops = searcher.all_ops
        op_detail_list, original_descs = [], []

        for op_name, op_info in all_ops.items():
            if "Formatter" in op_name:
                op_info["name"] = camel_to_snake(op_name)

            md_path = OPS_DOCS_DIR / op_info["type"] / f"{op_name}.md"
            existing_md = load_existing_op_md(md_path)
            op_tmpl, ex_list = self.handle_one(op_info, existing_md, explain_examples)

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

        # Save Cache
        self.cache = self._de_absolutize(self.cache)
        CACHE_PATH.write_text(json.dumps(self.cache, indent=4, ensure_ascii=False), encoding="utf-8")

        # Bilingual Batch Processing
        bilingual_descs = get_bilingual_descs(original_descs)
        desc_iter = iter(bilingual_descs)

        template = self.env.get_template("op_doc.md.j2")
        for name, tmpl, ex_list in op_detail_list:
            if not tmpl.get("desc"):
                tmpl["desc"] = next(desc_iter)

            content = template.render(**tmpl, examples=ex_list)
            out_path = OPS_DOCS_DIR / tmpl["type"] / f"{name}.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            print(f"[Generated] {out_path}")


if __name__ == "__main__":
    # fire.Fire(DocGenerator)
    DocGenerator().gen(rewrite_docstring=False, explain_examples=False)
