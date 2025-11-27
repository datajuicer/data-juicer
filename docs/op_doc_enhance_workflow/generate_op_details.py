#!/usr/bin/env python3
"""
Script to auto-generate operator documentation.
Extracts information from operator source code and test files to generate Markdown documentation.
"""

import json
import os
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from utils.model import chat
from utils.parse_class import extract_class_attr_paths
from utils.extractor import extract_test_info_from_path
from utils.router import route
from utils.view_model import to_legacy_view

from data_juicer.tools.op_search import OPSearcher
from rewrite_op_docstrings import update_op_docstrings_with_names

# -----------------------------------------------------------------------------
# Paths, constants, and template environment
# -----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = ROOT / "tests" / "ops"
OPS_DOCS_DIR = ROOT / "docs" / "operators"
OPS_DOCS_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATE_DIR = Path(__file__).parent / "templates"

MD_FLAGS = re.MULTILINE | re.DOTALL
PROMPT_BRIEF_DELIM = "\n\n-----\n\n"
NO_EXPLAIN_OPS = [
    "llm_task_relevance_filter",
    "in_context_influence_filter",
    "text_embd_similarity_filter",
    "audio_add_gaussian_noise_mapper",
    "image_blur_mapper",
    "image_captioning_from_gpt4v_mapper",
]

env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)

examples_cache = {}

examples_cache_path = Path(__file__).parent / "examples.json"
if examples_cache_path.exists():
    with open(examples_cache_path, "r") as f:
        examples_cache = json.load(f)

# -----------------------------------------------------------------------------
# Global prompts for LLM calls (centralized for easy adjustment)
# -----------------------------------------------------------------------------

PROMPTS = {
    "select_system": (
        "You are a senior data processing engineer.\n"
        "A test file for a specific data operator and candidate example methods are provided to you.\n"
        "Please select the most representative examples that best illustrate the behavior of the operator,\n"
        "covering typical usage and one important edge case as much as possible.\n"
        "For each selected example, write a short bilingual explanation describing the behavior of the operator:\n"
        "- Use simple, everyday language. If technical terms are used, provide layman's explanations so non-technical users can understand.\n"
        "- First describe in English, then on the next line give the corresponding Chinese translation.\n"
        "- Use op_desc and the test file to explain what the operator does to the input data (filtering/transforming logic).\n"
        "- Explain why the result (output data) comes from the input (input data): clearly state why certain items are kept, removed, or modified.\n"
        "- If the test file processes the output of the operator, making the provided output data not the original output of the operator, please clarify that the output data is not the original output, and describe how the original output was processed into the displayed result to avoid user misunderstanding.\n"
        "Only return a concise JSON containing the keys: 'selected' (a list of up to 2 method names)\n"
        "and 'explanations' (a mapping from method names to strings).\n"
        "Use a single newline character '\\n' to separate English and Chinese text,\n"
        "In the Chinese explanation, “Operator” = 算子.\n"
        "Do not include any additional text, code blocks, or comments."
    ),
    "select_user_template": (
        "The full content of the test file is as follows:\n\n"
        "{test_file_full}\n\n"
        "Candidate examples, including method names, code, and datasets:\n\n"
        "{briefs}\n\n"
        "Please respond with a JSON in the following format only:\n"
        "{json_example}"
        "Read the output data of the test file and candidate examples carefully. "
        "If you find that the original output of the operator has been additionally processed in the test file "
        "(such as calculating the size, transformation, etc.), "
        "Please explain what transformations have been made so that users do not mistake output data for "
        "the original output of the operator. "
        "For example: 'For clarity, we show the (width, height) of each video in the raw output; "
        "the actual raw output from the operator is four cropped videos.'"
        "if output data is that original output of the operator, no additional specification is require"
    ),
    "select_json_example": (
        '{"selected": ["test_xxx", "test_yyy"], "explanations": {"test_xxx": "English explanation.\\n中文解释。", '
        '"test_yyy": "English explanation.\\n中文解释。"}}'
    ),
    "translate_system": (
        "You are a professional bilingual technical translator (English -> Simplified Chinese) "
        "for machine learning documentation. Translate accurately and fluently for Chinese ML engineers.\n"
        "Requirements:\n"
        "- Keep Markdown formatting, lists, headings, indentation, and line breaks.\n"
        "- Do NOT translate any code, inline code in backticks, URLs, environment variables, CLI flags, API names, or model names.\n"
        "- Terminology constraints:\n"
        "  * 'operator' -> '算子' (case-insensitive when used as the ML operator term)\n"
        "  * 'Hugging Face' -> keep in English\n"
        "  * 'token' -> keep in English\n"
        "- Preserve technical accuracy; keep proper nouns and product names unchanged unless specified above.\n"
        "- Do not add any explanations or extra text.\n"
        "- Output only the translations, in the same order as input, and separated by the exact separator token provided.\n"
        "- Do not wrap output in quotes or code fences; do not add leading or trailing separators."
    ),
    "translate_user_template": (
        "You will receive multiple English text blocks that must be translated into Simplified Chinese.\n"
        "Blocks are separated by the exact separator token shown below. Translate each block independently.\n"
        "Return only the translated blocks, in the same order, separated by the exact same separator.\n"
        "Separator token:\n"
        "{separator}\n\n"
        "English blocks:\n"
        "{batch}\n"
    ),
}


def _build_example_brief(method, vals):
    """
    Build a compact text brief for a test example to feed into LLM prompts.
    vals keys: op_code, ds, tgt, samples, test_code.
    """
    parts = [f"method: {method}", "Here is what is shown to users (If this method is selected):"]
    if vals.op_code:
        parts.append(f"Code that executes this operator: {vals.op_code}")
    if vals.input:
        parts.append(f"input data: {vals.input}")
    if vals.output:
        parts.append(f"output data: {vals.output}")
    brief = "\n".join(parts)
    if len(brief) > 2000:
        return ""
    return brief


# -----------------------------------------------------------------------------
# Markdown parsing and rendering
# -----------------------------------------------------------------------------


def parse_existing_op_md(md_text: str) -> dict:
    """
    Parse Markdown previously rendered by op_doc.md.j2 and return structured data.

    Returns:
      {
        "name": str,
        "desc": str,
        "type": str,
        "tags": [str],
        "params": [{"name":..,"type":..,"default":..,"desc":..}, ...],
        "examples": [
          {"method": str, "op_code": str, "input": str, "output": str, "explanation": str},
          ...
        ],
        "code_links": {"source": str, "test": str}
      }

    Missing sections yield empty values.
    """
    res = {
        "name": "",
        "desc": "",
        "type": "",
        "tags": [],
        "params": [],
        "examples": dict(),
        "code_links": {"source": "", "test": ""},
    }
    text = md_text

    # Title and description
    m = re.search(r"^\#\s+(?P<name>.+?)\s*$", text, flags=MD_FLAGS)
    if m:
        res["name"] = m.group("name").strip()
        title_end = m.end()
    else:
        title_end = 0

    m_type = re.search(r"^Type\s*[^:]*:\s*\*\*(?P<type>.+?)\*\*\s*$", text, flags=MD_FLAGS)
    type_start = m_type.start() if m_type else len(text)
    if title_end < type_start:
        desc = text[title_end:type_start].strip()
        res["desc"] = desc.strip()
    if m_type:
        res["type"] = m_type.group("type").strip()

    # Tags line
    m_tag = re.search(r"^Tags\s*标签:\s*(?P<tags>.+?)\s*$", text, flags=MD_FLAGS)
    if m_tag:
        tags_line = m_tag.group("tags").strip()
        if tags_line:
            res["tags"] = [t.strip() for t in tags_line.split(",") if t.strip()]

    # Parameter table
    m_param_sec = re.search(r"^\#\#\s*🔧\s*Parameter Configuration.*?$", text, flags=MD_FLAGS)
    if m_param_sec:
        rows = []
        m_next = re.search(r"^\#\#\s*📊\s*Effect demonstration.*?$", text, flags=MD_FLAGS)
        sec_text = text[m_param_sec.end() : m_next.start() if m_next else len(text)]
        for line in sec_text.splitlines():
            line = line.rstrip()
            mrow = re.match(
                r"^\|\s*`(?P<name>[^`]+)`\s*\|\s*(?P<type>[^|]+?)\s*\|\s*`(?P<default>[^`]*)`\s*\|\s*(?P<desc>.*?)\s*\|$",
                line,
            )
            if mrow:
                rows.append(
                    {
                        "name": mrow.group("name").strip(),
                        "type": mrow.group("type").strip(),
                        "default": mrow.group("default").strip(),
                        "desc": mrow.group("desc").strip(),
                    }
                )
        res["params"] = rows

    # Examples section
    m_effect = re.search(r"^\#\#\s*📊\s*Effect demonstration.*?$", text, flags=MD_FLAGS)
    m_links = re.search(r"^\#\#\s*🔗\s*related links.*?$", text, flags=MD_FLAGS)
    if m_effect:
        sec = text[m_effect.end() : m_links.start() if m_links else len(text)]
        if "not available" not in sec and "暂无" not in sec:
            blocks = []
            for mth in re.finditer(r"^\#\#\#\s+(?P<method>.+?)\s*$", sec, flags=MD_FLAGS):
                blocks.append((mth.group("method").strip(), mth.start(), mth.end()))
            for i, (method, s, e) in enumerate(blocks):
                end = blocks[i + 1][1] if i + 1 < len(blocks) else len(sec)
                b = sec[e:end]

                # Optional operator code block
                op_code = ""
                mcode = re.search(r"```python\s*(?P<code>.*?)```", b, flags=MD_FLAGS)
                if mcode:
                    op_code = mcode.group("code").strip()

                # Input/output/explanation subsections
                minput = re.search(r"^####\s*📥\s*input data.*?$", b, flags=MD_FLAGS)
                moutput = re.search(r"^####\s*📤\s*output data.*?$", b, flags=MD_FLAGS)
                mexpl = re.search(r"^####\s*✨\s*explanation.*?$", b, flags=MD_FLAGS)

                input_text = output_text = explanation = ""
                if minput and moutput:
                    input_text = b[minput.end() : moutput.start()].strip()
                if moutput and mexpl:
                    output_text = b[moutput.end() : mexpl.start()].strip()
                elif moutput:
                    output_text = b[moutput.end() :].strip()

                if mexpl:
                    explanation = b[mexpl.end() :].strip()
                    explanation = explanation.strip() if "TODO" not in explanation else ""

                res["examples"][method] = {
                    "method": method,
                    "op_code": op_code,
                    "input": input_text,
                    "output": output_text,
                    "explanation": explanation,
                }

    # Related links
    if m_links:
        link_sec = text[m_links.end() :]
        m_src = re.search(r"\[source code.*?\]\((?P<src>[^)]+)\)", link_sec)
        m_tst = re.search(r"\[unit test.*?\]\((?P<test>[^)]+)\)", link_sec)
        res["code_links"]["source"] = (m_src.group("src") if m_src else "").strip()
        res["code_links"]["test"] = (m_tst.group("test") if m_tst else "").strip()

    return res


def load_existing_op_md(op_name: str, op_type: str):
    """Load and parse existing operator markdown if present."""
    # if op_type == "mapper":
    #     return None
    md_path = OPS_DOCS_DIR / op_type / f"{op_name}.md"
    if md_path.exists():
        try:
            return parse_existing_op_md(md_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[MD parse] {md_path} parse failed: {e}")
    return None


def render_op_doc(op_info, examples_data, template="op_doc.md.j2"):
    """Render operator documentation via Jinja2 template."""
    template = env.get_template(template)
    return template.render(**op_info, examples=examples_data)


# -----------------------------------------------------------------------------
# Parameter parsing utilities
# -----------------------------------------------------------------------------


def param_signature_to_list(sig, param_docs):
    """
    Convert inspect.Signature to a list of parameter info:
    [{name, type, default, desc}]
    """
    params_info = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        typ = str(param.annotation) if param.annotation != param.empty else ""
        default = param.default if param.default != param.empty else ""
        if isinstance(default, str):
            default = f"'{default}'"
        params_info.append(
            {
                "name": name,
                "type": typ,
                "default": default,
                "desc": param_docs.get(name, ""),
            }
        )
    return params_info


# -----------------------------------------------------------------------------
# Test processing
# -----------------------------------------------------------------------------


def process_test_file(pyfile: Path):
    """Parse a test file and extract test case information."""
    return extract_test_info_from_path(pyfile)


def find_test_file(op_name):
    """Locate the corresponding test file for a given operator name."""
    pattern = f"test_{op_name.lower()}.py"
    matches = list(TESTS_DIR.rglob(pattern))
    return matches[0] if matches else None


# -----------------------------------------------------------------------------
# File I/O
# -----------------------------------------------------------------------------


def save_md_file(op_name, content, op_type=".", lan=""):
    """Save generated markdown content to the docs/operators tree."""
    outpath = OPS_DOCS_DIR / op_type / f"{op_name}{lan}.md"
    outdir = outpath.parent
    outdir.mkdir(parents=True, exist_ok=True)
    outpath.write_text(content, encoding="utf-8")
    print(f"[Generated] {outpath}")


# -----------------------------------------------------------------------------
# Translation (EN -> ZH)
# -----------------------------------------------------------------------------


def optimize_text(text):
    """
    Optimize text formatting:
    - Merge ordinary newlines into one line
    - Keep list items that start with '- '
    - Keep blank lines (paragraph separation)
    """
    text = "\n".join([line.strip() for line in text.split("\n")])
    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        current_line = lines[i].strip()

        if not current_line:
            result.append("")
            i += 1
            continue

        merged_line = current_line
        i += 1

        while i < len(lines):
            next_line = lines[i].strip()

            if not next_line or next_line.startswith("- "):
                break

            merged_line += " " + next_line
            i += 1

        result.append(merged_line)

    return "\n".join(result)


def split_bilingual_text(text):

    def contains_chinese(text):
        """Detect whether text contains Chinese characters"""
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        return bool(chinese_pattern.search(text))

    def is_chinese_start_line(line):
        """Determine whether it is the beginning line of the Chinese version"""
        line = line.strip()
        if not line:
            return False

        first_five = line[:15]
        return contains_chinese(first_five)

    lines = text.split("\n")

    chinese_start_index = -1

    for i in range(len(lines)):
        current_line = lines[i].strip()

        if not current_line and i + 1 < len(lines):
            next_line = lines[i + 1]
            if is_chinese_start_line(next_line):
                chinese_start_index = i + 1
                break

    if chinese_start_index == -1:
        return text.strip(), ""

    english_lines = lines[:chinese_start_index]
    while english_lines and not english_lines[-1].strip():
        english_lines.pop()

    chinese_lines = lines[chinese_start_index:]

    english_text = "\n".join(english_lines).strip()
    chinese_text = "\n".join(chinese_lines).strip()

    return english_text, chinese_text


def get_op_desc_in_en_zh_batched(descs):
    """
    Translate a list of English descriptions to Chinese in batches using LLM
    and return merged bilingual descriptions. Batch split is based on length.
    Terminology rules:
      - operator -> 算子
      - Hugging Face -> keep English
      - token -> keep English
    """
    if not descs:
        return []
    separator = "\n\n******\n\n"
    limit = int(5e3)

    def _translate_batch_with_llm(batch_text, sep):
        system = PROMPTS.get("translate_system") or (
            "You are a professional bilingual technical translator (English -> Simplified Chinese)."
        )
        user = PROMPTS.get("translate_user_template", "Translate with separator {separator}:\n{batch}").format(
            separator=sep,
            batch=batch_text,
        )

        retry = 0
        last_err = None
        while retry < 3:
            try:
                content = chat(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ]
                )
                if isinstance(content, str):
                    return content
                else:
                    last_err = RuntimeError(f"Unexpected response type: {type(content)}, content={content}")
            except Exception as e:
                last_err = e
                print(f"[LLM translate] error: {e} retry={retry} len_batch={len(batch_text)}")
            retry += 1
        print(f"[LLM translate] failed after retries. last_err={last_err}")
        return None

    batch = separator.join(descs)
    if len(batch) > limit:
        split_idx = int(len(descs) / 2)
        res1 = get_op_desc_in_en_zh_batched(descs[:split_idx])
        res2 = get_op_desc_in_en_zh_batched(descs[split_idx:])
        return res1 + res2
    else:
        res = _translate_batch_with_llm(batch, separator)

    if not res:
        zhs = [""] * len(descs)
    else:
        print(f"[LLM Translate] ori_len: {len(batch)}")
        zhs = res.split(separator)
        assert len(zhs) == len(descs)

    assert len(zhs) == len(descs)
    return [desc + "\n\n" + zh.strip() for desc, zh in zip(descs, zhs)]


# -----------------------------------------------------------------------------
# Example selection and processing
# -----------------------------------------------------------------------------


def process_example(examples, attr_map, op_info, test_file_full, existing_examples=None):
    """
    Build examples list for rendering:
    - Filter out unusable cases.
    - Reuse preferred methods and explanations if provided, otherwise call LLM.
    - Route example IR to legacy view data for template.
    """
    if op_info["name"] in NO_EXPLAIN_OPS:
        return []
    examples_list = []
    usable = {}
    md_dir_abs = OPS_DOCS_DIR / op_info["type"]
    for m, vals in examples.items():
        if (vals["ds"] and vals["tgt"]) or vals["samples"]:
            example_ir = route(vals, attr_map, md_dir_abs, m)
            if example_ir is None:
                continue
            usable[m] = example_ir
    if not usable:
        return examples_list

    if existing_examples:
        select_methods = [m for m in existing_examples.keys() if m in usable]
        explanations = {m: existing_examples[m]["explanation"] for m in select_methods}
    else:
        # Selection + explanation via LLM based on full test file and pre-screened method names
        select_methods, explanations = select_and_explain_examples(
            usable,
            op_info=op_info,
            test_file_full=test_file_full,
        )

    for m in select_methods:
        vals = usable[m]
        view = to_legacy_view(vals)
        examples_list.append(
            {
                "method": m,
                "op_code": vals.op_code or "",
                "explanation": (explanations.get(m, "") or "").strip(),
                **view,
            }
        )
    return examples_list


def select_and_explain_examples(examples: dict, op_info: dict, test_file_full: str = ""):
    """
    Drive LLM to select and explain examples.

    Behavior:
    - Always provide the FULL test file content and the list of pre-screened method names.
    - If len(pre-screened) < 2, the prompt instructs the model to select all of them.
    - The model must return JSON with 'selected' and 'explanations'.

    Returns: (selected_methods: List[str], explanations: Dict[method, str])
    """
    if not examples:
        return [], {}

    briefs = []
    op_desc = ""
    if op_info.get("desc"):
        op_desc = op_info.get("desc")
        briefs.append(f"op_desc: {op_desc}")
    for m, v in examples.items():
        briefs.append(_build_example_brief(m, v))

    briefs_string = PROMPT_BRIEF_DELIM.join(briefs)
    if len(briefs_string) > 5000:
        briefs_string = briefs_string[:5000] + "..." + "subsequent omission"

    methods_all = list(examples.keys())

    system = PROMPTS["select_system"]
    user = PROMPTS["select_user_template"].format(
        test_file_full=test_file_full,
        briefs=briefs_string,
        json_example=PROMPTS["select_json_example"],
    )

    try:
        resp = chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        )
        data = resp if isinstance(resp, dict) else json.loads(str(resp))
        # Only keep methods that are in the pre-screened list; cap at k
        selected = [m for m in (data.get("selected") or []) if m in examples]
        if len(methods_all) < 2:
            # If fewer than 2 pre-screened, ensure we use all of them
            selected = methods_all
        else:
            selected = selected[:2] if selected else methods_all[:2]
        expl = data.get("explanations") or {}
        explanations = {m: str(expl.get(m, "")).strip() for m in selected}
        print(f"[LLM select+explain] selected: {selected}, explanations: {list(explanations.values())}")
        return selected, explanations
    except Exception as e:
        print(f"[LLM select+explain] parse error: {e} op: {op_info['name']}")
        return [], {}


def camel_to_snake(camel_str):
    """
    Convert camel naming to underscore naming
    For example: CsvFormatter -> csv_formatter
    """
    snake_str = re.sub("([a-z0-9])([A-Z])", r"\1_\2", camel_str)
    return snake_str.lower()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def rewrite_op_doc(op_name):
    results = update_op_docstrings_with_names([op_name])
    assert len(results[0]) == 1
    for result_info in results[0]:
        if result_info.get("new_docstring"):
            return optimize_text(result_info["new_docstring"])
    return None


def should_use_cache(new_method_info, cached_method_info):
    """Determine whether cached method information should be used"""
    if not cached_method_info:
        return False

    # Compare only op_code and ds
    new_op_code = new_method_info.get("op_code")
    new_ds = new_method_info.get("ds")

    cached_op_code = cached_method_info.get("op_code")
    cached_ds = cached_method_info.get("ds")

    return (new_op_code == cached_op_code and new_ds == cached_ds) or (new_ds is None and cached_ds is not None)


def main():
    """Generate documentation for all operators found by OPSearcher."""
    searcher = OPSearcher(include_formatter=True)
    all_ops = searcher.records_map
    op_detail_list = []
    original_descs = []

    def handle_one(op_info, existing_md=None):
        """Process a single operator into template-ready info and examples."""
        # Params
        params = param_signature_to_list(op_info["sig"], op_info["param_desc_map"])

        # Tests and examples
        examples_list = []
        test_file = find_test_file(op_info["name"])
        if test_file:
            test_file_full = test_file.read_text(encoding="utf-8")[:5000]
            new_examples = process_test_file(test_file)
            new_examples = {k: v for k, v in new_examples.items() if not any(x in k for x in ["parallel", "np"])}
            attr_map = extract_class_attr_paths(test_file)
            if examples_cache and examples_cache.get(op_info["name"]):
                final_examples = {}
                cached_op_examples = examples_cache.get(op_info["name"], {})

                for method_name, new_method_info in new_examples.items():
                    cached_method_info = cached_op_examples.get(method_name)

                    if should_use_cache(new_method_info, cached_method_info):
                        final_examples[method_name] = cached_method_info.copy()
                        if new_method_info.get("tgt") is not None:
                            final_examples[method_name]["tgt"] = new_method_info["tgt"]
                    else:
                        final_examples[method_name] = new_method_info

                examples = final_examples
            else:
                examples = new_examples
            # update examples cache
            examples_cache[op_info["name"]] = examples
            if existing_md and existing_md.get("examples"):
                existing_examples = existing_md["examples"]
            else:
                existing_examples = None
            examples_list = process_example(examples, attr_map, op_info, test_file_full, existing_examples)

        # Template data
        op_info_dir = (OPS_DOCS_DIR / op_info["type"]).relative_to(ROOT)
        source_path = Path(f"data_juicer/ops/{op_info['type']}/{op_info['name'].lower()}.py")
        if op_info["name"] == "general_fused_op":
            source_path = Path(f"data_juicer/ops/op_fusion.py")
        op_info_tmpl = {
            "name": op_info["name"],
            "type": op_info["type"],
            "tags": op_info["tags"],
            "params": params,
            "code_links": {
                "source": Path(os.path.relpath(source_path, op_info_dir)),
                "test": (Path(os.path.relpath(test_file, op_info_dir)) if test_file else ""),
            },
        }

        return op_info["name"], op_info_tmpl, examples_list

    # Iterate all operators
    for op_name, op_info in all_ops.items():
        op_info = op_info.to_dict()
        if "Formatter" in op_name:
            op_info["name"] = camel_to_snake(op_name)

        # Reuse existing markdown (if available)
        existing_md = load_existing_op_md(op_info["name"], op_info["type"])
        res = handle_one(op_info, existing_md)
        _, op_info_tmpl, examples_list = res
        cleaned_desc = optimize_text(op_info["desc"])
        if existing_md and existing_md.get("desc"):
            en_desc, zh_desc = split_bilingual_text(existing_md["desc"])
            if cleaned_desc.strip() != en_desc.strip() or not zh_desc:
                print(f"[Update] {op_info['name']}'s desc is updated")
                original_descs.append(cleaned_desc)
            else:
                op_info_tmpl["desc"] = en_desc + "\n\n" + zh_desc
        else:
            new_desc = rewrite_op_doc(op_name)
            if new_desc:
                cleaned_desc = new_desc
            original_descs.append(cleaned_desc)
        op_detail_list.append((op_info["name"], op_info_tmpl, examples_list))

    # save examples cache
    with open(examples_cache_path, "w") as f:
        json.dump(examples_cache, f, indent=4, ensure_ascii=False)

    bilingual_descs = get_op_desc_in_en_zh_batched(original_descs)
    iter_bilingual_descs = iter(bilingual_descs)
    for op_name, op_info_tmpl, examples_list in op_detail_list:
        if not op_info_tmpl.get("desc"):
            op_info_tmpl["desc"] = next(iter_bilingual_descs)
        md_content = render_op_doc(op_info_tmpl, examples_list)
        save_md_file(op_name, md_content, op_info_tmpl["type"])


if __name__ == "__main__":
    main()
