# scripts/rewrite_op_docstrings.py
# -*- coding: utf-8 -*-
import ast
from pathlib import Path

import shutil
import textwrap
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from utils.model import chat

ROOT = Path(__file__).resolve().parents[2]
OPS_DIR = ROOT / "data_juicer" / "ops"


def find_op_file(op_name):
    """Locate the corresponding operator file for a given operator name."""
    pattern = f"{op_name}.py"
    matches = list(OPS_DIR.rglob(pattern))
    return matches[0] if matches else None


# ---------------------------
# AST helpers
# ---------------------------


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _compute_offsets_by_line(source: str):
    lines = source.splitlines(keepends=True)
    offsets = [0]
    total = 0
    for line in lines:
        total += len(line)
        offsets.append(total)
    return lines, offsets


def _offset_of(line_no: int, col: int, line_offsets) -> int:
    return line_offsets[line_no - 1] + col


def _get_op_name_from_assigns(module: ast.Module) -> Optional[str]:
    for node in module.body:
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "OP_NAME":
                val = node.value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    return val.value
                if hasattr(ast, "Str") and isinstance(val, ast.Str):
                    return val.s
    return None


def _resolve_decorator_op_name(dec: ast.AST, op_name_from_const: Optional[str]) -> Optional[str]:
    if not isinstance(dec, ast.Call):
        return None
    func = dec.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr != "register_module":
        return None
    if not dec.args:
        return None
    arg0 = dec.args[0]
    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
        return arg0.value
    if hasattr(ast, "Str") and isinstance(arg0, ast.Str):
        return arg0.s
    if isinstance(arg0, ast.Name) and arg0.id == "OP_NAME":
        return op_name_from_const
    return None


def _get_docstring_expr_if_any(cls_node: ast.ClassDef) -> Optional[ast.Expr]:
    if not cls_node.body:
        return None
    first_stmt = cls_node.body[0]
    if isinstance(first_stmt, ast.Expr):
        val = first_stmt.value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return first_stmt
        if hasattr(ast, "Str") and isinstance(val, ast.Str):
            return first_stmt
    return None


def _escape_triple_double_quotes(s: str) -> str:
    s_list = s.split("\n")
    for i, line in enumerate(s_list):
        if i == 0:
            continue
        s_list[i] = "    " + line
    s = "\n".join(s_list)
    return s.replace('"""', '\\"""')


def _build_docstring_literal(content: str, indent: str) -> str:
    escaped = _escape_triple_double_quotes(content)
    return f'{indent}"""{escaped}"""\n'


def _slice_source_for_node(source: str, node: ast.AST) -> str:
    # Best effort to extract node source
    if hasattr(ast, "get_source_segment"):
        seg = ast.get_source_segment(source, node)
        if seg is not None:
            return seg
    lines, _ = _compute_offsets_by_line(source)
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)  # inclusive end line
    return "".join(lines[start:end])


@dataclass
class OperatorClassInfo:
    file_path: str
    op_name: str
    class_name: str
    cls_node: ast.ClassDef
    old_docstring: Optional[str]
    doc_expr: Optional[ast.Expr]
    class_source: str


def find_operator_classes_in_file(file_path: str) -> List[OperatorClassInfo]:
    src = _read_text(file_path)
    module = ast.parse(src)
    op_name_const = _get_op_name_from_assigns(module)

    results = []
    for node in module.body:
        if isinstance(node, ast.ClassDef):
            op_name_for_cls = None
            for dec in node.decorator_list:
                deco_op = _resolve_decorator_op_name(dec, op_name_const)
                if deco_op:
                    op_name_for_cls = deco_op
                    break
            if not op_name_for_cls:
                continue
            doc_expr = _get_docstring_expr_if_any(node)
            old_doc = ast.get_docstring(node, clean=False)
            class_src = _slice_source_for_node(src, node)
            results.append(
                OperatorClassInfo(
                    file_path=file_path,
                    op_name=op_name_for_cls,
                    class_name=node.name,
                    cls_node=node,
                    old_docstring=old_doc,
                    doc_expr=doc_expr,
                    class_source=class_src,
                )
            )
    return results


def replace_class_docstring_in_source(
    source: str,
    cls_node: ast.ClassDef,
    doc_expr: Optional[ast.Expr],
    new_doc: str,
) -> Tuple[str, str]:
    lines, line_offsets = _compute_offsets_by_line(source)
    if doc_expr is not None:
        start_line = doc_expr.lineno
        start_col = doc_expr.col_offset
        end_line = getattr(doc_expr, "end_lineno", None)
        end_col = getattr(doc_expr, "end_col_offset", None)
        if end_line is None or end_col is None:
            raise RuntimeError("Python AST lacks end_lineno/end_col_offset; please use Python 3.8+.")
        start_idx = _offset_of(start_line, start_col, line_offsets)
        end_idx = _offset_of(end_line, end_col, line_offsets)
        indent = " " * start_col
        replacement = _build_docstring_literal(new_doc, "")  # 缩进过多
        new_source = source[:start_idx] + replacement + source[end_idx:]
        action = "replaced"
    else:
        # insert as the first statement in class body
        if cls_node.body:
            first_stmt = cls_node.body[0]
            insert_line = first_stmt.lineno
            insert_col = first_stmt.col_offset
        else:
            insert_line = cls_node.lineno + 1
            insert_col = cls_node.col_offset + 4
        insert_offset = _offset_of(insert_line, 0, line_offsets)
        indent = " " * insert_col
        insertion = _build_docstring_literal(new_doc, indent)
        new_source = source[:insert_offset] + insertion + source[insert_offset:]
        action = "inserted"
    return new_source, action


# ---------------------------
# Prompting
# ---------------------------

DEFAULT_SYSTEM_PROMPT = (
    "You are a senior ML engineer. Write precise, implementation-faithful class docstrings for data processing "
    "operators so that both engineers and LLMs can understand usage without reading code. Use clear English.\n"
    "Style: Google-style docstring. Start with a one-sentence summary. Then details, behavior, and important notes.\n"
    "Constraints:\n"
    "- Describe what the operator does, how it decides to keep/filter samples, and how the key metric is computed with its default mode (e.g., character-based vs token-based).\n"
    "- If stats are cached, mention the field names as plain keys (e.g., 'alnum_ratio'), never internal constants (e.g., 'StatsKeys.alnum_ratio').\n"
    "- Do not list or restate constructor parameters, types, or defaults\n"
    "- Keep it concise (3–8 sentences). Bullet points allowed. \n"
    "- Hard wrap lines at 72 characters.\n"
    "- Preserve all substantive information from the original docstring; "
    "do not omit unique caveats or modes.\n"
    "- Preserve all links, references, citations, and documentation URLs from the original docstring.\n"
    "- If the operator name contains 'ray', explicitly mention it operates in Ray distributed mode.\n"
    "- Add a blank line after the first summary sentence.\n"
    "- If external models/tokenizers are used, mention them at a high level ('uses a Hugging Face tokenizer'), without specific model IDs unless strictly necessary.\n"
    "- Do not include triple quotes or code fences; return only the docstring body text."
)


def build_user_prompt_for_class(info: OperatorClassInfo) -> str:
    header = f"File: {info.file_path}\n" f"Operator name: {info.op_name}\n" f"Class: {info.class_name}\n"
    if info.old_docstring:
        old = info.old_docstring.strip()
        prev = f"Existing docstring (for reference):\n{old}\n"
    else:
        prev = "Existing docstring: <missing>\n"

    # Provide the exact class source for faithful generation
    code = info.class_source
    body = (
        "Here is the complete class source. Write a new, improved class docstring for this operator:\n"
        "----BEGIN CLASS SOURCE----\n"
        f"{code}\n"
        "----END CLASS SOURCE----\n"
    )
    return f"{header}\n{prev}\n{body}"


def call_model_to_generate_docstring(
    info: OperatorClassInfo,
    model_func: Callable,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    user_query = build_user_prompt_for_class(info)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    text = model_func(messages)
    # Be tolerant to different return shapes

    # sanitize: strip fences and triple quotes if model included them
    cleaned = text.strip()
    for fence in ("```", "'''", '"""'):
        if cleaned.startswith(fence) and cleaned.endswith(fence):
            cleaned = cleaned[len(fence) : -len(fence)].strip()
    cleaned = cleaned.replace('"""', '\\"""').strip()
    return cleaned


# ---------------------------
# Workflow
# ---------------------------


def _wrap_docstring(text: str, width: int = 88) -> str:
    paras = text.strip().split("\n\n")
    out = []
    for p in paras:
        lines = [line.rstrip() for line in p.splitlines() if line.strip() != "" or line == p]
        if any(line.lstrip().startswith(("-", "*")) for line in lines):
            buf = []
            for line in lines:
                s = line.strip()
                if s.startswith(("-", "*")):
                    bullet = s[0]
                    rest = s[1:].lstrip()
                    wrapped = textwrap.wrap(rest, width=width - 2)
                    if wrapped:
                        buf.append(f"{bullet} {wrapped[0]}")
                        for w in wrapped[1:]:
                            buf.append(f"  {w}")
                    else:
                        buf.append(f"{bullet}")
                else:
                    buf.extend(textwrap.wrap(s, width=width))
            out.append("\n".join(buf))
        else:
            joined = " ".join([line.strip() for line in lines]).strip()
            out.append("\n".join(textwrap.wrap(joined, width=width)) or joined)
    return "\n\n".join(out).strip()


def rewrite_file_ops(
    file_path: str,
    model_func: Callable,
    dry_run: bool = True,
    backup: bool = True,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[dict]:
    """
    Process one file: find operator classes, generate new docstrings, preview/replace.
    Returns a list of result dicts.
    """
    source = _read_text(file_path)

    infos = find_operator_classes_in_file(file_path)
    results = []
    if not infos:
        return results

    for info in infos:
        new_doc = call_model_to_generate_docstring(info, model_func=model_func, system_prompt=system_prompt)
        if not new_doc or new_doc.strip() == "":
            results.append(
                {
                    "file_path": file_path,
                    "class_name": info.class_name,
                    "op_name": info.op_name,
                    "action": "skipped",
                    "reason": "empty_model_output",
                }
            )
            continue

        # If unchanged, skip write
        if (info.old_docstring or "").strip() == new_doc.strip():
            results.append(
                {
                    "file_path": file_path,
                    "class_name": info.class_name,
                    "op_name": info.op_name,
                    "action": "unchanged",
                }
            )
            continue

        try:
            new_source, action = replace_class_docstring_in_source(
                source=source,
                cls_node=info.cls_node,
                doc_expr=info.doc_expr,
                new_doc=_wrap_docstring(new_doc),
            )
        except Exception as e:
            results.append(
                {
                    "file_path": file_path,
                    "class_name": info.class_name,
                    "op_name": info.op_name,
                    "action": "error",
                    "error": f"{e.__class__.__name__}: {e}",
                }
            )
            continue

        if dry_run:
            results.append(
                {
                    "file_path": file_path,
                    "class_name": info.class_name,
                    "op_name": info.op_name,
                    "action": f"dry_run_{action}",
                    "old_docstring": info.old_docstring,
                    "new_docstring": new_doc,
                }
            )
        else:
            if backup:
                shutil.copyfile(file_path, file_path + ".bak")
            _write_text(file_path, new_source)
            # update in-memory source for subsequent classes in same file
            source = new_source
            # also refresh AST positions if multiple classes exist; simplest is to re-parse
            results.append(
                {
                    "file_path": file_path,
                    "class_name": info.class_name,
                    "op_name": info.op_name,
                    "action": action,
                    "old_docstring": info.old_docstring,
                    "new_docstring": new_doc,
                }
            )

    return results


# ---------------------------
# main
# ---------------------------


def update_op_docstrings_with_files(modified_operator_files: List[str], system_prompt: str = DEFAULT_SYSTEM_PROMPT):

    if not modified_operator_files:
        print("No locally modified operator file found")
        return

    print(f"Found {len(modified_operator_files)} modified operator files:")
    for file_path in modified_operator_files:
        print(f"  - {file_path}")

    results = []
    # Process each file
    for file_path in modified_operator_files:
        print(f"\nProcessing file: {file_path}")
        try:
            result = rewrite_file_ops(file_path, chat, dry_run=False, backup=False, system_prompt=system_prompt)
            results.append(result)
            print(f"✅ Successfully updated: {file_path}")
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")

    return results


def update_op_docstrings_with_names(op_names: List[str], system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    files = []
    for op_name in op_names:
        file_path = find_op_file(op_name)
        if not file_path:
            print(f"Operator {op_name} not found")
            continue
        files.append(file_path)
    return update_op_docstrings_with_files(files, system_prompt=system_prompt)


if __name__ == "__main__":
    pass
