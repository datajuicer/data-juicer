#!/usr/bin/env python3
import ast
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, List

from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.constant import Fields, BatchMetaKeys, MetaKeys

# Base environment for evaluating well-known constants (read-only)
env: Dict[str, Any] = {
    "SpecialTokens": SpecialTokens,
    "Fields": Fields,
    "BatchMetaKeys": BatchMetaKeys,
    "MetaKeys": MetaKeys,
}


def _eval_expr(node: ast.AST, scope: Dict[str, Any]) -> Optional[Any]:
    """Resolve simple constants, names, and attribute lookups from scope."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return scope.get(node.id)
    if isinstance(node, ast.Attribute):
        obj = _eval_expr(node.value, scope)
        if obj is not None and hasattr(obj, node.attr):
            return getattr(obj, node.attr)
    return None


def _eval_fstring(node: ast.AST, scope: Dict[str, Any]) -> Optional[str]:
    """Evaluate f-strings if every interpolated value can be resolved."""
    if not isinstance(node, ast.JoinedStr):
        return None
    parts: List[str] = []
    for v in node.values:
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            parts.append(v.value)
        elif isinstance(v, ast.FormattedValue):
            val = _eval_expr(v.value, scope)
            if val is None:
                return None
            parts.append(str(val))
        else:
            return None
    return "".join(parts)


def _eval_path_expr(node: ast.AST, scope: Dict[str, Any]) -> Optional[str]:
    """
    Evaluate path expressions: string literals, __file__, names in scope,
    and os.path.* calls with statically resolvable string arguments.
    """
    # Plain string
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    # Name (e.g., __file__ or a previously resolved path var)
    if isinstance(node, ast.Name):
        v = scope.get(node.id)
        return v if isinstance(v, str) else None

    # os.path.* calls
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        func_attr = node.func
        if isinstance(func_attr.value, ast.Attribute) and isinstance(func_attr.value.value, ast.Name):
            if func_attr.value.value.id == "os" and func_attr.value.attr == "path":
                fname = func_attr.attr
                # Each arg must be a resolvable path fragment or f-string
                args: List[str] = []
                for a in node.args:
                    v = _eval_path_expr(a, scope)
                    if v is None:
                        v = _eval_fstring(a, scope)
                    if v is None:
                        return None
                    args.append(v)

                import os as _os
                if fname == "join":
                    return _os.path.normpath(_os.path.join(*args))
                if fname == "dirname":
                    return _os.path.dirname(args[0])
                if fname == "realpath":
                    return _os.path.realpath(args[0])
                if fname == "abspath":
                    return _os.path.abspath(args[0])
                if fname == "normpath":
                    return _os.path.normpath(args[0])
    return None


def _is_json_like(obj: Any) -> bool:
    """Whitelist literal types that are safe for serialization."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return True
    if isinstance(obj, (list, tuple)):
        return all(_is_json_like(x) for x in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _is_json_like(v) for k, v in obj.items())
    return False


def _eval_static(node: ast.AST, scope: Dict[str, Any]) -> Optional[Any]:
    """
    Best-effort static evaluation for class-level attributes.
    Supports:
      - constants (str/int/float/bool/None)
      - containers (list/tuple/set/dict with string keys)
      - f-strings
      - os.path.* calls
      - Name/Attribute from provided scope
    Returns Python values when fully resolved, else None.
    """
    # Path-like expressions first
    path_val = _eval_path_expr(node, scope)
    if path_val is not None:
        return path_val

    # f-strings
    s_val = _eval_fstring(node, scope)
    if s_val is not None:
        return s_val

    # Basic constants
    if isinstance(node, ast.Constant):
        return node.value

    # Containers
    if isinstance(node, ast.List):
        out = [_eval_static(e, scope) for e in node.elts]
        return out if all(v is not None for v in out) else None

    if isinstance(node, ast.Tuple):
        out = [_eval_static(e, scope) for e in node.elts]
        return out if all(v is not None for v in out) else None

    if isinstance(node, ast.Set):
        out = [_eval_static(e, scope) for e in node.elts]
        return list(out) if all(v is not None for v in out) else None

    if isinstance(node, ast.Dict):
        keys: List[str] = []
        vals: List[Any] = []
        for k, v in zip(node.keys, node.values):
            k_eval = _eval_static(k, scope)
            v_eval = _eval_static(v, scope)
            if not isinstance(k_eval, str) or v_eval is None:
                return None
            keys.append(k_eval)
            vals.append(v_eval)
        return dict(zip(keys, vals))

    # Simple addition for string concatenation
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        l = _eval_static(node.left, scope)
        r = _eval_static(node.right, scope)
        if isinstance(l, str) and isinstance(r, str):
            return l + r

    # Name/Attribute resolution
    if isinstance(node, ast.Name):
        return scope.get(node.id)

    if isinstance(node, ast.Attribute):
        base = _eval_static(node.value, scope)
        if base is not None and hasattr(base, node.attr):
            return getattr(base, node.attr)

    return None


def extract_class_attr_values(
    pyfile: Path,
    *,
    extra_env: Optional[Dict[str, Any]] = None,
    include_source_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate class-level attributes to literals where possible.
    - Returns a dict: {attr_name -> Python literal or (when fallback) source string}
    - When include_source_fallback=True, expressions that cannot be statically
      evaluated (e.g., TestMapper()) are returned as their source strings.
    The function is pure-static and does not execute user code.
    """
    src = pyfile.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(pyfile))

    scope: Dict[str, Any] = {"__file__": str(pyfile)}
    import os as _os
    scope["os"] = _os
    # Enrich with project-wide constants (Fields, etc.)
    scope.update(env)
    if extra_env:
        scope.update(extra_env)

    results: Dict[str, Any] = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    name = stmt.targets[0].id
                    value_node = stmt.value

                    # Try static evaluation
                    val = _eval_static(value_node, scope)
                    if val is not None and _is_json_like(val):
                        # Update scope so later attributes can refer to earlier ones
                        scope[name] = val
                        results[name] = val
                        continue

                    # Fallback to source string (non-literals like TestMapper())
                    if include_source_fallback:
                        try:
                            code = ast.unparse(value_node)
                        except Exception:
                            code = None
                        if code:
                            results[name] = code

    # Cleanup helper symbols
    results.pop("__file__", None)
    results.pop("os", None)
    return results


def extract_class_attr_paths(
    pyfile: Path,
    *,
    include_containers: bool = False,
    include_source_fallback: bool = False,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Backward-compatible API used by the extractor.
    - Default (include_containers=False): return only string-valued attributes
      (original behavior: typical for path-like constants).
    - include_containers=True: return all class attributes that can be safely
      evaluated to Python literals (list/dict/etc.), plus optional source fallback.
    - include_source_fallback=True: for expressions that are not statically
      evaluable, return their source string (e.g., 'TestMapper()').
    """
    values = extract_class_attr_values(
        pyfile,
        extra_env=extra_env,
        include_source_fallback=include_source_fallback,
    )
    if include_containers:
        return values
    return {k: v for k, v in values.items() if isinstance(v, str)}


class ASTTransformer(ast.NodeTransformer):
    """Replace self.xxx in strings and f-strings using provided attr_map."""

    def __init__(self, attr_map: Dict[str, str], scope: Dict[str, Any]):
        self.attr_map = attr_map   # Mapping for self.xxx → string
        self.scope = scope         # Scope for f-string variables

    def visit_Attribute(self, node: ast.Attribute):
        # Replace self.xxx with string constants when available
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr in self.attr_map:
                return ast.copy_location(ast.Constant(value=self.attr_map[node.attr]), node)
        return self.generic_visit(node)

    def visit_FormattedValue(self, node: ast.FormattedValue):
        # Replace f-string formatted values with evaluated constants
        node = self.generic_visit(node)
        expr_value = _eval_expr(node.value, self.scope)
        if expr_value is None:
            return node
        return ast.copy_location(ast.Constant(value=str(expr_value)), node)

    def visit_JoinedStr(self, node: ast.JoinedStr):
        # If every part of the f-string is constant, join into a plain string
        node = self.generic_visit(node)
        parts: List[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                return node
        return ast.copy_location(ast.Constant(value="".join(parts)), node)


def literal_eval_with_self(raw_code: str, attr_map: Dict[str, str]) -> Any:
    """
    Replace self.xxx with string constants in expression, then literal_eval.
    Only suitable for literal expressions (list/dict/strings/numbers).
    Returns Python value on success, or None if expression isn't a pure literal.
    """
    return literal_eval_universal(raw_code, extra_env=None, attr_map=attr_map)

def literal_eval_universal(expr_str, extra_env=None, attr_map=None):
    try:
        scope = env.copy()
        if extra_env:
            scope.update(extra_env)
        node = ast.parse(expr_str, mode='eval')
        
        if attr_map:
            transformer = ASTTransformer(attr_map, scope)
            node = transformer.visit(node)
            ast.fix_missing_locations(node)
        try:
            return ast.literal_eval(node.body)
        except (ValueError, SyntaxError, TypeError):
            pass
        return _eval_static(node.body, scope)
    except Exception:
        traceback.print_exc()
        return None