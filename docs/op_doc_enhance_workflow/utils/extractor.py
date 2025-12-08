#!/usr/bin/env python3
import ast
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.parse_class import (
    literal_eval_universal,  # 使用更通用的静态求值
    extract_class_attr_paths,
)


class Keys:
    """Symbol names used during extraction."""
    SAMPLES = "samples"
    DS_LIST = "ds_list"
    TGT_LIST = "tgt_list"
    OP = "op"
    TEXT = "text"
    TARGET = "target"
    TEST_PREFIX = "test"
    SELF = "self"
    SETUP = "setUp"


@dataclass
class MethodInfo:
    """Normalized info for a test method."""
    op_code: Optional[str] = None
    ds: Optional[str] = None
    tgt: Optional[str] = None
    samples: Optional[str] = None

    def merge(self, other: "MethodInfo") -> "MethodInfo":
        """Non-None fields from 'other' override None fields here."""
        return MethodInfo(
            op_code=other.op_code or self.op_code,
            ds=other.ds or self.ds,
            tgt=other.tgt or self.tgt,
            samples=other.samples or self.samples,
        )

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


class Env:
    """Static environment for name and self.attr substitution across methods."""
    def __init__(self) -> None:
        self.locals: Dict[str, ast.AST] = {}       # name -> AST expression
        self.self_attrs: Dict[str, ast.AST] = {}   # self.attr -> AST expression

    def clone(self) -> "Env":
        cp = Env()
        cp.locals.update(self.locals)
        cp.self_attrs.update(self.self_attrs)
        return cp


class _Substituter(ast.NodeTransformer):
    """Replace Name and self.attr nodes with env-provided AST expressions."""
    def __init__(self, env: Env) -> None:
        self.env = env

    def visit_Name(self, node: ast.Name) -> ast.AST:
        repl = self.env.locals.get(node.id)
        return ast.copy_location(repl, node) if repl is not None else node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == Keys.SELF:
            repl = self.env.self_attrs.get(node.attr)
            if repl is not None:
                return ast.copy_location(repl, node)
        return node


def _subst(node: ast.AST, env: Env) -> ast.AST:
    """Apply env-based substitution to an AST node."""
    return _Substituter(env).visit(ast.fix_missing_locations(node))


class TestInfoExtractor(ast.NodeVisitor):
    """
    Extract op_code / ds / tgt / samples for each test* method.
    """

    def __init__(
        self,
        source_code: str,
        file_path: Optional[str] = None,
        class_attr_seed: Optional[Dict[str, Any]] = None,
        assume_setup: bool = True,
        max_helper_depth: int = 1,
    ) -> None:
        try:
            self.tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Invalid source code syntax: {e}") from e

        # class_name -> { method_name -> FunctionDef }
        self._classes: Dict[str, Dict[str, ast.FunctionDef]] = {}
        # Only test* methods in results (只存方法名，不含类名)
        self.results: Dict[str, Dict[str, Optional[str]]] = {}

        self.file_path = Path(file_path) if file_path else None
        self.assume_setup = assume_setup
        self.max_helper_depth = max_helper_depth

        # Seed from class-level attributes (literals and/or source strings)
        if class_attr_seed is not None:
            self.class_attr_seed: Dict[str, Any] = dict(class_attr_seed)
        elif self.file_path:
            try:
                self.class_attr_seed = dict(
                    extract_class_attr_paths(
                        self.file_path,
                        include_containers=True,
                        include_source_fallback=True,
                    )
                )
            except Exception:
                self.class_attr_seed = {}
        else:
            self.class_attr_seed = {}

    def extract(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Entry point."""
        self.visit(self.tree)
        for _, methods in self._classes.items():
            self._process_class(methods)
        return self.results

    # ---------------- Collect methods per class ----------------

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        methods: Dict[str, ast.FunctionDef] = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods[item.name] = item
        self._classes.setdefault(node.name, {}).update(methods)

    # ---------------- Per-class processing ----------------

    def _process_class(self, methods: Dict[str, ast.FunctionDef]) -> None:
        setup_fn = methods.get(Keys.SETUP)

        # Class-level defaults (lowest priority) from seed
        class_defaults = self._class_defaults_from_seed(self.class_attr_seed)

        for name, fn in methods.items():
            if not name.startswith(Keys.TEST_PREFIX):
                continue

            env = Env()
            # 预热 self.xxx：用类属性中的字符串（例如路径字符串等），跳过 'op'
            for k, v in self.class_attr_seed.items():
                if k == Keys.OP:
                    continue
                if isinstance(v, str):
                    env.self_attrs[k] = ast.Constant(value=v)

            # 1) 隐式 setUp（若开启）
            setup_info = MethodInfo()
            if self.assume_setup and setup_fn:
                setup_info = self._run_setup(setup_fn, methods, env)

            # 2) 提取 test 函数（允许 helper 深度展开）
            test_info = self._extract_with_env(fn, methods, env, depth=0)

            # 合并：class_defaults < setUp < test
            info = class_defaults.merge(setup_info).merge(test_info)

            # 3) 字面量化：尽量转为纯字面值字符串，否则保留源码字符串
            info.op_code = self._finalize_literal(info.op_code, env)
            info.ds = self._finalize_literal(info.ds, env)
            info.tgt = self._finalize_literal(info.tgt, env)
            info.samples = self._finalize_literal(info.samples, env)

            self.results[name] = info.to_dict()

    def _run_setup(self, setup_fn: ast.FunctionDef, methods: Dict[str, ast.FunctionDef], env: Env) -> MethodInfo:
        """Simulate setUp() with default params (ignore 'self')."""
        params = [a.arg for a in setup_fn.args.args]
        if params and params[0] == Keys.SELF:
            params = params[1:]

        # 位置参数默认值
        defaults = list(setup_fn.args.defaults or [])
        if defaults:
            tail = params[-len(defaults):]
            for p, d in zip(tail, defaults):
                env.locals[p] = _subst(d, env)

        # 关键字专有参数默认值
        if setup_fn.args.kwonlyargs and setup_fn.args.kw_defaults:
            for a, d in zip(setup_fn.args.kwonlyargs, setup_fn.args.kw_defaults):
                if d is not None:
                    env.locals[a.arg] = _subst(d, env)

        # 在同一 env 执行 setUp，这样 self.* 能保留
        return self._extract_with_env(setup_fn, methods, env, depth=0)

    # ---------------- Build class-level defaults from seed ----------------

    def _class_defaults_from_seed(self, seed: Dict[str, Any]) -> MethodInfo:
        """Turn class-level ds_list/tgt_list/samples/op from seed into defaults."""
        info = MethodInfo()

        # op: 源码字符串或字面量字符串
        if Keys.OP in seed and isinstance(seed[Keys.OP], str):
            info.op_code = seed[Keys.OP]

        # samples：若为 Python 列表，尝试拆分；否则 repr 保存
        if Keys.SAMPLES in seed:
            val = seed[Keys.SAMPLES]
            if isinstance(val, list):
                ds, tgt, keep = self._split_samples_py(val)
                if ds is not None:
                    info.ds = info.ds or repr(ds)
                if tgt is not None:
                    info.tgt = info.tgt or repr(tgt)
                if ds is None and tgt is None and keep is not None:
                    info.samples = info.samples or repr(keep)
            else:
                info.samples = info.samples or repr(val)

        # ds_list / tgt_list：直接 repr（若已是字符串就保留字符串）
        if Keys.DS_LIST in seed:
            val = seed[Keys.DS_LIST]
            info.ds = info.ds or (val if isinstance(val, str) else repr(val))
        if Keys.TGT_LIST in seed:
            val = seed[Keys.TGT_LIST]
            info.tgt = info.tgt or (val if isinstance(val, str) else repr(val))

        return info

    # ---------------- Core extraction with env/propagation ----------------

    def _extract_with_env(
        self,
        node: ast.FunctionDef,
        methods: Dict[str, ast.FunctionDef],
        env: Env,
        depth: int,
    ) -> MethodInfo:
        """Sequentially walk statements, update env, and merge extracted info."""
        info = MethodInfo()

        for stmt in node.body:
            # Assignments (including simple Assign and AnnAssign)
            if isinstance(stmt, ast.Assign):
                value = _subst(stmt.value, env)
                # 处理所有 targets
                for target in stmt.targets:
                    self._handle_assignment_target(
                        target, value, info, env
                    )
                # RHS 自身为 self.helper(...) 调用
                if isinstance(stmt.value, ast.Call):
                    child_info = self._maybe_recurse_into_self_call(stmt.value, methods, env, depth)
                    if child_info:
                        info = info.merge(child_info)

            elif isinstance(stmt, ast.AnnAssign):
                # 注解赋值
                value = _subst(stmt.value, env) if stmt.value is not None else None
                self._handle_assignment_target(stmt.target, value, info, env)
                if stmt.value is not None and isinstance(stmt.value, ast.Call):
                    child_info = self._maybe_recurse_into_self_call(stmt.value, methods, env, depth)
                    if child_info:
                        info = info.merge(child_info)

            # Top-level self.helper(...) call (Expr)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                child_info = self._maybe_recurse_into_self_call(stmt.value, methods, env, depth)
                if child_info:
                    info = info.merge(child_info)

        return info

    def _handle_assignment_target(
        self,
        target: ast.AST,
        value: Optional[ast.AST],
        info: MethodInfo,
        env: Env,
    ) -> None:
        """Process a single assignment target with its RHS value."""
        if value is None:
            return

        # name = expr
        if isinstance(target, ast.Name):
            env.locals[target.id] = value

            if target.id == Keys.SAMPLES:
                ds_code, tgt_code, samples_code = self._split_samples(value, env)
                if ds_code:
                    info.ds = info.ds or ds_code
                if tgt_code:
                    info.tgt = info.tgt or tgt_code
                if ds_code is None and tgt_code is None and samples_code:
                    info.samples = info.samples or samples_code

            elif target.id == Keys.DS_LIST:
                code = self._code_string(value, env)
                if code:
                    info.ds = info.ds or code

            elif target.id == Keys.TGT_LIST:
                code = self._code_string(value, env)
                if code:
                    info.tgt = info.tgt or code

            elif target.id == Keys.OP:
                code = self._code_string(value, env)
                if code:
                    info.op_code = info.op_code or code

        # self.attr = expr
        elif (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == Keys.SELF
        ):
            env.self_attrs[target.attr] = value
            if target.attr == Keys.OP:
                code = self._code_string(value, env)
                if code:
                    info.op_code = info.op_code or code

        # (a, b, ...) = expr 解包（只处理 RHS 同长度 tuple/list）
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
            if len(target.elts) == len(value.elts):
                for t_el, v_el in zip(target.elts, value.elts):
                    self._handle_assignment_target(t_el, v_el, info, env)

    def _maybe_recurse_into_self_call(
        self,
        call: ast.Call,
        methods: Dict[str, ast.FunctionDef],
        env: Env,
        depth: int,
    ) -> Optional[MethodInfo]:
        """If call is self.helper(...), bind args, recurse (depth-limited), propagate self.*, return collected info."""
        if depth >= self.max_helper_depth:
            return None

        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == Keys.SELF
        ):
            helper_name = call.func.attr
            if helper_name.startswith(Keys.TEST_PREFIX):
                return None

            callee = methods.get(helper_name)
            if callee:
                child_env = env.clone()
                bindings = self._bind_call_args(callee, call, env)
                child_env.locals.update(bindings)

                child_info = self._extract_with_env(callee, methods, child_env, depth + 1)
                # Propagate self.* updates back to caller env
                env.self_attrs.update(child_env.self_attrs)
                return child_info
        return None

    # ---------------- Argument binding for helpers ----------------

    def _bind_call_args(self, func: ast.FunctionDef, call: ast.Call, parent_env: Env) -> Dict[str, ast.AST]:
        """Bind positional/keyword args at call site to callee parameter names (ignore 'self')."""
        # 普通位置参数
        params = [a.arg for a in func.args.args]
        if params and params[0] == Keys.SELF:
            params = params[1:]

        bound: Dict[str, ast.AST] = {}

        # 位置参数默认值 -> 尾部参数
        defs = list(func.args.defaults or [])
        if defs:
            tail = params[-len(defs):]
            for p, d in zip(tail, defs):
                bound[p] = _subst(d, parent_env)

        # 位置实参绑定
        for i, arg in enumerate(call.args[:len(params)]):
            bound[params[i]] = _subst(arg, parent_env)

        # 关键字专有参数默认值
        kwonly = [a.arg for a in (func.args.kwonlyargs or [])]
        kwdefs = list(func.args.kw_defaults or [])
        for a, d in zip(kwonly, kwdefs):
            if d is not None:
                bound[a] = _subst(d, parent_env)

        # 关键字实参绑定
        for kw in call.keywords or []:
            if kw.arg in params or kw.arg in kwonly:
                bound[kw.arg] = _subst(kw.value, parent_env)

        # vararg/kwarg 暂不展开（现网场景一般简单）
        return bound

    # ---------------- samples splitting (unified via static eval) ----------------

    def _split_samples(self, node: ast.AST, env: Env) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        code = self._code_string(node, env)
        if code is None:
            return None, None, None

        attr_map = self._self_attr_string_map(env)
        val = None
        try:
            val = literal_eval_universal(code, extra_env=None, attr_map=attr_map)
        except Exception:
            val = None

        if isinstance(val, list) and val:
            ok = all(isinstance(el, dict) and set(el.keys()) == {Keys.TEXT, Keys.TARGET} for el in val)
            if ok:
                ds = [{Keys.TEXT: el[Keys.TEXT]} for el in val]
                tgt = [{Keys.TEXT: el[Keys.TARGET]} for el in val]
                return repr(ds), repr(tgt), None

        return None, None, code

    def _split_samples_py(
        self, value: List[Any]
    ) -> Tuple[Optional[List[Any]], Optional[List[Any]], Optional[List[Any]]]:
        """Python-level split for class seed."""
        if not value:
            return value, None, None
        ok = all(isinstance(el, dict) and set(el.keys()) == {Keys.TEXT, Keys.TARGET} for el in value)
        if not ok:
            return None, None, value
        ds = [{Keys.TEXT: el[Keys.TEXT]} for el in value]
        tgt = [{Keys.TEXT: el[Keys.TARGET]} for el in value]
        return ds, tgt, None

    # ---------------- Utilities ----------------

    def _self_attr_string_map(self, env: Env) -> Dict[str, Any]:
        """
        构建 self.xxx -> 替换值 的字典，用于 literal_eval_universal 的 ASTTransformer。
        出于稳妥，只注入字符串常量；如需更激进，可将更多可静态评估的值注入。
        """
        attr_map: Dict[str, Any] = {}
        for k, v in env.self_attrs.items():
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                attr_map[k] = v.value
        return attr_map

    def _finalize_literal(self, code: Optional[str], env: Env) -> Optional[str]:
        """
        尝试用 literal_eval_universal 求值；成功则 repr(值)，失败则返回原 code。
        """
        if not code:
            return code
        attr_map = self._self_attr_string_map(env)
        try:
            val = literal_eval_universal(code, extra_env=None, attr_map=attr_map)
            return repr(val) if val is not None else code
        except Exception:
            return code

    def _code_string(self, node: Optional[ast.AST], env: Env) -> Optional[str]:
        """对 AST 节点先做 env 替换再 unparse 成源码字符串。"""
        if node is None:
            return None
        try:
            return ast.unparse(_subst(node, env))
        except Exception:
            return None


# ---------------- Convenience APIs ----------------

def extract_test_info_from_source(
    source_code: str,
    *,
    file_path: Optional[str] = None,
    class_attr_seed: Optional[Dict[str, Any]] = None,
    assume_setup: bool = True,
    max_helper_depth: int = 1,
) -> Dict[str, Dict[str, Optional[str]]]:
    extractor = TestInfoExtractor(
        source_code,
        file_path=file_path,
        class_attr_seed=class_attr_seed,
        assume_setup=assume_setup,
        max_helper_depth=max_helper_depth,
    )
    return extractor.extract()


def extract_test_info_from_path(
    py_path: str,
    *,
    class_attr_seed: Optional[Dict[str, Any]] = None,
    assume_setup: bool = True,
    max_helper_depth: int = 1,
) -> Dict[str, Dict[str, Optional[str]]]:
    p = Path(py_path)
    src = p.read_text(encoding="utf-8")
    extractor = TestInfoExtractor(
        src,
        file_path=str(p),
        class_attr_seed=class_attr_seed,
        assume_setup=assume_setup,
        max_helper_depth=max_helper_depth,
    )
    return extractor.extract()