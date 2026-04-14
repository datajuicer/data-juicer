import argparse
import inspect
import os
import sys
from typing import Annotated, Optional, get_type_hints

from pydantic import Field

from data_juicer.tools.mcp_tool import execute_op
from data_juicer.tools.op_search import OPSearcher
from data_juicer.utils.lazy_loader import LazyLoader

fastmcp = LazyLoader("mcp.server.fastmcp", "mcp[cli]")


def resolve_signature_annotations(func, sig: inspect.Signature) -> inspect.Signature:
    """Resolve postponed/string annotations into real runtime types.

    When a module uses ``from __future__ import annotations``, all
    annotations are stored as strings. This helper calls
    ``typing.get_type_hints`` on the original callable to obtain the
    real type objects and rebuilds the signature with them.
    """
    try:
        module = sys.modules.get(func.__module__, None) if hasattr(func, "__module__") else None
        globalns = module.__dict__ if module else {}
        hints = get_type_hints(func, globalns=globalns, localns=globalns)
    except Exception:
        hints = {}

    new_params = []
    for name, param in sig.parameters.items():
        resolved_annotation = hints.get(name, param.annotation)
        new_params.append(param.replace(annotation=resolved_annotation))

    return_annotation = hints.get("return", sig.return_annotation)
    return sig.replace(parameters=new_params, return_annotation=return_annotation)


# Dynamic MCP Tool Creation
def process_parameter(name: str, param: inspect.Parameter) -> inspect.Parameter:
    """
    Processes a function parameter:
    - Converts jsonargparse.typing.ClosedUnitInterval to a local equivalent annotation.
    """
    ClosedUnitInterval = Annotated[float, Field(ge=0.0, le=1.0, description="float restricted to be ≥0 and ≤1")]
    if param.annotation == getattr(sys.modules.get("jsonargparse.typing"), "ClosedUnitInterval", None):
        return param.replace(annotation=ClosedUnitInterval)
    return param


def create_operator_function(op, mcp):
    """Creates a callable function for a Data-Juicer operator class.

    This function dynamically creates a function that can be registered as an MCP tool,
    with proper signature and documentation based on the operator's __init__ method.
    """
    raw_sig = op["sig"]
    init_func = op.get("init_func")
    if init_func is not None:
        sig = resolve_signature_annotations(init_func, raw_sig)
    else:
        sig = raw_sig
    docstring = op["desc"]
    param_docstring = op["param_desc"]

    # Create new function signature with dataset_path as first parameter
    # Consider adding other common parameters later, such as export_psth
    fixed_params = [
        inspect.Parameter("dataset_path", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
        inspect.Parameter(
            "export_path",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Optional[str],
            default=None,
        ),
        inspect.Parameter(
            "np",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=Optional[int],
            default=None,
        ),
    ]
    op_params = [
        process_parameter(name, param)
        for name, param in sig.parameters.items()
        if name not in ("args", "kwargs", "self")
    ]
    # Merge all params, then reorder: required (no default) first,
    # optional (with default) second, to satisfy Python's signature rule.
    all_params = fixed_params + op_params
    required_params = [p for p in all_params if p.default is inspect.Parameter.empty]
    optional_params = [p for p in all_params if p.default is not inspect.Parameter.empty]
    new_parameters = required_params + optional_params
    new_signature = sig.replace(parameters=new_parameters, return_annotation=str)

    def func(*args, **kwargs):
        args_dict = {}
        bound_arguments = new_signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        export_path = bound_arguments.arguments.pop("export_path")
        dataset_path = bound_arguments.arguments.pop("dataset_path")
        np = bound_arguments.arguments.pop("np")
        args_dict = {k: v for k, v in bound_arguments.arguments.items() if v is not None}

        dj_cfg = {
            "dataset_path": dataset_path,
            "export_path": export_path,
            "process": [{op["name"]: args_dict}],
            "np": np,
        }
        return execute_op(dj_cfg)

    func.__signature__ = new_signature
    func.__doc__ = f"""{docstring}\n\n{param_docstring}\n"""
    func.__name__ = op["name"]

    decorated_func = mcp.tool()(func)

    return decorated_func


def create_mcp_server(port: str = "8000"):
    """
    Creates the FastMCP server and registers the tools.

    Args:
        port (str, optional): Port number. Defaults to "8000".
    """
    mcp = fastmcp.FastMCP("Data-Juicer Server", port=port)

    # Operator Management
    ops_list_path = os.getenv("DJ_OPS_LIST_PATH", None)
    if ops_list_path:
        with open(ops_list_path, "r", encoding="utf-8") as file:
            ops_list = [line.strip() for line in file if line.strip()]
    else:
        ops_list = None
    searcher = OPSearcher(ops_list)
    op_results = searcher.search()

    # Register all operators as MCP tools
    for op in op_results:
        _ = create_operator_function(op, mcp)

    return mcp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data-Juicer MCP Server")
    parser.add_argument(
        "--port", type=str, default="8000", help="Port number for the MCP server"
    )  # changed to str for consistency
    args = parser.parse_args()

    # Server configuration
    mcp = create_mcp_server(port=args.port)

    mcp.run(transport=os.getenv("SERVER_TRANSPORT", "sse"))
