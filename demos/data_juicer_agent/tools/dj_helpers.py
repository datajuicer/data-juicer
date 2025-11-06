# -*- coding: utf-8 -*-
import os
import os.path as osp
import json
import asyncio
from typing import Any
from agentscope.message import TextBlock
from agentscope.tool import ToolResponse
from .op_manager.op_retrieval import retrieve_ops

# Load tool information for formatting
TOOLS_INFO_PATH = osp.join(
    osp.dirname(__file__),
    "op_manager",
    "dj_funcs_all.json",
)


def _load_tools_info():
    """Load tools information from JSON file or create it if not exists"""
    if osp.exists(TOOLS_INFO_PATH):
        with open(TOOLS_INFO_PATH, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    else:
        from .op_manager.create_dj_func_info import dj_func_info

        with open(TOOLS_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump(dj_func_info, f)
        return dj_func_info


def _format_tool_names_to_class_entries(tool_names):
    """Convert tool names list to formatted class entries string"""
    if not tool_names:
        return ""

    tools_info = _load_tools_info()

    # Create a mapping from class_name to tool info for quick lookup
    tools_map = {tool["class_name"]: tool for tool in tools_info}

    formatted_entries = []
    for i, tool_name in enumerate(tool_names):
        if tool_name in tools_map:
            tool_info = tools_map[tool_name]
            class_entry = (
                f"{i+1}. {tool_info['class_name']}: {tool_info['class_desc']}"
            )
            class_entry += "\n" + tool_info["arguments"]
            formatted_entries.append(class_entry)

    return "\n".join(formatted_entries)


async def query_dj_operators(query: str, limit: int = 20) -> ToolResponse:
    """Query DataJuicer operators by natural language description.

    Retrieves relevant operators from DataJuicer library based on user query.
    Supports matching by functionality, data type, and processing scenarios.

    Args:
        query (str): Natural language operator query
        limit (int): Maximum number of operators to return (default: 20)

    Returns:
        ToolResponse: Tool response containing matched operators with names, descriptions, and parameters
    """

    try:
        # Retrieve operator names using existing functionality with limit
        # Use retrieval mode from environment variable if set
        retrieval_mode = os.environ.get("RETRIEVAL_MODE", "auto")
        tool_names = await retrieve_ops(
            query,
            limit=limit,
            mode=retrieval_mode,
        )

        if not tool_names:
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=f"No matching DataJuicer operators found for query: {query}\n"
                        f"Suggestions:\n"
                        f"1. Use more specific keywords like 'text filter', 'image processing'\n"
                        f"2. Check spelling and try alternative terms\n"
                        f"3. Try English keywords for better matching",
                    ),
                ],
            )

        # Format tool names to class entries
        retrieved_operators = _format_tool_names_to_class_entries(tool_names)

        # Format response
        result_text = f"🔍 DataJuicer Operator Query Results\n"
        result_text += f"Query: {query}\n"
        result_text += f"Limit: {limit} operators\n"
        result_text += f"{'='*50}\n\n"
        result_text += retrieved_operators

        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=result_text,
                ),
            ],
        )

    except Exception as e:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"Error querying DataJuicer operators: {str(e)}\n"
                    f"Please verify query parameters and retry.",
                ),
            ],
        )


async def execute_safe_command(
    command: str,
    timeout: int = 300,
    **kwargs: Any,
) -> ToolResponse:
    """Execute safe commands including DataJuicer commands and other safe system commands.
    Returns the return code, standard output and error within <returncode></returncode>,
    <stdout></stdout> and <stderr></stderr> tags.

    Args:
        command (`str`):
            The command to execute. Allowed commands include:
            - DataJuicer commands: dj-process, dj-analyze
            - File system commands: mkdir, ls, pwd, cat, echo, cp, mv, rm
            - Text processing: grep, head, tail, wc, sort, uniq
            - Archive commands: tar, zip, unzip
            - Other safe commands: which, whoami, date, find
        timeout (`float`, defaults to `300`):
            The maximum time (in seconds) allowed for the command to run.

    Returns:
        `ToolResponse`:
            The tool response containing the return code, standard output, and
            standard error of the executed command.
    """

    # Security check: only allow safe commands
    command_stripped = command.strip()

    # Define allowed command prefixes for security
    allowed_commands = [
        # DataJuicer commands
        "dj-process",
        "dj-analyze",
        # File system operations
        "mkdir",
        "ls",
        "pwd",
        "cat",
        "echo",
        "cp",
        "mv",
        "rm",
        # Text processing
        "grep",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        # Archive operations
        "tar",
        "zip",
        "unzip",
        # Information commands
        "which",
        "whoami",
        "date",
        "find",
        # Python commands
        "python",
        "python3",
        "pip",
        "uv",
    ]

    # Check if command starts with any allowed command
    command_allowed = False
    for allowed_cmd in allowed_commands:
        if command_stripped.startswith(allowed_cmd):
            # Additional security checks for potentially dangerous commands
            if allowed_cmd in ["rm", "mv"] and (
                "/" in command_stripped or ".." in command_stripped
            ):
                # Prevent dangerous path operations
                continue
            command_allowed = True
            break

    if not command_allowed:
        error_msg = f"Error: Command not allowed for security reasons. Allowed commands: {', '.join(allowed_commands)}. Received command: {command}"
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"<returncode>-1</returncode>"
                        f"<stdout></stdout>"
                        f"<stderr>{error_msg}</stderr>"
                    ),
                ),
            ],
        )

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        bufsize=0,
    )

    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        stdout, stderr = await proc.communicate()
        stdout_str = stdout.decode("utf-8")
        stderr_str = stderr.decode("utf-8")
        returncode = proc.returncode

    except asyncio.TimeoutError:
        stderr_suffix = (
            f"TimeoutError: The command execution exceeded "
            f"the timeout of {timeout} seconds."
        )
        returncode = -1
        try:
            proc.terminate()
            stdout, stderr = await proc.communicate()
            stdout_str = stdout.decode("utf-8")
            stderr_str = stderr.decode("utf-8")
            if stderr_str:
                stderr_str += f"\n{stderr_suffix}"
            else:
                stderr_str = stderr_suffix
        except ProcessLookupError:
            stdout_str = ""
            stderr_str = stderr_suffix

    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=(
                    f"<returncode>{returncode}</returncode>"
                    f"<stdout>{stdout_str}</stdout>"
                    f"<stderr>{stderr_str}</stderr>"
                ),
            ),
        ],
    )
