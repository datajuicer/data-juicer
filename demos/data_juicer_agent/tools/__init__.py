# -*- coding: utf-8 -*-
"""
Tools package for data-agent.

This module provides a unified entry point for all agent tools,
organized by agent type for easy access and management.
"""
from typing import List
from agentscope.agent import AgentBase
from agentscope.tool import (
    view_text_file,
    write_text_file,
)
from agentscope.tool import Toolkit

from .dj_helpers import execute_safe_command
from .router_helpers import agent_to_tool
from .dj_helpers import query_dj_operators
from .dj_dev_helpers import (
    get_basic_files,
    get_operator_example,
    configure_data_juicer_path,
)
from .mcp_helpers import get_mcp_toolkit


def create_toolkit(tools: List[str]):
    # Create toolkit and register tools
    toolkit = Toolkit()
    for tool in tools:
        toolkit.register_tool_function(tool)

    return toolkit


# DJ Agent tools
dj_tools = [
    execute_safe_command,
    view_text_file,
    write_text_file,
    query_dj_operators,
]

# DJ Development Agent tools - for developing DataJuicer operators
dj_dev_tools = [
    view_text_file,
    write_text_file,
    get_basic_files,
    get_operator_example,
    configure_data_juicer_path,
]

# MCP Agent tools - for advanced data processing with Recipe Flow MCP
mcp_tools = [
    view_text_file,
    write_text_file,
]


def agents2toolkit(agents: List[AgentBase]):
    tools = [agent_to_tool(agent) for agent in agents]
    return create_toolkit(tools)


dj_toolkit = create_toolkit(dj_tools)
dj_dev_toolkit = create_toolkit(dj_dev_tools)


# All available tools
all_toolkit = {
    "dj": dj_toolkit,
    "dj_dev": dj_dev_toolkit,
    "dj_mcp": get_mcp_toolkit,
    "router": agents2toolkit,
}

# Public API
__all__ = [
    "dj_tools",
    "dj_dev_tools",
    "mcp_tools",
    "all_tools",
    "agents2toolkit",
    "dj_toolkit",
    "dj_dev_toolkit",
    "get_mcp_toolkit",
    # Individual tools for direct import
    "execute_safe_command",
    "view_text_file",
    "write_text_file",
    "agent_to_tool",
    "query_dj_operators",
    "get_basic_files",
    "get_operator_example",
    "configure_data_juicer_path",
]
