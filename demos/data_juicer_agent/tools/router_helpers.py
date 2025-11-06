# -*- coding: utf-8 -*-
"""Router agent using implicit routing"""
from typing import Callable
from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.tool import ToolResponse


def agent_to_tool(
    agent: AgentBase,
    tool_name: str = None,
    description: str = None,
) -> Callable:
    """
    Convert any agent to a tool function that can be registered in toolkit.

    Args:
        agent: The agent instance to convert
        tool_name: Optional custom tool name (defaults to agent.name)
        description: Optional tool description (defaults to agent's docstring or sys_prompt)

    Returns:
        A tool function that can be registered with toolkit.register_tool_function()
    """
    # Get tool name and description
    if tool_name is None:
        tool_name = getattr(agent, "name", "agent_tool")

    if description is None:
        # Try to get description from agent's docstring or sys_prompt
        if hasattr(agent, "__doc__") and agent.__doc__:
            description = agent.__doc__.strip()
        elif hasattr(agent, "sys_prompt"):
            description = f"Agent: {agent.sys_prompt[:100]}..."
        elif hasattr(agent, "_sys_prompt"):
            description = f"Agent: {agent._sys_prompt[:100]}..."
        else:
            description = f"Tool function for {tool_name}"

    async def tool_function(task: str) -> ToolResponse:
        # Create message and call the agent
        msg = Msg("user", task, "user")
        result = await agent(msg)

        # Extract content from the result
        if hasattr(result, "get_content_blocks"):
            content = result.get_content_blocks("text")
            return ToolResponse(
                content=content,
                metadata={
                    "agent_name": getattr(agent, "name", "unknown"),
                    "task": task,
                },
            )
        else:
            raise ValueError(f"Not a valid Msg object: {result}")

    # Set function name and docstring
    tool_function.__name__ = f"call_{tool_name.lower().replace(' ', '_')}"
    tool_function.__doc__ = f"{description}\n\nArgs:\n    task (str): The task for {tool_name} to handle"

    return tool_function
