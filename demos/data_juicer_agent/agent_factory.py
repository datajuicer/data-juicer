# -*- coding: utf-8 -*-
"""
Agent Factory

Factory functions for creating and configuring agents with standardized toolkits.
"""

import os
from typing import Optional
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit
from agentscope.formatter import FormatterBase, OpenAIChatFormatter
from agentscope.model import ChatModelBase, OpenAIChatModel
from agentscope.memory import InMemoryMemory, MemoryBase


# Default configurations
DEFAULT_MODEL_CONFIG = {
    "model_name": "gpt-4o",
    "stream": False,
}


def get_default_model() -> OpenAIChatModel:
    """Create default OpenAI model instance."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    return OpenAIChatModel(api_key=api_key, **DEFAULT_MODEL_CONFIG)


def create_agent(
    name: str,
    sys_prompt: str,
    toolkit: Toolkit,
    description: Optional[str] = None,
    model: Optional[ChatModelBase] = None,
    formatter: Optional[FormatterBase] = None,
    memory: Optional[MemoryBase] = None,
    max_iters: int = 10,
    parallel_tool_calls: bool = False,
    **kwargs,
) -> ReActAgent:
    """
    Create a ReActAgent with standardized configuration.

    Args:
        name: Agent identifier
        sys_prompt: System prompt template (supports {name} placeholder)
        toolkit: Toolkit instance
        model: Language model (defaults to GPT-4o)
        formatter: Message formatter (defaults to OpenAIChatFormatter)
        memory: Memory instance (defaults to InMemoryMemory)
        max_iters: Maximum reasoning iterations
        parallel_tool_calls: Enable parallel tool execution
        **kwargs: Additional ReActAgent arguments

    Returns:
        Configured ReActAgent instance

    Example:
        >>> agent = create_agent(
        ...     name="sql_expert",
        ...     sys_prompt="You are {name}, a SQL database expert",
        ...     tools=sql_tools
        ... )
    """
    # Set defaults
    if model is None:
        model = get_default_model()
    if formatter is None:
        formatter = OpenAIChatFormatter()
    if memory is None:
        memory = InMemoryMemory()

    # Create agent
    agent = ReActAgent(
        name=name,
        sys_prompt=sys_prompt.format(name=name),
        model=model,
        formatter=formatter,
        toolkit=toolkit,
        memory=memory,
        max_iters=max_iters,
        parallel_tool_calls=parallel_tool_calls,
        **kwargs,
    )

    agent.__doc__ = description

    return agent
