# -*- coding: utf-8 -*-
import os
import fire
from typing import List

from agentscope.model import DashScopeChatModel
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.agent import UserAgent

from agent_factory import create_agent
from prompts import (
    DJ_SYS_PROMPT,
    DJ_DEV_SYS_PROMPT,
    ROUTER_SYS_PROMPT,
    MCP_SYS_PROMPT,
)
from tools import (
    dj_toolkit,
    dj_dev_toolkit,
    mcp_tools,
    get_mcp_toolkit,
    agents2toolkit,
)

# Create shared configuration
model = DashScopeChatModel(
    model_name="qwen-max",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    stream=True,
    enable_thinking=False,
)

dev_model = DashScopeChatModel(
    model_name="qwen3-coder-480b-a35b-instruct",
    api_key=os.environ["DASHSCOPE_API_KEY"],
    stream=True,
    enable_thinking=False,
)

formatter = DashScopeChatFormatter()
memory = InMemoryMemory()

user = UserAgent("User")


async def main(
    use_studio: bool = False,
    available_agents: List[str] = ["dj", "dj_dev"],
    retrieval_mode: str = "auto",
):
    """
    Main function for running the agent.

    :param use_studio: Whether to use agentscope studio.
    :param available_agents: List of available agents. Options: dj, dj_dev, dj_mcp
    :param retrieval_mode: Retrieval mode for operators. Options: auto, vector, llm
    """

    if "dj" in available_agents:
        # Set global retrieval mode for tools to use
        os.environ["RETRIEVAL_MODE"] = retrieval_mode
        print(f"Using retrieval mode: {retrieval_mode}")

    agents = []
    for agent_name in available_agents:
        if agent_name == "dj":
            # Create agents using unified create_agent function
            dj_agent = create_agent(
                "datajuicer_agent",
                DJ_SYS_PROMPT,
                dj_toolkit,
                (
                    "A professional data preprocessing AI assistant with the following core capabilities: \n"
                    "Tool Matching \n"
                    "- Query and validate suitable DataJuicer operators; \n"
                    "Configuration Generation \n"
                    "- Create YAML configuration files and preview data; \n"
                    "Task Execution - Run data processing pipelines and output results"
                ),
                model,
                formatter,
                memory,
            )
            agents.append(dj_agent)

        if agent_name == "dj_dev":
            # DJ Development Agent for operator development
            dj_dev_agent = create_agent(
                "dj_dev_agent",
                DJ_DEV_SYS_PROMPT,
                dj_dev_toolkit,
                (
                    "An expert DataJuicer development assistant specializing in creating new DataJuicer operators. \n"
                    "Core capabilities: \n"
                    "Reference Retrieval - fetch base classes and examples; \n"
                    "Environment Configuration - handle DATA_JUICER_PATH setup. if user provides a DataJuicer path requiring setup/update, please call this agent;\n; "
                    "Code Generation - write complete, convention-compliant operator code"
                ),
                dev_model,
                formatter,
                memory,
            )
            agents.append(dj_dev_agent)

        if agent_name == "dj_mcp":
            mcp_toolkit, _ = await get_mcp_toolkit()
            for tool in mcp_tools:
                mcp_toolkit.register_tool_function(tool)

            mcp_agent = create_agent(
                "mcp_datajuicer_agent",
                MCP_SYS_PROMPT,
                mcp_toolkit,
                (
                    "DataJuicer MCP Agent powered by Recipe Flow MCP server. \n"
                    "Core capabilities: \n"
                    "- Filter operators by tags/categories using MCP protocol; \n"
                    "- Real-time data processing pipeline execution. \n"
                ),
                model,
                formatter,
                memory,
            )
            agents.append(mcp_agent)

    # Router agent - uses agents2tools to dynamically generate tools from all agents
    router_agent = create_agent(
        "Router",
        ROUTER_SYS_PROMPT,
        agents2toolkit(agents),
        "A router agent that intelligently routes tasks to specialized DataJuicer agents",
        model,
        formatter,
        InMemoryMemory(),  # Router uses its own memory instance
    )

    if use_studio is True:
        import agentscope

        agentscope.init(
            studio_url="http://localhost:3000",
            project="data_agent",
        )

    msg = None
    while True:
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            break
        # Router agent handles the entire task with automatic multi-step routing
        msg = await router_agent(msg)


if __name__ == "__main__":
    # Example tasks
    # project_root = os.path.abspath(os.path.dirname(__file__))
    # task = (
    #     f"The data is stored in {project_root}/data/demo-dataset-images.jsonl. "
    #     "Among the samples, the text field length is less than 5 "
    #     "and the image size is less than 100Kb. "
    #     "And save the output results to the ./outputs path."
    # )
    #
    # DJ Development example task:
    # task = "I want to develop a new DataJuicer filter operator to filter out audio files without vocals"
    #
    fire.Fire(main)
