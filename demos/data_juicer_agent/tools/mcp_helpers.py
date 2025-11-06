# -*- coding: utf-8 -*-
import json
import os
import logging
from typing import Optional, List
import string

from agentscope.tool import Toolkit
from agentscope.mcp import (
    HttpStatefulClient,
    HttpStatelessClient,
    StdIOStatefulClient,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def _load_config(config_path: str) -> dict:
    """Load MCP configuration from file"""
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Loaded MCP configuration from {config_path}")
            return config
        else:
            logger.warning(
                f"Configuration file {config_path} not found, using default settings",
            )
            return _create_default_config()
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return _create_default_config()


def _create_default_config() -> dict:
    """Create default configuration"""
    return {
        "mcpServers": {
            "dj_recipe_flow": {
                "command": "python",
                "args": ["/home/test/data_juicer/tools/DJ_mcp_recipe_flow.py"],
                "env": {"SERVER_TRANSPORT": "stdio"},
            },
        },
    }


def _expand_env_vars(value: str) -> str:
    """Expand environment variables in configuration values"""
    if isinstance(value, str):
        template = string.Template(value)
        try:
            return template.substitute(os.environ)
        except KeyError as e:
            logger.warning(f"Environment variable not found: {e}")
            return value
    return value


async def _create_clients(config: dict, toolkit: Toolkit):
    """Create MCP clients based on configuration"""
    server_configs = config.get("mcpServers", {})
    clients = []

    for server_name, server_config in server_configs.items():
        try:
            # Handle StdIO client
            if "command" in server_config:
                command = server_config["command"]
                args = server_config.get("args", [])
                env = server_config.get("env", {})

                # Expand environment variables
                expanded_args = [_expand_env_vars(arg) for arg in args]
                expanded_env = {k: _expand_env_vars(v) for k, v in env.items()}

                client = StdIOStatefulClient(
                    name=server_name,
                    command=command,
                    args=expanded_args,
                    env=expanded_env,
                )

                await client.connect()
                await toolkit.register_mcp_client(client)

            # Handle HTTP clients
            elif "url" in server_config:
                url = _expand_env_vars(server_config["url"])
                transport = server_config.get("transport", "sse")
                stateful = server_config.get("stateful", True)

                if stateful:
                    client = HttpStatefulClient(
                        name=server_name,
                        transport=transport,
                        url=url,
                    )
                    await client.connect()
                    await toolkit.register_mcp_client(client)
                else:
                    client = HttpStatelessClient(
                        name=server_name,
                        transport=transport,
                        url=url,
                    )
                    await toolkit.register_mcp_client(client)

            else:
                raise ValueError("Invalid server configuration")

            clients.append(client)
        except Exception as e:
            if "Invalid server configuration" in str(e):
                raise e
            logger.error(f"Failed to create client {server_name}: {e}")

    return clients


async def get_mcp_toolkit(config_path: Optional[str] = None) -> Toolkit:
    """Get toolkit with all MCP tools registered"""
    config_path = config_path or root_path + "/configs/mcp_config.json"
    config = _load_config(config_path)
    toolkit = Toolkit()

    clients = await _create_clients(config, toolkit)

    return toolkit, clients
