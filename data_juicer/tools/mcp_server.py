#!/usr/bin/env python3

import argparse
import os
import sys


def main():
    """Data-Juicer MCP Server CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Data-Juicer MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  granular-ops    Launch MCP server with individual operator tools
  recipe-flow     Launch MCP server with recipe-based workflow tools

Examples:
  dj-mcp granular-ops --transport stdio
  dj-mcp recipe-flow --transport sse --port 8000
        """,
    )

    parser.add_argument("mode", choices=["granular-ops", "recipe-flow"], help="MCP server mode to launch")

    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol for MCP server (default: stdio)",
    )

    parser.add_argument("--port", type=int, default=8080, help="Port number for HTTP-based transports (default: 8080)")

    args = parser.parse_args()

    # Set environment variable for transport
    os.environ["SERVER_TRANSPORT"] = args.transport

    try:
        if args.mode == "granular-ops":
            from data_juicer.tools.DJ_mcp_granular_ops import create_mcp_server

        elif args.mode == "recipe-flow":
            from data_juicer.tools.DJ_mcp_recipe_flow import create_mcp_server

        print(f"Starting Data-Juicer MCP Server ({args.mode} mode)")
        print(f"Transport: {args.transport}, Port: {args.port}")

        mcp = create_mcp_server(port=str(args.port))
        mcp.run(transport=args.transport)

    except ImportError as e:
        print(f"Error: Missing dependencies for MCP server. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
