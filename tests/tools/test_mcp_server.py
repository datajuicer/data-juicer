import ast
import os
import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import MagicMock, patch

from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)
from mcp.types import TextContent

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from loguru import logger


class MCPServerCLITest(DataJuicerTestCaseBase):
    """Tests for the mcp_server.py CLI entry point (main())."""

    def test_granular_ops_mode_default_transport(self):
        """granular-ops mode should launch with streamable-http by default."""
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "granular-ops"]):
            with patch("data_juicer.tools.DJ_mcp_granular_ops.create_mcp_server") as mock_create:
                mock_server = MagicMock()
                mock_create.return_value = mock_server
                main()
                mock_create.assert_called_once_with(port="8080")
                mock_server.run.assert_called_once_with(transport="streamable-http")

    def test_recipe_flow_mode_default_transport(self):
        """recipe-flow mode should launch with streamable-http by default."""
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "recipe-flow"]):
            with patch("data_juicer.tools.DJ_mcp_recipe_flow.create_mcp_server") as mock_create:
                mock_server = MagicMock()
                mock_create.return_value = mock_server
                main()
                mock_create.assert_called_once_with(port="8080")
                mock_server.run.assert_called_once_with(transport="streamable-http")

    def test_stdio_transport(self):
        """--transport stdio should be passed through to mcp.run()."""
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "granular-ops", "--transport", "stdio"]):
            with patch("data_juicer.tools.DJ_mcp_granular_ops.create_mcp_server") as mock_create:
                mock_server = MagicMock()
                mock_create.return_value = mock_server
                main()
                mock_server.run.assert_called_once_with(transport="stdio")

    def test_sse_transport(self):
        """--transport sse should be passed through to mcp.run()."""
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "granular-ops", "--transport", "sse"]):
            with patch("data_juicer.tools.DJ_mcp_granular_ops.create_mcp_server") as mock_create:
                mock_server = MagicMock()
                mock_create.return_value = mock_server
                main()
                mock_server.run.assert_called_once_with(transport="sse")

    def test_custom_port_passed_to_create(self):
        """--port should be forwarded to create_mcp_server as a string."""
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "recipe-flow", "--port", "9090"]):
            with patch("data_juicer.tools.DJ_mcp_recipe_flow.create_mcp_server") as mock_create:
                mock_server = MagicMock()
                mock_create.return_value = mock_server
                main()
                mock_create.assert_called_once_with(port="9090")

    def test_invalid_mode_exits(self):
        """An invalid mode should cause argparse to exit with a non-zero code."""
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "invalid-mode"]):
            with self.assertRaises(SystemExit) as context:
                main()
            self.assertNotEqual(context.exception.code, 0)

    def test_invalid_transport_exits(self):
        """An invalid --transport value should cause argparse to exit."""
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "granular-ops", "--transport", "websocket"]):
            with self.assertRaises(SystemExit) as context:
                main()
            self.assertNotEqual(context.exception.code, 0)

    def test_import_error_exits_with_code_1(self):
        """ImportError during server creation should exit with code 1."""
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "granular-ops"]):
            with patch(
                "data_juicer.tools.DJ_mcp_granular_ops.create_mcp_server",
                side_effect=ImportError("missing dep"),
            ):
                with self.assertRaises(SystemExit) as context:
                    main()
                self.assertEqual(context.exception.code, 1)

    def test_server_transport_env_var_set(self):
        """SERVER_TRANSPORT env var should be set to the chosen transport."""
        from data_juicer.tools.mcp_server import main

        with patch("sys.argv", ["dj-mcp", "granular-ops", "--transport", "stdio"]):
            with patch("data_juicer.tools.DJ_mcp_granular_ops.create_mcp_server") as mock_create:
                mock_server = MagicMock()
                mock_create.return_value = mock_server
                main()
                self.assertEqual(os.environ.get("SERVER_TRANSPORT"), "stdio")


class MCPServerTest(IsolatedAsyncioTestCase, DataJuicerTestCaseBase):

    test_dataset_path = "./demos/data/demo-dataset.jsonl"

    async def search_ops_test(self):
        """Test the search_ops method"""
        from data_juicer.tools.DJ_mcp_recipe_flow import create_mcp_server

        mcp = create_mcp_server()

        async with client_session(mcp._mcp_server) as client:
            # Test with op_type and tags (default "tags" search_mode)
            result = await client.call_tool(
                "search_ops",
                {"op_type": "filter", "tags": ["text", "cpu"]},
            )
            self.assertEqual(len(result.content), 1)
            content = result.content[0]
            self.assertIsInstance(content, TextContent)
            try:
                dict_content = ast.literal_eval(content.text)
                self.assertIsInstance(dict_content, dict)
                logger.info(f"ops count: {len(dict_content)}")
            except (ValueError, SyntaxError):
                self.fail("content.text is not a valid dictionary string")

            # Test with no parameters (returns all ops)
            result = await client.call_tool("search_ops", {})
            self.assertGreater(len(result.content), 0)
            content = result.content[0]
            self.assertIsInstance(content, TextContent)
            try:
                dict_content = ast.literal_eval(content.text)
                self.assertIsInstance(dict_content, dict)
                logger.info(f"ops count: {len(dict_content)}")
            except (ValueError, SyntaxError):
                self.fail("content.text is not a valid dictionary string")

            # Test regex search mode
            result = await client.call_tool(
                "search_ops",
                {"query": "text_length", "search_mode": "regex"},
            )
            self.assertEqual(len(result.content), 1)
            content = result.content[0]
            self.assertIsInstance(content, TextContent)
            try:
                dict_content = ast.literal_eval(content.text)
                self.assertIsInstance(dict_content, dict)
                self.assertGreater(len(dict_content), 0)
                logger.info(f"regex search ops count: {len(dict_content)}")
            except (ValueError, SyntaxError):
                self.fail("content.text is not a valid dictionary string")

            # Test bm25 search mode
            result = await client.call_tool(
                "search_ops",
                {"query": "filter text by length", "search_mode": "bm25", "top_k": 5},
            )
            self.assertEqual(len(result.content), 1)
            content = result.content[0]
            self.assertIsInstance(content, TextContent)
            try:
                dict_content = ast.literal_eval(content.text)
                self.assertIsInstance(dict_content, dict)
                logger.info(f"bm25 search ops count: {len(dict_content)}")
            except (ValueError, SyntaxError):
                self.fail("content.text is not a valid dictionary string")

    async def run_data_recipe_test(self):
        """Test the run_data_recipe method"""
        from data_juicer.tools.DJ_mcp_recipe_flow import create_mcp_server

        mcp = create_mcp_server()

        async with client_session(mcp._mcp_server) as client:
            # Test with valid parameters
            result = await client.call_tool(
                "run_data_recipe",
                {
                    "process": [
                        {"text_length_filter": {"min_len": 10}},
                        {
                            "language_id_score_filter": {
                                "lang": "zh",
                                "min_score": 0.8,
                            }
                        },
                    ],
                    "dataset_path": self.test_dataset_path,
                    "np": 2,
                },
            )
            self.assertFalse(result.isError)
            logger.info(f"result: {result.content[0].text}")

    async def test_granular_ops(self):
        """Test the text_length_filter operator"""
        from data_juicer.tools.DJ_mcp_granular_ops import create_mcp_server

        mcp = create_mcp_server()

        async with client_session(mcp._mcp_server) as client:
            # Test with valid parameters
            result = await client.call_tool(
                "text_length_filter",
                {
                    "dataset_path": self.test_dataset_path,
                    "min_len": 10,
                    "max_len": 50,
                },
            )
            self.assertFalse(result.isError)
            logger.info(f"result: {result.content[0].text}")

            # Test with list_tools
            result = await client.list_tools()
            self.assertGreater(len(result.tools), 2)
            logger.info(f"tools count: {len(result.tools)}")

    async def test_recipe_flow(self):
        """Test the recipe_flow method"""
        from data_juicer.tools.DJ_mcp_recipe_flow import create_mcp_server

        mcp = create_mcp_server()

        async with client_session(mcp._mcp_server) as client:
            result = await client.list_tools()
            # recipe-flow registers 5 tools: get_global_config_schema,
            # get_dataset_load_strategies, search_ops, run_data_recipe,
            # analyze_dataset
            self.assertEqual(len(result.tools), 5)
            tool_names = {tool.name for tool in result.tools}
            expected_tools = {
                "get_global_config_schema",
                "get_dataset_load_strategies",
                "search_ops",
                "run_data_recipe",
                "analyze_dataset",
            }
            self.assertEqual(tool_names, expected_tools)

        await self.search_ops_test()
        await self.run_data_recipe_test()


if __name__ == "__main__":
    unittest.main()
