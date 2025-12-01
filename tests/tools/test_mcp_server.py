import ast
import unittest
from unittest import IsolatedAsyncioTestCase
from mcp.shared.memory import (
    create_connected_server_and_client_session as client_session,
)
from mcp.types import TextContent
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase
from loguru import logger


class MCPServerTest(IsolatedAsyncioTestCase, DataJuicerTestCaseBase):

    test_dataset_path = "./demos/data/demo-dataset.jsonl"

    async def get_data_processing_ops(self):
        """Test the get_data_processing_ops method"""
        from data_juicer.tools.DJ_mcp_recipe_flow import create_mcp_server

        mcp = create_mcp_server()

        async with client_session(mcp._mcp_server) as client:
            # Test with op_type and tags
            result = await client.call_tool(
                "get_data_processing_ops",
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

            # Test with no parameters
            result = await client.call_tool("get_data_processing_ops", {})
            self.assertGreater(len(result.content), 0)
            content = result.content[0]
            self.assertIsInstance(content, TextContent)
            try:
                dict_content = ast.literal_eval(content.text)
                self.assertIsInstance(dict_content, dict)
                logger.info(f"ops count: {len(dict_content)}")
            except (ValueError, SyntaxError):
                self.fail("content.text is not a valid dictionary string")

    async def run_data_recipe(self):
        """Test the run_data_recipe method"""
        from data_juicer.tools.DJ_mcp_recipe_flow import create_mcp_server

        mcp = create_mcp_server()

        async with client_session(mcp._mcp_server) as client:
            # Test with valid parameters
            result = await client.call_tool(
                "run_data_recipe",
                {
                    "dataset_path": self.test_dataset_path,
                    "process": [
                        {"text_length_filter": {"min_len": 10}},
                        {
                            "language_id_score_filter": {
                                "lang": "zh",
                                "min_score": 0.8,
                            }
                        },
                    ],
                    "np": 2,
                },
            )
            self.assertFalse(result.isError)
            logger.info(f"result: {result.content[0].text}")

    # Test for granular_ops
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
            # Test with valid parameters
            result = await client.list_tools()
            self.assertGreater(len(result.tools), 1)

        await self.get_data_processing_ops()
        await self.run_data_recipe()


if __name__ == "__main__":
    # nest_asyncio.apply()
    unittest.main()
