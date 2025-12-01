import argparse
import os
from typing import Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from data_juicer.tools.mcp_tool import execute_op
from data_juicer.tools.op_search import OPSearcher

# Operator Management
ops_list_path = os.getenv("DJ_OPS_LIST_PATH", None)
if ops_list_path:
    with open(ops_list_path, "r", encoding="utf-8") as file:
        ops_list = [line.strip() for line in file if line.strip()]
else:
    ops_list = None
searcher = OPSearcher(ops_list)


# Operator Management
def get_data_processing_ops(
    op_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    match_all: bool = True,
) -> dict:
    """
    Retrieves a list of available data processing operators based on the specified type and tags.
    Operators are a collection of basic processes that assist in data modification,
    cleaning, filtering, deduplication, etc.

    If both tags and ops_type are None, return a list of all operators.

    The following `op_type` values are supported:
    - aggregator: Aggregate for batched samples, such as summary or conclusion.
    - deduplicator: Detects and removes duplicate samples.
    - filter: Filters out low-quality samples.
    - grouper: Group samples to batched samples.
    - mapper: Edits and transforms samples.
    - selector: Selects top samples based on ranking.

    The `tags` parameter specifies the characteristics of the data or the required resources.
    Available tags are:

    Modality Tags:
        - text: process text data specifically.
        - image: process image data specifically.
        - audio: process audio data specifically.
        - video: process video data specifically.
        - multimodal: process multimodal data.

    Resource Tags:
        - cpu: only requires CPU resource.
        - gpu: requires GPU/CUDA resource as well.

    Model Tags:
        - api: equipped with API-based models (e.g. ChatGPT, GPT-4o).
        - vllm: equipped with models supported by vLLM.
        - hf: equipped with models from HuggingFace Hub.

    Tags are used to refine the search for suitable operators based on specific data processing needs.

    :param op_type: The type of data processing operator to retrieve.
                     If None, no ops_type-based filtering is applied.
                     If specified, must be one of the values listed. Defaults to None.
    :param tags: An optional list of tags to filter operators.  See the tag list above for options.
                 If None, no tag-based filtering is applied. Defaults to None.
    :param match_all: If True, only operators matching all specified tags are returned.
                      If False, operators matching any of the specified tags are returned. Defaults to True.
    :returns: A dict containing detailed information about the available operators
    """
    op_results = searcher.search(tags=tags, op_type=op_type, match_all=match_all)

    ops_dict = dict()
    for op in op_results:
        ops_dict[op["name"]] = "\n".join([op["desc"], op["param_desc"], "Parameters: ", str(op["sig"])])

    return ops_dict


def run_data_recipe(
    dataset_path: str,
    process: list[Dict],
    export_path: Optional[str] = None,
    np: int = 1,
) -> str:
    """
    Run data recipe.
    If you want to run one or more DataJuicer data processing operators, you can
    use this tool. Supported operators and their arguments should be obtained
    through the `get_data_processing_ops` tool.

    :param dataset_path: Path to the dataset to be processed.
    :param process: List of processing operations to be executed sequentially.
                Each element is a dictionary with operator name as key and
                its configuration as value. Multiple operators can be chained.
    :param export_path: Path to export the processed dataset. Defaults to None,
                       which exports to './outputs' directory.
    :param np: Number of processes to use. Defaults to 1.

    Example:
        # First get available filter operators for text data
        >>> available_ops = get_data_processing_ops(
        ...     op_type="filter",
        ...     tags=["text"]
        ... )

        # Then run a data recipe with selected filters:
        # 1. First filter samples with text length 10-50
        # 2. Then filter English samples with language confidence score >= 0.8
        >>> run_data_recipe(
        ...     "/path/to/dataset.jsonl",
        ...     [
        ...         {
        ...             "text_length_filter": {
        ...                 "min_len": 10,
        ...                 "max_len": 50
        ...             }
        ...         },
        ...         {
        ...             "language_id_score_filter": {
        ...                 "lang": "en",
        ...                 "min_score": 0.8
        ...             }
        ...         }
        ...     ]
        ... )
    """
    dj_cfg = dict(locals())
    return execute_op(dj_cfg)


def create_mcp_server(port: str = "8000"):
    """
    Creates the FastMCP server and registers the tools.

    Args:
        port (str, optional): Port number. Defaults to "8000".
    """
    mcp = FastMCP("Data-Juicer Server", port=port)

    mcp.tool()(get_data_processing_ops)
    mcp.tool()(run_data_recipe)

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
