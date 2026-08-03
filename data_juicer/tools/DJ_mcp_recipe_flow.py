import argparse
import os
from typing import Dict, List, Optional

from data_juicer.tools.mcp_tool import execute_analyze, execute_op
from data_juicer.tools.op_search import OPSearcher
from data_juicer.utils.lazy_loader import LazyLoader

fastmcp = LazyLoader("mcp.server.fastmcp", "mcp[cli]")

# Operator Management
ops_list_path = os.getenv("DJ_OPS_LIST_PATH", None)
if ops_list_path:
    with open(ops_list_path, "r", encoding="utf-8") as file:
        ops_list = [line.strip() for line in file if line.strip()]
else:
    ops_list = None
searcher = OPSearcher(ops_list)


def get_global_config_schema() -> dict:
    """
    Get the full schema of all available global configuration options
    for Data-Juicer.

    Returns a dictionary where each key is a config parameter name,
    and the value is a dict containing:
    - type: the expected type of the parameter (e.g. "bool", "int", "str")
    - default: the default value
    - description: a human-readable description of the parameter

    Use this tool to discover what configuration options can be passed
    to run_data_recipe via the extra_config parameter. This dynamically
    reflects the latest Data-Juicer configuration, so it will always
    be up-to-date even as new config options are added.

    :returns: A dict mapping config parameter names to their schema info
    """
    from data_juicer.config.config import build_base_parser

    parser = build_base_parser()

    if parser is None:
        return {"error": "Failed to initialize config parser"}

    # Internal parameters that should not be exposed to users
    excluded_params = {
        "config",
        "auto",
        "help",
        "print_config",
    }

    schema = {}
    for action in parser._actions:
        # Skip suppressed or internal actions
        if not action.option_strings:
            continue

        # Use the longest option string as the parameter name
        param_name = max(action.option_strings, key=len).lstrip("-")
        dest = action.dest

        if dest in excluded_params or param_name in excluded_params:
            continue

        # Determine type name
        type_name = "str"
        if action.type is not None:
            if hasattr(action.type, "__name__"):
                type_name = action.type.__name__
            elif hasattr(action.type, "__class__"):
                type_name = str(action.type)
            else:
                type_name = str(action.type)
        elif isinstance(action.const, bool):
            type_name = "bool"

        # Handle choices
        choices = None
        if action.choices:
            choices = list(action.choices)

        entry = {
            "type": type_name,
            "default": action.default,
            "description": action.help or "",
        }
        if choices:
            entry["choices"] = choices

        schema[param_name] = entry

    return schema


def get_dataset_load_strategies() -> dict:
    """
    Get all available dataset loading strategies supported by Data-Juicer.

    Returns information about each strategy including its executor type,
    data type, data source, required/optional configuration fields, and
    description. Use this tool to understand how to configure the 'dataset'
    parameter in run_data_recipe for different data sources (e.g., local
    files, HuggingFace, S3, ModelScope, etc.).

    The 'dataset' parameter in run_data_recipe accepts a dict with:
    - configs: a list of dataset config dicts, each containing a 'type'
      field that maps to a data source strategy (e.g., 'local', 'huggingface')
    - max_sample_num: optional max number of samples to load

    Each dataset config dict should follow the required/optional fields
    described in the returned strategy information.

    :returns: A dict mapping strategy identifiers to their configuration info
    """
    from data_juicer.core.data.load_strategy import DataLoadStrategyRegistry

    strategies_info = {}

    for strategy_key, strategy_class in DataLoadStrategyRegistry._strategies.items():
        identifier = f"{strategy_key.executor_type}/" f"{strategy_key.data_type}/" f"{strategy_key.data_source}"

        # Extract CONFIG_VALIDATION_RULES if available
        validation_rules = getattr(strategy_class, "CONFIG_VALIDATION_RULES", {})

        # Extract class docstring
        description = strategy_class.__doc__ or ""
        description = description.strip()

        entry = {
            "executor_type": strategy_key.executor_type,
            "data_type": strategy_key.data_type,
            "data_source": strategy_key.data_source,
            "description": description,
            "class_name": strategy_class.__name__,
        }

        if validation_rules:
            entry["required_fields"] = validation_rules.get("required_fields", [])
            entry["optional_fields"] = validation_rules.get("optional_fields", [])
            # Convert field_types to string representation for serialization
            field_types = validation_rules.get("field_types", {})
            entry["field_types"] = {
                key: (val.__name__ if hasattr(val, "__name__") else str(val)) for key, val in field_types.items()
            }

        strategies_info[identifier] = entry

    return strategies_info


def search_ops(
    query: Optional[str] = None,
    op_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    match_all: bool = True,
    search_mode: str = "tags",
    top_k: int = 10,
) -> dict:
    """
    Search for available data processing operators.

    Operators are a collection of basic processes that assist in data
    modification, cleaning, filtering, deduplication, etc.

    Supports multiple search modes:
    - "basic": filter by op_type and/or tags (default, original behavior).
      If both tags and op_type are None, returns all operators.
    - "regex": Python regex pattern matching against OP names,
      descriptions, and parameters. Requires the query parameter.
    - "bm25": BM25 text relevance ranking for natural language queries.
      Returns top_k most relevant operators. Requires the query parameter.

    op_type and tags can be combined with any search_mode as additional
    filters to narrow down results.

    The following `op_type` values are supported:
    - aggregator: Aggregate for batched samples, such as summary or conclusion.
    - deduplicator: Detects and removes duplicate samples.
    - filter: Filters out low-quality samples.
    - grouper: Group samples to batched samples.
    - mapper: Edits and transforms samples.
    - selector: Selects top samples based on ranking.
    - pipeline: Applies dataset-level processing; both input and output are datasets.

    The `tags` parameter specifies the characteristics of the data or the
    required resources. Available tags are:

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

    :param query: Search query string. Required for "regex" and "bm25"
        modes. For "regex" mode, this should be a Python regex pattern.
        For "bm25" mode, this should be a natural language
        description of the desired functionality.
    :param op_type: The type of data processing operator to filter by.
        If None, no type-based filtering is applied. Defaults to None.
    :param tags: An optional list of tags to filter operators.
        If None, no tag-based filtering is applied. Defaults to None.
    :param match_all: If True, only operators matching all specified tags
        are returned. If False, operators matching any tag are returned.
        Defaults to True.
    :param search_mode: The search strategy to use. One of "tags",
        "regex", or "bm25". Defaults to "tags".
    :param top_k: Maximum number of results to return for "bm25" mode.
        Defaults to 10. Ignored for other modes.
    :returns: A dict containing detailed information about the matched
        operators, keyed by operator name.
    """
    if search_mode == "regex":
        if not query:
            return {"error": "query is required for regex search mode"}
        op_results = searcher.search_by_regex(
            query=query,
            tags=tags,
            op_type=op_type,
            match_all=match_all,
        )
    elif search_mode == "bm25":
        if not query:
            return {"error": "query is required for bm25 search mode"}
        op_results = searcher.search_by_bm25(
            query=query,
            top_k=top_k,
            tags=tags,
            op_type=op_type,
            match_all=match_all,
        )
    else:
        # Default "tags" mode: filter by op_type and tags
        op_results = searcher.search(tags=tags, op_type=op_type, match_all=match_all)

    ops_dict = {}
    for op in op_results:
        ops_dict[op["name"]] = "\n".join([op["desc"], op["param_desc"], "Parameters: ", str(op["sig"])])

    return ops_dict


def run_data_recipe(
    process: list[Dict],
    dataset_path: Optional[str] = None,
    dataset: Optional[Dict] = None,
    export_path: Optional[str] = None,
    np: int = 1,
    extra_config: Optional[Dict] = None,
) -> str:
    """
    Run a data processing recipe using Data-Juicer operators.

    If you want to run one or more DataJuicer data processing operators,
    use this tool. Supported operators and their arguments should be
    obtained through the `search_ops` tool.

    For advanced configuration options (e.g., enabling tracing, op fusion,
    checkpoint, multimodal keys, etc.), first call `get_global_config_schema`
    to discover available options, then pass them via `extra_config`.

    For loading datasets from different sources (e.g., HuggingFace, S3),
    first call `get_dataset_load_strategies` to discover available loading
    strategies and their required fields, then pass the configuration via
    the `dataset` parameter.

    :param process: List of processing operations to be executed
        sequentially. Each element is a dictionary with operator name as
        key and its configuration as value.
    :param dataset_path: Path to the dataset to be processed. This is the
        simplest way to specify input data (local file path).
    :param dataset: Optional dataset configuration dict for advanced data
        loading. Supports multiple data sources (local, HuggingFace, S3,
        etc.). Format follows Data-Juicer's dataset config schema:
        {"configs": [{"type": "local", "path": "..."}, ...],
         "max_sample_num": 10000}
        Use `get_dataset_load_strategies` to discover available options.
        When provided alongside dataset_path, both are passed to
        Data-Juicer (dataset_path serves as a fallback).
    :param export_path: Path to export the processed dataset. Defaults to
        None, which exports to './outputs' directory.
    :param np: Number of processes to use. Defaults to 1.
    :param extra_config: Optional dict of additional global configuration
        options. Use `get_global_config_schema` to discover all available
        options. Example: {"open_tracer": true, "trace_num": 20,
        "op_fusion": true, "text_keys": "instruction"}

    Example:
        # Basic usage: filter text samples
        >>> run_data_recipe(
        ...     "/path/to/dataset.jsonl",
        ...     [{"text_length_filter": {"min_len": 10, "max_len": 50}}]
        ... )

        # Advanced usage with tracing and HuggingFace dataset
        >>> run_data_recipe(
        ...     dataset_path="",
        ...     process=[{"language_id_score_filter": {"lang": "en"}}],
        ...     dataset={
        ...         "configs": [{
        ...             "type": "huggingface",
        ...             "path": "tatsu-lab/alpaca",
        ...             "split": "train"
        ...         }]
        ...     },
        ...     extra_config={
        ...         "open_tracer": True,
        ...         "trace_num": 20,
        ...         "text_keys": "instruction"
        ...     }
        ... )
    """
    dj_cfg = {
        "dataset_path": dataset_path,
        "process": process,
        "export_path": export_path,
        "np": np,
    }

    if dataset is not None:
        dj_cfg["dataset"] = dataset

    if extra_config is not None:
        for key, value in extra_config.items():
            dj_cfg[key] = value

    return execute_op(dj_cfg)


def analyze_dataset(
    process: list[Dict],
    dataset_path: Optional[str] = None,
    dataset: Optional[Dict] = None,
    export_path: Optional[str] = None,
    np: int = 1,
    percentiles: Optional[List[float]] = None,
    extra_config: Optional[Dict] = None,
) -> str:
    """
    Analyze a dataset using Data-Juicer's Analyzer pipeline.

    This tool computes statistics for the specified filter and tagging
    operators on the dataset, then performs overall analysis, column-wise
    analysis, and correlation analysis. It generates stats tables and
    distribution figures to help understand the dataset characteristics
    before applying actual data processing.

    This is the equivalent of the ``dj-analyze`` command. Use it to
    understand your dataset's quality distribution, identify outliers,
    and determine appropriate filter thresholds before running
    ``run_data_recipe``.

    Supported operators and their arguments should be obtained through
    the ``search_ops`` tool. Only filter-type and tagging-type operators
    will produce meaningful analysis results.

    :param process: List of filter/tagging operations to compute stats
        for. Each element is a dictionary with operator name as key and
        its configuration as value. Only filter and tagging operators
        produce analysis stats.
    :param dataset_path: Path to the dataset to be analyzed. This is the
        simplest way to specify input data (local file path).
    :param dataset: Optional dataset configuration dict for advanced data
        loading. Same format as in run_data_recipe.
    :param export_path: Path to export the analyzed dataset with stats.
        Defaults to None, which exports to './outputs' directory.
    :param np: Number of processes to use. Defaults to 1.
    :param percentiles: List of percentiles to compute for the dataset
        distribution analysis. Defaults to [0.25, 0.5, 0.75].
    :param extra_config: Optional dict of additional global configuration
        options. Use ``get_global_config_schema`` to discover all available
        options. Analysis-specific options include:
        - export_original_dataset (bool): whether to export the original
          dataset with stats (default: False)
        - save_stats_in_one_file (bool): whether to save all stats into
          one file (default: False)

    Example:
        # Analyze text length and language distribution
        >>> analyze_dataset(
        ...     dataset_path="/path/to/dataset.jsonl",
        ...     process=[
        ...         {"text_length_filter": {"min_len": 10, "max_len": 1000}},
        ...         {"language_id_score_filter": {"lang": "en"}}
        ...     ],
        ...     percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        ... )

    :returns: A message indicating where the analysis results are saved,
        including the export path and the analysis directory containing
        stats tables and distribution figures.
    """
    dj_cfg = {
        "dataset_path": dataset_path,
        "process": process,
        "export_path": export_path,
        "np": np,
    }

    if percentiles is not None:
        dj_cfg["percentiles"] = percentiles

    if dataset is not None:
        dj_cfg["dataset"] = dataset

    if extra_config is not None:
        for key, value in extra_config.items():
            dj_cfg[key] = value

    return execute_analyze(dj_cfg)


def create_mcp_server(port: str = "8000"):
    """
    Creates the FastMCP server and registers the tools.

    Args:
        port (str, optional): Port number. Defaults to "8000".
    """
    mcp = fastmcp.FastMCP("Data-Juicer Server", port=port)

    mcp.tool()(get_global_config_schema)
    mcp.tool()(get_dataset_load_strategies)
    mcp.tool()(search_ops)
    mcp.tool()(run_data_recipe)
    mcp.tool()(analyze_dataset)

    return mcp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data-Juicer MCP Server")
    parser.add_argument(
        "--port",
        type=str,
        default="8000",
        help="Port number for the MCP server",
    )
    args = parser.parse_args()

    mcp = create_mcp_server(port=args.port)
    mcp.run(transport=os.getenv("SERVER_TRANSPORT", "sse"))
