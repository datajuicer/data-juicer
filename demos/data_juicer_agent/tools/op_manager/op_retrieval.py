# -*- coding: utf-8 -*-
import os
import os.path as osp
import json
import logging
import pickle
import hashlib
import time
from typing import Optional

from langchain_community.vectorstores import FAISS

TOOLS_INFO_PATH = osp.join(osp.dirname(__file__), "dj_funcs_all.json")
CACHE_RETRIEVED_TOOLS_PATH = osp.join(osp.dirname(__file__), "cache_retrieve")
VECTOR_INDEX_CACHE_PATH = osp.join(osp.dirname(__file__), "vector_index_cache")

# Global variable to cache the vector store
_cached_vector_store: Optional[FAISS] = None
_cached_tools_info: Optional[list] = None
_cached_file_hash: Optional[str] = None

RETRIEVAL_PROMPT = """You are a professional tool retrieval assistant responsible for filtering the top {limit} most relevant tools from a large tool library based on user requirements. Execute the following steps:

# Requirement Analysis
    Carefully read the user's [requirement description], extract core keywords, functional objectives, usage scenarios, and technical requirements (such as real-time performance, data types, industry domains, etc.).

# Tool Matching
    Perform multi-dimensional matching based on the following tool attributes:
    - Tool name and functional description
    - Supported input/output formats
    - Applicable industry or scenario tags
    - Technical implementation principles (API, local deployment, AI model types)
    - Relevance ranking

# Use weighted scoring mechanism (example weights):
    - Functional match (40%)
    - Scenario compatibility (30%)
    - Technical compatibility (20%)
    - User rating/usage rate (10%)

# Deduplication and Optimization
    Exclude the following low-quality results:
    - Tools with duplicate functionality (keep only the best one)
    - Tools that cannot meet basic requirements
    - Tools missing critical parameter descriptions

# Constraints
    - Strictly control output to a maximum of {limit} tools
    - Refuse to speculate on unknown tool attributes
    - Maintain accuracy of domain expertise

# Output Format
    Return a JSON format TOP{limit} tool list containing:
    [
        {{
            "rank": 1,
            "tool_name": "Tool Name",
            "description": "Core functionality summary",
            "relevance_score": 98.7,
            "key_match": ["Matching keywords/features"]
        }}
    ]
    Output strictly in JSON array format, and only output the JSON array format tool list.
"""


def fast_text_encoder(text: str) -> str:
    """Fast encoding using xxHash algorithm"""
    import xxhash

    hasher = xxhash.xxh64(seed=0)
    hasher.update(text.encode("utf-8"))

    # Return 16-bit hexadecimal string
    return hasher.hexdigest()


async def retrieve_ops_lm(user_query, limit=20):
    """Tool retrieval using language model - returns list of tool names"""
    hash_id = fast_text_encoder(user_query + str(limit))

    # Ensure cache directory exists
    os.makedirs(CACHE_RETRIEVED_TOOLS_PATH, exist_ok=True)

    cache_tools_path = osp.join(CACHE_RETRIEVED_TOOLS_PATH, f"{hash_id}.json")
    if osp.exists(cache_tools_path):
        with open(cache_tools_path, "r", encoding="utf-8") as f:
            return json.loads(f.read())

    if osp.exists(TOOLS_INFO_PATH):
        with open(TOOLS_INFO_PATH, "r", encoding="utf-8") as f:
            dj_func_info = json.loads(f.read())
            tool_descriptions = [
                f"{t['class_name']}: {t['class_desc']}" for t in dj_func_info
            ]
            tools_string = "\n".join(tool_descriptions)
    else:
        from create_dj_func_info import dj_func_info

        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), ".."),
        )

        with open(os.path.join(project_root, TOOLS_INFO_PATH), "w") as f:
            f.write(json.dumps(dj_func_info))

        tool_descriptions = [
            f"{t['class_name']}: {t['class_desc']}" for t in dj_func_info
        ]
        tools_string = "\n".join(tool_descriptions)

    from agentscope.model import DashScopeChatModel
    from agentscope.message import Msg
    from agentscope.formatter import DashScopeChatFormatter

    model = DashScopeChatModel(
        model_name="qwen-turbo",
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        stream=False,
    )

    formatter = DashScopeChatFormatter()

    # Update retrieval prompt to use the specified limit
    retrieval_prompt_with_limit = RETRIEVAL_PROMPT.format(limit=limit)

    user_prompt = (
        retrieval_prompt_with_limit
        + """
User requirement description:
{user_query}

Available tools:
{tools_string}
""".format(
            user_query=user_query,
            tools_string=tools_string,
        )
    )

    msgs = [
        Msg(name="user", role="user", content=user_prompt),
    ]

    formatted_msgs = await formatter.format(msgs)

    response = await model(formatted_msgs)

    msg = Msg(name="assistant", role="assistant", content=response.content)
    retrieved_tools_text = msg.get_text_content()
    retrieved_tools = json.loads(retrieved_tools_text)

    # Extract tool names and validate they exist
    tool_names = []
    for tool_info in retrieved_tools:
        if not isinstance(tool_info, dict) or "tool_name" not in tool_info:
            logging.warning(f"Invalid tool info format: {tool_info}")
            continue

        tool_name = tool_info["tool_name"]

        # Verify tool exists in dj_func_info
        tool_exists = any(t["class_name"] == tool_name for t in dj_func_info)
        if not tool_exists:
            logging.error(f"Tool not found: `{tool_name}`, skipping!")
            continue

        tool_names.append(tool_name)

    # Cache the result
    with open(cache_tools_path, "w", encoding="utf-8") as f:
        json.dump(tool_names, f)

    return tool_names


def _get_file_hash(file_path: str) -> str:
    """Get file content hash using SHA256"""
    try:
        with open(file_path, "rb") as f:
            file_content = f.read()
            return hashlib.sha256(file_content).hexdigest()
    except (OSError, IOError):
        return ""


def _load_cached_index() -> bool:
    """Load cached vector index from disk"""
    global _cached_vector_store, _cached_tools_info, _cached_file_hash

    try:
        # Ensure cache directory exists
        os.makedirs(VECTOR_INDEX_CACHE_PATH, exist_ok=True)

        index_path = osp.join(VECTOR_INDEX_CACHE_PATH, "faiss_index")
        metadata_path = osp.join(VECTOR_INDEX_CACHE_PATH, "metadata.json")

        if not all(os.path.exists(p) for p in [index_path, metadata_path]):
            return False

        # Check if cached index matches current tools info file
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        cached_hash = metadata.get("tools_info_hash", "")
        current_hash = _get_file_hash(TOOLS_INFO_PATH)

        if current_hash != cached_hash:
            return False

        # Load cached data
        from langchain_community.embeddings import DashScopeEmbeddings

        embeddings = DashScopeEmbeddings(
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
            model="text-embedding-v1",
        )

        _cached_vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        _cached_file_hash = cached_hash

        logging.info("Successfully loaded cached vector index")
        return True

    except Exception as e:
        logging.warning(f"Failed to load cached index: {e}")
        return False


def _save_cached_index():
    """Save vector index to disk cache"""
    global _cached_vector_store, _cached_file_hash

    try:
        # Ensure cache directory exists
        os.makedirs(VECTOR_INDEX_CACHE_PATH, exist_ok=True)

        index_path = osp.join(VECTOR_INDEX_CACHE_PATH, "faiss_index")
        metadata_path = osp.join(VECTOR_INDEX_CACHE_PATH, "metadata.json")

        # Save vector store
        if _cached_vector_store:
            _cached_vector_store.save_local(index_path)

        # Save metadata
        metadata = {
            "tools_info_hash": _cached_file_hash,
            "created_at": time.time(),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        logging.info("Successfully saved vector index to cache")

    except Exception as e:
        logging.error(f"Failed to save cached index: {e}")


def _build_vector_index():
    """Build and cache vector index"""
    global _cached_vector_store, _cached_file_hash

    with open(TOOLS_INFO_PATH, "r", encoding="utf-8") as f:
        tools_info = json.loads(f.read())

    tool_descriptions = [
        f"{t['class_name']}: {t['class_desc']}" for t in tools_info
    ]

    from langchain_community.embeddings import DashScopeEmbeddings

    embeddings = DashScopeEmbeddings(
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        model="text-embedding-v1",
    )

    metadatas = [{"index": i} for i in range(len(tool_descriptions))]
    vector_store = FAISS.from_texts(
        tool_descriptions,
        embeddings,
        metadatas=metadatas,
    )

    # Cache the results
    _cached_vector_store = vector_store
    _cached_file_hash = _get_file_hash(TOOLS_INFO_PATH)

    # Save to disk cache
    _save_cached_index()

    logging.info("Successfully built and cached vector index")


def retrieve_ops_vector(user_query, limit=20):
    """Tool retrieval using vector search with caching - returns list of tool names"""
    global _cached_vector_store

    # Try to load from cache first
    if not _load_cached_index():
        logging.info("Building new vector index...")
        _build_vector_index()

    # Perform similarity search
    retrieved_tools = _cached_vector_store.similarity_search(
        user_query,
        k=limit,
    )
    retrieved_indices = [doc.metadata["index"] for doc in retrieved_tools]

    with open(TOOLS_INFO_PATH, "r", encoding="utf-8") as f:
        tools_info = json.loads(f.read())

    # Extract tool names from retrieved indices
    tool_names = []
    for raw_idx in retrieved_indices:
        tool_info = tools_info[raw_idx]
        tool_names.append(tool_info["class_name"])

    return tool_names


async def retrieve_ops(
    user_query: str,
    limit: int = 20,
    mode: str = "auto",
) -> list:
    """
    Tool retrieval with configurable mode

    Args:
        user_query: User query string
        limit: Maximum number of tools to retrieve
        mode: Retrieval mode - "llm", "vector", or "auto" (default: "auto")
              - "llm": Use language model only
              - "vector": Use vector search only
              - "auto": Try LLM first, fallback to vector search on failure

    Returns:
        List of tool names
    """
    if mode == "llm":
        try:
            return await retrieve_ops_lm(user_query, limit=limit)
        except Exception as e:
            logging.error(f"LLM retrieval failed: {str(e)}")
            return []

    elif mode == "vector":
        try:
            return retrieve_ops_vector(user_query, limit=limit)
        except Exception as e:
            logging.error(f"Vector retrieval failed: {str(e)}")
            return []

    elif mode == "auto":
        try:
            return await retrieve_ops_lm(user_query, limit=limit)
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            try:
                return retrieve_ops_vector(user_query, limit=limit)
            except Exception as fallback_e:
                logging.error(
                    f"Tool retrieval failed: {str(e)}, fallback retrieval also failed: {str(fallback_e)}",
                )
                return []

    else:
        raise ValueError(
            f"Invalid mode: {mode}. Must be 'llm', 'vector', or 'auto'",
        )


if __name__ == "__main__":
    import asyncio

    user_query = (
        "Clean special characters from text and filter samples with excessive length. Mask sensitive information and filter unsafe content including adult/terror-related terms."
        + "Additionally, filter out small images, perform image tagging, and remove duplicate images."
    )

    # Test different modes
    print("=== Testing LLM mode ===")
    tool_names_llm = asyncio.run(
        retrieve_ops(user_query, limit=10, mode="llm"),
    )
    print("Retrieved tool names (LLM):")
    print(tool_names_llm)

    print("\n=== Testing Vector mode ===")
    tool_names_vector = asyncio.run(
        retrieve_ops(user_query, limit=10, mode="vector"),
    )
    print("Retrieved tool names (Vector):")
    print(tool_names_vector)

    print("\n=== Testing Auto mode (default) ===")
    tool_names_auto = asyncio.run(
        retrieve_ops(user_query, limit=10, mode="auto"),
    )
    print("Retrieved tool names (Auto):")
    print(tool_names_auto)
