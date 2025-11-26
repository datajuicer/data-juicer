# DJ_service
English | [中文页面](DJ_service_ZH.md)

To further enhance the user experience of Data-Juicer, we have introduced a new service features based on API (Service API) and a MCP server, This enables users to integrate and utilize Data-Juicer's powerful operator pool in a more convenient way. With these service features, users can quickly build data processing pipelines without needing to delve into the underlying implementation details of the framework, and seamlessly integrate with existing systems. Additionally, users can achieve environment isolation between different projects through this service. This document will provide a detailed explanation of how to launch and use this service feature, helping you get started quickly and fully leverage the potential of Data-Juicer.

## API Service
### Start the Service
Run the following code:
```bash
uvicorn service:app
```

### API Calls
The API supports calling all functions and classes in Data-Juicer's `__init__.py` (including calling specific functions of a class). Functions are called via GET, while classes are called via POST.

#### Protocol

##### URL Path
For GET requests to call functions, the URL path corresponds to the function reference path in the Data-Juicer library. For example, `from data_juicer.config import init_configs` maps to the path `data_juicer/config/init_configs`. For POST requests to call a specific function of a class, the URL path is constructed by appending the function name to the class path in the Data-Juicer library. For instance, calling the `compute_stats_batched` function of the `TextLengthFilter` operator corresponds to the path `data_juicer/ops/filter/TextLengthFilter/compute_stats_batched`.

##### Parameters
When making GET and POST calls, parameters are automatically converted into lists. Additionally, query parameters do not support dictionary transmission. Therefore, if the value of a parameter is a list or dictionary, we uniformly transmit it using `json.dumps` and prepend a special symbol `<json_dumps>` to distinguish it from regular strings.

##### Special Cases
1. For the `cfg` parameter, we default to transmitting it using `json.dumps`, without needing to prepend the special symbol `<json_dumps>`.
2. For the `dataset` parameter, users can pass the path of the dataset on the server, and the server will load the dataset.
3. Users can set the `skip_return` parameter. When set to `True`, the result of the function call will not be returned, avoiding errors caused by network transmission issues.

#### Function Calls
GET requests are used for function calls, with the URL path corresponding to the function reference path in the Data-Juicer library. Query parameters are used to pass the function arguments.

For example, the following Python code can be used to call Data-Juicer's `init_configs` function to retrieve all parameters of Data-Juicer:

```python
import requests
import json

json_prefix = '<json_dumps>'
url = 'http://localhost:8000/data_juicer/config/init_configs'
params = {"args": json_prefix + json.dumps(['--config', './demos/process_simple/process.yaml'])}
response = requests.get(url, params=params)
print(json.loads(response.text))
```

The corresponding curl command is as follows:

```bash
curl -G "http://localhost:8000/data_juicer/config/init_configs" \
     --data-urlencode "args=--config" \
     --data-urlencode "args=./demos/process_simple/process.yaml"
```

#### Class Function Calls
POST requests are used for class function calls, with the URL path constructed by appending the function name to the class path in the Data-Juicer library. Query parameters are used to pass the function arguments, while JSON fields are used to pass the arguments required for the class constructor.

For example, the following Python code can be used to call Data-Juicer's `TextLengthFilter` operator:

```python
import requests
import json

json_prefix = '<json_dumps>'
url = 'http://localhost:8000/data_juicer/ops/filter/TextLengthFilter/compute_stats_batched'
params = {'samples': json_prefix + json.dumps({'text': ['12345', '123'], '__dj__stats__': [{}, {}]})}
init_json = {'min_len': 4, 'max_len': 10}
response = requests.post(url, params=params, json=init_json)
print(json.loads(response.text))
```

The corresponding curl command is as follows:

```bash
curl -X POST \
  "http://localhost:8000/data_juicer/ops/filter/TextLengthFilter/compute_stats_batched?samples=%3Cjson_dumps%3E%7B%22text%22%3A%20%5B%2212345%22%2C%20%22123%22%5D%2C%20%22__dj__stats__%22%3A%20%5B%7B%7D%2C%20%7B%7D%5D%7D" \
  -H "Content-Type: application/json" \
  -d '{"min_len": 4, "max_len": 10}'
```

**Note**: If you need to call the `run` function of the `Executor` or `Analyzer` classes for data processing and data analysis, you must first call the `init_configs` or `get_init_configs` function to obtain the complete Data-Juicer parameters to construct these two classes. For more details, refer to the demonstration below.

### Demonstration
We have integrated [AgentScope](https://github.com/agentscope-ai/agentscope) to enable users to invoke Data-Juicer operators for data cleaning through natural language. The operators are invoked via an API service. For the specific code, please refer to [here](../demos/api_service).

## MCP Server

### Overview

The Data-Juicer MCP server provides data processing operators to assist in tasks such as data cleaning, filtering, deduplication, and more. To accommodate different use cases, we offer two server options:

- Recipe-Flow: Allows filtering operators by type and tags, and supports combining multiple operators into a data recipe for execution.
- Granular-Operators: Provides each operator as an independent tool, allowing you to flexibly specify a list of operators to use via environment variables, thus building a customized data processing pipeline.

Please note that the Data-Juicer MCP server is currently in early development. Its features and available tools may change and expand as we continue to develop and improve the server.

The server supports two deployment methods: **stdio** and **SSE**. The **stdio** method does not support multiprocessing. If you require multiprocessing or multithreading capabilities, you must use the **SSE** deployment method. Configuration details for each method are provided below.

### Recipe-Flow

1. `get_data_processing_ops`
   - Retrieves a list of available data processing operators based on the specified type and tags (if unspecified, returns all operators)
   - Input:
     - `op_type` (str, optional): The type of data processing operator to retrieve
     - `tags` (List[str], optional): A list of tags to filter operators
     - `match_all` (bool): Whether all specified tags must match. Default is True
   - Returns: A dictionary containing details about the available operators

2. `run_data_recipe`
   - Executes a data recipe
   - Input:
     - `dataset_path` (str): The path to the dataset to be processed
     - `process` (List[Dict]): A list of processing steps to execute, where each dictionary contains an operator name and a parameter dictionary
     - `export_path` (str, optional): The path to export the dataset to. Default is None, meaning the dataset will be exported to './outputs'
   - Returns: A string representing the execution result

For specific data processing requests, the MCP client should first call `get_data_processing_ops` to obtain relevant operator information, select the operators that meet the requirements, and then call `run_data_recipe` to execute the chosen operator combination.

### Granular-Operators

By default, this MCP server returns all Data-Juicer operator tools, each running independently.

To control the operator tools returned by the MCP server, specify the environment variable `DJ_OPS_LIST_PATH`:
1. Create a `.txt` file.
2. Add operator names to the file. For example:
```text
text_length_filter
flagged_words_filter
image_nsfw_filter
text_pair_similarity_filter
```
3. Set the path to the operator list as the environment variable `DJ_OPS_LIST_PATH`.

### Configuration

The following configuration examples demonstrate how to set up the two MCP server types using the stdio and SSE methods. These examples are for illustrative purposes only and should be adapted to the specific MCP client's configuration format.

#### stdio

Suitable for quick local testing and simple scenarios. Add the following to the MCP client's configuration file (e.g., `claude_desktop_config.json` or similar):

##### Using uvx

Run the latest version of Data-Juicer MCP directly from the repository without manual local installation.

- Recipe-Flow mode:
  ```json
  {
    "mcpServers": {
      "DJ_recipe_flow": {
        "command": "uvx",
        "args": [
          "--from",
          "git+https://github.com/datajuicer/data-juicer",
          "dj-mcp",
          "recipe-flow"
        ]
      }
    }
  }
  ```

- Granular-Operators mode:
  ```json
  {
    "mcpServers": {
      "DJ_granular_ops": {
        "command": "uvx",
        "args": [
          "--from",
          "git+https://github.com/datajuicer/data-juicer",
          "dj-mcp",
          "granular-ops",
          "--transport",
          "stdio"
        ],
        "env": {
          "DJ_OPS_LIST_PATH": "/path/to/ops_list.txt"
        }
      }
    }
  }
  ```
  Note: If `DJ_OPS_LIST_PATH` is not set, all operators are returned by default.

##### Local Installation

1. Clone the Data-Juicer repository locally:
   ```bash
   git clone https://github.com/datajuicer/data-juicer.git
   ```
2. Run Data-Juicer MCP using uv:
- Recipe-Flow mode:
  ```json
  {
    "mcpServers": {
      "DJ_recipe_flow": {
        "transport": "stdio",
        "command": "uv",
        "args": [
          "run",
          "--directory",
          "/abs/path/to/data-juicer",
          "dj-mcp",
          "recipe-flow"
        ]
      }
    }
  }
  ```
- Granular-Operators mode:
  ```json
  {
    "mcpServers": {
      "DJ_granular_ops": {
        "transport": "stdio",
        "command": "uv",
        "args": [
          "run",
          "--directory",
          "/abs/path/to/data-juicer",
          "dj-mcp",
          "granular-ops"
        ],
        "env": {
          "DJ_OPS_LIST_PATH": "/path/to/ops_list.txt"
        }
      }
    }
  }
  ```

#### SSE

To use SSE deployment, first start the MCP server separately.

1. Run the MCP server: Execute the MCP server script and specify the port number:
   - Using uvx:
     ```bash
     uvx --from git+https://github.com/datajuicer/data-juicer dj-mcp <MODE: recipe-flow/granular-ops> --transport sse --port 8080
     ```
   - Local execution:
     ```bash
     uv run dj-mcp <MODE: recipe-flow/granular-ops> --transport sse --port 8080
     ```

2. Configure your MCP client: Add the following to the MCP client's configuration file:
   ```json
   {
     "mcpServers": {
       "DJ_MCP": {
         "url": "http://127.0.0.1:8080/sse"
       }
     }
   }
   ```

Notes:
- URL: The `url` should point to the SSE endpoint of the running server (typically `http://127.0.0.1:<port>/sse`). Adjust the port number if a different value was used when starting the server.
- Separate server process: The SSE server must be running before the MCP client attempts to connect.
- Firewall: Ensure the firewall allows connections to the specified port.