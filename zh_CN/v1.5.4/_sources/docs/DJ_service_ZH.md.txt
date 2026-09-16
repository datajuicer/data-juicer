# DJ_服务化
中文 | [English Page](DJ_service.md) 

为了进一步提升Data-Juicer用户体验，我们新增了基于 API 的服务功能（Service API）和 MCP 服务，使用户能够以更便捷的方式集成和使用 Data-Juicer 的强大算子池。通过服务功能，用户无需深入了解框架的底层实现细节，即可快速构建数据处理流水线，并与现有系统无缝对接。用户也可通过该服务实现不同project之间的环境隔离。本文档将详细介绍如何启动和使用这两种服务功能，帮助您快速上手并充分发挥 Data-Juicer 的潜力。

## API服务化
### 启动服务
执行如下代码：
```bash
uvicorn service:app
```

### API调用
API支持调用Data-Juicer所有`__init__.py`中的函数和类（调用类的某个函数）。函数调用GET调用，类通过POST调用。

#### 协议

#### url路径
采用GET调用函数，url路径与Data-Juicer库中函数引用路径一致，如`from data_juicer.config import init_configs`对应路径为`data_juicer/config/init_configs`。采用POST调用类的某个函数，url路径与Data-Juicer库中类路径拼接上函数名，如调用`TextLengthFIlter`算子的`compute_stats_batched`函数，对应路径为`data_juicer/ops/filter/TextLengthFilter/compute_stats_batched`。

##### 参数
进行GET和POST调用时，会自动将参数转化为list，同时，查询参数不支持字典传输。于是，如果传递的参数的value是list或dict，我们统一用`json.dumps`传输，并在前面添加特殊符号`<json_dumps>`与一般的string进行区分。

##### 特例
1. 针对`cfg`参数，我们默认采用`json.dumps`传输，无需添加特殊符号`<json_dumps>`。
2. 针对`dataset`参数，允许用户传输`dataset`在服务器上的路径，sever将加载dataset。
3. 允许用户设定`skip_return`参数，为`True`时将不返回函数调用的结果，避免一些网络无法传输带来的错误。

#### 函数调用
采用GET调用，url路径与Data-Juicer库中函数引用路径一致，查询参数用来传递函数的参数。

例如，可用如下python代码调用Data-Juicer的参数init函数`init_configs`，获取Data-Juicer所有参数。

```python
import requests
import json

json_prefix = '<json_dumps>'
url = 'http://localhost:8000/data_juicer/config/init_configs'
params = {"args": json_prefix + json.dumps(['--config', './demos/process_simple/process.yaml'])}
response = requests.get(url, params=params)
print(json.loads(response.text))
```

对应的curl代码如下：

```bash
curl -G "http://localhost:8000/data_juicer/config/init_configs" \
     --data-urlencode "args=--config" \
     --data-urlencode "args=./demos/process_simple/process.yaml"
```

#### 类的函数调用
采用POST调用，url路径与Data-Juicer库中类路径拼接上函数名，查询参数用来传递函数的参数，JSON字段用来传递类构造函数所需参数。

例如，可用如下python代码调用Data-Juicer的`TextLengthFIlter`算子。
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

对应的curl代码如下：

```bash
curl -X POST \
  "http://localhost:8000/data_juicer/ops/filter/TextLengthFilter/compute_stats_batched?samples=%3Cjson_dumps%3E%7B%22text%22%3A%20%5B%2212345%22%2C%20%22123%22%5D%2C%20%22__dj__stats__%22%3A%20%5B%7B%7D%2C%20%7B%7D%5D%7D" \
  -H "Content-Type: application/json" \
  -d '{"min_len": 4, "max_len": 10}'
```


注：如果需要调用`Executor`类或`Analyzer`类的`run`函数进行数据处理和数据分析，需要先调用`init_configs`或`get_init_configs`函数获取完整的Data-Juicer参数来构造这两个类。具体可参考如下演示。

### 演示
我们结合[AgentScope](https://github.com/agentscope-ai/agentscope)实现了用户通过自然语言调用Data-Juicer算子进行数据清洗的功能，算子采用API服务的方式进行调用。具体代码请参考[这里](../demos/api_service)。

## MCP服务器

### 概览

Data-Juicer MCP 服务器提供数据处理算子，以协助完成数据清洗、过滤、去重等任务。为了适应不同的使用场景，我们提供两种服务器供选用：

- **Recipe-Flow（数据菜谱）**：允许根据算子的类型和标签进行筛选，并支持将多个算子组合成一个数据菜谱来运行。
- **Granular-Operators（细粒度算子）**：将每个算子作为一个独立的工具提供，可以灵活地通过环境变量指定需要使用的算子列表，从而构建定制化的数据处理管道。

请注意，Data-Juicer MCP 服务器的功能和可用工具可能会随着我们继续开发和改进服务器而发生变化和扩展。

支持两种部署方式：stdio 和 streamable-http。stdio 方法不支持多进程。如果需要多进程或多线程功能，则必须使用 streamable-http 部署方法。以下提供了每种方法的配置详细信息。

### Recipe-Flow

Recipe-Flow 模式提供以下 MCP 工具：

#### 1. search_ops
- 搜索可用的数据处理算子，支持多种搜索模式
- 输入：
  - `query` (str, optional): 搜索查询字符串。"keyword" 和 "bm25" 模式下必填
  - `op_type` (str, optional): 要检索的数据处理算子类型（aggregator / deduplicator / filter / grouper / mapper / selector / pipeline）
  - `tags` (List[str], optional): 用于过滤算子的标签列表（模态标签：text / image / audio / video / multimodal；资源标签：cpu / gpu；模型标签：api / vllm / hf）
  - `match_all` (bool): 是否所有指定的标签都必须匹配。默认为 True
  - `search_mode` (str): 搜索策略，可选 "tags"（按类型和标签过滤，默认）、"regex"（正则表达式匹配）、"bm25"（BM25 文本相关性排序）
  - `top_k` (int): "bm25" 模式下返回的最大结果数。默认为 10
- 返回：包含匹配算子详细信息的字典

#### 2. get_global_config_schema
- 动态获取 Data-Juicer 所有可用的全局配置项的元信息（参数名、类型、默认值、描述）
- 无需输入参数
- 返回：配置项 schema 字典，可用于了解 `run_data_recipe` 和 `analyze_dataset` 的 `extra_config` 参数支持哪些配置选项

#### 3. get_dataset_load_strategies
- 动态获取 Data-Juicer 支持的所有数据集加载策略及其配置规则
- 无需输入参数
- 返回：各策略的 executor_type、data_type、data_source、必填/可选字段等信息，可用于了解 `run_data_recipe` 和 `analyze_dataset` 的 `dataset` 参数如何配置不同数据源（本地文件、HuggingFace、S3 等）

#### 4. run_data_recipe
- 执行数据处理菜谱（等价于 `dj-process`）
- 输入：
  - `process` (List[Dict]): 要执行的处理步骤列表，字典包含算子名称和参数字典
  - `dataset_path` (str, optional): 要处理的数据集路径（本地文件的简单模式）
  - `dataset` (Dict, optional): 高级数据集配置，支持多种数据源。格式：`{"configs": [{"type": "local", "path": "..."}], "max_sample_num": 10000}`。可通过 `get_dataset_load_strategies` 了解可用选项
  - `export_path` (str, optional): 导出数据集的路径，默认为 None，导出到 './outputs'
  - `np` (int): 使用的进程数。默认为 1
  - `extra_config` (Dict, optional): 额外的全局配置选项。可通过 `get_global_config_schema` 了解所有可用选项。例如：`{"open_tracer": true, "trace_num": 20, "op_fusion": true}`
- 返回：执行结果的字符串

#### 5. analyze_dataset
- 分析数据集的质量分布（等价于 `dj-analyze`）
- 对指定的 filter/tagging 算子计算统计信息，执行整体分析、列分析和相关性分析，生成统计表和分布图
- 输入：
  - `process` (List[Dict]): 要计算统计信息的 filter/tagging 算子列表
  - `dataset_path` (str, optional): 要分析的数据集路径
  - `dataset` (Dict, optional): 高级数据集配置（同 `run_data_recipe`）
  - `export_path` (str, optional): 导出分析结果的路径
  - `np` (int): 使用的进程数。默认为 1
  - `percentiles` (List[float], optional): 分布分析的百分位数列表。默认为 [0.25, 0.5, 0.75]
  - `extra_config` (Dict, optional): 额外的全局配置选项（同 `run_data_recipe`）
- 返回：分析结果保存路径的字符串

#### 典型工作流

1. **搜索算子**：调用 `search_ops` 查找合适的算子（支持按标签过滤、关键词搜索或 BM25 自然语言搜索）
2. **了解配置**：按需调用 `get_global_config_schema` 和 `get_dataset_load_strategies` 了解可用的配置选项和数据源
3. **分析数据集**：调用 `analyze_dataset` 分析数据集质量分布，确定合适的过滤阈值
4. **执行处理**：根据分析结果，调用 `run_data_recipe` 执行实际的数据处理

### Granular-Operators

默认情况下，该 MCP 服务器将返回所有Data-Juicer算子工具，每个工具都独立运行。

可通过指定环境变量 `DJ_OPS_LIST_PATH` 控制 MCP 服务器返回的算子工具：
1. 创建一个 `.txt` 文件
2. 将算子名称添加到文件中，例如：
```text
text_length_filter
flagged_words_filter
image_nsfw_filter
text_pair_similarity_filter
```

3. 将算子列表的路径设置为环境变量 `DJ_OPS_LIST_PATH`

### 配置

以下配置示例演示了如何使用 stdio 和 SSE 方法设置两种不同的 MCP 服务器。这些示例仅用于说明目的，应根据特定 MCP 客户端的配置格式进行调整。

#### stdio

适用于快速本地测试和简单场景。将以下内容添加到 MCP 客户端的配置文件中（例如，claude_desktop_config.json 或类似的配置文件）：

##### 使用 uvx

直接从存储库运行最新版本的 Data-Juicer MCP，无需手动进行本地安装。

- **Recipe-Flow模式**：
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

- **Granular-Operators模式**：
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
  注意：若不设置`DJ_OPS_LIST_PATH`，则默认返回所有算子。

##### 本地安装

1. 将 Data-Juicer 仓库克隆到本地：
   ```bash
   git clone https://github.com/datajuicer/data-juicer.git
   ```
2. 使用 uv 运行 Data-Juicer MCP：
- Recipe-Flow 模式:
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
- Granular-Operators 模式:
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


#### streamable-http

要使用 streamable-http 部署，首先需要单独启动 MCP 服务器。

1. 运行 MCP 服务器：执行 MCP 服务器脚本，指定端口号：
   - uvx 启动:
     ```bash
     uvx --from git+https://github.com/datajuicer/data-juicer dj-mcp <MODE: recipe-flow/granular-ops> --transport streamable-http --port 8080
     ```
   - 本地启动:
     ```bash
     uv run dj-mcp <MODE: recipe-flow/granular-ops> --transport streamable-http --port 8080
     ```

2. 配置您的 MCP 客户端：将以下内容添加到 MCP 客户端的配置文件中：
   ```json
   {
     "mcpServers": {
       "DJ_MCP": {
         "url": "http://127.0.0.1:8080/mcp"
       }
     }
   }
   ```

**注意事项**：
- URL：`url` 应指向正在运行的服务器的 mcp 端点（通常为 `http://127.0.0.1:<port>/mcp`）。如果在启动服务器时使用了不同的值，请调整端口号。
- 单独的服务器进程：streamable-http 服务器必须在 MCP 客户端尝试连接之前运行。
- 防火墙：确保防火墙允许连接到指定的端口。