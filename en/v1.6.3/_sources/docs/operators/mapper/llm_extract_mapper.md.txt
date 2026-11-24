# llm_extract_mapper

Extract structured fields from text using an LLM; write results to meta. Part of the llm_* semantic ops family.

This operator uses an LLM to extract user-defined fields from each sample's text (or multiple input keys). You provide an `output_schema` (key → extraction instruction). Results are written to `meta[meta_output_key]` or to individual meta keys. Supports structured (JSON) and unstructured (e.g. plain text, jsonl) input. Token/cost usage is recorded in `meta[llm_semantic_usage]` (prompt_tokens, completion_tokens, total_tokens, optional cost_estimate).

使用 LLM 从文本中提取用户定义的结构化字段；结果写入 meta。支持结构化与非结构化输入，并记录 token/cost 用量。

Type 算子类型: **mapper**

Tags 标签: gpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `input_keys` | list | required | Sample keys to build input text (e.g. `["text"]` or `["query","response"]`). |
| `output_schema` | dict | required | `{output_key: "extraction instruction"}`. |
| `api_or_hf_model` | str | `'gpt-4o'` | Model name for API or HuggingFace. |
| `meta_output_key` | str, optional | `'llm_extract'` | If set, write full result to `meta[meta_output_key]`. |
| `knowledge_grounding_key` | str, optional | `None` | Optional sample key for per-sample grounding. |
| `knowledge_grounding_fixed` | str, optional | `None` | Optional fixed grounding string. |
| `is_hf_model` | bool | `False` | If true, use HuggingFace/Transformers. |
| `enable_vllm` | bool | `False` | If true, use vLLM backend. |
| `api_endpoint` | str, optional | `None` | URL endpoint for the API. |
| `response_path` | str, optional | `None` | Path to extract content from API response. |
| `system_prompt` | str, optional | `None` | Override default extraction system prompt. |
| `try_num` | int | `3` | Retries on parse/API failure. |
| `model_params` | dict | `{}` | Parameters for model init. |
| `sampling_params` | dict | `{}` | Sampling params (e.g. temperature, top_p). |

## 📊 Effect demonstration 效果演示

The examples below match the [unit tests](../../../tests/ops/mapper/test_llm_extract_mapper.py). **Concrete field values depend on the model and API**; only shape and keys are guaranteed.  
下列示例与单元测试场景一致；**具体抽取内容随模型与接口变化**，文档中仅示意典型结果。

### test_extract_default
```python
LLMExtractMapper(
    input_keys=["text"],
    output_schema={
        "topic": "One short phrase: main topic.",
        "sentiment": "One word: positive, negative, or neutral.",
    },
    api_or_hf_model="gpt-4o",
    meta_output_key="llm_extract",
    try_num=2,
)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The stock market rose today. Investors are optimistic.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Bad weather caused delays. Many people were upset.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The stock market rose today. Investors are optimistic.</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>llm_extract</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'><pre style="margin:0; white-space:pre-wrap;">{"topic": "stock market / finance", "sentiment": "positive"}</pre></td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>llm_semantic_usage</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'><pre style="margin:0; white-space:pre-wrap;">{"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}</pre></td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Bad weather caused delays. Many people were upset.</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>llm_extract</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'><pre style="margin:0; white-space:pre-wrap;">{"topic": "weather / travel disruption", "sentiment": "negative"}</pre></td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>llm_semantic_usage</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'><pre style="margin:0; white-space:pre-wrap;">{"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}</pre></td></tr></table></div></div>

#### ✨ explanation 解释
`dataset.map(op.process, batch_size=1)` fills `meta["llm_extract"]` with keys from `output_schema`. When the API returns usage, `meta["llm_semantic_usage"]` holds token counts (and optional cost).  
对每条样本调用 `map` 后，`meta` 中写入抽取结果；若接口返回用量，则同时写入 `llm_semantic_usage`。

### test_extract_empty_input
```python
LLMExtractMapper(
    input_keys=["text"],
    output_schema={"topic": "Main topic.", "sentiment": "Sentiment."},
    api_or_hf_model="gpt-4o",
    meta_output_key="llm_extract",
)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;"></pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;"></pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>llm_extract</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'><pre style="margin:0; white-space:pre-wrap;">{"topic": null, "sentiment": null}</pre></td></tr></table></div></div>

#### ✨ explanation 解释
Empty concatenated input skips the LLM call and sets all schema fields to `null` as in the unit test.  
空输入时不调用模型，各字段为 `null`，与单测一致。

## 🌐 DashScope / OpenAI-compatible 环境变量

使用阿里云 DashScope **OpenAI 兼容模式**（REST）时，可只配环境变量，无需在 recipe 里写 key：

```bash
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export OPENAI_API_KEY=<你的 DashScope API Key>
# 或使用：export DASHSCOPE_API_KEY=<同上>
```

默认算子模型名为 `gpt-4o`，在 DashScope 上不可用。若检测到上述兼容 Base URL，会自动改用 **`qwen-plus`**（可通过 `DASHSCOPE_DEFAULT_MODEL` 或 `OPENAI_DEFAULT_MODEL` 覆盖），或在配置里显式设置 `api_or_hf_model: qwen-plus`。

`OPENAI_API_URL` 与 `OPENAI_BASE_URL` 等价（任选其一）。

## 📊 Cost / usage 成本与用量

Each sample gets `meta[llm_semantic_usage]` with:
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `cost_estimate` (optional, when pricing is available)

## 🔗 Related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/llm_extract_mapper.py)
- [semantic ops 语义工具](../../../data_juicer/utils/llm_semantic_ops.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_llm_extract_mapper.py) (`TestLLMExtractMapper`)
- [Return operator list 返回算子列表](../../Operators.md)
