# llm_condition_filter

Filter by user-given natural language condition (LLM yes/no). Part of the llm_* semantic ops family.

This operator uses an LLM to decide whether each sample satisfies a free-form condition string (e.g. "contains a question", "is in formal tone"). It writes the result to `stats.llm_condition_filter_result` and keeps samples where the result is True. Token/cost usage is recorded in `stats.llm_semantic_usage`.

按用户给定的自然语言条件进行过滤（LLM 回答是/否）；满足条件的样本保留，并在 stats 中记录用量。

Type 算子类型: **filter**

Tags 标签: gpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `text_key` | str | `'text'` | Sample key for the text to evaluate. |
| `condition` | str | `''` | Natural language condition (e.g. "contains a question"). |
| `api_or_hf_model` | str | `'gpt-4o'` | Model name. |
| `knowledge_grounding_key` | str, optional | `None` | Optional sample key for per-sample grounding. |
| `knowledge_grounding_fixed` | str, optional | `None` | Optional fixed grounding string. |
| `is_hf_model` | bool | `False` | If true, use HuggingFace. |
| `enable_vllm` | bool | `False` | If true, use vLLM. |
| `api_endpoint` | str, optional | `None` | API endpoint. |
| `response_path` | str, optional | `None` | Path to extract content from API response. |
| `try_num` | int | `3` | Retries on API/parse failure; treat as False after all fail. |
| `model_params` | dict | `{}` | Model init params. |
| `sampling_params` | dict | `{}` | Sampling params. |

## 📊 Effect demonstration 效果演示

Workflow in tests: ensure `stats` column exists → `dataset.map(op.compute_stats, …)` → `dataset.filter(op.process, …)`. **LLM judgments are illustrative**; only the presence of `stats` keys and filter behavior are fixed.  
与单测一致：先写 `stats` 再 `map` 再 `filter`。**是否保留某条样本由模型判断**，下表为典型示意。

### test_condition_question
```python
LLMConditionFilter(
    text_key="text",
    condition="The text contains a clear question.",
    api_or_hf_model="gpt-4o",
    try_num=2,
)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">What is the capital of France?</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The capital of France is Paris.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">How do I install Python?</pre></div>

#### 📤 output data 输出数据
After `filter`, only samples with `stats.llm_condition_filter_result is True` remain (typically the two question sentences; exact count may vary).  
`filter` 之后仅保留判断为真的样本（通常为含明确问句的条目，具体条数依模型而定）。

<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">What is the capital of France?</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>stats</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>llm_condition_filter_result</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>True</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>llm_semantic_usage</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'><pre style="margin:0; white-space:pre-wrap;">{"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}</pre></td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">How do I install Python?</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>stats</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>llm_condition_filter_result</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>True</td></tr></table></div></div>

#### ✨ explanation 解释
`compute_stats` writes `llm_condition_filter_result` and optional `llm_semantic_usage`; `process` drops rows where the flag is `False`.  
先 `map(compute_stats)` 写入布尔结果与用量，再 `filter(process)` 去掉不满足条件的样本。

### test_empty_text
```python
LLMConditionFilter(
    text_key="text",
    condition="The text is non-empty.",
    api_or_hf_model="gpt-4o",
)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;"></pre></div>

#### 📤 output data 输出数据
After `map`, `stats.llm_condition_filter_result` is `False`; after `filter`, the dataset has **0** rows.  
`map` 后结果为 `False`，`filter` 后数据集为空。

#### ✨ explanation 解释
Empty `text` avoids calling the LLM; the condition is treated as not satisfied.  
空文本不调用模型，直接判为不满足。

### test_empty_condition
```python
LLMConditionFilter(text_key="text", condition="", api_or_hf_model="gpt-4o")
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Some content.</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Some content.</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>stats</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>llm_condition_filter_result</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>True</td></tr></table></div></div>

#### ✨ explanation 解释
When `condition` is empty, all samples pass without an LLM call (`result` is `True`).  
条件串为空时不调用 LLM，全部样本保留。

## 🌐 DashScope / OpenAI-compatible 环境变量

与 `llm_extract_mapper` 相同：设置 `OPENAI_BASE_URL`（或 `OPENAI_API_URL`）为 DashScope 兼容地址，以及 `OPENAI_API_KEY` 或 `DASHSCOPE_API_KEY`。检测到 DashScope 兼容端点时，默认的 `gpt-4o` 会映射为 `qwen-plus`，可用 `DASHSCOPE_DEFAULT_MODEL` 覆盖。

## 📊 Cost / usage 成本与用量

Each sample gets `stats.llm_semantic_usage` with:
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `cost_estimate` (optional)

## 🔗 Related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/llm_condition_filter.py)
- [semantic ops 语义工具](../../../data_juicer/utils/llm_semantic_ops.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_condition_filter.py) (`TestLLMConditionFilter`)
- [Return operator list 返回算子列表](../../Operators.md)
