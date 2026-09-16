# agent_insight_llm_mapper

Synthesize stats + LLM eval text into ``meta.agent_insight_llm`` (JSON).

Intended to run **after** filters/mappers that populate ``stats`` and
``agent_bad_case_signal_mapper``. Use ``run_for_tiers`` to limit API cost.

Output is best-effort JSON; raw model text is stored in
``meta.agent_insight_llm_raw`` if parsing fails.

将 stats 和 LLM eval 文本综合到 ``meta.agent_insight_llm``（JSON）中。

旨在填充 ``stats`` 和
``agent_bad_case_signal_mapper`` 的 filter/mapper 之**后**运行。使用 ``run_for_tiers`` 来限制 API 成本。

输出为尽力生成的 JSON；如果解析失败，原始模型文本将存储在
``meta.agent_insight_llm_raw`` 中。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | str | `'gpt-4o'` |  |
| `api_endpoint` | Optional[str] | `None` |  |
| `response_path` | Optional[str] | `None` |  |
| `system_prompt` | Optional[str] | `None` |  |
| `query_key` | str | `'query'` |  |
| `response_key` | str | `'response'` |  |
| `query_preview_max_chars` | int | `500` |  |
| `response_preview_max_chars` | int | `500` |  |
| `run_for_tiers` | Optional[List[str]] | `None` |  |
| `try_num` | PositiveInt | `2` |  |
| `model_params` | Dict | `{}` |  |
| `sampling_params` | Dict | `{}` |  |
| `preferred_output_lang` | str | `'en'` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/agent_insight_llm_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_agent_insight_llm_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)