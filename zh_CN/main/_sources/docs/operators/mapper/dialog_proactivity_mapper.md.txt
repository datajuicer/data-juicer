# dialog_proactivity_mapper

Balance helpful initiative against rambling or filler.

平衡有益的主动性与冗长或废话。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | str | `'qwen-turbo'` |  |
| `api_endpoint` | Optional[str] | `None` |  |
| `response_path` | Optional[str] | `None` |  |
| `history_key` | str | `'dialog_history'` |  |
| `query_key` | str | `'query'` |  |
| `response_key` | str | `'response'` |  |
| `text_key` | str | `'text'` |  |
| `max_round` | NonNegativeInt | `8` |  |
| `max_query_chars_for_prompt` | NonNegativeInt | `6000` |  |
| `max_response_chars_for_prompt` | NonNegativeInt | `8000` |  |
| `trajectory_text_max_chars` | NonNegativeInt | `12000` |  |
| `tool_types_key` | str | `'agent_tool_types'` |  |
| `primary_tool_key` | str | `'primary_tool_type'` |  |
| `try_num` | PositiveInt | `2` |  |
| `overwrite` | bool | `False` |  |
| `model_params` | Optional[Dict] | `None` |  |
| `sampling_params` | Optional[Dict] | `None` |  |
| `preferred_output_lang` | str | `'en'` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/dialog_proactivity_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_dialog_quality_llm.py)
- [Return operator list 返回算子列表](../../Operators.md)