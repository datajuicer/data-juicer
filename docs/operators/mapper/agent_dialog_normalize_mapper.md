# agent_dialog_normalize_mapper

Normalize agent format (messages + choices) to DJ fields.

Outputs: text, dialog_history, query, response; optionally meta tags
agent_tool_types, agent_skill_types, agent_turn_count. When
``copy_lineage_fields`` is True, also copies request_model, pt,
total_cost_time, and (when ``copy_request_id``) the first non-empty
id among ``request_id_keys`` from the sample root into meta for cohort
analysis and stable drill-down links. Always records last user/assistant
message indices (in the raw ``messages`` list) when present.
Supports multi-format tool_calls (e.g. tool_calls[].function.name as in
OpenAI / demos/local/demo-agent-data-content.json) and configurable
user/assistant labels.
Optional ``history_*_max_chars`` caps keep head+tail with an explicit
middle-omitted marker so ``dialog_history``, flattened ``text``, and last
``query`` / ``response`` stay aligned; ``meta.agent_dialog_history_compressed``
is set when any cap fires.

将 agent 格式（messages + choices）规范化为 DJ 字段。

输出：text、dialog_history、query、response；可选的 meta 标签包括
agent_tool_types、agent_skill_types、agent_turn_count。当
``copy_lineage_fields`` 为 True 时，还会将 request_model、pt、
total_cost_time 以及（当 ``copy_request_id`` 为 True 时）样本根目录中 ``request_id_keys`` 里的第一个非空
id 复制到 meta 中，用于队列分析和稳定的下钻链接。始终记录最后一条 user/assistant
消息的索引（在原始 ``messages`` 列表中，如果存在的话）。
支持多格式 tool_calls（例如 OpenAI / demos/local/demo-agent-data-content.json 中的 tool_calls[].function.name）以及可配置的
user/assistant 标签。
可选的 ``history_*_max_chars`` 上限会保留头部和尾部，并带有明确的
中间省略标记，从而使 ``dialog_history``、展平后的 ``text`` 以及最后的
``query`` / ``response`` 保持对齐；当触发任何上限时，会设置 ``meta.agent_dialog_history_compressed``。

Type 算子类型: **mapper**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `messages_key` | <class 'str'> | `'messages'` |  |
| `choices_key` | <class 'str'> | `'choices'` |  |
| `text_key` | <class 'str'> | `'text'` |  |
| `history_key` | <class 'str'> | `'dialog_history'` |  |
| `query_key` | <class 'str'> | `'query'` |  |
| `response_key` | <class 'str'> | `'response'` |  |
| `extract_tool_skill_tags` | <class 'bool'> | `True` |  |
| `include_system_in_first_user` | <class 'bool'> | `False` |  |
| `user_label` | <class 'str'> | `'User'` |  |
| `assistant_label` | <class 'str'> | `'Assistant'` |  |
| `copy_lineage_fields` | <class 'bool'> | `True` |  |
| `copy_request_id` | <class 'bool'> | `True` |  |
| `request_id_keys` | typing.List[str] | `['request_id', 'trace_id', 'id']` |  |
| `history_tool_result_max_chars` | <class 'int'> | `10000` |  |
| `history_max_assistant_trace_chars` | <class 'int'> | `0` |  |
| `history_max_user_chars` | <class 'int'> | `0` |  |
| `history_compress_head_ratio` | <class 'float'> | `0.62` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/agent_dialog_normalize_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_agent_dialog_normalize_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)