# agent_tool_type_mapper

Set primary_tool_type and dominant_tool_types from meta.agent_tool_types.

从 meta.agent_tool_types 设置 primary_tool_type 和 dominant_tool_types。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tool_types_meta_key` | <class 'str'> | `'agent_tool_types'` |  |
| `primary_key` | <class 'str'> | `'primary_tool_type'` |  |
| `dominant_key` | <class 'str'> | `'dominant_tool_types'` |  |
| `top_k_dominant` | <class 'int'> | `5` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/agent_tool_type_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_agent_tool_type_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)