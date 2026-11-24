# tool_success_tagger_mapper

Set meta tool_success_count, tool_fail_count, tool_success_ratio.

Scans messages for role=tool; configurable success/error patterns.

设置 meta 中的 tool_success_count、tool_fail_count、tool_success_ratio。

扫描 messages 中 role=tool 的内容；可配置成功/错误模式。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `messages_key` | <class 'str'> | `'messages'` |  |
| `tool_role_names` | typing.Optional[typing.List[str]] | `None` |  |
| `success_patterns` | typing.Optional[typing.List[str]] | `None` |  |
| `error_patterns` | typing.Optional[typing.List[str]] | `None` |  |
| `store_per_tool_results` | <class 'bool'> | `True` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/tool_success_tagger_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_tool_success_tagger_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)