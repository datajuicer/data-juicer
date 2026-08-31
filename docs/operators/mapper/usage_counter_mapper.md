# usage_counter_mapper

Write token usage to meta from choices/usage (OpenAI/Anthropic-style).

Collects every non-empty usage dict found (top-level ``usage_key``,
``response_metadata``, each ``choices[]`` entry, nested message usage).
By default, **deduplicates** identical usage snapshots before summing: same
``(prompt_tokens, completion_tokens, total_tokens or prompt+completion)``
only counts once (typical when ``response_usage`` mirrors ``choices[0].usage``).
Set ``dedupe_identical_usage: false`` to restore legacy double-counting.

将 token 使用情况从 choices/usage 写入 meta（OpenAI/Anthropic 风格）。

收集找到的每个非空 usage 字典（顶层 ``usage_key``、``response_metadata``、每个 ``choices[]`` 条目、嵌套的 message usage）。默认情况下，在求和前对相同的 usage 快照进行**去重**：相同的 ``(prompt_tokens, completion_tokens, total_tokens or prompt+completion)`` 只计算一次（当 ``response_usage`` 与 ``choices[0].usage`` 相同时的典型情况）。设置 ``dedupe_identical_usage: false`` 以恢复旧版的双重计算行为。

Type 算子类型: **mapper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `choices_key` | <class 'str'> | `'choices'` |  |
| `usage_key` | <class 'str'> | `'usage'` |  |
| `response_metadata_key` | <class 'str'> | `'response_metadata'` |  |
| `dedupe_identical_usage` | <class 'bool'> | `True` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/usage_counter_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_usage_counter_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)