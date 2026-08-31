# agent_skill_insight_mapper

Summarize agent_tool_types and agent_skill_types into insights via LLM.

Reads ``meta[agent_tool_types]`` and ``meta[agent_skill_types]`` (from
``agent_dialog_normalize_mapper``), calls the API for 3–5 **concrete**
capability phrases (about 10 Chinese characters or ~4–8 English words
each; avoid vague 'read/write / processing'), and stores them in
``meta[agent_skill_insights]``. Run after normalize. Override
``system_prompt`` for locale-specific label style.

通过 LLM 将 agent_tool_types 和 agent_skill_types 总结为洞察。

读取 ``meta[agent_tool_types]`` 和 ``meta[agent_skill_types]``（来自
``agent_dialog_normalize_mapper``），调用 API 生成 3-5 个**具体**的
能力短语（每个约 10 个汉字或 4-8 个英文单词；避免使用模糊的“读/写/处理”），并将其存储在
``meta[agent_skill_insights]`` 中。在 normalize 之后运行。覆盖
``system_prompt`` 以使用特定区域的标签样式。

Type 算子类型: **mapper**

Tags 标签: cpu, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_model` | <class 'str'> | `'gpt-4o'` |  |
| `tool_types_key` | <class 'str'> | `'agent_tool_types'` |  |
| `skill_types_key` | <class 'str'> | `'agent_skill_types'` |  |
| `insights_key` | <class 'str'> | `'agent_skill_insights'` |  |
| `api_endpoint` | typing.Optional[str] | `None` |  |
| `response_path` | typing.Optional[str] | `None` |  |
| `system_prompt` | typing.Optional[str] | `None` |  |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `2` |  |
| `model_params` | typing.Dict | `{}` |  |
| `sampling_params` | typing.Dict | `{}` |  |
| `preferred_output_lang` | <class 'str'> | `'en'` |  |
| `kwargs` |  | `` |  |




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/agent_skill_insight_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_agent_skill_insight_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)