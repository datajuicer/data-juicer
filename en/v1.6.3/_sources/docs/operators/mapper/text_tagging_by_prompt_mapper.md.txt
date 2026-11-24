# text_tagging_by_prompt_mapper

Mapper to generate text tags using prompt with LLM. Other opensourced models with good instruction following ability also works.

使用 LLM 通过 prompt 生成文本标签的 Mapper。其他具有良好指令遵循能力的开源模型也可使用。

Type 算子类型: **mapper**

Tags 标签: gpu, vllm, hf, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'Qwen/Qwen2.5-7B-Instruct'` |  |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `prompt` | <class 'str'> | `'\n请对下面的example文本回复的任务类别进行检测,并进行分类。\n备选的分类包括：{tag_list}。\n只回复对应的分类,不回复其他内容。\nexample文本:\n{text}\n'` |  |
| `tag_list` | typing.List[str] | `['数学', '代码', '翻译', '角色扮演', '开放领域问答', '特定领域问答', '提取', '生成', '头脑风暴', '分类', '总结', '改写', '其他']` |  |
| `enable_vllm` | <class 'bool'> | `True` |  |
| `tensor_parallel_size` | <class 'int'> | `None` |  |
| `max_model_len` | <class 'int'> | `None` |  |
| `max_num_seqs` | <class 'int'> | `256` |  |
| `sampling_params` | typing.Dict | `None` |  |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/text_tagging_by_prompt_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_text_tagging_by_prompt_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)