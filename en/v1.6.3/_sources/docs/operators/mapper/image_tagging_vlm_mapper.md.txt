# image_tagging_vlm_mapper

Mapper to generates image tags. This operator generates tags based on the content of given images. The tags are generated using a vlm model and stored in the specified field name. If the tags are already present in the sample, the operator skips processing.

用于生成图像标签的 Mapper。该算子根据给定图像的内容生成标签，使用 vlm 模型生成标签并存储到指定字段名中。如果样本中已存在标签，则跳过处理。

Type 算子类型: **mapper**

Tags 标签: gpu, api, vllm, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'Qwen/Qwen2.5-VL-7B-Instruct'` | API model name or HF model name. |
| `is_api_model` | <class 'bool'> | `False` | Whether the model is an API model. If true, use openai api to generate tags, otherwise use vllm. |
| `tag_field_name` | <class 'str'> | `'image_tags'` | the field name to store the tags. It's "image_tags" in default. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. Defaults to 'choices.0.message.content'. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API call error or output parsing error. |
| `kwargs` |  | `''` | Extra keyword arguments. |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_tagging_vlm_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_tagging_vlm_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)