# image_captioning_from_gpt4v_mapper

Generates text captions for images using the GPT-4 Vision model.

This operator generates text based on the provided images and specified parameters. It supports different modes of text generation, including 'reasoning', 'description', 'conversation', and 'custom'. The generated text can be added to the original sample or replace it, depending on the `keep_original_sample` parameter. The operator uses a Hugging Face tokenizer and the GPT-4 Vision API to generate the text. The `any_or_all` parameter determines whether all or any of the images in a sample must meet the generation criteria for the sample to be kept. If `user_prompt_key` is set, it will use the prompt from the sample; otherwise, it will use the `user_prompt` parameter. If both are set, `user_prompt_key` takes precedence.

使用 GPT-4 Vision 模型为图像生成文本描述。

该算子根据提供的图像和指定的参数生成文本。它支持不同的文本生成模式，包括'reasoning'、'description'、'conversation'和'custom'。根据`keep_original_sample`参数，生成的文本可以添加到原始样本中或替换它。该算子使用 Hugging Face 的 tokenizer 和 GPT-4 Vision API 生成文本。`any_or_all` 参数决定样本中的所有或任何图像是否必须满足生成条件才能保留样本。如果设置了`user_prompt_key`，则会使用样本中的提示；否则，将使用`user_prompt`参数。如果两者都设置，则`user_prompt_key`优先。

Type 算子类型: **mapper**

Tags 标签: cpu, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `mode` | <class 'str'> | `'description'` | mode of text generated from images, can be one of ['reasoning', 'description', 'conversation', 'custom'] |
| `api_key` | <class 'str'> | `''` | the API key to authenticate the request. |
| `max_token` | <class 'int'> | `500` | the maximum number of tokens to generate. Default is 500. |
| `temperature` | typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=1)])] | `1.0` | controls the randomness of the output (range from 0 to 1). Default is 0. |
| `system_prompt` | <class 'str'> | `''` | a string prompt used to set the context of a conversation and provide global guidance or rules for the gpt4-vision so that it can  generate responses in the expected way. If `mode` set to `custom`, the parameter will be used. |
| `user_prompt` | <class 'str'> | `''` | a string prompt to guide the generation of gpt4-vision for each samples. It's "" in default, which means no prompt provided. |
| `user_prompt_key` | typing.Optional[str] | `None` | the key name of fields in samples to store prompts for each sample. It's used for set different prompts for different samples. If it's none, use prompt in parameter "prompt". It's None in default. |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If it's set to False, there will be only generated text in the final datasets and the original text will be removed. It's True in default. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all images. 'any': keep this sample if any images meet the condition. 'all': keep this sample only if all images meet the condition. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_captioning_from_gpt4v_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_captioning_from_gpt4v_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)