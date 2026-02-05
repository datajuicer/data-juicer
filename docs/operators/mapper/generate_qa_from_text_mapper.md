# generate_qa_from_text_mapper

Generates question and answer pairs from text using a specified model.

This operator uses a Hugging Face model to generate QA pairs from the input text. It supports both Hugging Face and vLLM models for inference. The recommended models, such as 'alibaba-pai/pai-llama3-8b-doc2qa', are trained on Chinese data and are suitable for Chinese text. The operator can limit the number of generated QA pairs per text and allows custom output patterns for parsing the model's response. By default, it uses a regular expression to extract questions and answers from the model's output. If no QA pairs are extracted, a warning is logged.

使用指定模型从文本生成问题和答案对。

此算子使用Hugging Face模型从输入文本生成QA对。它支持使用Hugging Face和vLLM模型进行推理。推荐的模型，如'alibaba-pai/pai-llama3-8b-doc2qa'，是在中文数据上训练的，适合处理中文文本。算子可以限制每段文本生成的QA对数量，并允许自定义输出模式来解析模型的响应。默认情况下，它使用正则表达式从模型的输出中提取问题和答案。如果没有提取到QA对，将记录一条警告。

Type 算子类型: **mapper**

Tags 标签: gpu, vllm, hf, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'alibaba-pai/pai-qwen1_5-7b-doc2qa'` | Huggingface model ID. |
| `max_num` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | The max num of returned QA sample for each text. Not limit if it is None. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression pattern to extract questions and answers from model response. |
| `enable_vllm` | <class 'bool'> | `False` | Whether to use vllm for inference acceleration. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the model. |
| `sampling_params` | typing.Optional[typing.Dict] | `None` | Sampling parameters for text generation, e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. The default data format parsed by this interface is as follows: Model Input:     蒙古国的首都是乌兰巴托（Ulaanbaatar）     冰岛的首都是雷克雅未克（Reykjavik） Model Output:     蒙古国的首都是乌兰巴托（Ulaanbaatar）     冰岛的首都是雷克雅未克（Reykjavik）     Human: 请问蒙古国的首都是哪里？     Assistant: 你好，根据提供的信息，蒙古国的首都是乌兰巴托（Ulaanbaatar）。     Human: 冰岛的首都是哪里呢？     Assistant: 冰岛的首都是雷克雅未克（Reykjavik）。     ... |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/generate_qa_from_text_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_generate_qa_from_text_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)