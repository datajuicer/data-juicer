# generate_qa_from_examples_mapper

Generates question and answer pairs from examples using a Hugging Face model.

This operator generates QA pairs based on provided seed examples. The number of generated samples is determined by the length of the empty dataset configured in the YAML file. The operator uses a Hugging Face model to generate new QA pairs, which are then filtered based on their similarity to the seed examples. Samples with a similarity score below the specified threshold are kept. The similarity is computed using the ROUGE-L metric. The operator requires a seed file in chatml format, which provides the initial QA examples. The generated QA pairs must follow specific formatting rules, such as maintaining the same format as the input examples and ensuring that questions and answers are paired correctly.

使用Hugging Face模型从示例生成问题和答案对。

此算子基于提供的种子示例生成QA对。生成的样本数量由YAML文件中配置的空数据集长度决定。算子使用Hugging Face模型生成新的QA对，然后根据它们与种子示例的相似性进行筛选。相似度低于指定阈值的样本会被保留。相似度计算使用ROUGE-L指标。算子需要一个chatml格式的种子文件，提供初始的QA示例。生成的QA对必须遵循特定的格式规则，例如保持与输入示例相同的格式，并确保问题和答案正确配对。

Type 算子类型: **mapper**

Tags 标签: gpu, vllm, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'Qwen/Qwen2.5-7B-Instruct'` | Huggingface model ID. |
| `seed_file` | <class 'str'> | `''` | Path to the seed file in chatml format. |
| `example_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of selected examples. Randomly select N examples from "seed_file" and put them into prompt as QA examples. |
| `similarity_threshold` | <class 'float'> | `0.7` | The similarity score threshold between the generated samples and the seed examples. Range from 0 to 1. Samples with similarity score less than this threshold will be kept. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for guiding the generation task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the input prompt. It must include one placeholder '{}', which will be replaced by `example_num` formatted examples defined by `example_template`. |
| `example_template` | typing.Optional[str] | `None` | Template for formatting one QA example. It must include one placeholder '{}', which will be replaced by one formatted qa_pair. |
| `qa_pair_template` | typing.Optional[str] | `None` | Template for formatting a single QA pair within each example. Must include two placeholders '{}' for the question and answer. |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression pattern to extract questions and answers from model response. |
| `enable_vllm` | <class 'bool'> | `False` | Whether to use vllm for inference acceleration. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the model. |
| `sampling_params` | typing.Optional[typing.Dict] | `None` | Sampling parameters for text generation. e.g {'temperature': 0.9, 'top_p': 0.95} |
| `kwargs` |  | `''` | Extra keyword arguments. |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/generate_qa_from_examples_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_generate_qa_from_examples_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)