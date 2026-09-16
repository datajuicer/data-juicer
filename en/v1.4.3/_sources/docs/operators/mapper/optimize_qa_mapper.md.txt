# optimize_qa_mapper

Mapper to optimize question-answer pairs.

This operator refines and enhances the quality of question-answer pairs. It uses a Hugging Face model to generate more detailed and accurate questions and answers. The input is formatted using a template, and the output is parsed using a regular expression. The system prompt, input template, and output pattern can be customized. If VLLM is enabled, the operator accelerates inference on CUDA devices.

优化问答对的映射器。

该算子改进和提高问答对的质量。它使用Hugging Face模型生成更详细和准确的问题和答案。输入使用模板格式化，输出通过正则表达式解析。可以自定义系统提示、输入模板和输出模式。如果启用了VLLM，算子将在CUDA设备上加速推理。

Type 算子类型: **mapper**

Tags 标签: cpu, vllm, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'Qwen/Qwen2.5-7B-Instruct'` | Hugging Face model ID. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for guiding the optimization task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the input for the model. |
| `qa_pair_template` | typing.Optional[str] | `None` | Template for formatting the question and |
| `output_pattern` | typing.Optional[str] | `None` | Regular expression pattern to extract question |
| `enable_vllm` | <class 'bool'> | `False` | Whether to use VLLM for inference acceleration. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the model. |
| `sampling_params` | typing.Optional[typing.Dict] | `None` | Sampling parameters for text generation (e.g., |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/optimize_qa_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_optimize_qa_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)