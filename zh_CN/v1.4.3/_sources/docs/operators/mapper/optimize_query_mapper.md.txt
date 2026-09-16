# optimize_query_mapper

Optimize queries in question-answer pairs to make them more specific and detailed.

This mapper refines the questions in a QA pair, making them more specific and detailed while ensuring that the original answer can still address the optimized question. It uses a predefined system prompt for the optimization process. The optimized query is extracted from the raw output by stripping any leading or trailing whitespace. The mapper utilizes a CUDA accelerator for faster processing.

优化问答对中的查询，使其更具体和详细。

该映射器改进问答对中的问题，使其更具体和详细，同时确保原始答案仍能回答优化后的问题。它使用预定义的系统提示进行优化过程。优化后的查询通过去除任何前导或尾随空格从原始输出中提取。映射器利用CUDA加速器进行更快的处理。

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
- [source code 源代码](../../../data_juicer/ops/mapper/optimize_query_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_optimize_query_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)