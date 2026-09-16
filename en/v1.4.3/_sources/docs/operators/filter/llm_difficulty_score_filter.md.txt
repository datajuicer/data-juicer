# llm_difficulty_score_filter

Filter to keep samples with high difficulty scores estimated by an LLM.

This operator uses a Hugging Face LLM to evaluate the difficulty of each sample. The LLM analyzes the sample across multiple dimensions, including linguistic complexity, conceptual depth, prior knowledge, step complexity, and ambiguity. Each dimension is scored on a 1-5 scale, with 5 being the highest difficulty. The final difficulty score is computed as the average of these dimension scores. Samples are kept if their difficulty score falls within the specified range (min_score to max_score). The key metric 'llm_difficulty_score' is stored in the sample's stats, along with detailed records and flags.

用于保留由大型语言模型估计出高难度分数的样本的过滤器。

该算子使用Hugging Face的大型语言模型评估每个样本的难度。LLM从多个维度分析样本，包括语言复杂性、概念深度、先验知识、步骤复杂性和模糊性。每个维度的评分范围为1-5分，5分为最高难度。最终的难度分数是这些维度分数的平均值。如果样本的难度分数在指定范围内（min_score到max_score），则保留该样本。关键指标'llm_difficulty_score'存储在样本的stats中，同时还有详细的记录和标志。

Type 算子类型: **filter**

Tags 标签: cpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'gpt-4o'` | API or huggingface model name. |
| `min_score` | <class 'float'> | `0.5` | The min score threshold to keep the sample. |
| `max_score` | <class 'float'> | `1.0` | The max score threshold to keep the sample. |
| `is_hf_model` | <class 'bool'> | `False` | If true, use huggingface model. Otherwise, use API. |
| `api_endpoint` | typing.Optional[str] | `None` | URL endpoint for the API. |
| `response_path` | typing.Optional[str] | `None` | Path to extract content from the API response. |
| `input_keys` | typing.List[str] | `['text']` | Sub set of keys in the sample. Support data with |
| `field_names` | typing.List[str] | `['Text']` | Corresponding field names for input keys. |
| `system_prompt` | typing.Optional[str] | `None` | System prompt for the task. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `field_template` | typing.Optional[str] | `None` | Template for each field in the prompt. |
| `try_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of retry attempts when there is an API |
| `enable_vllm` | <class 'bool'> | `False` | If true, use VLLM for loading hugging face or |
| `model_params` | typing.Dict | `{}` | Parameters for initializing the API model. |
| `sampling_params` | typing.Dict | `{}` | Extra parameters passed to the API call. |
| `dim_required_keys` | typing.Optional[typing.List[str]] | `None` | A list of keys used to calculate the average |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/llm_difficulty_score_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_difficulty_score_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)