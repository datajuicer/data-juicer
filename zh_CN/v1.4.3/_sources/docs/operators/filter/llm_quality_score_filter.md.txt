# llm_quality_score_filter

Filter to keep samples with a high quality score estimated by a language model.

This operator uses a language model to evaluate the quality of each sample across multiple dimensions, including accuracy, grammar, informativeness, and coherence. The LLM provides a numerical score for each dimension on a 1-5 scale, where 1 is the lowest and 5 is the highest. The overall quality score is used to decide whether to keep or filter out the sample based on the specified minimum and maximum score thresholds. The evaluation results are cached in the 'llm_quality_score' and 'llm_quality_record' fields. Important flags and tags from the LLM's analysis may also be stored in the sample's stats.

用于保留由语言模型估计出高质量分数的样本的过滤器。

该算子使用语言模型评估每个样本在多个维度上的质量，包括准确性、语法、信息量和连贯性。LLM为每个维度提供一个1-5分的数值评分，其中1分最低，5分最高。总体质量分数用于根据指定的最小和最大分数阈值来决定是否保留或过滤掉样本。评估结果缓存在'llm_quality_score'和'llm_quality_record'字段中。LLM分析的重要标志和标签也可能存储在样本的stats中。

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
- [source code 源代码](../../../data_juicer/ops/filter/llm_quality_score_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_quality_score_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)