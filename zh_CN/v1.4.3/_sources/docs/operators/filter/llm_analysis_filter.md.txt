# llm_analysis_filter

Base filter class for leveraging LLMs to analyze and filter data samples.

This operator uses an LLM to score and tag each sample across multiple quality dimensions. It supports both API-based and Hugging Face models. The LLM evaluates the sample on clarity, relevance, usefulness, and fluency, providing scores from 1 to 5. Tags are assigned to categorize the sample, and a recommendation is made to keep, review, or discard the sample. The average score is computed based on the required dimension keys. Samples are kept if their average score falls within the specified min and max score thresholds. The key metric 'llm_analysis_score' is cached in the sample's stats.

用于利用大语言模型分析和筛选数据样本的基础过滤器类。

该算子使用大语言模型对每个样本在多个质量维度上进行评分和标记。它支持基于API的模型和Hugging Face模型。大语言模型会对样本的清晰度、相关性、有用性和流利度进行评估，提供1到5分的评分。会分配标签来对样本进行分类，并提出保留、审查或丢弃样本的建议。平均分会根据所需的维度键计算得出。如果样本的平均分落在指定的最小和最大分数阈值之间，则保留该样本。关键指标'llm_analysis_score'会被缓存在样本的统计信息中。

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
- [source code 源代码](../../../data_juicer/ops/filter/llm_analysis_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_analysis_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)