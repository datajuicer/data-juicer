# llm_task_relevance_filter

Filter to keep samples with high relevance scores to validation tasks estimated by an LLM.

This operator evaluates the relevance of each sample to a specified validation task using an LLM. The LLM scores the sample on multiple dimensions, including topical relevance, linguistic style match, task match, knowledge alignment, and potential utility. Each dimension is scored on a 1-5 scale, with 5 being the highest. The key metric, 'llm_task_relevance', is the average score across these dimensions. Samples are kept if their average score meets or exceeds the specified minimum threshold. The operator uses either an API or a Hugging Face model for evaluation. If no validation dataset or task description is provided, the 'prepare_valid_feature' method must be called manually before applying the filter.

筛选并保留LLM估计的与验证任务高度相关的样本。

该算子使用LLM评估每个样本与指定验证任务的相关性。LLM从多个维度对样本进行评分，包括主题相关性、语言风格匹配、任务匹配、知识对齐和潜在实用性。每个维度按1-5的评分，5为最高。关键指标'llm_task_relevance'是这些维度分数的平均值。如果样本的平均分数达到或超过指定的最低阈值，则保留该样本。该算子使用API或Hugging Face模型进行评估。如果没有提供验证数据集或任务描述，则必须在应用过滤前手动调用'prepare_valid_feature'方法。

Type 算子类型: **filter**

Tags 标签: gpu, vllm, hf, api

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'gpt-4o'` | API or huggingface model name. |
| `min_score` | <class 'float'> | `0.5` | The lowest score threshold to keep the sample. |
| `is_hf_model` | <class 'bool'> | `False` | Indicates if the model is from HuggingFace. |
| `valid_dataset` | typing.Optional[typing.List[typing.Dict]] | `None` | The dataset to use for validation. |
| `task_desc` | typing.Optional[str] | `None` | The description of the validation task. If valid_dataset=None and task_desc=None, 'self.prepare_valid_feature' should be manually called before applying the filter. |
| `n_shot` | typing.Optional[int] | `None` | The number of shots in validation. |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/llm_task_relevance_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_task_relevance_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)