# text_embd_similarity_filter

Filter to keep texts whose average embedding similarity to a set of given validation texts falls within a specific range.

This operator computes the cosine similarity between the text embeddings and a set of validation text embeddings. It keeps samples where the average similarity score is within the specified range. The key metric, 'text_embd_similarity', is computed as the mean cosine similarity. The operator supports both API-based and Hugging Face model- based embeddings. If no valid dataset is provided, the `prepare_valid_feature` method must be called manually before applying the filter.

用于保留与一组给定验证文本的平均嵌入相似度在特定范围内的文本的过滤器。

该算子计算文本嵌入与一组验证文本嵌入之间的余弦相似度。如果平均相似度得分在指定范围内，则保留样本。关键指标 'text_embd_similarity' 计算为平均余弦相似度。算子支持基于 API 和基于 Hugging Face 模型的嵌入。如果没有提供有效的数据集，则必须在应用过滤之前手动调用 `prepare_valid_feature` 方法。

Type 算子类型: **filter**

Tags 标签: gpu, api, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | <class 'str'> | `'text-embedding-v4'` | API or huggingface embedding model name. |
| `is_hf_model` | <class 'bool'> | `False` | Indicates if the model is from HuggingFace. |
| `api_endpoint` | <class 'str'> | `'embeddings'` | Embedding URL endpoint for the API. |
| `response_path` | <class 'str'> | `'data.0.embedding'` | Path to extract content from the API response. Defaults to 'data.0.embedding' for embedding model. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the API model. |
| `min_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `0.1` | The min average similarity to keep samples. |
| `max_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `1.0` | The max average similarity to keep samples. |
| `valid_dataset` | typing.Optional[typing.List[typing.Dict]] | `None` | The dataset to use for validation. If None, 'self.prepare_valid_feature' should be manually called before applying the filter. |
| `ebd_dim` | <class 'int'> | `4096` | The embedding's dimension via API. API specific parameter, i.e., if is_hf_model=True, this parameter will not take effect. |
| `pooling` | typing.Optional[str] | `None` | strategy to extract embedding from the hidden states. https://arxiv.org/abs/2503.01807 None: default option, the hidden state of the last token. "mean": uniform mean of hidden states. "weighted_mean": weighted mean of hidden states. https://arxiv.org/abs/2202.08904 HF_MODEL specific parameter, i.e., if is_hf_model=False, this parameter will not take effect. |
| `input_template` | typing.Optional[str] | `None` | Template for building the model input. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/text_embd_similarity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_text_embd_similarity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)