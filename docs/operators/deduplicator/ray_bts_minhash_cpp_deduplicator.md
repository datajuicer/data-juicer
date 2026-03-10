# ray_bts_minhash_cpp_deduplicator

A MinHash LSH deduplicator that operates in Ray distributed mode with C++ acceleration.

Same as ray_bts_minhash_deduplicator but with tokenization and MinHash signature computation implemented in C++ for improved performance.

This operator uses the MinHash LSH technique to identify and remove near-duplicate samples from a dataset. It supports various tokenization methods, including space, punctuation, character, and sentencepiece. The Jaccard similarity threshold is used to determine if two samples are considered duplicates. If the Jaccard similarity of two samples is greater than or equal to the specified threshold, one of the samples is filtered out. The operator computes the MinHash values for each sample and uses a union- find algorithm to group similar samples. The key metric, Jaccard similarity, is computed based on the shingling of the text.

一种在 Ray 分布式模式下运行并采用 C++ 加速的 MinHash LSH 去重器。

与 ray_bts_minhash_deduplicator 相同，但其分词和 MinHash 签名计算使用 C++ 实现，以提升性能。

该算子使用 MinHash LSH 技术识别并从数据集中移除近似重复样本。它支持多种分词方法，包括空格、标点符号、字符和 sentencepiece。使用 Jaccard 相似度阈值来判断两个样本是否被视为重复：若两个样本的 Jaccard 相似度大于或等于指定阈值，则其中一个样本将被过滤掉。该算子为每个样本计算 MinHash 值，并使用并查集算法对相似样本进行分组。关键指标 Jaccard 相似度基于文本的 shingling 计算得出。

Type 算子类型: **deduplicator**

Tags 标签: cpu, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tokenization` | <class 'str'> | `'space'` | tokenization method for sample texts. It should be one of [space, punctuation, character, sentencepiece]. For English-like languages, we recommend to use 'space', for Chinese-like languages, we recommend to use 'character', and for multiple languages, we recommend to use 'sentencepiece'. If using 'sentencepiece', please provided the model path in the 'tokenizer_model' field. |
| `window_size` | typing.Annotated[int, Gt(gt=0)] | `5` | window size of shingling |
| `lowercase` | <class 'bool'> | `True` | whether to convert text to lower case first |
| `ignore_pattern` | typing.Optional[str] | `None` | whether to ignore sub-strings with specific pattern when computing minhash |
| `tokenizer_model` | typing.Optional[str] | `None` | path for the sentencepiece model, used for sentencepiece tokenization. |
| `num_permutations` | typing.Annotated[int, Gt(gt=0)] | `256` | number of permutations in minhash computing |
| `jaccard_threshold` | typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=1)])] | `0.7` | the min jaccard similarity threshold in near-duplicate detection. When the jaccard similarity of two sample texts is >= this threshold, they are regarded as similar samples and this op will only keep one of them after deduplication |
| `num_bands` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | number of bands in LSH. Default it's None, and it will be determined by an optimal params computation algorithm by minimize the weighted sum of probs of False Positives and False Negatives |
| `num_rows_per_band` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | number of rows in each band in LSH. Default it's None, and it will be determined by an optimal params computation algorithm |
| `union_find_parallel_num` | typing.Union[int, str] | `'auto'` |  |
| `union_threshold` | typing.Optional[int] | `256` |  |
| `max_pending_edge_buffer_task` | typing.Optional[int] | `20` |  |
| `num_edge_buffer_task_returns` | typing.Optional[int] | `10` |  |
| `max_pending_filter_tasks` | typing.Optional[int] | `20` |  |
| `num_filter_task_returns` | typing.Optional[int] | `10` |  |
| `merge_batch_size` | typing.Optional[int] | `1000` |  |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/deduplicator/ray_bts_minhash_cpp_deduplicator.py)
- [unit test 单元测试]()
- [Return operator list 返回算子列表](../../Operators.md)