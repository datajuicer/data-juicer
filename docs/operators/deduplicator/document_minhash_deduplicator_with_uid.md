# document_minhash_deduplicator_with_uid

A Deduplicator that performs document-level deduplication using MinHashLSH.

Unlike `DocumentMinhashDeduplicator`, this class requires the dataset to include an additional column named '__dj__uid' of type int, with unique values for each sample. This column is essential for supporting incremental deduplication scenarios.

For example, consider a scenario where you have an already deduplicated dataset A and a new dataset B that you wish to add. If you want to perform joint deduplication on both A and B while prioritizing the retention of data from A, you can ensure that all '__dj__uid' values in B are greater than those in A. Then, by applying this deduplicator to the combined dataset, duplicates will be resolved in favor of the entries from A.

一种使用 MinHashLSH 执行文档级去重的 Deduplicator。

与 `DocumentMinhashDeduplicator` 不同，此类要求数据集中包含一个名为 '__dj__uid' 的额外列，其类型为 int，并且每个样本具有唯一值。该列对于支持增量去重场景至关重要。

例如，假设你已有一个去重后的数据集 A，以及一个希望添加的新数据集 B。如果你希望对 A 和 B 联合执行去重操作，同时优先保留 A 中的数据，你可以确保 B 中所有 '__dj__uid' 的值均大于 A 中的值。然后，将此去重器应用于合并后的数据集，重复项将优先保留来自 A 的条目。

Type 算子类型: **deduplicator**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `tokenization` | <class 'str'> | `'space'` | tokenization method for sample texts. It should be one of [space, punctuation, character, sentencepiece]. For English-like languages, we recommend to use 'space', for Chinese-like languages, we recommend to use 'character', and for multiple languages, we recommend to use 'sentencepiece'. If using 'sentencepiece', please provided the model path in the 'tokenizer_model' field. |
| `window_size` | typing.Annotated[int, Gt(gt=0)] | `5` | window size of shingling |
| `lowercase` | <class 'bool'> | `True` | whether to convert text to lower case first |
| `ignore_pattern` | typing.Optional[str] | `None` | whether to ignore sub-strings with specific pattern when computing minhash |
| `num_permutations` | typing.Annotated[int, Gt(gt=0)] | `256` | number of permutations in minhash computing |
| `jaccard_threshold` | typing.Annotated[float, FieldInfo(annotation=NoneType, required=True, metadata=[Ge(ge=0), Le(le=1)])] | `0.7` | the min jaccard similarity threshold in near-duplicate detection. When the jaccard similarity of two sample texts is >= this threshold, they are regarded as similar samples and this op will only keep one of them after deduplication |
| `num_bands` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | number of bands in LSH. Default it's None, and it will be determined by an optimal params computation algorithm by minimize the weighted sum of probs of False Positives and False Negatives |
| `num_rows_per_band` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | number of rows in each band in LSH. Default it's None, and it will be determined by an optimal params computation algorithm |
| `tokenizer_model` | typing.Optional[str] | `None` | path for the sentencepiece model, used for sentencepiece tokenization. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/deduplicator/document_minhash_deduplicator.py)
- [unit test 单元测试]()
- [Return operator list 返回算子列表](../../Operators.md)