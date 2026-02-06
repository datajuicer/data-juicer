# ray_bts_minhash_deduplicator_with_uid

A MinhashLSH deduplicator based on RAY.

Unlike `RayBTSMinhashDeduplicator`, this class requires the input dataset to contain an additional column named '__dj__uid' of type int, where each value is unique across samples. This column serves two main purposes:

1. **Reduced I/O Overhead**: Compared to RayBTSMinhashDeduplicator, this class does not persist intermediate results, thereby reducing disk read and write operations.

2. **Support for Incremental Deduplication**: The '__dj__uid' column enables the deduplicator to perform incremental deduplication. This is particularly useful in scenarios where you already have a deduplicated dataset (e.g., dataset A) and want to add a new dataset (e.g., dataset B) while ensuring that duplicates are resolved in favor of the original data.

For example, consider a scenario where you have an already deduplicated dataset A and a new dataset B that you wish to add. If you want to perform joint deduplication on both A and B while prioritizing the retention of data from A, you can ensure that all '__dj__uid' values in B are greater than those in A. Then, by applying this deduplicator to the combined dataset, duplicates will be resolved in favor of the entries from A.

基于 RAY 的 MinhashLSH 去重器。

与 `RayBTSMinhashDeduplicator` 不同，此类要求输入数据集包含一个名为 '__dj__uid' 的额外列，其类型为 int，且每个样本的值在整个数据集中唯一。该列主要有两个用途：

1. **降低 I/O 开销**：与 RayBTSMinhashDeduplicator 相比，此类不持久化中间结果，从而减少了磁盘读写操作。

2. **支持增量去重**：'__dj__uid' 列使去重器能够执行增量去重。这在以下场景中特别有用：你已经有一个去重后的数据集（例如数据集 A），并希望添加一个新数据集（例如数据集 B），同时确保在去重时优先保留原始数据。

例如，假设你已有一个去重后的数据集 A 和一个希望添加的新数据集 B。如果你希望对 A 和 B 联合去重，并优先保留 A 中的数据，可以确保 B 中所有 '__dj__uid' 的值都大于 A 中的值。然后，将此去重器应用于合并后的数据集，重复项将优先保留来自 A 的条目。

Type 算子类型: **deduplicator**

Tags 标签: cpu, text

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
| `union_find_parallel_num` | typing.Union[int, str] | `'auto'` | number of parallel workers for union-find algorithm. Default it's 'auto', and it will be determined by half of the number of CPUs. |
| `union_threshold` | typing.Optional[int] | `256` | threshold for minhash values group to perform union-find algorithm. Default it's 256. |
| `max_pending_edge_buffer_task` | typing.Optional[int] | `20` | max number of pending edge buffer ray tasks. Default it's 20. |
| `num_edge_buffer_task_returns` | typing.Optional[int] | `10` | number of edge buffer tasks for `ray.wait` to return. Default it's 10. |
| `max_pending_filter_tasks` | typing.Optional[int] | `20` | max number of pending filter ray tasks. Default it's 20. |
| `num_filter_task_returns` | typing.Optional[int] | `10` | number of filter tasks for `ray.wait` to return. Default it's 10. |
| `merge_batch_size` | typing.Optional[int] | `1000` | batch size for BTS operations. Default it's 1000. |
| `minhash_batch_size` | typing.Union[int, str, NoneType] | `'auto'` | batch size for MinHash computation. If "auto", it will be set to default value on CPU(1024), or auto calculated per available GPU memory and memory_per_sample setting for GPU. |
| `memory_per_sample` | typing.Optional[float] | `0.1` | estimated memory needed per sample in MB. Used to calculate batch size based on available GPU memory. Default is 0.1 MB per sample. |
| `actor_memory` | typing.Optional[int] | `None` | Memory reservation per BTSUnionFind/EdgeBuffer actor in bytes. For billion-row scale, use 20_000_000_000 (20GB). Default is None (no reservation). |
| `task_memory` | typing.Optional[int] | `None` | Memory reservation per map_batches task in bytes. For billion-row scale, use 2_000_000_000 (2GB). Default is None (no reservation). |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/deduplicator/ray_bts_minhash_deduplicator.py)
- [unit test 单元测试]()
- [Return operator list 返回算子列表](../../Operators.md)