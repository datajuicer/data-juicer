# 分布式模糊去重工具
复现与GPT-3论文相似的模糊去重方法并应用到您的Web数据集。

**模糊去重的一般描述**：
这里的模糊去重方法主要指的是 [GPT-3论文](https://arxiv.org/pdf/2005.14165.pdf)附录A中提到的模糊去重方法。
> 为了进一步提高模型质量并防止过拟合（随着模型容量的增加越来越重要），我们使用Spark的MinHashLSH实现对每个数据集中的文档进行了模糊去重（即移除了与其他文档高度重合的文档），使用了10个哈希，使用的**特征与上面用于分类的特征相同**。我们还从Common Crawl中模糊移除了WebText。总体而言，这使数据集的大小平均减少了10％。
正如论文中提到的，使用的特征与前文描述的质量分类器（[quality_classifier tools](../../data_juicer/tools/quality_classifier/README.md)）中所用的一致。
整个工具包基于PySpark。
- 分词器：由于pyspark的标准分词器无法很好地处理中文等语言的文本，所以使用了PySpark的[标准分词器](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html#tokenizer)或[sentencepiece](https://github.com/google/sentencepiece)模型。
- 特征提取器：[HashingTF](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.HashingTF.html)
- minhashLSH：[MinHashLSH](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.MinHashLSH.html)

## 使用方法
使用`spark_dedup.py`对文档进行模糊去重。
```shell
python spark_dedup.py \
    <dataset_path> \
    <result_path> \
    [--tokenizer <tokenizer_type>] \
    [--num_features <num_features>] \
    [--num_hashtables <num_hashtables>] \
    [--text_key <text_key>] \
    [--master_url <master_url>]
# 打印使用信息
python spark_dedup.py --help

```

- `dataset_path`：输入数据集路径。路径的后缀应该是`[json, jsonl, parquet]`中的一个。
- `result_path`：存储带有预测结果数据集的路径。路径的后缀应该是`[json, jsonl, parquet]`中的一个。
- `tokenizer`：（可选。默认值：None）用于对将要分类的文本进行分词的分词器。如果为None，将使用PySpark的[标准分词器](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html#tokenizer)。此外，你可以使用我们提供的分词器`[zh.sp.model, code.sp.model]`中的一个，或者你可以将其设置为你自己的[sentencepiece](https://github.com/google/sentencepiece)模型的路径。
- `num_features`：HashingTF生成的特征数量。默认值为1047576，如megatron-turing-nlg论文中所述。
- `num_hashtables`：（可选。默认值：10）MinHashLSH中使用的哈希数量。默认使用10个哈希，如GPT-3论文中所述。
- `text_key`：（可选。默认值："text"）输入数据集中用于存储待分类文本的字段名称。
- `master_url`：（可选。默认值：None）用于Spark配置的master URL。如果为空，则默认运行在"local[*]"模式下。
