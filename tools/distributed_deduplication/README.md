# Distributed Fuzzy Deduplication Tools

Help you reproduce and apply fuzzy deduplication to your web datasets similar to GPT-3 paper.

**The General Description about Fuzzy Deduplication**:

The fuzzy deduplication method here mainly refer to the fuzzy deduplication method mentioned in the Appendix A of [GPT-3 paper](https://arxiv.org/pdf/2005.14165.pdf).

> To further improve model quality and prevent overfitting (which becomes increasingly important as model capacity increases), we fuzzily deduplicated documents (i.e. removed documents with high overlap with other documents) within each dataset using Spark’s MinHashLSH implementation with 10 hashes, using **the same features as were used for classification above**. We also fuzzily removed WebText from Common Crawl. Overall this decreased dataset size by an average of 10%.

As the paper mentioned, the features used are the same as were used for  quality classification, as described in [quality_classifier tools](../../data_juicer/tools/quality_classifier/README.md).

The whole toolkit is based on PySpark.

- tokenizer: Since the standard tokenizer of pyspark have trouble tokenizing text in languages such as Chinese, the [standard Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html#tokenizer) of PySpark or [sentencepiece](https://github.com/google/sentencepiece) model are used.
- feature extractor: [HashingTF](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.HashingTF.html)
- minhashLSH: [MinHashLSH](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.MinHashLSH.html)


## Usage

Use `spark_dedup.py` to fuzzily deduplicate documents.

```shell
python spark_dedup.py \
    <dataset_path> \
    <result_path> \
    [--tokenizer <tokenizer_type>] \
    [--num_features <num_features>] \
    [--num_hashtables <num_hashtables>] \
    [--text_key <text_key>] \
    [--master_url <master_url>]

# print the usage message
python spark_dedup.py --help
```

- `dataset_path`: the input dataset path. The suffix of the path should be one of the `[json, jsonl, parquet]`.
- `result_path`: the path to store the dataset with prediction results. The suffix of the path should be one of the `[json, jsonl, parquet]`.
- `tokenizer`: (Optional. Default: None) the tokenizer to tokenize texts to be classified. If it's None, the [standard Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html#tokenizer) of PySpark will be used. Besides, you can use one of the tokenizers we provide `[zh.sp.model, code.sp.model]`. Or you can set it to a path to your own [sentencepiece](https://github.com/google/sentencepiece) model.
- `num_features`: the number of features that HashingTF generates. Default with 1047576 as mentioned in megatron-turing-nlg paper.
- `num_hashtables`: (Optional. Default: 10) the number of hashes used in MinHashLSH. Default with 10 hashes as mentioned in the GPT3 paper.
- `text_key`: (Optional. Default: "text") the field name to store texts to be classified in the input dataset.
- `master_url`: (Optional. Default: None) the master url for spark config. If None, then run with "local[*]"
