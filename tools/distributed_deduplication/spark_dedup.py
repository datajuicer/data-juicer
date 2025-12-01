import sys
import time
from typing import Optional

import fire
from loguru import logger
from pyspark.ml.feature import HashingTF, MinHashLSH, Tokenizer
from pyspark.sql import functions as F
from pyspark.sql.functions import posexplode

from data_juicer.tools.quality_classifier.qc_utils import (
    export_result,
    load_dataset,
    tokenize_dataset,
)
from tools.distributed_deduplication.dedup_utils import (
    find_components,
    generate_edges,
    init_spark,
)


@logger.catch
def dedup_dataset(
    dataset_path: str,
    result_path: str,
    tokenizer: Optional[str] = None,
    num_features: int = 1047576,
    num_hashtables: int = 10,
    text_key: str = "text",
    master_url: Optional[str] = None,
):
    """
    Perform fuzzy text deduplication on the given dataset.
    :param dataset_path: the path to the dataset to perform deduplication,
        The suffix of the path should be one of the json, jsonl, parquet.
    :param result_path: the path to store the predicted result dataset.
    :param tokenizer: what tokenizer to use to tokenize texts. It's None in
        default, which means using the standard Tokenizer of PySpark. You can
        use one of ["zh.sp.model", "code.sp.model"] we provided, or you can set
        it to the path to your own sentencepiece model.
    :param num_features: the number of features that HashingTF generates.
        Default with 1047576 as mentioned in megatron-turing-nlg paper.
    :param num_hashtables: the number of hashes used in MinHashLSH.
        Default with 10 hashes as mentioned in the GPT3 paper.
    :param text_key: the field key name to hold texts to be classified. It's
        "text" in default.
    :param master_url: the master url for spark config. Default is None.
        If None, then run with local[*].
    """
    # for inited cluster,
    # provide master url such as "spark://master:7077"
    spark = init_spark(master_url=master_url)
    ds = load_dataset(spark, dataset_path, text_key=text_key)
    ds = ds.withColumn("id", F.monotonically_increasing_id()).cache()
    df = ds

    if tokenizer:
        ds = tokenize_dataset(ds, tokenizer)
    else:
        ds = Tokenizer(inputCol="text", outputCol="words").transform(ds)

    hashingTF = HashingTF(inputCol="words", outputCol="features", numFeatures=num_features)
    ds = hashingTF.transform(ds)

    minHash = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=num_hashtables)
    model = minHash.fit(ds)

    ds = model.transform(ds)

    ds = ds.select("id", posexplode("hashes").alias("band_idx", "hash_vector"))

    record = ds.rdd.map(lambda x: (x["band_idx"], int(x["hash_vector"][0]), x["id"]))

    edges = (
        record.groupBy(lambda x: (x[0], x[1]))
        .flatMap(lambda x: generate_edges([i[2] for i in x[1]]))
        .distinct()
        .cache()
    )

    results = find_components(edges)
    if len(results) == 0:
        logger.info("No components found.")
        sys.exit(0)

    components = spark.createDataFrame(results, schema=["id", "component"]).sort(["component", "id"])
    components.show()
    df = df.join(components, on="id", how="left")
    df = df.filter(F.col("component").isNull()).drop("id", "component").cache()
    export_result(df, result_path)


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(dedup_dataset)
    etime = time.time()
    logger.info(f"Execution Done, Total time {etime - stime}")
