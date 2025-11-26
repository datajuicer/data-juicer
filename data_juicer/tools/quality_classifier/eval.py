# This tool is used for evaluating a quality classifier on your own datasets
# based on PySpark.
#
# We provide several trained models for you. Please refer to the comments at
# the beginning of predict tool for more details.
#
# This tool needs several arguments:
#   - positive_datasets: the paths to the positive datasets. It could be a
#       string for a single dataset, e.g. 'pos.parquet', or a list of strings
#       for several datasets, e.g. '["pos1.parquet", "pos2.parquet"]'.
#   - negative_datasets: the paths to the negative datasets. It could be a
#       string for a single dataset, e.g. 'neg.parquet', or a list of strings
#       for several datasets, e.g. '["neg1.parquet", "neg2.parquet"]'.
#   - model: quality classifier name to apply. It's "gpt3" in default. You can
#       use one of ["gpt3", "chinese", "code"] we provided, or you can set it
#       to the path to your own model trained using the train.py tool.
#   - tokenizer: what tokenizer to use to tokenize texts. It's None in default,
#       which means using the standard Tokenizer of PySpark. You can use one of
#       ["zh.sp.model", "code.sp.model"] we provided, or you can set it to the
#       path to your own sentencepiece model.
#   - text_key: the field key name to hold texts to be classified. It's "text"
#       in default.

import fire
from loguru import logger

from data_juicer.tools.quality_classifier.qc_utils import (
    eval,
    init_spark,
    load_datasets,
)


@logger.catch(reraise=True)
def main(positive_datasets=None, negative_datasets=None, model="my_quality_model", tokenizer=None, text_key="text"):
    """
    Evaluate a trained quality classifier using specific positive/negative
    datasets
    :param positive_datasets: the paths to the positive datasets. It could be a
        string for a single dataset, e.g. 'pos.parquet', or a list of strings
        for multiple datasets, e.g. '["pos1.parquet", "pos2.parquet"]'
    :param negative_datasets: the paths to the negative datasets. It could be a
        string for a single dataset, e.g. 'neg.parquet', or a list of strings
        for multiple datasets, e.g. '["neg1.parquet", "neg2.parquet"]'
    :param model: quality classifier name to apply. It's "my_quality_model" in
        default. You can use one of ["gpt3", "chinese", "code"] we provided, or
        you can set it to the path to your own model trained using the train.py
        tool
    :param tokenizer: what tokenizer to use to tokenize texts. It's None in
        default, which means using the standard Tokenizer of PySpark. You can
        use one of ["zh.sp.model", "code.sp.model"] we provided, or you can set
        it to the path to your own sentencepiece model
    :param text_key: the field key name to hold texts to be classified. It's
        "text" in default
    :return:
    """
    # convert a single dataset to a dataset list
    if positive_datasets is None:
        positive_datasets = []
    if negative_datasets is None:
        negative_datasets = []
    if isinstance(positive_datasets, str):
        positive_datasets = [positive_datasets]
    if isinstance(negative_datasets, str):
        negative_datasets = [negative_datasets]

    # initialize a spark session
    spark = init_spark()

    # load positive and negative datasets
    pos = load_datasets(spark, positive_datasets, text_key=text_key, label=1, only_text=True)
    neg = load_datasets(spark, negative_datasets, text_key=text_key, label=0, only_text=True)

    # merge positive and negative datasets
    if pos is not None and neg is not None:
        ds = pos.unionAll(neg)
    elif pos is not None:
        ds = pos
    elif neg is not None:
        ds = neg
    else:
        logger.error("Empty dataset.")
        exit(0)

    # start evaluation
    logger.info(f"Number of samples: {ds.count()}")
    eval(model, ds, tokenizer)


if __name__ == "__main__":
    fire.Fire(main)
