# This tool is used for training a quality classifier for your own datasets
# based on PySpark.
#
# After training, this tool will generate a classifier model in a specific
# directory. You can use it to evaluate or predict on other datasets using eval
# and predict tools.
#
# This tool needs several arguments:
#   - positive_datasets: the paths to the positive datasets. It could be a
#       string for a single dataset, e.g. 'pos.parquet', or a list of strings
#       for several datasets, e.g. '["pos1.parquet", "pos2.parquet"]'.
#   - negative_datasets: the paths to the negative datasets. It could be a
#       string for a single dataset, e.g. 'neg.parquet', or a list of strings
#       for several datasets, e.g. '["neg1.parquet", "neg2.parquet"]'.
#   - output_model_path: the path to store the trained quality classifier. It's
#       "my_quality_model" in default.
#   - num_training_samples: number of samples used to train the model. It's 0
#       in default, which means using all samples in datasets to train.
#   - train_test_split_ratio: ratio to split train and test set. It's 0.8 in
#       default.
#   - tokenizer: what tokenizer to use to tokenize texts. It's None in default,
#       which means using the standard Tokenizer of PySpark. You can use one of
#       ["zh.sp.model", "code.sp.model"] we provided, or you can set it to the
#       path to your own sentencepiece model.
#   - evaluation: whether to evaluate the model after training using test set.
#       It's True in default.
#   - text_key: the field key name to hold texts to be classified. It's "text"
#       in default.

import fire
from loguru import logger

from data_juicer.tools.quality_classifier.qc_utils import (
    eval,
    init_spark,
    load_datasets,
    shuffle,
    train,
)


@logger.catch(reraise=True)
def main(
    positive_datasets,
    negative_datasets,
    output_model_path="my_quality_model",
    num_training_samples=0,
    train_test_split_ratio=0.8,
    tokenizer=None,
    evaluation=True,
    text_key="text",
):
    """
    Train a quality classifier using your own pos/neg datasets
    :param positive_datasets: the paths to the positive datasets. It could be a
        string for a single dataset, e.g. 'pos.parquet', or a list of strings
        for several datasets, e.g. '["pos1.parquet", "pos2.parquet"]'
    :param negative_datasets: the paths to the negative datasets. It could be a
        string for a single dataset, e.g. 'neg.parquet', or a list of strings
        for several datasets, e.g. '["neg1.parquet", "neg2.parquet"]'
    :param output_model_path: the path to store the trained quality classifier.
        It's "my_quality_model" in default
    :param num_training_samples: number of samples used to train the model.
        It's 0 in default, which means using all samples in datasets to train
    :param train_test_split_ratio: ratio to split train and test set. It's 0.8
        in default
    :param tokenizer: what tokenizer to use to tokenize texts. It's None in
        default, which means using the standard Tokenizer of PySpark. You can
        use one of ["zh.sp.model", "code.sp.model"] we provided, or you can set
        it to the path to your own sentencepiece model
    :param evaluation: whether to evaluate the model after training using test
        set. It's True in default
    :param text_key: the field key name to hold texts to be classified. It's
        "text" in default
    :return:
    """
    # convert a single dataset to a dataset list
    if isinstance(positive_datasets, str):
        positive_datasets = [positive_datasets]
    if isinstance(negative_datasets, str):
        negative_datasets = [negative_datasets]

    # initialize a spark session
    spark = init_spark()

    # load positive and negative datasets
    pos = load_datasets(spark, positive_datasets, text_key=text_key, label=1, only_text=True)
    neg = load_datasets(spark, negative_datasets, text_key=text_key, label=0, only_text=True)

    if pos is None or neg is None:
        logger.error("Empty dataset in positive/negative dataset list...")
        exit(1)

    # sample a part of positive/negative samples to train
    if num_training_samples > 0:
        logger.info(f"Only use {num_training_samples} pairs samples to train.")
        pos = shuffle(pos).limit(num_training_samples)
        neg = shuffle(neg).limit(num_training_samples)

    # merge pos and neg samples
    ds = pos.unionAll(neg)
    # split the merged dataset into training and test set
    train_set, test_set = ds.randomSplit([train_test_split_ratio, 1.0 - train_test_split_ratio], seed=42)
    logger.info(f"Number of training samples: {train_set.count()}, " f"test samples: {test_set.count()}")

    # start the ML pipeline to train the classifier
    train(output_model_path, train_set, tokenizer)

    # evaluate the trained model on test set
    if evaluation:
        eval(output_model_path, test_set, tokenizer)


if __name__ == "__main__":
    fire.Fire(main)
