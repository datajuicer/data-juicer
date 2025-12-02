# This tool is used for predicting a document score for text samples using
# quality classifier models we provided, including:
#   - gpt3: A GPT3 quality classifier reproduced from scratch by us based on
#       PySpark. It's trained over CC as negative samples and Wikipedia-en,
#       Books, OpenWebText as positive samples.
#   - chinese: A quality classifier for Chinese. It's trained over Chinese
#       texts sampled from CC as negative samples and Wudao, Wikipedia-zh as
#       positive samples.
#   - code: A quality classifier for codes. It's trained over code samples that
#       have stars >= 1372 as positive samples and random samples from left
#       data as negative samples. Stars count 1372 splits a nearly 700w subset
#       with most stars.
# All these 3 classifiers are trained using the same training pipeline as GPT3
# based on PySpark but with different tokenizers and keeping methods:
#   - gpt3: standard Tokenizer from spark & GPT3 keeping method based on pareto
#   - chinese: sentencepiece tokenizer for Chinese & label
#   - code: sentencepiece tokenizer for code & label
#
# This tool needs several arguments:
#   - dataset_path: the path to the dataset you want to predict doc_scores for.
#   - result_path: the path to store the predicted result dataset.
#   - model: quality classifier name to apply. It's "gpt3" in default. You can
#       use one of ["gpt3", "chinese", "code"] we provided, or you can set it
#       to the path to your own model trained using the train.py tool.
#   - tokenizer: what tokenizer to use to tokenize texts. It's None in default,
#       which means using the standard Tokenizer of PySpark. You can use one of
#       ["zh.sp.model", "code.sp.model"] we provided, or you can set it to the
#       path to your own sentencepiece model.
#   - keep_method: the method to label should_keep field for each sample. It's
#       "gpt3" in default. Should be one of ["gpt3", "label"].
#   - text_key: the field key name to hold texts to be classified. It's "text"
#       in default.
#   - overall_stats: whether to output an overall stats report on predicted
#       document scores. It's False in default.
#
# Recommended arguments for provided trained models:
#   - gpt3:
#       - model: gpt3
#       - tokenizer: None
#       - keep_method: gpt3
#   - chinese:
#       - model: chinese
#       - tokenizer: zh.sp.model
#       - keep_method: label
#   - code:
#       - model: code
#       - tokenizer: code.sp.model
#       - keep_method: label
#
# Notice:
#   1. The configs of SparkSession in function init_spark can be modified to be
#       more suitable for your own machine. See function init_spark in
#       qc_utils.py.
#   2. Random factors are involved in "gpt3" model. So you might get different
#       should_keep label in different running processes. But you should get
#       same doc_score predictions in different running processes.

import os

import fire
from loguru import logger

from data_juicer.tools.quality_classifier.qc_utils import (
    export_result,
    init_spark,
    load_dataset,
    predict,
    prepare_model,
)


@logger.catch(reraise=True)
def predict_score(
    dataset_path, result_path, model="gpt3", tokenizer=None, keep_method="gpt3", text_key="text", overall_stats=False
):
    """
    Use specific quality classifier to predict document scores on your dataset
    :param dataset_path: the path to the dataset you want to predict for
    :param result_path: the path to store the predicted result dataset
    :param model: quality classifier name to apply. It's "gpt3" in default. You
        can use one of ["gpt3", "chinese", "code"] we provided, or you can set
        it to the path to your own model trained using the train.py tool
    :param tokenizer: what tokenizer to use to tokenize texts. It's None in
        default, which means using the standard Tokenizer of PySpark. You can
        use one of ["zh.sp.model", "code.sp.model"] we provided, or you can set
        it to the path to your own sentencepiece model
    :param keep_method: the method to label should_keep field for each sample.
        It's "gpt3" in default. Should be one of ["gpt3", "label"]
    :param text_key: the field key name to hold texts to be classified. It's
        "text" in default
    :param overall_stats: whether to output an overall stats report on
        predicted document scores. It's False in default
    :return:
        None if overall_stats is False
        average quality score of the document if overall_stats is True
    """
    # set default tokenizers for default models
    if model == "chinese":
        tokenizer = "zh.sp.model"
        keep_method = "label"
    if model == "code":
        tokenizer = "code.sp.model"
        keep_method = "label"
    if model == "gpt3":
        tokenizer = None
        keep_method = "gpt3"

    # initialize a spark session
    if "_JAVA_OPTIONS" in os.environ and "-Djava.net.preferIPv6Addresses=true" in os.environ["_JAVA_OPTIONS"]:
        os.environ["_JAVA_OPTIONS"] = os.environ["_JAVA_OPTIONS"].replace(
            "-Djava.net.preferIPv6Addresses=true", "-Djava.net.preferIPv6Addresses=false"
        )
    spark = init_spark()
    # load the quality classifier model
    model = prepare_model(model_name=model)
    # load dataset
    ds = load_dataset(spark, dataset_path, text_key=text_key)
    # start to predict
    pred = predict(model, ds, tokenizer=tokenizer, keep_method=keep_method)
    # export prediction result to specific path
    export_result(pred, result_path)

    if overall_stats:
        # generate overall statistics on doc scores
        overall = pred.select("doc_score").toPandas().describe(include="all")
        # export to result report file
        overall.to_csv(os.path.join(result_path, "overall.csv"))
        overall.to_markdown(os.path.join(result_path, "overall.md"))
        return overall


if __name__ == "__main__":
    fire.Fire(predict_score)
