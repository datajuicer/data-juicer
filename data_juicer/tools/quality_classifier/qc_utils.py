import os
import zipfile

import numpy as np
import wget
from loguru import logger
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, udf
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType

from data_juicer.utils.cache_utils import DATA_JUICER_MODELS_CACHE
from data_juicer.utils.model_utils import MODEL_LINKS, prepare_sentencepiece_for_lang


def init_spark(spark_executor_memory=None, spark_driver_memory=None, spark_executor_memoryOverhead=None):
    """
    Initialize a spark session. You can set parameters such as memory, number
    of partitions, timeout and so on here
    :return: A spark session instance.
    """
    if not spark_executor_memory:
        spark_executor_memory = "64g"
    if not spark_driver_memory:
        spark_driver_memory = "64g"
    if not spark_executor_memoryOverhead:
        spark_executor_memoryOverhead = "20000"
    spark = (
        SparkSession.builder.master("local[*]")
        .config("spark.driver.memory", spark_driver_memory)
        .config("spark.executor.memory", spark_executor_memory)
        .config("spark.sql.shuffle.partitions", "300")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.executor.memoryOverhead", spark_executor_memoryOverhead)
        .config("spark.network.timeout", "10000s")
        .config("spark.executor.heartbeatInterval", "3600s")
        .getOrCreate()
    )
    logger.info("Spark initialization done.")
    return spark


def prepare_model(model_name, model_path=DATA_JUICER_MODELS_CACHE):
    """
    Prepare the specific model from model cache path or the remote oss
    :param model_name: name of the quality classifier model
    :param model_path: the path to store the model to be loaded
    :return: a loaded PipelineModel
    """
    udm = False
    if model_name not in ["gpt3", "chinese", "code"]:
        # use user-specific model
        real_model_path = model_name
        udm = True
    else:
        # use prepared models we provided
        model_name = "%s_quality_model" % model_name
        real_model_path = os.path.join(model_path, model_name)
    logger.info(f"Preparing scorer model in [{real_model_path}]...")
    if os.path.exists(real_model_path) and os.path.isdir(real_model_path):
        return PipelineModel.load(real_model_path)
    if udm:
        logger.error(f"Customized model [{real_model_path}] cannot be loaded.")
        exit(0)
    # No specific models in local file systems. Download them from remote.
    os.makedirs(model_path, exist_ok=True)
    wget.download(
        os.path.join(MODEL_LINKS, f"{model_name}.zip"), os.path.join(model_path, f"{model_name}.zip"), bar=None
    )
    # extract the compressed model file into a model directory
    with zipfile.ZipFile(os.path.join(model_path, f"{model_name}.zip")) as zp:
        zp.extractall(os.path.join(model_path))
    return PipelineModel.load(real_model_path)


def load_dataset(spark, ds_path, text_key="text", only_text=False):
    """
    Load a single dataset using PySpark. Only support 'json', 'jsonl', or
    'parquet' files for now
    :param spark: spark session
    :param ds_path: dataset path
    :param text_key: the name of the column that stores the contents of texts
    :param only_text: whether to load texts only and drop other columns.
    :return: a data frame
    """
    # load dataset using different methods according to the suffix
    logger.info(f"Loading dataset from [{ds_path}]...")
    if ds_path.endswith(".json") or ds_path.endswith(".jsonl"):
        df = spark.read.json(ds_path)
    elif ds_path.endswith(".parquet"):
        df = spark.read.parquet(ds_path)
    else:
        raise NotImplementedError(
            "Dataset type is not supported for now. "
            "Suffix of dataset file should be one of "
            "[.json, .jsonl, .parquet]"
        )
    # rename the column that stores texts to "text" if necessary
    if text_key != "text":
        df = df.withColumnRenamed(text_key, "text")
    # whether to keep "text" column only
    if only_text:
        return df.select("text")
    else:
        return df


def load_datasets(spark, ds_paths, text_key="text", label=None, only_text=True):
    """
    Load a list of datasets. Only support 'json', 'jsonl', or 'parquet' files
    for now
    :param spark: spark session
    :param ds_paths: a list of datasets to be loaded.
    :param text_key: the name of the column that stores the contents of texts
    :param label: the label set to these datasets. Used in training pipeline
    :param only_text: whether to load texts only and drop other columns.
    :return: a data frame
    """
    if len(ds_paths) == 0:
        logger.warning("No dataset path provided.")
        return None
    # load each dataset in order and union them all
    base_ds = load_dataset(spark, ds_paths[0], text_key, only_text)
    for i in range(1, len(ds_paths)):
        base_ds = base_ds.unionAll(load_dataset(spark, ds_paths[i], text_key, only_text))
    if label is not None:
        # add labels for training pipeline
        return base_ds.selectExpr("text", "%d as label" % label)
    else:
        return base_ds


def shuffle(df):
    """
    Shuffle a data frame
    :param df: input data frame
    :return: shuffled data frame
    """
    temp_df = df.withColumn("rand", rand(seed=42))
    df_rnd = temp_df.orderBy(temp_df.rand)
    return df_rnd.drop(df_rnd.rand)


def export_result(ds, res_path):
    """
    Export a dataset to specified path. Only support 'json', 'jsonl', or
    'parquet' export formats for now
    :param ds: the dataset to be exported
    :param res_path: the path to store the exported dataset
    :return:
    """
    logger.info(f"Exporting predicted result to [{res_path}]")
    if res_path.endswith(".json") or res_path.endswith(".jsonl"):
        ds.write.mode("overwrite").format("json").save(res_path)
    elif res_path.endswith(".parquet"):
        ds.write.mode("overwrite").format("parquet").save(res_path)
    else:
        ds.write.mode("overwrite").save(res_path)


def get_keep_method_udf(keep_method):
    """
    Given the name of keep method, return a PySpark user-defined function of
    this kind of keep method. Only support 'gpt3' or 'label' for now
    :param keep_method: name of keep method
    :return: a PySpark udf of specified keep method
    """
    if keep_method == "label":
        return udf(lambda score: int(score > 0.5), IntegerType())
    elif keep_method == "gpt3":
        pareto = 9
        return udf(lambda score: int(score > 1 - np.random.pareto(pareto)), IntegerType())
    else:
        raise NotImplementedError(f"Keep method [{keep_method}] is not " f"implemented for now.")


def tokenize_dataset(ds, tokenizer):
    """
    Tokenize the texts in input dataset using specified tokenizer
    :param ds: dataset to be tokenized
    :param tokenizer: tokenizer used to tokenize texts
    :return: a dataset with an extra column "words" that stores the tokenized
        texts
    """
    tkn = prepare_sentencepiece_for_lang("", tokenizer)
    # create a PySpark udf to tokenize the dataset
    tokenizer_udf = udf(lambda text: tkn.encode_as_pieces(text), ArrayType(StringType()))
    logger.info("Tokenize texts using specific tokenizer...")
    return ds.withColumn("words", tokenizer_udf(col("text")))


def train(output_model_path, ds, tokenizer=None):
    """
    Train a quality classifier with training dataset and export the trained
    model to a specified path
    :param output_model_path: the path to store the trained model
    :param ds: training dataset
    :param tokenizer: specified sentencepiece tokenizer. It's None in default,
        which means using the standard Tokenizer in PySpark
    :return:
    """
    logger.info("Preparing training quality classifier model...")
    if tokenizer:
        # tokenizer is not standard Tokenizer in PySpark, need to apply it
        # explicitly
        ds = tokenize_dataset(ds, tokenizer)

    # model
    hashingTF = HashingTF(inputCol="words", outputCol="features")
    lr = LogisticRegression()
    if tokenizer is None:
        # using standard Tokenizer in PySpark
        std_tokenizer = Tokenizer(inputCol="text", outputCol="words")
        pipeline = Pipeline(stages=[std_tokenizer, hashingTF, lr])
    else:
        # using extra sentencepiece tokenizer, which will not included in the
        # final PipelineModel
        pipeline = Pipeline(stages=[hashingTF, lr])

    logger.info("Start training...")
    model = pipeline.fit(ds)

    logger.info("Trained model saving...")
    model.write().overwrite().save(output_model_path)


def eval(model_path, ds, tokenizer=None):
    """
    Evaluate a quality classifier model on specified dataset
    :param model_path: the path to the model to be evaluated
    :param ds: evaluation dataset
    :param tokenizer: specified sentencepiece tokenizer. It's None in default,
        which means using the standard Tokenizer in PySpark
    :return:
    """
    logger.info("Preparing to evaluate...")
    if tokenizer:
        # tokenizer is not standard Tokenizer in PySpark, need to apply it
        # explicitly
        ds = tokenize_dataset(ds, tokenizer)

    logger.info("Start evaluation...")
    model = prepare_model(model_path)
    pred = model.transform(ds)
    # get positive and negative samples
    P = pred.filter("label = 1")
    N = pred.filter("label = 0")
    # get TP, FP, TN, FN samples
    TP = P.filter("prediction = 1").count() + 1
    FP = N.filter("prediction = 1").count() + 1
    TN = N.filter("prediction = 0").count() + 1
    FN = P.filter("prediction = 0").count() + 1
    # compute precision, recall and F1 metrics
    precision = 1.0 * TP / (TP + FP)
    recall = 1.0 * TP / P.count()
    F1 = 2.0 * precision * recall / (precision + recall)
    logger.info(f"TP: {TP}, FN: {FN}")
    logger.info(f"FP: {FP}, TN: {TN}")
    logger.info(f"P: {precision}, R: {recall}, F1: {F1}")


def predict(model, ds, tokenizer=None, keep_method="label"):
    """
    Predict document scores for a dataset using a trained quality classifier
    model
    :param model: the model used to predict
    :param ds: the dataset to be predicted
    :param tokenizer: specified sentencepiece tokenizer. It's None in default,
        which means using the standard Tokenizer in PySpark
    :param keep_method: name of keep method to label the "should_keep" column
    :return:
    """
    logger.info("Start scoring dataset...")
    if tokenizer:
        # tokenizer is not standard Tokenizer in PySpark, need to apply it
        # explicitly
        ds = tokenize_dataset(ds, tokenizer)

    prediction = model.transform(ds)

    # A UDF to extract doc scores from probability vectors
    def extract_prob(v):
        try:
            return float(v[1])
        except ValueError:
            return None

    # extract the predicted probability as the doc_score
    extract_prob_udf = udf(extract_prob, DoubleType())
    doc_score = prediction.withColumn("doc_score", extract_prob_udf(col("probability")))

    # A UDF to get the bool value indicating whether this sample should be kept
    should_keep_label_udf = get_keep_method_udf(keep_method)
    should_keep = doc_score.withColumn("should_keep", should_keep_label_udf(col("doc_score")))
    # drop extra useless columns
    return should_keep.drop("words", "features", "rawPrediction", "probability", "prediction")
