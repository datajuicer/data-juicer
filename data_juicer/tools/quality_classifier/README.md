# Data Scoring
English | [中文](README_ZH.md)

## Data Scoring Capabilities

Data-Juicer provides a set of data scoring capabilities to help you evaluate your datasets.

- All [Filter Operators](../../../docs/Operators.md) include a sub-process called `compute_stats`, which calculates statistical measurements based on a pre-defined functional goal of the operator. These measurements are typically derived using simple rules, auxiliary models, or advanced algorithms, such as for perplexity, length, modality matching scores, etc. The `Analyzer` will automatically aggregate these statistics and report them in the resulting dataset.
  
- Prompt-based LLM scoring operators are also available, such as `llm_difficulty_score_filter` and `llm_quality_score_filter`. These operators come with default prompts for general use cases but also offer flexibility for users to customize their own models or specific requirements.

- Additionally, we provide a toolkit to reproduce the GPT-3 quality classifier, as described in the following section.


## Quality Classifier Toolkit (GPT-3 Reproduced)

Help you reproduce and apply quality classifier to your web datasets similar to GPT-3 quality classifier.

The whole toolkit is based on PySpark. And the basic structure of quality classifiers here consists of:
- tokenizer: the [standard Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html##tokenizer) of PySpark or [sentencepiece](https://github.com/google/sentencepiece) model
- feature extractor: [HashingTF](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.HashingTF.html##hashingtf)
- classifier: [LogisticRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html##logisticregression)

### Usage

#### Predict with existing classifiers

Use `predict.py` to predict a document score of "quality" and a label for each sample to indicate whether this sample should be kept according to the score.

```shell
## predict doc_score for a dataset
python predict.py \
    <dataset_path> \
    <result_path> \
    [--model <model_path>] \
    [--tokenizer <tokenizer_type>] \
    [--keep_method <keep_method>] \
    [--text_key <text_key>] \
    [--overall_stats]

## print the usage message
python predict.py --help
```

- `dataset_path`: the input dataset path. The suffix of the path should be one of the `[json, jsonl, parquet]`.
- `result_path`: the path to store the dataset with prediction results. The suffix of the path should be one of the `[json, jsonl, parquet]`.
- `model_path`: (Optional. Default: "gpt3") the path to the model used to predict. You can use one of the models we provide `[gpt3, chinese, code]`. Or you can use the model trained by yourself using the `train.py` script.
- `tokenizer`: (Optional. Default: None) the tokenizer to tokenize texts to be classified. If it's None, the [standard Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html##tokenizer) of PySpark will be used. Besides, you can use one of the tokenizers we provide `[zh.sp.model, code.sp.model]`. Or you can set it to a path to your own [sentencepiece](https://github.com/google/sentencepiece) model.
- `keep_method`: (Optional. Default: "gpt3") the method used to decide whether a sample should be kept according to the doc_score. Should be one of `[gpt3, label]`.
- `text_key`: (Optional. Default: "text") the field name to store texts to be classified in the input dataset.
- `overall_stats`: (Optional. Default: False) whether to generate an overall stats report of document scores.

#### Train your own quality classifier

Use `train.py` to train your own quality classifier for your datasets.

```shell
## train a quality classifier for your own dataset
python train.py \
    <positive_datasets>] \
    <negative_datasets>] \
    [--output_model_path <model_name>] \
    [--num_training_samples <num_training_samples>] \
    [--train_test_split_ratio <train_test_split_ratio>] \
    [--tokenizer <tokenizer_type>] \
    [--evaluation <evaluation>] \
    [--text_key <text_key>]

## print the usage message
python train.py --help
```

- `positive_datasets`: the paths to the positive datasets. It could be a string for a single dataset, e.g. `'pos.parquet'`, or a list of strings for multiple datasets, e.g. `'["pos1.parquet", "pos2.parquet"]'`.
- `negative_datasets`: the paths to the negative datasets. Similar to `positive_datasets`.
- `output_model_path`: (Optional. Default: "my_quality_model") the path to store the trained classifier.
- `num_training_samples`: (Optional. Default: 0) number of samples used to train the model for pos/neg datasets respectively. Default 0 means using all samples to train.
- `train_test_split_ratio`: (Optional. Default: 0.8) ratio to split training set, and the rest of samples will be test set used to evaluate.
- `tokenizer`: (Optional. Default: None) the tokenizer to tokenize texts to be classified. If it's None, the [standard Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html##tokenizer) of PySpark will be used. Besides, you can use one of the tokenizers we provide `[zh.sp.model, code.sp.model]`. Or you can set it to a path to your own [sentencepiece](https://github.com/google/sentencepiece) model.
- `evaluation`: (Optional, Default: True) whether to evaluate the trained classifier using the test set after training.
- `text_key`: (Optional. Default: "text") the field name to store texts to be classified in the input dataset.

#### Evaluate a quality classifier

Use `eval.py` to evaluate a quality classifier to report Precision, Recall, and F1 metrics.

```shell
## evaluate a quality classifier on your own dataset
python eval.py \
    [--positive_datasets <positive_datasets>] \
    [--negative_datasets <negative_datasets>] \
    [--model <model_path>] \
    [--tokenizer <tokenizer_type>] \
    [--text_key <text_key>]

## print the usage message
python eval.py --help
```

- `positive_datasets`: (Optional. Default: None) the paths to the positive datasets. It could be a string for a single dataset, e.g. `'pos.parquet'`, or a list of strings for multiple datasets, e.g. `'["pos1.parquet", "pos2.parquet"]'`.
- `negative_datasets`: (Optional. Default: None) the paths to the negative datasets. Similar to `positive_datasets`.
- `model_path`: (Optional. Default: "my_quality_model") the path to the model to be evaluated. You can evaluate one of the models we provide `[gpt3, chinese, code]`. Or you can evaluate the model trained by yourself using the `train.py` script.
- `tokenizer`: (Optional. Default: None) the tokenizer to tokenize texts to be classified. If it's None, the [standard Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html##tokenizer) of PySpark will be used. Besides, you can use one of the tokenizers we provide `[zh.sp.model, code.sp.model]`. Or you can set it to a path to your own [sentencepiece](https://github.com/google/sentencepiece) model.
- `text_key`: (Optional. Default: "text") the field name to store texts to be classified in the input dataset.

### Model Zoo

We provide 3 models we trained before: `gpt3`, `chinese`, `code`. Each model has its tokenizer and keep method. Tokenizers "xx.sp.model" are trained on the training data using [sentencepiece](https://github.com/google/sentencepiece).

| model     | tokenizer          | keep method      | positive datasets                                  | negative datasets                        |
|-----------|--------------------|------------------|----------------------------------------------------|------------------------------------------|
| `gpt3`    | standard Tokenizer | pareto           | Wikipedia-en & books1 & OpenWebText2               | CommonCrawl                              |
| `chinese` | zh.sp.model        | label            | Wikipedia-zh & Wudao                               | Samples in Chinese from CommonCrawl      |
| `code`    | code.sp.model      | label            | Samples with max_stars_count >= 1372 from TheStack | Random samples from the rest of TheStack |

- `gpt3`: GPT-3 quality classifier reproduced by us.
- `chinese`: A Chinese quality classifier trained by the same pipeline as `gpt3`, but with different tokenizer and training data.
- `code`: (Experimental) A code quality classifier trained by the same pipeline as `gpt3`, but with different tokenizer and training data. We only keep "programming" and "markup" language types of samples for training.
- Experiments of these classifiers on corresponding test sets are shown in the table below:

| model     | Precision  | Recall | F1     |
|-----------|------------|--------|--------|
| `gpt3`    | 96.82%     | 98.14% | 97.47% |
| `chinese` | 98.00%     | 99.30% | 98.64% |
| `code`    | 71.23%     | 54.21% | 61.56% |

- Keep ratios of `gpt3` and `chiense` classifiers on CommonCrawl are shown in the table below:

| model                                | keep ratio @ label  | keep ratio @ pareto |
|--------------------------------------|---------------------|---------------------|
| GPT-3 quality classifier (estimated) | -                   | ~1.3%               |
| `gpt3`                               | 3.22%               | 1.41%               |
| `chinese`                            | 1.81%               | -                   |

### More about Quality Classifier

#### Method

The quality classifiers here mainly refer to the GPT-3 quality classifier mentioned in the Appendix A of GPT-3 paper:

> In order to improve the quality of Common Crawl, we developed an automatic filtering method to remove low quality documents. Using the original WebText as a proxy for high-quality documents, we trained a classifier to distinguish these from raw Common Crawl. We then used this classifier to re-sample Common Crawl by prioritizing documents which were predicted by the classifier to be higher quality. The classifier is trained using logistic regression classifier with features from Spark’s standard tokenizer and HashingTF 10. For the positive examples, we used a collection of curated datasets such as WebText, Wikiedia, and our web books corpus as the positive examples, and for the negative examples, we used unfiltered Common Crawl. We used this classifier to score Common Crawl documents. We kept each document in our dataset iff
>
>     np.random.pareto(α) > 1 − document_score
>
> We chose α = 9 in order to take mostly documents the classifier scored highly, but still include some documents that were out of distribution. α was chosen to match the distribution of scores from our classifier on WebText. We found this re-weighting increased quality as measured by loss on a range of out-of-distribution generative text samples.

#### Tokenizers

- Standard Tokenizer in Spark: split texts by whitespaces.
- zh/code.sp.model: trained using sentencepiece.

#### Keep Methods
- label: `doc_score > 0.5`
- pareto: `doc_score > 1 - np.random.pareto(α), α = 9`
