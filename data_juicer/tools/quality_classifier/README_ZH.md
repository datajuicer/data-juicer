# 给数据打分
中文 | [English](README.md)

## 数据打分能力

Data-Juicer 提供了一组数据打分能力，可帮助您评估数据集。

- 所有 [Filter OPs](../../../docs/Operators.md) 都包含一个名为 `compute_stats` 的子过程，该过程根据运算符的预定义功能目标，计算统计测量值。这些测量值通常使用简单规则、辅助模型或高级算法得出，例如困惑度、长度、模态匹配分数等。`Analyzer` 将自动汇总这些统计数据并将其报告在结果数据集中。

- 我们也提供了基于提示词的 LLM 评分算子，例如 `llm_difficulty_score_filter` 和 `llm_quality_score_filter`。这些算子带有针对一般用例的默认提示，但也为用户提供了灵活性，以自定义特定模型或特定要求。

- 此外，我们还提供了一个工具包来复现 GPT-3 质量分类器，如下一节所述。

## 复现GPT3的质量分类器套件

帮助您复现类似于 GPT-3 质量分类器并将其应用到您的 Web 数据集。

整个工具包基于PySpark，分类器的基本模块包括：

- tokenizer: PySpark 的 [standard Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html##tokenizer) 或 [sentencepiece](https://github.com/google/sentencepiece) 模型
- feature extractor: [HashingTF](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.HashingTF.html##hashingtf)
- classifier: [LogisticRegression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.LogisticRegression.html##logisticregression)

### 用法

#### 使用现有的分类器进行预测

使用 `predict.py` 来预测一个文档的“质量”分数，并为每个样本添加一个标签，以根据分数判断是否应该保留该样本。

```shell
## 预测数据集的 doc_score
python predict.py \
    <dataset_path> \
    <result_path> \
    [--model <model_path>] \
    [--tokenizer <tokenizer_type>] \
    [--keep_method <keep_method>] \
    [--text_key <text_key>] \
    [--overall_stats]

## 打印帮助信息
python predict.py --help
```

- `dataset_path`: 输入数据集路径。要求路径的后缀为 `[json, jsonl, parquet]` 之一。
- `result_path`: 存储带有预测结果的数据集的路径。要求路径的后缀为`[json, jsonl, parquet]`之一。
- `model_path`: (可选，默认为 `gpt3`) 用于预测的模型的路径。您可以使用我们提供的模型之一`[gpt3, chinese,code]`。或者您可以使用`train.py`脚本使用自己训练的模型。
- `tokenizer`: (可选，默认为 None) 用于标记要分类的文本的标记器。 如果为 None，则将使用 PySpark 的 [standard Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html##tokenizer)。 此外，您可以使用我们提供的标记器`[zh.sp.model, code.sp.model]`之一。您也可以将其设置为您自己的 [sentencepiece](https://github.com/google/sentencepiece) 模型的路径。
- `keep_method`: (可选，默认为 `gpt3`) 根据 doc_score 决定是否保留样本的方法。应为 `[gpt3, label]` 之一。
- `text_key`: (可选，默认为 `text`) 用于存储输入数据集中需要被分类的文本的字段名称。
- `overall_stats`: (可选，默认为 False) 是否生成文档分数的汇总统计报告。

#### 训练自己的质量分类器

使用`train.py`在您的数据集上训练您自己的质量分类器。

```shell
## 为自己的数据集训练质量分类器
python train.py \
    <positive_datasets>] \
    <negative_datasets>] \
    [--output_model_path <model_name>] \
    [--num_training_samples <num_training_samples>] \
    [--train_test_split_ratio <train_test_split_ratio>] \
    [--tokenizer <tokenizer_type>] \
    [--evaluation <evaluation>] \
    [--text_key <text_key>]

## 打印帮助信息
python train.py --help
```

- `positive_datasets`: 正样本数据集的路径。可以是单个数据集的字符串，例如 `'pos.parquet'`，或多个数据集的字符串列表，例如 `'["pos1.parquet", "pos2.parquet"]'`。
- `negative_datasets`: 负样本数据集的路径，配置方法与 `positive_datasets` 相似。
- `output_model_path`: (可选，默认值为 `my_quality_model`) 存储训练好的分类器的路径。
- `num_training_samples`: (可选，默认值为 0) 分别用于训练 正/负样本数据集模型的样本数量。 默认0表示使用所有样本进行训练。
- `train_test_split_ratio`: (可选，默认值为0.8) 分割训练集的比率，其余样本将作为测试集用于评估。
- `tokenizer`: (可选，默认值为None) 用于对要分类的文本进行标记的标记生成器。如果为 None，则将使用 PySpark 的[标准 Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html##tokenizer) 此外，您可以使用我们提供的标记器`[zh.sp.model，code.sp.model]`之一。也可以将其设置为您自己的 [sentencepiece](https://github.com/google/sentencepiece) 模型的路径。
- `evaluation`: (可选，默认值为 True) 是否在训练后使用测试集评估训练好的分类器。
- `text_key`: (可选，默认值为 `text`) 用于存储输入数据集中需要被分类的文本的字段名称。

#### 评估质量分类器

使用`eval.py`以报告精度、召回率和 F1 指标来评估质量分类器。

```shell
## 在自己的数据集上评估质量分类器
python eval.py \
    [--positive_datasets <positive_datasets>] \
    [--negative_datasets <negative_datasets>] \
    [--model <model_path>] \
    [--tokenizer <tokenizer_type>] \
    [--text_key <text_key>]

## 打印帮助信息
python eval.py --help
```

- `positive_datasets`: (Optional. Default: None) the paths to the positive datasets. It could be a string for a single dataset, e.g. `'pos.parquet'`, or a list of strings for multiple datasets, e.g. `'["pos1.parquet", "pos2.parquet"]'`.
- `negative_datasets`: (Optional. Default: None) the paths to the negative datasets. Similar to `positive_datasets`.
- `model_path`: (Optional. Default: "my_quality_model") the path to the model to be evaluated. You can evaluate one of the models we provide `[gpt3, chinese, code]`. Or you can evaluate the model trained by yourself using the `train.py` script.
- `tokenizer`: (Optional. Default: None) the tokenizer to tokenize texts to be classified. If it's None, the [standard Tokenizer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Tokenizer.html##tokenizer) of PySpark will be used. Besides, you can use one of the tokenizers we provide `[zh.sp.model, code.sp.model]`. Or you can set it to a path to your own [sentencepiece](https://github.com/google/sentencepiece) model.
- `text_key`: (Optional. Default: "text") the field name to store texts to be classified in the input dataset.

### Model Zoo

我们提供了已训练好的三个模型：`gpt3`，`chinese`，`code`。每个模型都有其 tokenizer 和 keep method。其中Tokenizer `xx.sp.model` 使用 [sentencepiece](https://github.com/google/sentencepiece) 的训练数据进行训练。

| model     | tokenizer          | keep method      | positive datasets                                  | negative datasets                        |
|-----------|--------------------|------------------|----------------------------------------------------|------------------------------------------|
| `gpt3`    | standard Tokenizer | pareto           | Wikipedia-en & books1 & OpenWebText2               | CommonCrawl                              |
| `chinese` | zh.sp.model        | label            | Wikipedia-zh & Wudao                               | Samples in Chinese from CommonCrawl      |
| `code`    | code.sp.model      | label            | Samples with max_stars_count >= 1372 from TheStack | Random samples from the rest of TheStack |

- `gpt3`: 我们复现的 GPT-3质量分类器。
- `chinese`: 通过与`gpt3`相同的流程训练的中文质量分类器，但使用不同的标记器和训练数据。
- `code`: (Experimental) 通过与`gpt3`相同的流程进行训练，但使用不同的标记器和训练数据得到的代码质量分类器。我们只保留 “programming” 和 “markup” 语言类型的样本进行训练。
- 这些分类器在相应测试集上的实验如下表所示：

| model     | Precision  | Recall | F1     |
|-----------|------------|--------|--------|
| `gpt3`    | 96.82%     | 98.14% | 97.47% |
| `chinese` | 98.00%     | 99.30% | 98.64% |
| `code`    | 71.23%     | 54.21% | 61.56% |

- Common Crawl 上 `gpt3`和 `chinese` 分类器的 keep ratio 如下表所示：

| model                                | keep ratio @ label  | keep ratio @ pareto |
|--------------------------------------|---------------------|---------------------|
| GPT-3 quality classifier (estimated) | -                   | ~1.3%               |
| `gpt3`                               | 3.22%               | 1.41%               |
| `chinese`                            | 1.81%               | -                   |

### 有关质量分类器的更多信息

#### 方法

这里的质量分类器主要参考GPT-3论文附录A中提到的GPT-3质量分类器：

> In order to improve the quality of Common Crawl, we developed an automatic filtering method to remove low quality documents. Using the original WebText as a proxy for high-quality documents, we trained a classifier to distinguish these from raw Common Crawl. We then used this classifier to re-sample Common Crawl by prioritizing documents which were predicted by the classifier to be higher quality. The classifier is trained using logistic regression classifier with features from Spark’s standard tokenizer and HashingTF 10. For the positive examples, we used a collection of curated datasets such as WebText, Wikiedia, and our web books corpus as the positive examples, and for the negative examples, we used unfiltered Common Crawl. We used this classifier to score Common Crawl documents. We kept each document in our dataset iff
>
>     np.random.pareto(α) > 1 − document_score
>
> We chose α = 9 in order to take mostly documents the classifier scored highly, but still include some documents that were out of distribution. α was chosen to match the distribution of scores from our classifier on WebText. We found this re-weighting increased quality as measured by loss on a range of out-of-distribution generative text samples.

#### Tokenizers

- Spark 中的标准 Tokenizer: 根据空白字符分割文本.
- zh/code.sp.model: 使用 sentencepiece 训练得到。

#### Keep Methods

- label: `doc_score > 0.5`
- pareto: `doc_score > 1 - np.random.pareto(α), α = 9`
