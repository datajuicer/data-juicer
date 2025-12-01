from typing import Dict, Optional

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, TAGGING_OPS, Mapper

OP_NAME = "query_sentiment_detection_mapper"


@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class QuerySentimentDetectionMapper(Mapper):
    """Predicts user's sentiment label ('negative', 'neutral', 'positive') in a query.

    This mapper takes input from the specified query key and outputs the predicted sentiment
    label and its corresponding score. The results are stored in the Data-Juicer meta field
    under 'query_sentiment_label' and 'query_sentiment_label_score'. It uses a Hugging Face
    model for sentiment detection. If a Chinese-to-English translation model is provided, it
    first translates the query from Chinese to English before performing sentiment analysis."""

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        hf_model: str = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",  # noqa: E501 E131
        zh_to_en_hf_model: Optional[str] = "Helsinki-NLP/opus-mt-zh-en",
        model_params: Dict = {},
        zh_to_en_model_params: Dict = {},
        *,
        label_key: str = MetaKeys.query_sentiment_label,
        score_key: str = MetaKeys.query_sentiment_score,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_model: Huggingface model ID to predict sentiment label.
        :param zh_to_en_hf_model: Translation model from Chinese to English.
            If not None, translate the query from Chinese to English.
        :param model_params: model param for hf_model.
        :param zh_to_en_model_params: model param for zh_to_hf_model.
        :param label_key: The key name in the meta field to store the
            output label. It is 'query_sentiment_label' in default.
        :param score_key: The key name in the meta field to store the
            corresponding label score. It is 'query_sentiment_label_score'
            in default.
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.label_key = label_key
        self.score_key = score_key

        self.model_key = prepare_model(
            model_type="huggingface",
            pretrained_model_name_or_path=hf_model,
            return_pipe=True,
            pipe_task="text-classification",
            **model_params,
        )

        if zh_to_en_hf_model is not None:
            self.zh_to_en_model_key = prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path=zh_to_en_hf_model,
                return_pipe=True,
                pipe_task="translation",
                **zh_to_en_model_params,
            )
        else:
            self.zh_to_en_model_key = None

    def process_batched(self, samples, rank=None):
        metas = samples[Fields.meta]
        if self.label_key in metas[0] and self.score_key in metas[0]:
            return samples

        queries = samples[self.query_key]

        if self.zh_to_en_model_key is not None:
            translator, _ = get_model(self.zh_to_en_model_key, rank, self.use_cuda())
            results = translator(queries)
            queries = [item["translation_text"] for item in results]

        classifier, _ = get_model(self.model_key, rank, self.use_cuda())
        results = classifier(queries)
        labels = [r["label"] for r in results]
        scores = [r["score"] for r in results]

        for i in range(len(metas)):
            metas[i][self.label_key] = labels[i]
            metas[i][self.score_key] = scores[i]

        return samples
