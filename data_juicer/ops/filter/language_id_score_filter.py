from typing import List, Union

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter

OP_NAME = "language_id_score_filter"


@OPERATORS.register_module(OP_NAME)
class LanguageIDScoreFilter(Filter):
    """Filter to keep samples in a specific language with a confidence score above a threshold.

    This operator uses a FastText model to identify the language of each sample. It keeps
    samples that are in the specified language(s) and have a language identification
    confidence score greater than or equal to the minimum score. If no specific language is
    provided, it only filters based on the confidence score. The language ID and its
    confidence score are stored in the 'lang' and 'lang_score' fields of the sample's stats,
    respectively."""

    def __init__(self, lang: Union[str, List[str]] = "", min_score: float = 0.8, *args, **kwargs):
        """
        Initialization method.

        :param lang: Samples in which languages to keep.
        :param min_score: The min language identification confidence
            scores of samples to keep.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if not lang:
            # lang is [], '' or None
            self.lang = None
        elif isinstance(lang, str):
            # lang is a single language string
            self.lang = [lang]
        else:
            # lang is a list of multiple languages
            self.lang = lang
        self.min_score = min_score
        self.model_key = prepare_model(model_type="fasttext")

    def compute_stats_single(self, sample, *args, **kwargs):
        # check if it's computed already
        if StatsKeys.lang in sample[Fields.stats] and StatsKeys.lang_score in sample[Fields.stats]:
            return sample

        text = sample[self.text_key].lower().replace("\n", " ")
        ft_model = get_model(self.model_key)
        if ft_model is None:
            err_msg = "Model not loaded. Please retry later."
            raise ValueError(err_msg)
        pred = ft_model.predict(text)
        lang_id = pred[0][0].replace("__label__", "")
        lang_score = pred[1][0]

        sample[Fields.stats][StatsKeys.lang] = lang_id
        sample[Fields.stats][StatsKeys.lang_score] = lang_score

        return sample

    def process_single(self, sample):
        if self.lang:
            return sample[Fields.stats][StatsKeys.lang] in self.lang and self.get_keep_boolean(
                sample[Fields.stats][StatsKeys.lang_score], self.min_score
            )
        else:
            return self.get_keep_boolean(sample[Fields.stats][StatsKeys.lang_score], self.min_score)
