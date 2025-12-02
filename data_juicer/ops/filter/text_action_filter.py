from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import remove_special_tokens
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter

OP_NAME = "text_action_filter"


@OPERATORS.register_module(OP_NAME)
class TextActionFilter(Filter):
    """Filter to keep texts that contain a minimum number of actions.

    This operator uses a Spacy model to detect actions in the text. It keeps samples if the
    number of detected actions meets or exceeds the specified minimum. The supported
    languages are English ('en') and Chinese ('zh'). The 'num_action' statistic is computed
    and cached for each sample. Actions are identified based on part-of-speech (POS) tags
    and specific tags for verbs."""

    def __init__(self, lang: str = "en", min_action_num: int = 1, *args, **kwargs):
        """
        Initialization method.

        :param lang: language of the text in the samples. 'en' for detection of
            actions in English and 'zh' for detection of actions in Chinese.
        :param min_action_num: The min action number in the filtering. samples
            will be filtered if their action number in the text is below this
            parameter.
        """
        super().__init__(*args, **kwargs)
        LazyLoader.check_packages(["spacy-pkuseg"], "--no-deps")

        if lang not in ["en", "zh"]:
            raise ValueError(
                f"Language [{lang}] is not supported in action detection." f'Can only be one of ["en", "zh"].'
            )
        self.lang = lang
        self.model_key = prepare_model(model_type="spacy", lang=lang)
        self.action_poss = ["VERB"]
        self.action_tags = ["VV", "VB", "VBP", "VBZ", "VBD", "VBG", "VBN"]
        self.min_action_num = min_action_num

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.num_action in sample[Fields.stats]:
            return sample

        text = remove_special_tokens(sample[self.text_key])

        # process text via spacy and count the actions in text
        model = get_model(self.model_key)
        doc = model(text)
        num_action = 0
        for token in doc:
            if token.pos_ in self.action_poss and token.tag_ in self.action_tags:
                num_action += 1
        sample[Fields.stats][StatsKeys.num_action] = num_action

        return sample

    def process_single(self, sample):
        num_action = sample[Fields.stats][StatsKeys.num_action]
        return self.get_keep_boolean(num_action, self.min_action_num)
