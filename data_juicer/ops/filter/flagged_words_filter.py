# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
#
# The flagged words list comes from https://huggingface.co/spaces/huggingface/text-data-filtering
# and https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
# --------------------------------------------------------

from typing import List

from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ...utils.asset_utils import ASSET_DIR, load_words_asset
from ..base_op import OPERATORS, Filter
from ..common import SPECIAL_CHARACTERS, get_words_from_document, words_refinement
from ..op_fusion import INTER_WORDS

OP_NAME = "flagged_words_filter"


@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)
class FlaggedWordFilter(Filter):
    """Filter to keep samples with flagged-word ratio in a specified range.

    This operator filters out samples based on the ratio of flagged words. It uses a list of
    flagged words, which can be language-specific or combined from multiple languages. The
    flagged-word ratio is computed as the number of flagged words divided by the total
    number of words in the sample. If tokenization is enabled, a Hugging Face tokenizer is
    used to split the text into words. The operator supports word augmentation for certain
    languages, which can be configured. The key metric, 'flagged_words_ratio', is cached and
    reused if already computed. Samples are kept if their flagged-word ratio falls within
    the specified min and max ratio."""

    _batched_op = True

    def __init__(
        self,
        lang: str = "en",
        tokenization: bool = False,
        min_ratio: float = 0.0,
        max_ratio: float = 0.045,
        flagged_words_dir: str = ASSET_DIR,
        use_words_aug: bool = False,
        words_aug_group_sizes: List[PositiveInt] = [2],
        words_aug_join_char: str = "",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param lang: Consider flagged words in what language. If lang ==
            "all", we will adopt the one merged from all the available
            languages
        :param tokenization: Whether to use model to tokenize documents
        :param min_ratio: The min filter ratio in this op.
        :param max_ratio: The max filter ratio in this op.
        :param flagged_words_dir: The directory storing the
            flagged_words file(s) whose name includes "flagged_words"
            and in json format
        :param use_words_aug: Whether to augment words, especially for
            Chinese and Vietnamese
        :param words_aug_group_sizes: The group size of words to augment
        :param words_aug_join_char: The join char between words to
            augment
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.lang = lang
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.use_words_aug = use_words_aug
        self.words_aug_group_sizes = words_aug_group_sizes
        self.words_aug_join_char = words_aug_join_char
        self.model_key = None

        self.FLAGGED_WORDS = load_words_asset(words_dir=flagged_words_dir, words_type="flagged_words")

        if "all" not in self.FLAGGED_WORDS:
            self.FLAGGED_WORDS["all"] = [val for vals in self.FLAGGED_WORDS.values() for val in vals]
        if tokenization:
            self.model_key = prepare_model(model_type="sentencepiece", lang=lang)

    def compute_stats_batched(self, samples, context=False):
        # check if it's computed already
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]
        words_key = f"{InterVars.words}-{self.model_key}"
        tokenizer = get_model(self.model_key)
        for idx, stat in enumerate(samples_stats):
            if StatsKeys.flagged_words_ratio in stat:
                continue
            if context and words_key in samples[Fields.context][idx]:
                words = samples[Fields.context][idx][words_key]
            else:
                words = get_words_from_document(
                    samples_list[idx], token_func=tokenizer.encode_as_pieces if tokenizer else None
                )
                if context:
                    samples[Fields.context][idx][words_key] = words
            # try to get refined words from context
            refined_words_key = (
                f"{InterVars.refined_words}"
                "-True-SPECIAL_CHARS-"
                f"{self.use_words_aug}-"
                f"{self.words_aug_group_sizes}-"
                f"{self.words_aug_join_char}"
            )
            if context and refined_words_key in samples[Fields.context][idx]:
                words = samples[Fields.context][idx][refined_words_key]
            else:
                words = words_refinement(
                    words,
                    lower_case=True,
                    strip_chars=SPECIAL_CHARACTERS,
                    use_words_aug=self.use_words_aug,
                    words_aug_group_sizes=self.words_aug_group_sizes,
                    words_aug_join_char=self.words_aug_join_char,
                )
                if context:
                    samples[Fields.context][idx][refined_words_key] = words

            flagged_words_ratio = (
                (len([word for word in words if word in self.FLAGGED_WORDS[self.lang]]) / len(words))
                if len(words) != 0
                else 0.0
            )

            if flagged_words_ratio > 1.0:
                flagged_words_ratio = 1.0

            samples_stats[idx][StatsKeys.flagged_words_ratio] = flagged_words_ratio

        return samples

    def process_batched(self, samples):
        return list(
            map(
                lambda stat: self.get_keep_boolean(stat[StatsKeys.flagged_words_ratio], self.min_ratio, self.max_ratio),
                samples[Fields.stats],
            )
        )
