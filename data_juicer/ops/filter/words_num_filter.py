import sys

from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..common import SPECIAL_CHARACTERS, get_words_from_document, words_refinement
from ..op_fusion import INTER_WORDS

OP_NAME = "words_num_filter"


@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)
class WordsNumFilter(Filter):
    """Filter to keep samples with a total word count within a specified range.

    This operator filters samples based on the number of words they contain. It retains
    samples if their word count is within the given minimum and maximum limits. If
    tokenization is enabled, it uses a Hugging Face tokenizer to count words. The key metric
    `num_words` is computed and stored in the sample's stats under the `num_words` field. If
    the word count is already cached, it reuses the cached value to avoid redundant
    computation."""

    _batched_op = True

    def __init__(
        self,
        lang: str = "en",
        tokenization: bool = False,
        min_num: int = 10,
        max_num: int = sys.maxsize,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param lang: sample in which language.
        :param tokenization: whether to use model to tokenize documents
        :param min_num: The min filter word number in this op, samples
            will be filtered if their word number is below this
            parameter.
        :param max_num: The max filter word number in this op, samples
            will be filtered if their word number exceeds this
            parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_num = min_num
        self.max_num = max_num
        self.model_key = None
        self.lang = lang

        if tokenization:
            self.model_key = prepare_model(model_type="sentencepiece", lang=lang)

    def compute_stats_batched(self, samples, *args, **kwargs):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]
        words_key = f"{InterVars.words}-{self.model_key}"

        for idx, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.num_words in stat:
                continue
            context = kwargs.get("context", False)
            if context and words_key in samples[Fields.context][idx]:
                words = samples[Fields.context][idx][words_key]
            else:
                tokenizer = get_model(self.model_key)
                words = get_words_from_document(
                    samples_list[idx], token_func=tokenizer.encode_as_pieces if tokenizer else None
                )
                if context:
                    samples[Fields.context][idx][words_key] = words
            words = words_refinement(words, strip_chars=SPECIAL_CHARACTERS)
            samples_stats[idx][StatsKeys.num_words] = len(words)

        return samples

    def process_batched(self, samples):
        assert isinstance(samples[Fields.stats], list)
        return map(
            lambda stat: self.get_keep_boolean(stat[StatsKeys.num_words], self.min_num, self.max_num),
            samples[Fields.stats],
        )
