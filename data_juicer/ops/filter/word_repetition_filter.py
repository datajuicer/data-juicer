# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from pydantic import PositiveInt

from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..common import SPECIAL_CHARACTERS, get_words_from_document, words_refinement
from ..op_fusion import INTER_WORDS

OP_NAME = "word_repetition_filter"


@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)
class WordRepetitionFilter(Filter):
    """Filter to keep samples with word-level n-gram repetition ratio within a specific range.

    This operator calculates the word-level n-gram repetition ratio for each sample and
    filters out those that do not fall within the specified range. The n-gram length and the
    min/max ratio thresholds are configurable. If tokenization is enabled, a Hugging Face
    tokenizer is used to tokenize the text. The key metric, `word_rep_ratio`, is computed as
    the ratio of repeated n-grams to the total number of n-grams. This ratio is then
    compared against the min and max ratio thresholds to decide whether to keep or filter
    the sample. If the ratio is outside the specified range, the sample is filtered out."""

    _batched_op = True

    def __init__(
        self,
        lang: str = "en",
        tokenization: bool = False,
        rep_len: PositiveInt = 10,
        min_ratio: float = 0.0,
        max_ratio: float = 0.5,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param lang: sample in which language.
        :param tokenization: whether to use model to tokenize documents
        :param rep_len: Repetition length for word-level n-gram.
        :param min_ratio: The min filter ratio in this op, samples will
            be filtered if their word-level n-gram repetition ratio is
            below this parameter.
        :param max_ratio: The max filter ratio in this op, samples will
            be filtered if their word-level n-gram repetition ratio
            exceeds this parameter.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.n = rep_len
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.model_key = None
        self.lang = lang

        if tokenization:
            self.model_key = prepare_model(model_type="sentencepiece", lang=lang)

    def compute_stats_batched(self, samples, context=False):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]
        words_key = f"{InterVars.words}-{self.model_key}"

        for idx, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.word_rep_ratio in stat:
                continue
            # try to get words from context
            if context and words_key in samples[Fields.context][idx]:
                words = samples[Fields.context][idx][words_key]
            else:
                tokenizer = get_model(self.model_key)
                words = get_words_from_document(
                    samples_list[idx], token_func=tokenizer.encode_as_pieces if tokenizer else None
                )
                if context:
                    samples[Fields.context][idx][words_key] = words

            # try to get refined words from context
            refined_words_key = f"{InterVars.refined_words}-" f"True-SPECIAL_CHARS-False-[2]-"
            if context and refined_words_key in samples[Fields.context][idx]:
                words = samples[Fields.context][idx][refined_words_key]
            else:
                words = words_refinement(words, lower_case=True, strip_chars=SPECIAL_CHARACTERS)
                if context:
                    samples[Fields.context][idx][refined_words_key] = words
            word_ngrams = [" ".join(words[i : i + self.n]) for i in range(len(words) - self.n + 1)]
            freq_word_ngrams = {}
            for word_ngram in word_ngrams:
                freq_word_ngrams[word_ngram] = freq_word_ngrams.get(word_ngram, 0) + 1

            if len(freq_word_ngrams) == 0:
                samples_stats[idx][StatsKeys.word_rep_ratio] = 0.0
                continue

            freq_word_ngrams = list(freq_word_ngrams.values())
            rep_more_than_one = [freq for freq in freq_word_ngrams if freq > 1]
            samples_stats[idx][StatsKeys.word_rep_ratio] = (
                (sum(rep_more_than_one) / sum(freq_word_ngrams)) if sum(freq_word_ngrams) != 0 else 0.0
            )

        return samples

    def process_batched(self, samples):
        assert isinstance(samples[Fields.stats], list)
        return map(
            lambda stat: self.get_keep_boolean(stat[StatsKeys.word_rep_ratio], self.min_ratio, self.max_ratio),
            samples[Fields.stats],
        )
