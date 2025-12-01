# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------

from data_juicer.utils.constant import Fields, InterVars, StatsKeys
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..common import get_words_from_document
from ..op_fusion import INTER_WORDS

OP_NAME = "perplexity_filter"


@OPERATORS.register_module(OP_NAME)
@INTER_WORDS.register_module(OP_NAME)
class PerplexityFilter(Filter):
    """Filter to keep samples with perplexity score in a specified range.

    This operator computes the perplexity of text samples using a Hugging Face tokenizer and
    a KenLM language model. It keeps samples with perplexity scores within the specified
    minimum and maximum values. The perplexity is calculated character-based by default. If
    the perplexity is already computed, it will be reused from the 'perplexity' field in the
    sample's stats. The operator supports batched operations for efficiency."""

    _batched_op = True

    def __init__(self, lang: str = "en", min_ppl: float = 0, max_ppl: float = 1500, *args, **kwargs):
        """
        Initialization method.

        :param lang: Compute perplexity for samples in which language.
        :param min_ppl: The min filter perplexity in this op.
        :param max_ppl: The max filter perplexity in this op.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_ppl = min_ppl
        self.max_ppl = max_ppl
        self.lang = lang
        self.sp_model_key = prepare_model(model_type="sentencepiece", lang=lang)
        self.kl_model_key = prepare_model(model_type="kenlm", lang=lang)

    def compute_stats_batched(self, samples, context=False):
        samples_list = samples[self.text_key]
        samples_stats = samples[Fields.stats]
        words_key = f"{InterVars.words}-{self.sp_model_key}"
        tokenizer = get_model(self.sp_model_key)

        for idx, stat in enumerate(samples_stats):
            # check if it's computed already
            if StatsKeys.perplexity in stat:
                continue
            # tokenization
            if context and words_key in samples[Fields.context][idx]:
                words = samples[Fields.context][idx][words_key]
            else:
                words = get_words_from_document(
                    samples_list[idx], token_func=tokenizer.encode_as_pieces if tokenizer else None
                )
                if context:
                    samples[Fields.context][idx][words_key] = words
            text = " ".join(words)
            # compute perplexity
            logits, length = 0, 0
            kenlm_model = get_model(self.kl_model_key)
            for line in text.splitlines():
                logits += kenlm_model.score(line)
                length += len(line.split()) + 1
            ppl = (10.0 ** (-logits / length)) if length != 0 else 0.0
            samples_stats[idx][StatsKeys.perplexity] = round(ppl, 1)

        return samples

    def process_batched(self, samples):
        assert isinstance(samples[Fields.stats], list)
        return map(
            lambda stat: self.get_keep_boolean(stat[StatsKeys.perplexity], self.min_ppl, self.max_ppl),
            samples[Fields.stats],
        )
