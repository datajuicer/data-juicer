from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.nltk_utils import patch_nltk_pickle_security

from ..base_op import OPERATORS, Mapper
from ..common import get_sentences_from_document

OP_NAME = "sentence_split_mapper"


@OPERATORS.register_module(OP_NAME)
class SentenceSplitMapper(Mapper):
    """Splits text samples into individual sentences based on the specified language.

    This operator uses an NLTK-based tokenizer to split the input text into sentences. The
    language for the tokenizer is specified during initialization. The original text in each
    sample is replaced with a list of sentences. This operator processes samples in batches
    for efficiency. Ensure that the `lang` parameter is set to the appropriate language code
    (e.g., "en" for English) to achieve accurate sentence splitting."""

    _batched_op = True

    def __init__(self, lang: str = "en", *args, **kwargs):
        """
        Initialization method.

        :param lang: split sentence of text in which language.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.lang = lang

        # Ensure NLTK pickle security patch is applied
        patch_nltk_pickle_security()

        # Prepare the sentence tokenizer model
        self.model_key = prepare_model(model_type="nltk", lang=lang)

    def process_batched(self, samples):
        # Get the sentence tokenizer model
        nltk_model = get_model(self.model_key)

        samples[self.text_key] = [
            get_sentences_from_document(text, model_func=nltk_model.tokenize if nltk_model else None)
            for text in samples[self.text_key]
        ]

        return samples
