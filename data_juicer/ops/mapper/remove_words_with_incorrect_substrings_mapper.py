from typing import List, Optional

from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..common import (
    SPECIAL_CHARACTERS,
    get_words_from_document,
    merge_on_whitespace_tab_newline,
    split_on_newline_tab_whitespace,
    strip,
)

OP_NAME = "remove_words_with_incorrect_substrings_mapper"


@OPERATORS.register_module(OP_NAME)
class RemoveWordsWithIncorrectSubstringsMapper(Mapper):
    """Mapper to remove words containing specified incorrect substrings.

    This operator processes text by removing words that contain any of the specified
    incorrect substrings. By default, it removes words with substrings like "http", "www",
    ".com", "href", and "//". The operator can operate in tokenized or non-tokenized mode.
    In tokenized mode, it uses a Hugging Face tokenizer to tokenize the text before
    processing. The key metric is not computed; this operator focuses on filtering out
    specific words.

    - If `tokenization` is True, the text is tokenized using a Hugging Face
    tokenizer, and words are filtered based on the specified substrings.
    - If `tokenization` is False, the text is split into sentences and words,
    and words are filtered based on the specified substrings.
    - The filtered text is then merged back into a single string.

    The operator processes samples in batches and updates the text in place."""

    _batched_op = True

    def __init__(
        self, lang: str = "en", tokenization: bool = False, substrings: Optional[List[str]] = None, *args, **kwargs
    ):
        """
        Initialization method.

        :param lang: sample in which language
        :param tokenization: whether to use model to tokenize documents
        :param substrings: The incorrect substrings in words.
        :param args: extra args
        :param kwargs: extra args
        """
        if substrings is None:
            substrings = ["http", "www", ".com", "href", "//"]
        super().__init__(*args, **kwargs)
        self.tokenization = tokenization
        self.substrings = substrings
        self.lang = lang
        if tokenization:
            self.model_key = prepare_model(model_type="sentencepiece", lang=lang)

    def should_keep_word_with_incorrect_substrings(self, word, substrings):
        word = strip(word, SPECIAL_CHARACTERS)
        should_keep = all([(i_substr not in word) for i_substr in substrings])
        return should_keep

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            if self.tokenization:
                tokenizer = get_model(self.model_key)
                sentences = get_words_from_document(text, token_func=tokenizer.encode_as_pieces if tokenizer else None)
                words = [
                    word.replace("▁", "")
                    for word in sentences
                    if self.should_keep_word_with_incorrect_substrings(word.replace("▁", ""), self.substrings)
                ]
                if len(words) != len(sentences):
                    text = "".join(words)
            else:
                sentences = split_on_newline_tab_whitespace(text)
                sentences = [
                    [
                        [
                            word
                            for word in subsentence
                            if self.should_keep_word_with_incorrect_substrings(word, self.substrings)
                        ]
                        for subsentence in sentence
                    ]
                    for sentence in sentences
                ]
                text = merge_on_whitespace_tab_newline(sentences)

            samples[self.text_key][idx] = text

        return samples
