# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# --------------------------------------------------------
from typing import Dict

import regex as re


class UnionFind:
    def __init__(self):
        """Initialization method."""
        self.parent: Dict[int, int] = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        self.parent[px] = self.parent[py] = min(px, py)


def strip(document, strip_characters):
    """
    Way faster than document.strip(strip_characters) since strip_characters is
    now a set instead of a str, and it contains a lot of elements (all the
    emojis).

    :param document: document to be processed
    :param strip_characters: characters used for stripping document
    :return: stripped document
    """
    if not document:
        return document
    beg_ind = 0
    end_ind = len(document)
    for i in range(len(document)):
        if document[i] in strip_characters:
            beg_ind += 1
        else:
            break
    for i in range(1, len(document) + 1):
        if document[-i] in strip_characters:
            end_ind -= 1
        else:
            break
    document_stripped = document[beg_ind:end_ind]
    return document_stripped


def split_on_whitespace(document, new_line=False, tab=False):
    """
    This method also removes concatenated spaces.

    :param document: document to be split
    :param new_line: whether to split document with '\\\\n'
    :param tag: whether to split document with '\\\\t'
    :return: word list obtained after splitting document
    """
    sep = [" "] + new_line * ["\n"] + tab * ["\t"]
    sep = "|".join(sep)
    split_document = re.split(sep, document)
    split_document = [word for word in split_document if word]
    return split_document


def split_on_newline_tab_whitespace(document):
    """
    This method is used to split the document into different levels of sub-
    sentences.

    First split on "\\\\n", then on "\\\\t", then on " ".
    :param document: document to be split
    :return: sentence list obtained after splitting document
    """
    sentences = document.split("\n")
    sentences = [sentence.split("\t") for sentence in sentences]
    sentences = [[split_on_whitespace(subsentence) for subsentence in sentence] for sentence in sentences]
    return sentences


def merge_on_whitespace_tab_newline(sentences):
    """
    This method is used to merge different levels of sub-sentences into one
    document. Invert the method split_on_newline_tab_whitespace. Removes
    concatenated separators.

    :param sentences: sentence list to be merged
    :return: document obtained after merging sub-sentences
    """
    sentences = [[" ".join(subsentence) for subsentence in sentence if subsentence] for sentence in sentences]
    sentences = ["\t".join(sentence) for sentence in sentences if sentence]
    if not sentences:
        return ""
    document = "\n".join(sentences)
    return document


def words_augmentation(words, group_size, join_char):
    """
    Augment words, especially for Chinese (without a space between words) and
    Vietnamese (with a space between syllables).

    :param word: word list to be augmented
    :param group_size: the size of word groups that need to be merged
    :param join_char: characters to be added between word group
    :return: word list after augment
    """
    augmentation = [join_char.join(words[i : i + group_size]) for i in range(len(words) - group_size + 1)]
    return augmentation


def get_words_from_document(
    document,
    token_func=None,
    new_line=True,
    tab=True,
):
    """
    Get words from a document. Useful to compute ratios, like the
    stopwords ratio.

    :param document: document that need to split words.
    :param token_func: function of tokenizer, if specified, the function
     will be used for split document into different tokens.
    :param new_line: whether to use '\\\\n' to split words.
    :param tab: whether to use '\\\\t' to split words.
    :return: word list obtained from document
    """
    if token_func:
        words = token_func(document)
    else:
        words = split_on_whitespace(document, new_line, tab)
    return words


def words_refinement(
    words, lower_case=False, strip_chars=None, use_words_aug=False, words_aug_group_sizes=[2], words_aug_join_char=""
):
    """
    Refine split words. Non reversible since the document is split on
    multiple characters, words are stripped of special characters and
    characters are converted to lower case.

    :param words: the word list to be augmented
    :param lower_case: whether to convert word to lowercase
    :param strip_chars: chars that need to be stripped in words
    :param use_words_aug: whether to use word augmentation
    :param words_aug_group_sizes: the size of word groups that need to
        be merged
    :param words_aug_join_char: characters to be added between word
        group
    :return: refined words or word list
    """

    if lower_case:
        words = [word.lower() for word in words]
    if strip_chars:
        words = [strip(word, strip_chars) for word in words]
        words = [word for word in words if word]
    if use_words_aug:
        augmentation = [
            words_augmentation(words, group_size, words_aug_join_char) for group_size in words_aug_group_sizes
        ]
        augmentation = [word for augm in augmentation for word in augm]
        words = words + augmentation
    return words


def get_sentences_from_document(document, model_func=None):
    """
    Get sentences from a document.

    :param document: document that need to split sentences
    :param model_func: function of sentence model, if specified, the
        function will be used for splitting document into different
        sentences.
    :return: document with the sentences separated by '\\\\n'
    """
    if model_func:
        sentences = model_func(document)
    else:
        sentences = document.splitlines()
    return "\n".join(sentences)


def split_text_by_punctuation(text):
    """
    Split text by any zh and en punctuation

    :param text: text to be split.
    :return: sub texts split by any zh and en punctuation
    """
    # any zh and en punctuation
    punctuation_pattern = r'[\u3000-\u303f\uff00-\uffef]|[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'  # noqa: E501

    result = re.split(punctuation_pattern, text)
    result = [s.strip() for s in result if s.strip()]

    if not result:
        return [text]

    return result
