import ast
from collections.abc import Callable

import regex as re

from ..base_op import OPERATORS, Mapper


def split_sentence(text):
    text = re.sub(r"([.。！!？\?])([^’”])", r"\1\n\2", text)  # noqa
    text = re.sub(r"(\.{6})([^’”])", r"\1\n\2", text)  # noqa
    text = re.sub("(\…{2})([^’”])", r"\1\n\2", text)  # noqa
    text = re.sub(r"([.。!！？\?\.{6}\…{2}][’”])([^’”])", r"\1\n\2", text)  # noqa
    return text.split("\n")


def _wrap_tokenizer(fn: Callable[[str], list[str]]) -> Callable[[str], list[str]]:
    """Wrap a tokenizer to match split_sentence's whitespace convention.

    split_sentence preserves leading whitespace on each fragment (e.g. ["Hello.", " Goodbye."]), so the downstream
    ``new_sent += sentence`` concatenation produces correct spacing. Custom tokenizers will typically return clean
    tokens without leading whitespace, so this wrapper prepends a space to every sentence after the first.
    """

    def wrapped(line: str) -> list[str]:
        sentences = fn(line)
        return [(" " + s if i > 0 and not s[:1].isspace() else s) for i, s in enumerate(sentences)]

    return wrapped


@OPERATORS.register_module("remove_repeat_sentences_mapper")
class RemoveRepeatSentencesMapper(Mapper):
    """Mapper to remove repeat sentences in text samples.

    This operator processes text samples to remove duplicate sentences. It splits the text
    into lines and then further splits each line into sentences. Sentences are considered
    duplicates if they are identical after optional case normalization and special character
    removal. The operator uses a hash set to track unique sentences. Sentences shorter than
    `min_repeat_sentence_length` are not deduplicated. If `ignore_special_character` is
    enabled, special characters (all except Chinese, letters, and numbers) are ignored when
    checking for duplicates. The resulting text is reassembled with unique sentences."""

    _batched_op = True

    def __init__(
        self,
        lowercase: bool = False,
        ignore_special_character: bool = True,
        min_repeat_sentence_length: int = 2,
        tokenizer: Callable[[str], list[str]] | str | None = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param lowercase: Whether to convert sample text to lower case
        :param ignore_special_character: Whether to ignore special
            characters when judging repeated sentences. Special characters
            are all characters except Chinese characters, letters and
            numbers.
        :param min_repeat_sentence_length: Sentences shorter than this
            length will not be deduplicated. If ignore_special_character is
            set to True, then special characters are not included in this
            length.
        :param tokenizer: Custom sentence tokenizer. Can be a callable
            that takes a string and returns a list of sentence strings,
            or a lambda string for YAML configs (e.g.
            ``"lambda text: __import__('nltk').sent_tokenize(text)"``).
            If None, uses the built-in regex-based splitter.
        :param args: extra args
        :param kwargs: extra args
        """

        super().__init__(*args, **kwargs)
        self.lowercase = lowercase
        self.min_repeat_sentence_length = min_repeat_sentence_length
        self.remove_regex = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5\n\t ]") if ignore_special_character else None

        if tokenizer is None:
            self._tokenize = split_sentence
        elif callable(tokenizer):
            self._tokenize = _wrap_tokenizer(tokenizer)
        elif isinstance(tokenizer, str):
            self._tokenize = _wrap_tokenizer(self._create_tokenizer(tokenizer))
        else:
            raise ValueError(f"tokenizer must be None, a callable, or a lambda string, " f"got {type(tokenizer)}")

    @staticmethod
    def _create_tokenizer(tokenizer_str: str) -> Callable[[str], list[str]]:
        """Parse and validate a tokenizer lambda string."""
        try:
            node = ast.parse(tokenizer_str, mode="eval")
            if not isinstance(node.body, ast.Lambda):
                raise ValueError("Input string must be a valid lambda function.")
            if len(node.body.args.args) != 1:
                raise ValueError("Lambda function must have exactly one argument.")
            compiled_code = compile(node, "<string>", "eval")
            return eval(compiled_code, {"__builtins__": __builtins__})
        except Exception as e:
            raise ValueError(f"Invalid tokenizer lambda: {e}")

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            lines = [e for e in text.split("\n")]
            new_lines = []
            hash_set = set([])
            for line in lines:
                new_sent = ""
                if line:
                    sentences = self._tokenize(line)
                    for sentence in sentences:
                        copy = sentence.strip()
                        if self.lowercase:
                            copy = copy.lower()
                        if self.remove_regex:
                            copy = self.remove_regex.sub("", copy)

                        if len(copy) < self.min_repeat_sentence_length:
                            new_sent += sentence
                        elif copy not in hash_set:
                            new_sent += sentence
                            hash_set.add(copy)
                new_lines.append(new_sent)

            samples[self.text_key][idx] = "\n".join(new_lines)

        return samples
