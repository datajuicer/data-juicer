import regex as re

from ..base_op import OPERATORS, Mapper


def split_sentence(text):
    text = re.sub("([.。！!？\?])([^’”])", r"\1\n\2", text)  # noqa
    text = re.sub("(\.{6})([^’”])", r"\1\n\2", text)  # noqa
    text = re.sub("(\…{2})([^’”])", r"\1\n\2", text)  # noqa
    text = re.sub("([.。!！？\?\.{6}\…{2}][’”])([^’”])", r"\1\n\2", text)  # noqa
    return text.split("\n")


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
        :param args: extra args
        :param kwargs: extra args
        """

        super().__init__(*args, **kwargs)
        self.lowercase = lowercase
        self.min_repeat_sentence_length = min_repeat_sentence_length
        self.remove_regex = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5\n\t ]") if ignore_special_character else None

    def process_batched(self, samples):
        for idx, text in enumerate(samples[self.text_key]):
            lines = [e for e in text.split("\n")]
            new_lines = []
            hash_set = set([])
            for line in lines:
                new_sent = ""
                if line:
                    sentences = split_sentence(line)
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
