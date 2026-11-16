import random
import unittest

from data_juicer.ops.common import (
    SPECIAL_CHARACTERS,
    get_words_from_document,
    words_refinement,
)
from data_juicer.ops.filter.words_num_filter import WordsNumFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


def reference_count(text):
    words = get_words_from_document(text)
    words = words_refinement(words, strip_chars=SPECIAL_CHARACTERS)
    return len(words)


class WordsNumFilterStreamingTest(DataJuicerTestCaseBase):

    def _run(self, texts, op=None):
        op = op or WordsNumFilter()
        samples = {
            "text": texts,
            Fields.stats: [{} for _ in texts],
        }
        op.compute_stats_batched(samples)
        return [stat[StatsKeys.num_words] for stat in samples[Fields.stats]]

    def test_matches_reference_on_mixed_separators(self):
        texts = [
            "alpha beta gamma",
            "alpha\nbeta\tgamma",
            "alpha  beta   gamma",
            "alpha\r\nbeta",
            "",
            "   \n\t  ",
            "!!! ...",
            "alpha, beta! gamma?",
            "中文 分词 测试",
            "🙂 alpha 🙂",
        ]
        got = self._run(texts)
        expected = [reference_count(text) for text in texts]
        self.assertEqual(got, expected)

    def test_matches_reference_on_random_text(self):
        rng = random.Random(20261115)
        vocab = ["alpha", "beta", "gamma", "!!!", "...", "中文", "🙂", ",", "x"]
        separators = [" ", "  ", "\n", "\t", " \n "]
        texts = []
        for _ in range(60):
            parts = [rng.choice(vocab) for _ in range(rng.randint(0, 40))]
            text = ""
            for part in parts:
                text += part + rng.choice(separators)
            texts.append(text)
        got = self._run(texts)
        expected = [reference_count(text) for text in texts]
        self.assertEqual(got, expected)

    def test_cached_stat_is_not_recomputed(self):
        op = WordsNumFilter()
        samples = {
            "text": ["alpha beta gamma"],
            Fields.stats: [{StatsKeys.num_words: 999}],
        }
        op.compute_stats_batched(samples)
        self.assertEqual(samples[Fields.stats][0][StatsKeys.num_words], 999)

    def test_context_path_still_uses_word_list(self):
        op = WordsNumFilter()
        samples = {
            "text": ["alpha beta gamma"],
            Fields.stats: [{}],
            Fields.context: [{}],
        }
        op.compute_stats_batched(samples, context=True)
        self.assertEqual(samples[Fields.stats][0][StatsKeys.num_words], 3)
        self.assertIn("__dj__words-None-text", samples[Fields.context][0])


    def test_matches_reference_across_chunk_boundaries(self):
        from data_juicer.ops.filter import words_num_filter as mod

        original = mod._WORD_COUNT_CHUNK_CHARS
        mod._WORD_COUNT_CHUNK_CHARS = 7
        try:
            texts = [
                "alpha beta gamma delta",
                "a" * 20 + " b",
                "中文" * 20 + " 测试",
                "one\r\ntwo\tthree four",
                "x y z",
                "longwordwithoutseparators",
            ]
            got = self._run(texts)
            expected = [reference_count(text) for text in texts]
            self.assertEqual(got, expected)
        finally:
            mod._WORD_COUNT_CHUNK_CHARS = original


if __name__ == "__main__":
    unittest.main()
