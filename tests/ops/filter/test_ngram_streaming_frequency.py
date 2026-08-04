import gc
import random
import tracemalloc
import unittest

import numpy as np

from data_juicer.ops.filter.character_repetition_filter import (
    CharacterRepetitionFilter,
)
from data_juicer.ops.filter.word_repetition_filter import WordRepetitionFilter
from data_juicer.ops.op_fusion import GeneralFusedOP
from data_juicer.utils.constant import Fields, InterVars, StatsKeys

_OCCURRENCE_COUNT = 100_000
_MAX_STREAMING_PEAK_BYTES = 2 * 1024 * 1024


def _tracemalloc_peak(call):
    gc.collect()
    tracemalloc.start()
    try:
        call()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _reference_character_ratio(text, n):
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    frequencies = {}
    for ngram in ngrams:
        frequencies[ngram] = frequencies.get(ngram, 0) + 1

    if not frequencies:
        return 0.0

    frequencies = sorted(frequencies.values(), reverse=True)
    num_repeated = min(
        int(np.sqrt(len(frequencies))),
        len(frequencies) - frequencies.count(1),
    )
    return sum(frequencies[:num_repeated]) / sum(frequencies)


def _reference_word_ratio(words, n):
    ngrams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    frequencies = {}
    for ngram in ngrams:
        frequencies[ngram] = frequencies.get(ngram, 0) + 1

    if not frequencies:
        return 0.0

    values = list(frequencies.values())
    return sum(frequency for frequency in values if frequency > 1) / sum(values)


def _word_context(op, word_rows):
    words_key = f"{InterVars.words}-{op.model_key}"
    refined_words_key = f"{InterVars.refined_words}-True-SPECIAL_CHARS-False-[2]-"
    return [
        {
            words_key: words,
            refined_words_key: words,
        }
        for words in word_rows
    ]


class NgramStreamingFrequencyTest(unittest.TestCase):
    def test_character_and_word_frequency_run_through_general_fusion(self):
        op = GeneralFusedOP(
            batch_size=2,
            fused_op_list=[
                {
                    "character_repetition_filter": {
                        "rep_len": 2,
                        "max_ratio": 1.0,
                        "auto_op_parallelism": False,
                        "num_proc": 1,
                    }
                },
                {
                    "word_repetition_filter": {
                        "rep_len": 2,
                        "max_ratio": 1.0,
                        "auto_op_parallelism": False,
                        "num_proc": 1,
                    }
                },
            ],
            auto_op_parallelism=False,
            num_proc=1,
        )
        samples = {
            "text": ["a b a b", ""],
            Fields.stats: [{}, {}],
        }

        output = op.process_batched(samples)

        self.assertEqual(
            output,
            {
                "text": ["a b a b", ""],
                Fields.stats: [
                    {
                        StatsKeys.char_rep_ratio: 2 / 3,
                        StatsKeys.word_rep_ratio: 2 / 3,
                    },
                    {
                        StatsKeys.char_rep_ratio: 0.0,
                        StatsKeys.word_rep_ratio: 0.0,
                    },
                ],
            },
        )
        self.assertEqual(samples[Fields.stats], [{}, {}])

    def test_character_frequency_matches_eager_reference_on_adversarial_inputs(self):
        rng = random.Random(20260804)
        alphabet = "aA 0\n界🙂"
        texts = [
            "",
            "a",
            "aaaaaa",
            "abababa",
            "界界界界",
            "a\x00a\x00a",
            "🙂🙃🙂🙃🙂",
        ]
        texts.extend("".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 80))) for _ in range(80))

        for n in (1, 2, 3, 10, 100):
            with self.subTest(n=n):
                op = CharacterRepetitionFilter(
                    rep_len=n,
                    auto_op_parallelism=False,
                    num_proc=1,
                )
                samples = {
                    op.text_key: texts,
                    Fields.stats: [{} for _ in texts],
                }

                output = op.compute_stats_batched(samples)

                self.assertIs(output, samples)
                self.assertEqual(
                    [stat[StatsKeys.char_rep_ratio] for stat in samples[Fields.stats]],
                    [_reference_character_ratio(text, n) for text in texts],
                )

    def test_character_frequency_preserves_cached_stats(self):
        op = CharacterRepetitionFilter(
            rep_len=2,
            auto_op_parallelism=False,
            num_proc=1,
        )
        samples = {
            op.text_key: ["ignored", "abab"],
            Fields.stats: [{StatsKeys.char_rep_ratio: 0.125}, {}],
        }

        op.compute_stats_batched(samples)

        self.assertEqual(
            [stat[StatsKeys.char_rep_ratio] for stat in samples[Fields.stats]],
            [0.125, 2 / 3],
        )

    def test_character_frequency_does_not_retain_all_occurrences(self):
        op = CharacterRepetitionFilter(
            rep_len=10,
            auto_op_parallelism=False,
            num_proc=1,
        )
        samples = {
            op.text_key: ["a" * (_OCCURRENCE_COUNT + op.n - 1)],
            Fields.stats: [{}],
        }

        peak_bytes = _tracemalloc_peak(lambda: op.compute_stats_batched(samples))

        self.assertEqual(samples[Fields.stats][0][StatsKeys.char_rep_ratio], 1.0)
        self.assertLess(peak_bytes, _MAX_STREAMING_PEAK_BYTES)

    def test_word_frequency_matches_eager_reference_on_adversarial_inputs(self):
        rng = random.Random(20260804)
        vocabulary = ["a", "A", "界", "🙂", "", "a b", "\x00"]
        word_rows = [
            [],
            ["a"],
            ["a"] * 8,
            ["a", "b", "a", "b"],
            ["a b", "c", "a", "b c"],
            ["界", "🙂", "界", "🙂"],
        ]
        word_rows.extend([rng.choice(vocabulary) for _ in range(rng.randrange(0, 80))] for _ in range(80))

        for n in (1, 2, 3, 10, 100):
            with self.subTest(n=n):
                op = WordRepetitionFilter(
                    rep_len=n,
                    auto_op_parallelism=False,
                    num_proc=1,
                )
                samples = {
                    op.text_key: [""] * len(word_rows),
                    Fields.stats: [{} for _ in word_rows],
                    Fields.context: _word_context(op, word_rows),
                }

                output = op.compute_stats_batched(samples, context=True)

                self.assertIs(output, samples)
                self.assertEqual(
                    [stat[StatsKeys.word_rep_ratio] for stat in samples[Fields.stats]],
                    [_reference_word_ratio(words, n) for words in word_rows],
                )

    def test_word_frequency_preserves_cached_stats(self):
        op = WordRepetitionFilter(
            rep_len=2,
            auto_op_parallelism=False,
            num_proc=1,
        )
        word_rows = [["ignored"], ["a", "b", "a", "b"]]
        samples = {
            op.text_key: [""] * len(word_rows),
            Fields.stats: [{StatsKeys.word_rep_ratio: 0.125}, {}],
            Fields.context: _word_context(op, word_rows),
        }

        op.compute_stats_batched(samples, context=True)

        self.assertEqual(
            [stat[StatsKeys.word_rep_ratio] for stat in samples[Fields.stats]],
            [0.125, 2 / 3],
        )

    def test_word_frequency_preserves_join_error_without_partial_stats(self):
        op = WordRepetitionFilter(
            rep_len=2,
            auto_op_parallelism=False,
            num_proc=1,
        )
        word_rows = [["a", "b", 7, "c"]]
        samples = {
            op.text_key: [""],
            Fields.stats: [{}],
            Fields.context: _word_context(op, word_rows),
        }

        with self.assertRaisesRegex(
            TypeError,
            "sequence item 1: expected str instance, int found",
        ):
            op.compute_stats_batched(samples, context=True)

        self.assertEqual(samples[Fields.stats], [{}])

    def test_word_frequency_does_not_retain_all_occurrences(self):
        op = WordRepetitionFilter(
            rep_len=10,
            auto_op_parallelism=False,
            num_proc=1,
        )
        words = ["word"] * (_OCCURRENCE_COUNT + op.n - 1)
        samples = {
            op.text_key: [""],
            Fields.stats: [{}],
            Fields.context: _word_context(op, [words]),
        }

        peak_bytes = _tracemalloc_peak(lambda: op.compute_stats_batched(samples, context=True))

        self.assertEqual(samples[Fields.stats][0][StatsKeys.word_rep_ratio], 1.0)
        self.assertLess(peak_bytes, _MAX_STREAMING_PEAK_BYTES)


if __name__ == "__main__":
    unittest.main()
