import random

import pytest

from data_juicer.ops.common.special_characters import VARIOUS_WHITESPACES
from data_juicer.ops.mapper.whitespace_normalization_mapper import (
    WhitespaceNormalizationMapper,
)


@pytest.mark.parametrize("character", sorted(VARIOUS_WHITESPACES))
def test_each_configured_character(character):
    op = WhitespaceNormalizationMapper()
    text = "left" + character * 3 + "right"
    assert op.process_batched({"text": [text]})["text"] == ["left   right"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", ""),
        (" \t\n\r", ""),
        ("  a  b  ", "a  b"),
        ("\nline1\nline2\r\nline3\vline4\fline5\n", "line1\nline2\r\nline3\vline4\fline5"),
        ("\u200bword\u200b", " word "),
        ("x\ufeffy\u0085z", "x\ufeffy\u0085z"),
        ("中文🙂e\u0301", "中文🙂e\u0301"),
    ],
)
def test_preserve_normalization_boundaries(text, expected):
    op = WhitespaceNormalizationMapper(text_key="content")
    samples = {"content": [text], "id": [7]}
    result = op.process_batched(samples)
    assert result is samples
    assert result == {"content": [expected], "id": [7]}


def test_mixed_unicode_batch_matches_character_reference():
    rng = random.Random(42)
    alphabet = sorted(VARIOUS_WHITESPACES) + list("abc 中文🙂\n\r\v\f\ufeff\u0085e\u0301")
    texts = ["".join(rng.choices(alphabet, k=size)) for size in range(0, 2048, 17)]
    expected = ["".join(" " if c in VARIOUS_WHITESPACES else c for c in text.strip()) for text in texts]
    assert WhitespaceNormalizationMapper().process_batched({"text": texts})["text"] == expected


def test_empty_batch():
    assert WhitespaceNormalizationMapper().process_batched({"text": []}) == {"text": []}
