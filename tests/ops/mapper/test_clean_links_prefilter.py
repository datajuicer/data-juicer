import random

import pytest
import regex as re

from data_juicer.ops.mapper.clean_links_mapper import CleanLinksMapper


def original_result(op, text):
    if not re.search(op.pattern, text, flags=re.DOTALL):
        return text
    return re.sub(op.pattern, op.repl, text, flags=re.DOTALL)


@pytest.mark.parametrize("repl", ["", "[LINK]", r"<\1>"])
def test_default_url_prefixes_and_nonlinks(repl):
    texts = [
        "",
        "ordinary words without links",
        "中文句子。全角冒号：仍应保留",
        "www.example.com/path",
        "www2.example.com/path",
        "example.com/path",
        "http://localhost/path",
        "mailto:user@example.com",
        "FTP://server/path",
        "before https://example.com/(one(two)) after",
        "prefix\nhttps://example.com/a\nend",
        "nothing: here",
        "version 1.2 and a sentence.",
        "https://example.com/a and ftp://example.org/b",
    ]
    op = CleanLinksMapper(repl=repl)
    expected = [original_result(op, text) for text in texts]
    samples = {"text": texts.copy()}
    assert op.process_batched(samples) is samples
    assert samples["text"] == expected


@pytest.mark.parametrize("pattern", ["REMOVE", "r'REMOVE'", r"(word)", r"^prefix", r"(?=word)"])
def test_custom_patterns_without_url_delimiters(pattern):
    op = CleanLinksMapper(pattern=pattern, repl="replacement", text_key="content")
    texts = ["REMOVE this word", "prefix word", "unchanged", ""]
    expected = [original_result(op, text) for text in texts]
    assert op.process_batched({"content": texts.copy()})["content"] == expected


def test_modified_pattern_does_not_use_default_prefilter():
    op = CleanLinksMapper()
    op.pattern = "REMOVE"
    assert op.process_batched({"text": ["REMOVE this"]})["text"] == [" this"]


def test_no_match_does_not_evaluate_invalid_replacement():
    op = CleanLinksMapper(repl=r"\g<missing>")
    texts = ["ordinary words", "nothing. here"]
    assert op.process_batched({"text": texts.copy()})["text"] == texts


def test_seeded_unicode_nonlinks_match_original():
    rng = random.Random(42)
    alphabet = "abcXYZ019 -_/()\n\t中文🙂。：：\u200b"
    texts = ["".join(rng.choices(alphabet, k=size)) for size in range(0, 2048, 17)]
    op = CleanLinksMapper()
    expected = [original_result(op, text) for text in texts]
    assert op.process_batched({"text": texts.copy()})["text"] == expected


def test_empty_batch():
    assert CleanLinksMapper().process_batched({"text": []}) == {"text": []}
