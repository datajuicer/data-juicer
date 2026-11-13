import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.extract_html_content_mapper import ExtractHtmlContentMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ExtractHtmlContentMapperTest(DataJuicerTestCaseBase):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Why Rivers Meander | Example Science</title>
  <meta name="author" content="Jane Doe">
  <meta name="description" content="A short explanation of river meanders.">
  <meta property="og:site_name" content="Example Science">
  <meta property="article:published_time" content="2024-05-17">
</head>
<body>
  <div class="cookie-banner">We use cookies to improve your experience. Accept all cookies</div>
  <header><nav><ul><li><a href="/">Home</a></li><li><a href="/news">News</a></li></ul></nav></header>
  <aside class="sidebar"><h3>Trending now</h3><ul><li><a href="/x">Ten celebrity diets you must try</a></li></ul></aside>
  <main>
    <article>
      <h1>Why Rivers Meander</h1>
      <p>Rivers rarely flow in straight lines. Over time, small irregularities in the channel cause water to flow faster on the outside of a bend, eroding the bank there.</p>
      <p>On the inside of the bend the water slows down and deposits sediment, building a point bar. This combination of erosion and deposition makes the bend grow wider year after year.</p>
      <h2>Key factors</h2>
      <ul>
        <li>The slope of the valley floor</li>
        <li>The amount of sediment carried by the river</li>
      </ul>
      <table>
        <tr><th>River</th><th>Sinuosity</th></tr>
        <tr><td>Mississippi</td><td>1.9</td></tr>
      </table>
      <p>Eventually a meander can be cut off entirely, leaving behind a crescent-shaped oxbow lake that slowly fills with sediment and <a href="/plants">vegetation</a>.</p>
    </article>
    <section id="comments" class="comments">
      <h3>Comments</h3>
      <div class="comment"><p>Great article, I finally understand oxbow lakes after visiting one last summer near my hometown.</p></div>
    </section>
  </main>
  <footer><p>Copyright 2024 Example Science. All rights reserved.</p><a href="/privacy">Privacy policy</a></footer>
</body>
</html>"""

    article_sentences = [
        "Why Rivers Meander",
        "Rivers rarely flow in straight lines.",
        "building a point bar.",
        "The slope of the valley floor",
        "leaving behind a crescent-shaped oxbow lake",
    ]
    boilerplate = [
        "We use cookies",
        "Trending now",
        "Ten celebrity diets",
        "Copyright 2024",
        "Privacy policy",
    ]

    def _run(self, op, samples):
        dataset = Dataset.from_list(samples)
        return op.run(dataset)

    def _extract_one(self, op, **extra):
        return self._run(op, [{"text": self.html, **extra}])[0]

    def assert_article_kept(self, text):
        for sentence in self.article_sentences:
            self.assertIn(sentence, text)

    def assert_boilerplate_removed(self, text):
        for noise in self.boilerplate:
            self.assertNotIn(noise, text)

    def test_default_removes_boilerplate(self):
        text = self._extract_one(ExtractHtmlContentMapper())["text"]
        self.assert_article_kept(text)
        self.assert_boilerplate_removed(text)
        self.assertNotIn("Great article", text)
        self.assertNotIn("<p>", text)
        self.assertIn("Mississippi", text)

    def test_markdown_output(self):
        op = ExtractHtmlContentMapper(output_format="markdown")
        text = self._extract_one(op)["text"]
        self.assertTrue(text.startswith("# Why Rivers Meander"))
        self.assertIn("## Key factors", text)
        self.assertIn("- The slope of the valley floor", text)
        self.assertIn("| Mississippi | 1.9 |", text)
        self.assert_boilerplate_removed(text)

    def test_markdown_links_resolved_with_url(self):
        op = ExtractHtmlContentMapper(output_format="markdown", include_links=True, url_key="url")
        text = self._extract_one(op, url="https://example.org/science/rivers")["text"]
        self.assertIn("[vegetation](https://example.org/plants)", text)

    def test_txt_link_targets(self):
        for include_links in (False, True):
            with self.subTest(include_links=include_links):
                op = ExtractHtmlContentMapper(output_format="txt", include_links=include_links, url_key="url")
                text = self._extract_one(op, url="https://example.org/science/rivers")["text"]
                self.assert_article_kept(text)
                self.assertEqual("[vegetation](https://example.org/plants)" in text, include_links)

    def test_nested_url_links_and_metadata(self):
        op = ExtractHtmlContentMapper(
            output_format="markdown",
            include_links=True,
            extract_metadata=True,
            url_key="page.source.url",
            num_proc=2,
        )
        page = {"source": {"url": "https://example.org/science/rivers"}, "keep": "original"}
        dataset = self._run(op, [{"text": self.html, "page": page}] * 4)
        for sample in dataset:
            self.assertIn("[vegetation](https://example.org/plants)", sample["text"])
            metadata = sample[Fields.meta][MetaKeys.html_metadata]
            self.assertEqual(metadata["url"], "https://example.org/science/rivers")
            self.assertEqual(metadata["hostname"], "example.org")
            self.assertEqual(sample["page"], page)

    def test_literal_url_key_takes_precedence(self):
        op = ExtractHtmlContentMapper(
            output_format="markdown",
            include_links=True,
            extract_metadata=True,
            url_key="page.url",
        )
        sample = self._extract_one(
            op,
            **{
                "page.url": "https://literal.example.org/article",
                "page": {"url": "https://nested.example.org/article"},
            },
        )
        self.assertIn("[vegetation](https://literal.example.org/plants)", sample["text"])
        self.assertEqual(sample[Fields.meta][MetaKeys.html_metadata]["url"], "https://literal.example.org/article")

    def test_missing_and_invalid_nested_urls_are_ignored(self):
        cases = ({}, {"page": None}, {"page": "not a dictionary"}, {"page": {}}, {"page": {"url": 42}})
        for extra in cases:
            with self.subTest(extra=extra):
                op = ExtractHtmlContentMapper(
                    output_format="markdown",
                    include_links=True,
                    extract_metadata=True,
                    url_key="page.url",
                )
                sample = self._extract_one(op, **extra)
                self.assert_article_kept(sample["text"])
                metadata = sample[Fields.meta][MetaKeys.html_metadata]
                self.assertEqual(metadata["url"], "")
                self.assertEqual(metadata["hostname"], "")

    def test_exclude_tables(self):
        text = self._extract_one(ExtractHtmlContentMapper(include_tables=False))["text"]
        self.assert_article_kept(text)
        self.assertNotIn("Mississippi", text)

    def test_include_comments(self):
        text = self._extract_one(ExtractHtmlContentMapper(include_comments=True))["text"]
        self.assert_article_kept(text)
        self.assertIn("Great article", text)

    def test_focus_modes_and_fast(self):
        for kwargs in [{"focus": "precision"}, {"focus": "recall"}, {"fast": True}]:
            with self.subTest(**kwargs):
                text = self._extract_one(ExtractHtmlContentMapper(**kwargs))["text"]
                self.assert_article_kept(text)
                self.assert_boilerplate_removed(text)

    def test_extract_metadata(self):
        op = ExtractHtmlContentMapper(extract_metadata=True, url_key="url")
        sample = self._extract_one(op, url="https://example.org/science/rivers")
        metadata = sample[Fields.meta][MetaKeys.html_metadata]
        self.assertEqual(metadata["title"], "Why Rivers Meander")
        self.assertEqual(metadata["author"], "Jane Doe")
        self.assertEqual(metadata["date"], "2024-05-17")
        self.assertEqual(metadata["sitename"], "Example Science")
        self.assertEqual(metadata["description"], "A short explanation of river meanders.")
        self.assertEqual(metadata["url"], "https://example.org/science/rivers")
        self.assertEqual(metadata["hostname"], "example.org")
        self.assert_article_kept(sample["text"])
        self.assertFalse(sample["text"].startswith("---"))

    def test_metadata_schema_is_stable_for_empty_pages(self):
        op = ExtractHtmlContentMapper(
            extract_metadata=True,
            metadata_field_name="page_meta",
        )
        dataset = self._run(op, [{"text": self.html}, {"text": "<html><body></body></html>"}])
        full, empty = dataset[0][Fields.meta]["page_meta"], dataset[1][Fields.meta]["page_meta"]
        self.assertEqual(set(full.keys()), set(empty.keys()))
        self.assertEqual(empty["title"], "")
        self.assertEqual(empty["tags"], [])

    def test_no_metadata_by_default(self):
        sample = self._extract_one(ExtractHtmlContentMapper())
        self.assertNotIn(MetaKeys.html_metadata, sample[Fields.meta])

    def test_invalid_url_is_ignored(self):
        op = ExtractHtmlContentMapper(
            output_format="markdown",
            include_links=True,
            extract_metadata=True,
            url_key="url",
        )
        sample = self._extract_one(op, url=42)
        self.assert_article_kept(sample["text"])
        metadata = sample[Fields.meta][MetaKeys.html_metadata]
        self.assertEqual(metadata["url"], "")
        self.assertEqual(metadata["hostname"], "")

    def test_no_main_content(self):
        op = ExtractHtmlContentMapper()
        dataset = self._run(op, [{"text": "<html><body></body></html>"}, {"text": ""}, {"text": "plain text"}])
        self.assertEqual(dataset["text"], ["", "", ""])

    def test_non_string_text_is_kept_with_stable_metadata(self):
        op = ExtractHtmlContentMapper(extract_metadata=True)
        dataset = self._run(
            op,
            [
                {"text": None},
                {"text": self.html},
                {"text": None, Fields.meta: None},
                {"text": None, Fields.meta: {MetaKeys.html_metadata: None}},
            ],
        )
        expected_keys = set(dataset[1][Fields.meta][MetaKeys.html_metadata])
        for index in (0, 2, 3):
            self.assertIsNone(dataset[index]["text"])
            metadata = dataset[index][Fields.meta][MetaKeys.html_metadata]
            self.assertEqual(set(metadata), expected_keys)
            self.assertEqual(metadata["title"], "")
            self.assertEqual(metadata["tags"], [])
        self.assert_article_kept(dataset[1]["text"])

    def test_multiprocess(self):
        dataset = self._run(ExtractHtmlContentMapper(num_proc=2), [{"text": self.html}] * 4)
        for text in dataset["text"]:
            self.assert_article_kept(text)
            self.assert_boilerplate_removed(text)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            ExtractHtmlContentMapper(output_format="xml")
        with self.assertRaises(ValueError):
            ExtractHtmlContentMapper(focus="fast")


if __name__ == "__main__":
    unittest.main()
