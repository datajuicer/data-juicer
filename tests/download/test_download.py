import unittest
import tempfile
import os
import shutil
import json
import bz2
import threading
import contextlib
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

from data_juicer.download.downloader import download_and_extract
from data_juicer.download.wikipedia import (
    get_wikipedia_urls,
    WikipediaDownloader, WikipediaIterator, WikipediaExtractor
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


# Field schema that Wikipedia download/extraction promises to produce.
_WIKI_OUTPUT_FORMAT = {
    "text": str,
    "title": str,
    "id": str,
    "url": str,
    "language": str,
    "source_id": str,
    "filename": str,
}

class TestDownload(DataJuicerTestCaseBase):
    def setUp(self):
        super().setUp()
        # Creates a temporary directory that persists until you delete it
        self.temp_dir = tempfile.mkdtemp(prefix='dj_test_')

    def tearDown(self):
        # Clean up the temporary directory after each test
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        super().tearDown()

    @contextlib.contextmanager
    def _local_http_server(self, root):
        """Serve *root* over HTTP on a random port and yield its URL prefix."""
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=root))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

    def test_wikipedia_urls(self):
        dump_date = "20241101"

        with tempfile.TemporaryDirectory() as root:
            dump_dir = os.path.join(root, "enwiki", dump_date)
            os.makedirs(dump_dir)
            files = [
                "enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2",
                "enwiki-20241101-pages-articles-multistream2.xml-p41243p151573.bz2",
                "enwiki-20241101-pages-articles-multistream3.xml-p311574p311329.bz2",
            ]
            with open(os.path.join(dump_dir, "dumpstatus.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "jobs": {
                            "articlesmultistreamdump": {
                                "files": {name: {} for name in files}
                            }
                        }
                    },
                    f,
                )

            with self._local_http_server(root) as prefix:
                expected_urls = [
                    f"{prefix}/enwiki/{dump_date}/{name}"
                    for name in files
                ]
                urls = get_wikipedia_urls(
                    wikidumps_index_prefix=prefix,
                    dump_date=dump_date,
                )

        self.assertEqual(urls, expected_urls)

    def test_wikipedia_urls_latest_index_and_invalid_dump_json(self):
        import lxml  # noqa: F401

        with tempfile.TemporaryDirectory() as root:
            wiki_dir = os.path.join(root, "enwiki")
            dump_dir = os.path.join(wiki_dir, "20250101")
            bad_dump_dir = os.path.join(wiki_dir, "19000101")
            os.makedirs(dump_dir)
            os.makedirs(bad_dump_dir)
            with open(os.path.join(wiki_dir, "index.html"), "w", encoding="utf-8") as f:
                f.write('<a href="20241201/">20241201/</a><a href="20250101/">20250101/</a><a href="latest/">latest/</a>')
            with open(os.path.join(dump_dir, "dumpstatus.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "jobs": {
                            "articlesmultistreamdump": {
                                "files": {
                                    "enwiki-20250101-pages-articles.xml.bz2": {},
                                    "enwiki-20250101-pages-meta-history.xml.bz2": {},
                                    "checksums.txt": {},
                                }
                            }
                        }
                    },
                    f,
                )
            with open(os.path.join(bad_dump_dir, "dumpstatus.json"), "w", encoding="utf-8") as f:
                f.write("not-json")

            with self._local_http_server(root) as prefix:
                expected = [
                    f"{prefix}/enwiki/20250101/enwiki-20250101-pages-articles.xml.bz2",
                    f"{prefix}/enwiki/20250101/enwiki-20250101-pages-meta-history.xml.bz2",
                ]
                self.assertEqual(get_wikipedia_urls(wikidumps_index_prefix=prefix), expected)
                with self.assertRaises(ValueError):
                    get_wikipedia_urls(wikidumps_index_prefix=prefix, dump_date="19000101")

    def test_wikipedia_download_components_extract_small_local_dump(self):
        dump_date = "20250101"
        dump_file = f"enwiki-{dump_date}-pages-articles.xml.bz2"

        with tempfile.TemporaryDirectory() as root:
            dump_dir = os.path.join(root, "enwiki", dump_date)
            os.makedirs(dump_dir)

            xml = b"""<mediawiki>
            <page><title>Main Page</title><ns>0</ns><id>1</id>
            <revision><text>Article body [[Category:Science]]
            [[File:Skip.jpg]] <ref>drop me</ref></text></revision></page>
            <page><title>Second Page</title><ns>0</ns><id>2</id>
            <revision><text>Second body</text></revision></page>
            </mediawiki>"""
            with bz2.open(os.path.join(dump_dir, dump_file), "wb") as f:
                f.write(xml)

            with self._local_http_server(root) as prefix:
                raw_dir = os.path.join(self.temp_dir, "raw")
                os.makedirs(raw_dir)
                result = download_and_extract(
                    [f"{prefix}/enwiki/{dump_date}/{dump_file}"],
                    [os.path.join(self.temp_dir, f"{dump_file}.jsonl")],
                    WikipediaDownloader(download_dir=raw_dir),
                    WikipediaIterator(language="en"),
                    WikipediaExtractor(language="en"),
                    _WIKI_OUTPUT_FORMAT,
                    item_limit=1,
                )

        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["title"], "Main Page")
        self.assertEqual(row["id"], "1")
        self.assertEqual(row["language"], "en")
        self.assertEqual(row["url"], "https://en.wikipedia.org/wiki/Main%20Page")
        self.assertIn("Article body", row["text"])
        self.assertIn("Science", row["text"])
        self.assertNotIn("Skip.jpg", row["text"])
        self.assertNotIn("drop me", row["text"])
        self.assertEqual(row["filename"], f"{dump_file}.jsonl")
        self.assertEqual(sorted(result.features.keys()), sorted(_WIKI_OUTPUT_FORMAT.keys()))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, f"{dump_file}.jsonl")))
        self.assertEqual(os.listdir(raw_dir), [])

    def test_wikipedia_downloader_iterator_and_extractor_without_network(self):
        dl_dir = os.path.join(self.temp_dir, "raw")
        os.makedirs(dl_dir, exist_ok=True)
        downloader = WikipediaDownloader(dl_dir, verbose=True)
        url = "https://dumps.wikimedia.org/enwiki/20250101/file.xml.bz2"
        expected_path = os.path.join(dl_dir, "enwiki-20250101-file.xml.bz2")

        with open(expected_path, "wb") as f:
            f.write(b"cached")
        self.assertEqual(downloader.download(url), expected_path)

        xml = b"""<mediawiki>
        <page><title>Main Page</title><ns>0</ns><id>1</id>
        <revision><text>Raw text</text></revision></page>
        <page><title>Talk Page</title><ns>1</ns><id>2</id>
        <revision><text>Talk text</text></revision></page>
        <page><title>Redirect</title><ns>0</ns><id>3</id><redirect />
        <revision><text>Redirect text</text></revision></page>
        <page><title>Empty</title><ns>0</ns><id>4</id>
        <revision><text /></revision></page>
        </mediawiki>"""
        bz2_path = os.path.join(self.temp_dir, "wiki.xml.bz2")
        with bz2.open(bz2_path, "wb") as f:
            f.write(xml)

        rows = list(WikipediaIterator(language="en", log_frequency=1).iterate(bz2_path))
        self.assertEqual(len(rows), 1)
        meta, raw = rows[0]
        self.assertEqual(meta["title"], "Main Page")
        self.assertEqual(meta["url"], "https://en.wikipedia.org/wiki/Main%20Page")
        self.assertEqual(raw, "Raw text")

        _, text = WikipediaExtractor(language="en").extract(
            "__NOTOC__ [[File:Example.jpg]] [[Category:Science]] <ref>remove</ref> body"
        )
        self.assertNotIn("NOTOC", text)
        self.assertNotIn("Example.jpg", text)
        self.assertNotIn("remove", text)
        self.assertIn("Science", text)
        self.assertIn("body", text)


class ValidateSnapshotFormatTest(DataJuicerTestCaseBase):

    def test_none_is_valid(self):
        from data_juicer.download.downloader import validate_snapshot_format
        validate_snapshot_format(None)

    def test_valid_format(self):
        from data_juicer.download.downloader import validate_snapshot_format
        validate_snapshot_format("2020-50")
        validate_snapshot_format("2024-01")
        validate_snapshot_format("2024-53")

    def test_invalid_format_no_dash(self):
        from data_juicer.download.downloader import validate_snapshot_format
        with self.assertRaises(ValueError) as ctx:
            validate_snapshot_format("202050")
        self.assertIn("Invalid snapshot format", str(ctx.exception))

    def test_invalid_format_extra_parts(self):
        from data_juicer.download.downloader import validate_snapshot_format
        with self.assertRaises(ValueError):
            validate_snapshot_format("2020-50-01")

    def test_invalid_format_letters(self):
        from data_juicer.download.downloader import validate_snapshot_format
        with self.assertRaises(ValueError):
            validate_snapshot_format("abcd-ef")

    def test_year_too_low(self):
        from data_juicer.download.downloader import validate_snapshot_format
        with self.assertRaises(ValueError) as ctx:
            validate_snapshot_format("1999-01")
        self.assertIn("Year must be between", str(ctx.exception))

    def test_year_too_high(self):
        from data_juicer.download.downloader import validate_snapshot_format
        with self.assertRaises(ValueError) as ctx:
            validate_snapshot_format("2101-01")
        self.assertIn("Year must be between", str(ctx.exception))

    def test_week_zero(self):
        from data_juicer.download.downloader import validate_snapshot_format
        with self.assertRaises(ValueError) as ctx:
            validate_snapshot_format("2020-00")
        self.assertIn("Week must be between", str(ctx.exception))

    def test_week_too_high(self):
        from data_juicer.download.downloader import validate_snapshot_format
        with self.assertRaises(ValueError) as ctx:
            validate_snapshot_format("2020-54")
        self.assertIn("Week must be between", str(ctx.exception))

    def test_boundary_valid_year(self):
        from data_juicer.download.downloader import validate_snapshot_format
        validate_snapshot_format("2000-01")
        validate_snapshot_format("2100-53")

    def test_boundary_valid_week(self):
        from data_juicer.download.downloader import validate_snapshot_format
        validate_snapshot_format("2024-01")
        validate_snapshot_format("2024-53")

    def test_empty_string(self):
        from data_juicer.download.downloader import validate_snapshot_format
        with self.assertRaises(ValueError):
            validate_snapshot_format("")


if __name__ == '__main__':
    unittest.main()
