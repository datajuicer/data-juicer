import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import shutil
import json
from datasets import Dataset
from data_juicer.download.wikipedia import (
    get_wikipedia_urls, download_wikipedia,
    WikipediaDownloader, WikipediaIterator, WikipediaExtractor
)
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


# Field schema that download_wikipedia promises to produce.
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

    def test_wikipedia_urls(self):
        dump_date = "20241101"
        expected_urls = [
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2",
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream2.xml-p41243p151573.bz2",
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream3.xml-p311574p311329.bz2"
        ]
        
        with patch('requests.get') as mock_get:
            def mock_get_response(*args, **kwargs):
                url = args[0]
                mock_response = MagicMock()
                
                if 'dumpstatus.json' in url:
                    mock_response.content = bytes(json.dumps({
                        "jobs": {
                            "articlesmultistreamdump": {
                                "files": {
                                    "enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2": {
                                        "url": expected_urls[0]
                                    },
                                    "enwiki-20241101-pages-articles-multistream2.xml-p41243p151573.bz2": {
                                        "url": expected_urls[1]
                                    },
                                    "enwiki-20241101-pages-articles-multistream3.xml-p311574p311329.bz2": {
                                        "url": expected_urls[2]
                                    }
                                }
                            }
                        }
                    }), 'utf-8')
                else:
                    mock_response.content = bytes("""
                    <html>
                        <body>
                            <a href="20241101/">20241101/</a>
                        </body>
                    </html>
                    """, 'utf-8')
                
                return mock_response
                
            mock_get.side_effect = mock_get_response
            
            urls = get_wikipedia_urls(dump_date=dump_date)

            self.assertEqual(urls, expected_urls)

    @patch('data_juicer.download.wikipedia.get_wikipedia_urls')
    @patch('data_juicer.download.wikipedia.download_and_extract')
    def test_wikipedia_download(self, mock_download_and_extract, mock_get_urls):
        # download_wikipedia promises to: (1) fetch urls for the language/date,
        # (2) clip them by url_limit, (3) derive one output path per url,
        # (4) hand real downloader/iterator/extractor + the field schema to
        # download_and_extract, and (5) return its dataset unchanged.
        dump_date = "20241101"
        url_limit = 1
        item_limit = 50

        mock_urls = [
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2",
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream2.xml-p41243p151573.bz2",
        ]
        mock_get_urls.return_value = mock_urls

        expected_output_paths = [
            os.path.join(
                self.temp_dir,
                "enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2.jsonl",
            )
        ]

        returned_dataset = Dataset.from_dict({
            'text': [f"Article {i}" for i in range(10)],
            'title': [f"Title {i}" for i in range(10)],
            'id': [str(i) for i in range(10)],
            'url': [f"https://en.wikipedia.org/wiki/Title_{i}" for i in range(10)],
            'language': ['en'] * 10,
            'source_id': ['enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2'] * 10,
            'filename': ['enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2.jsonl'] * 10,
        })
        mock_download_and_extract.return_value = returned_dataset

        result = download_wikipedia(
            self.temp_dir,
            dump_date=dump_date,
            url_limit=url_limit,
            item_limit=item_limit,
        )

        mock_get_urls.assert_called_once_with(language='en', dump_date=dump_date)

        mock_download_and_extract.assert_called_once()
        call_args = mock_download_and_extract.call_args[0]
        # urls clipped to url_limit (only the first of two)
        self.assertEqual(call_args[0], mock_urls[:url_limit])
        self.assertEqual(call_args[1], expected_output_paths)
        self.assertIsInstance(call_args[2], WikipediaDownloader)
        self.assertIsInstance(call_args[3], WikipediaIterator)
        self.assertIsInstance(call_args[4], WikipediaExtractor)
        self.assertEqual(call_args[5], _WIKI_OUTPUT_FORMAT)
        # item_limit is forwarded as a keyword argument
        self.assertEqual(
            mock_download_and_extract.call_args[1].get('item_limit'), item_limit)

        # the dataset from download_and_extract is returned unchanged
        self.assertIs(result, returned_dataset)
        self.assertEqual(len(result), 10)
        self.assertEqual(
            sorted(result.features.keys()),
            sorted(_WIKI_OUTPUT_FORMAT.keys()),
        )


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
