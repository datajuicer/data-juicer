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
            
            # Verify returned URLs
            assert len(urls) == 3
            assert urls[0] == expected_urls[0]
            assert urls[1] == expected_urls[1]
            assert urls[2] == expected_urls[2]


    @patch('data_juicer.download.wikipedia.get_wikipedia_urls')
    @patch('data_juicer.download.downloader.download_and_extract')
    @patch('data_juicer.download.wikipedia.download_and_extract')  # Add this patch too
    def test_wikipedia_download(self, mock_download_and_extract_wiki, mock_download_and_extract, mock_get_urls):
        dump_date = "20241101"
        url_limit = 1
        item_limit = 50

        # Mock the URLs returned
        mock_urls = [
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2"
        ]
        mock_get_urls.return_value = mock_urls

        # Create expected output paths
        output_paths = [
            os.path.join(self.temp_dir, "enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2.jsonl")
        ]

        # Create mock dataset
        mock_dataset = Dataset.from_dict({
            'text': [f"Article {i}" for i in range(10)],
            'title': [f"Title {i}" for i in range(10)],
            'id': [str(i) for i in range(10)],
            'url': [f"https://en.wikipedia.org/wiki/Title_{i}" for i in range(10)],
            'language': ['en'] * 10,
            'source_id': ['enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2'] * 10,
            'filename': ['enwiki-20241101-pages-articles-multistream1.xml-p1p41242.bz2.jsonl'] * 10
        })

        # Set return value for both mocks
        mock_download_and_extract.return_value = mock_dataset
        mock_download_and_extract_wiki.return_value = mock_dataset

        # Add print statements to debug
        print("Before calling download_wikipedia")
        
        # Run the function
        result = download_wikipedia(
            self.temp_dir,
            dump_date=dump_date,
            url_limit=url_limit,
            item_limit=item_limit
        )

        print("After calling download_wikipedia")
        
        # Print mock call counts
        print(f"mock_download_and_extract.call_count: {mock_download_and_extract.call_count}")
        print(f"mock_download_and_extract_wiki.call_count: {mock_download_and_extract_wiki.call_count}")

        # Verify the calls
        mock_get_urls.assert_called_once_with(language='en', dump_date=dump_date)
        
        # Try both mocks
        if mock_download_and_extract.call_count > 0:
            mock = mock_download_and_extract
        else:
            mock = mock_download_and_extract_wiki
            
        # Verify download_and_extract was called with correct arguments
        mock.assert_called_once()
        call_args = mock.call_args[0]
        assert call_args[0] == mock_urls[:url_limit]  # urls (limited by url_limit)
        assert call_args[1] == output_paths  # output_paths
        assert isinstance(call_args[2], WikipediaDownloader)  # downloader
        assert isinstance(call_args[3], WikipediaIterator)  # iterator
        assert isinstance(call_args[4], WikipediaExtractor)  # extractor
        
        # Verify the output format
        expected_format = {
            'text': str,
            'title': str,
            'id': str,
            'url': str,
            'language': str,
            'source_id': str,
            'filename': str,
        }
        assert call_args[5] == expected_format  # output_format

        # Verify the result
        assert isinstance(result, Dataset)
        assert len(result) == 10
        assert all(field in result.features for field in expected_format.keys())


if __name__ == '__main__':
    unittest.main()
