import json
import os
import subprocess
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Optional, Tuple, Union
from urllib.parse import urljoin

import pandas as pd
import regex as re
import requests
from bs4 import BeautifulSoup
from datasets import Dataset

from data_juicer.utils.file_utils import (
    read_single_partition,
    single_partition_write_with_filename,
)


class DocumentDownloader(ABC):
    """Abstract class for downloading remote data to disk"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def download(self, url):
        pass


class DocumentIterator(ABC):
    """
    Abstract iterator class for reading in raw records that have been
    downloaded to disk
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def iterate(self, file_path):
        pass


class DocumentExtractor(ABC):
    """Abstract class for extracting text from records read from disk"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def extract(self, content):
        pass


def _download_and_extract_single_partition(
    paths: Tuple[str, str],
    downloader: DocumentDownloader,
    iterator: DocumentIterator,
    extractor: DocumentExtractor,
    output_type: str,
    keep_raw_download: bool,
    force_download: bool,
    input_meta: Union[str, dict] = None,
    meta: Union[str, dict] = None,
    item_limit=None,
) -> pd.DataFrame:
    url, output_path = paths

    if os.path.exists(output_path) and not force_download:
        partition = read_single_partition([output_path], filetype=output_type, add_filename=True)
        return partition

    downloaded_file = downloader.download(url)
    records = []
    # Iterate over all records in file
    item_count = 0
    for item in iterator.iterate(downloaded_file):
        item_count += 1
        if item_limit and item_count > item_limit:
            break
        record_meta, content = item
        # Extract the text from the record
        extracted = extractor.extract(content)
        if extracted is not None:
            text_meta, text = extracted
            if text is not None:
                line = {
                    "text": text,
                    **text_meta,
                    **record_meta,
                }
                records.append(line)

    partition = pd.DataFrame(records)
    filename = os.path.basename(output_path)
    output_dir = os.path.dirname(output_path)
    partition["filename"] = filename
    single_partition_write_with_filename(partition, output_dir, output_type=output_type)
    if not keep_raw_download:
        os.remove(downloaded_file)

    return partition


def download_and_extract(
    urls: List[str],
    output_paths: List[str],
    downloader: DocumentDownloader,
    iterator: DocumentIterator,
    extractor: DocumentExtractor,
    output_format: dict,
    output_type: str = "jsonl",
    keep_raw_download=False,
    force_download=False,
    input_meta: Union[str, dict] = None,
    item_limit=None,
) -> Dataset:
    """
    Downloads and extracts a dataset

    Args:
      urls: A list of urls to download the dataset from
      output_paths: A list of paths to save the final extracted output to.
        The raw output of the downloader will be saved using the path given by downloader.download(url).
      downloader: A DocumentDownloader that handles retrieving each file from its url and saving it to storage
      iterator: A DocumentIterator that handles iterating through the downloaded file's format
      extractor: A DocumentExtractor that handles extracting the data from its raw format into text
      output_format: A dictionary mappings columns to datatypes for the fields of each datapoint after extraction.
      output_type: The file type to save the dataset as.
      keep_raw_download: Whether to keep the pre-extracted download file.
      force_download: If False, will skip processing all files in output_paths that already exist and
        directly read from them instead.
      input_meta: A dictionary or a string formatted as a dictionary, which outlines
        the field names and their respective data types within the JSONL input file.
      item_limit: limit on number of items downloaded; for sampling and testing purposes

    Returns:
      A HuggingFace DataSet of the downloaded data
    """  # noqa: E501
    if len(urls) == 0:
        raise ValueError("No urls were provided to download")

    if len(urls) != len(output_paths):
        raise ValueError(
            f"Different number of urls and output_paths. " f"{len(urls)} urls vs {len(output_paths)} output_paths"
        )

    output_format = dict(sorted(output_format.items()))
    part = partial(
        _download_and_extract_single_partition,
        downloader=downloader,
        iterator=iterator,
        extractor=extractor,
        output_type=output_type,
        keep_raw_download=keep_raw_download,
        force_download=force_download,
        input_meta=input_meta,
        meta=output_format,
        item_limit=item_limit,
    )
    combined_df = pd.concat(map(part, zip(urls, output_paths)))  # list of DataFrames
    return Dataset.from_pandas(combined_df)


def get_wikipedia_urls(
    language="en",
    wikidumps_index_prefix="https://dumps.wikimedia.org",
    dump_date: Optional[str] = None,
) -> List[str]:
    """
    Retrieves all urls pointing to the latest Wikipedia dumps

    Args:
        language: Desired language of the Wikipedia dump.
        wikidumps_index_prefix: The base url for all wikipedia dumps
        dump_date: A string formatted as "YYYYMMDD" for the wikipedia dump to use.
          If None, latest dump is used.
    """  # noqa: E501
    wiki_index_url = urljoin(wikidumps_index_prefix, f"{language}wiki")
    if not dump_date:
        # First get the index
        raw_wiki_index = requests.get(wiki_index_url)
        wiki_index = raw_wiki_index.content.decode("utf-8")
        wiki_index_parsed = BeautifulSoup(wiki_index, "lxml")

        # Get all dumps available in the index
        dumps = wiki_index_parsed.find_all("a")
        dump_date = dumps[-2].text
    else:
        # A trailing / is needed for the url
        dump_date = dump_date + "/"

    # Get the json dump data
    wiki_latest_dump = urljoin(wiki_index_url + "/", dump_date)
    wiki_latest_dump_status = urljoin(wiki_latest_dump, "dumpstatus.json")
    raw_dump_data = requests.get(wiki_latest_dump_status)
    try:
        dump_data = json.loads(raw_dump_data.content)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"No wikipedia dump found for {dump_date[:-1]}")

    # Get all multistream files within the dump data
    wikipedia_urls = []
    for ifile in dump_data["jobs"]["articlesmultistreamdump"]["files"]:
        if "xml" in ifile:
            url = urljoin(wiki_latest_dump, ifile)
            wikipedia_urls.append(url)

    return wikipedia_urls


def get_arxiv_urls():
    command = "s5cmd --request-payer=requester ls s3://arxiv/src/ | grep '.tar'"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)

    if result.returncode != 0:
        raise RuntimeError(f"Unable to get arxiv urls: {result.stderr}")

    urls = result.stdout.split()[3::4]
    urls.sort()

    return urls


def validate_snapshot_format(snapshot: Optional[str]) -> None:
    """
    Validate snapshot format 'YYYY-WW'.

    Args:
        snapshot: Snapshot string in format 'YYYY-WW' or None

    Raises:
        ValueError: If format is invalid
    """
    if snapshot is None:
        return

    # Check basic format with regex
    pattern = r"^\d{4}-\d{2}$"
    if not re.match(pattern, snapshot):
        raise ValueError(f"Invalid snapshot format: {snapshot}. " "Expected format: 'YYYY-WW' (e.g., '2020-50')")

    # Parse year and week
    try:
        year, week = map(int, snapshot.split("-"))

        # Validate year
        if not (2000 <= year <= 2100):  # Reasonable year range
            raise ValueError(f"Year must be between 2000 and 2100, got {year}")

        # Validate week number (1-53)
        if not (1 <= week <= 53):
            raise ValueError(f"Week must be between 1 and 53, got {week}")

    except ValueError as e:
        if str(e).startswith("Week") or str(e).startswith("Year"):
            raise
        raise ValueError(f"Invalid snapshot format: {snapshot}. " "Expected format: 'YYYY-WW' (e.g., '2020-50')")
