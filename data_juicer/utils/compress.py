import os
import re
import shutil
from abc import ABC, abstractmethod
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from datasets import Dataset
from datasets.utils.extract import Extractor as HF_Extractor
from datasets.utils.filelock import FileLock as HF_FileLock
from loguru import logger

from data_juicer.utils import cache_utils


class FileLock(HF_FileLock):
    """
    File lock for compression or decompression, and
    remove lock file automatically.
    """

    def _release(self):
        super()._release()
        try:
            # logger.debug(f'Remove {self.lock_file}')
            os.remove(self.lock_file)
        # The file is already deleted and that's what we want.
        except OSError:
            pass
        return None


class Extractor(HF_Extractor):
    """
    Extract content from a compressed file.
    """

    @classmethod
    def extract(
        cls,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
        extractor_format: str,
    ):
        """
        Extract content from a compressed file.

        :param input_path: path to compressed file.
        :param output_path: path to uncompressed file.
        :param extractor_format: extraction format,
            see supported algorithm in `Extractor` of huggingface dataset.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Prevent parallel extractions
        lock_path = str(Path(output_path).with_suffix(".lock"))
        with FileLock(lock_path):
            shutil.rmtree(output_path, ignore_errors=True)
            extractor = cls.extractors[extractor_format]
            return extractor.extract(input_path, output_path)


class BaseCompressor(ABC):
    """
    Base class that compresses a file.
    """

    @staticmethod
    @abstractmethod
    def compress(input_path: Union[Path, str], output_path: Union[Path, str]):
        """
        Compress input file and save to output file.

        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """
        ...


class ZstdCompressor(BaseCompressor):
    """
    This class compresses a file using the `zstd` algorithm.
    """

    @staticmethod
    def compress(input_path: Union[Path, str], output_path: Union[Path, str]):
        """
        Compress input file and save to output file.

        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """

        import zstandard as zstd

        cctx = zstd.ZstdCompressor()
        with open(input_path, "rb") as ifh, open(output_path, "wb") as ofh:
            cctx.copy_stream(ifh, ofh)


class Lz4Compressor(BaseCompressor):
    """
    This class compresses a file using the `lz4` algorithm.
    """

    @staticmethod
    def compress(input_path: Union[Path, str], output_path: Union[Path, str]):
        """
        Compress a input file and save to output file.

        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """
        import lz4.frame

        with open(input_path, "rb") as input_file:
            with lz4.frame.open(output_path, "wb") as compressed_file:
                shutil.copyfileobj(input_file, compressed_file)


class GzipCompressor(BaseCompressor):
    """
    This class compresses a file using the `gzip` algorithm.
    """

    @staticmethod
    def compress(input_path: Union[Path, str], output_path: Union[Path, str]):
        """
        Compress input file and save to output file.

        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """
        import gzip

        with open(input_path, "rb") as input_file:
            with gzip.open(output_path, "wb") as compressed_file:
                shutil.copyfileobj(input_file, compressed_file)


class Compressor:
    """
    This class that contains multiple compressors.
    """

    compressors: Dict[str, Type[BaseCompressor]] = {
        "gzip": GzipCompressor,
        # "zip": ZipCompressor,
        # "xz": XzCompressor,
        # "rar": RarCompressor,
        "zstd": ZstdCompressor,
        # "bz2": Bzip2Compressor,
        # "7z": SevenZipCompressor,
        "lz4": Lz4Compressor,
    }

    @classmethod
    def compress(
        cls,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
        compressor_format: str,
    ):
        """
        Compress input file and save to output file.

        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        :param compressor_format: compression format,
            see supported algorithm in `compressors`.
        """

        assert compressor_format in cls.compressors.keys()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Prevent parallel extractions
        lock_path = str(Path(output_path).with_suffix(".lock"))
        with FileLock(lock_path):
            shutil.rmtree(output_path, ignore_errors=True)
            compressor = cls.compressors[compressor_format]
            compressor.compress(input_path, output_path)


class CompressManager:
    """
    This class is used to compress or decompress a input file
    using compression format algorithms.
    """

    def __init__(self, compressor_format: str = "zstd"):
        """
        Initialization method.

        :param compressor_format: compression format algorithms,
            default `zstd`.
        """

        assert compressor_format in Compressor.compressors.keys()
        self.compressor_format = compressor_format
        self.compressor = Compressor
        self.extractor = Extractor

    def compress(
        self,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
    ):
        """
        Compress input file and save to output file.

        :param input_path: path to uncompressed file.
        :param output_path: path to compressed file.
        """
        self.compressor.compress(input_path, output_path, self.compressor_format)

    def decompress(
        self,
        input_path: Union[Path, str],
        output_path: Union[Path, str],
    ):
        """
        Decompress input file and save to output file.

        :param input_path: path to compressed file.
        :param output_path: path to uncompressed file.
        """
        self.extractor.extract(input_path, output_path, self.compressor_format)


class CacheCompressManager:
    """
    This class is used to compress or decompress huggingface cache files
    using compression format algorithms.
    """

    def __init__(self, compressor_format: str = "zstd"):
        """
        Initialization method.

        :param compressor_format: compression format algorithms,
            default `zstd`.
        """
        self.compressor_format = compressor_format
        self.compressor_extension = "." + compressor_format
        self.compress_manager = CompressManager(compressor_format=compressor_format)
        self.pattern = re.compile(r"_\d{5}_of_")

    def _get_raw_filename(self, filename: Union[Path, str]):
        """
        Get a uncompressed file name from a compressed file.
        :param filename: path to compressed file.
        :return: path to uncompressed file.
        """
        assert filename.endswith(self.compressor_format)
        return str(filename)[: -len(self.compressor_extension)]

    def _get_compressed_filename(self, filename: Union[Path, str]):
        """
        Get a compressed file name from a uncompressed file.
        :param filename: path to uncompressed file.
        :return: path to compressed file.
        """
        return str(filename) + self.compressor_extension

    def _get_cache_directory(self, ds):
        """
        Get dataset cache directory.
        :param ds: input dataset.
        :return: dataset cache directory.
        """
        current_cache_files = [os.path.abspath(cache_file["filename"]) for cache_file in ds.cache_files]
        if not current_cache_files:
            return None
        cache_directory = os.path.dirname(current_cache_files[0])
        return cache_directory

    def _get_cache_file_names(
        self, cache_directory: str, fingerprints: Union[str, List[str]] = None, extension=".arrow"
    ):
        """
        Get all cache files in the dataset cache directory with fingerprints,
        which ends with specified extension.

        :param cache_directory: dataset cache directory.
        :param fingerprints: fingerprints of cache files. String or List are
            accepted. If `None`, we will find all cache files which starts with
            `cache-` and ends with specified extension.
        :param extension: extension of cache files, default `.arrow`
        :return: list of file names
        """
        if cache_directory is None:
            return []
        if fingerprints is None:
            fingerprints = [""]
        if isinstance(fingerprints, str):
            fingerprints = [fingerprints]

        files: List[str] = os.listdir(cache_directory)
        f_names = []
        for f_name in files:
            for fingerprint in fingerprints:
                if f_name.startswith(f"cache-{fingerprint}") and f_name.endswith(extension):
                    f_names.append(f_name)
        return f_names

    def compress(self, prev_ds: Dataset, this_ds: Dataset = None, num_proc: int = 1):
        """
        Compress cache files with fingerprint in dataset cache directory.

        :param prev_ds: previous dataset whose cache files need to be
            compressed here.
        :param this_ds: Current dataset that is computed from the previous
            dataset. There might be overlaps between cache files of them, so we
            must not compress cache files that will be used again in the
            current dataset. If it's None, it means all cache files of previous
            dataset should be compressed.
        :param num_proc: number of processes to compress cache files.
        """
        # remove cache files from the list of cache files to be compressed
        prev_cache_names = [item["filename"] for item in prev_ds.cache_files]
        this_cache_names = [item["filename"] for item in this_ds.cache_files] if this_ds else []
        caches_to_compress = list(set(prev_cache_names) - set(this_cache_names))

        files_to_remove = []
        files_printed = set()
        if num_proc > 1:
            pool = Pool(num_proc)
        for full_name in caches_to_compress:
            # ignore the cache file of the original dataset and only consider
            # the cache files of following OPs
            if not os.path.basename(full_name).startswith("cache-"):
                continue
            # If there are no specified cache files, just skip
            if not os.path.exists(full_name):
                continue
            compress_filename = self._get_compressed_filename(full_name)
            formatted_cache_name = self.format_cache_file_name(compress_filename)

            if not os.path.exists(compress_filename):
                if formatted_cache_name not in files_printed:
                    logger.info(f"Compressing cache file to {formatted_cache_name}")
                if num_proc > 1:
                    pool.apply_async(
                        self.compress_manager.compress,
                        args=(
                            full_name,
                            compress_filename,
                        ),
                    )
                else:
                    self.compress_manager.compress(full_name, compress_filename)
            else:
                if formatted_cache_name not in files_printed:
                    logger.debug(f"Found compressed cache file {formatted_cache_name}")
            files_printed.add(formatted_cache_name)
            files_to_remove.append(full_name)
        if num_proc > 1:
            pool.close()
            pool.join()

        # clean up raw cache file
        for file_path in files_to_remove:
            logger.debug(f"Removing cache file {file_path}")
            os.remove(file_path)

    def decompress(self, ds: Dataset, fingerprints: Union[str, List[str]] = None, num_proc: int = 1):
        """
        Decompress compressed cache files with fingerprint in
        dataset cache directory.

        :param ds: input dataset.
        :param fingerprints: fingerprints of cache files. String or List are
            accepted. If `None`, we will find all cache files which starts with
            `cache-` and ends with compression format.
        :param num_proc: number of processes to decompress cache files.
        """
        cache_directory = self._get_cache_directory(ds)
        if cache_directory is None:
            return

        # find compressed cache files with given fingerprints
        f_names = self._get_cache_file_names(
            cache_directory=cache_directory, fingerprints=fingerprints, extension=self.compressor_extension
        )
        files_printed = set()
        if num_proc > 1:
            pool = Pool(num_proc)
        for f_name in f_names:
            full_name = os.path.abspath(os.path.join(cache_directory, f_name))
            raw_filename = self._get_raw_filename(full_name)
            formatted_cache_name = self.format_cache_file_name(raw_filename)

            if not os.path.exists(raw_filename):
                if formatted_cache_name not in files_printed:
                    logger.info(f"Decompressing cache file to " f"{formatted_cache_name}")
                    files_printed.add(formatted_cache_name)
                if num_proc > 1:
                    pool.apply_async(
                        self.compress_manager.decompress,
                        args=(
                            full_name,
                            raw_filename,
                        ),
                    )
                else:
                    self.compress_manager.decompress(full_name, raw_filename)
            else:
                if formatted_cache_name not in files_printed:
                    logger.debug(f"Found uncompressed cache files " f"{formatted_cache_name}")
        if num_proc > 1:
            pool.close()
            pool.join()

    def format_cache_file_name(self, cache_file_name: Optional[str]) -> Optional[str]:
        """
        Use `*` to replace the sub rank in a cache file name.
        :param cache_file_name: a cache file name.
        """

        if not cache_file_name:
            return cache_file_name

        cache_file_name = re.sub(pattern=self.pattern, repl=r"_*_of_", string=cache_file_name)
        return cache_file_name

    def cleanup_cache_files(self, ds):
        """
        Clean up all compressed cache files in dataset cache directory,
        which starts with `cache-` and ends with compression format
        :param ds: input dataset.
        """
        cache_directory = self._get_cache_directory(ds)
        if cache_directory is None:
            return
        f_names = self._get_cache_file_names(cache_directory=cache_directory, extension=self.compressor_extension)
        files_printed = set()
        for f_name in f_names:
            full_name = os.path.abspath(os.path.join(cache_directory, f_name))
            formatted_cache_name = self.format_cache_file_name(full_name)
            if formatted_cache_name not in files_printed:
                logger.debug(f"Clean up cache file {formatted_cache_name}")
                files_printed.add(formatted_cache_name)
            os.remove(full_name)
        return len(f_names)


class CompressionOff:
    """Define a range that turn off the cache compression temporarily."""

    def __enter__(self):
        """
        Record the original cache compression method and turn it off.
        """
        from . import cache_utils

        self.original_cache_compress = cache_utils.CACHE_COMPRESS
        cache_utils.CACHE_COMPRESS = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore the original cache compression method.
        """
        from . import cache_utils

        cache_utils.CACHE_COMPRESS = self.original_cache_compress


def compress(prev_ds, this_ds=None, num_proc=1):
    if cache_utils.CACHE_COMPRESS:
        CacheCompressManager(cache_utils.CACHE_COMPRESS).compress(prev_ds, this_ds, num_proc)


def decompress(ds, fingerprints=None, num_proc=1):
    if cache_utils.CACHE_COMPRESS:
        CacheCompressManager(cache_utils.CACHE_COMPRESS).decompress(ds, fingerprints, num_proc)


def cleanup_compressed_cache_files(ds):
    if cache_utils.CACHE_COMPRESS is None:
        for fmt in Compressor.compressors.keys():
            CacheCompressManager(fmt).cleanup_cache_files(ds)
    else:
        CacheCompressManager(cache_utils.CACHE_COMPRESS).cleanup_cache_files(ds)
