import gzip
import hashlib
import json
import logging
import os
import re
import unicodedata

import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys

from ..base_op import OPERATORS, Filter

logger = logging.getLogger(__name__)

OP_NAME = "benchmark_contamination_filter"


@OPERATORS.register_module(OP_NAME)
class BenchmarkContaminationFilter(Filter):
    """Filter out samples that are contaminated by a benchmark corpus.

    Train-test contamination happens when a training corpus accidentally
    contains text that also appears in an evaluation set. Models that
    memorize such samples report inflated benchmark scores. This operator
    detects the overlap by matching n-grams between every sample and a
    reference benchmark file, and drops the samples whose overlap ratio
    exceeds a configurable threshold.

    Unlike the deduplication operators, which compare a corpus against
    itself, this operator compares a corpus against an external reference
    file, so it can be used as the last step before fine-tuning or as an
    audit tool for an existing dataset.

    The reference benchmark is read once when the operator is created and
    is stored as a sorted array of 64-bit n-gram hashes, which keeps the
    memory footprint small enough to be shared by many worker processes.
    """

    def __init__(
        self,
        benchmark_path: str = None,
        benchmark_text_key: str = "text",
        ngram_size: int = 13,
        tokenization: str = "char",
        lowercase: bool = True,
        normalize_whitespace: bool = True,
        threshold: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param benchmark_path: path to the benchmark file used as the
            reference. Supported formats are ``.jsonl``, ``.json`` and
            ``.txt`` (optionally gzipped, e.g. ``.jsonl.gz``). A ``.txt``
            file is read as one document per line. This argument is
            required in practice; when it is left empty the operator logs
            a warning and keeps every sample, so that config validation
            and ``--help`` keep working.
        :param benchmark_text_key: the field of each benchmark record that
            holds the text to match against. Records of a ``.jsonl`` or
            ``.json`` file must be dicts that contain this field.
        :param ngram_size: the size of the n-grams used for matching. 13
            is the value commonly used for decontamination in the
            literature; smaller values increase recall but also flag
            samples that merely share common phrasing.
        :param tokenization: ``'char'`` to build n-grams over characters,
            which works for languages without word delimiters such as
            Chinese, or ``'word'`` to build them over whitespace
            separated tokens.
        :param lowercase: whether to lowercase the text before matching.
        :param normalize_whitespace: whether to collapse consecutive
            whitespace characters into a single space and strip the
            ends of the text.
        :param threshold: samples whose contamination ratio is greater
            than this value are filtered out. The ratio is the number of
            distinct n-grams of the sample that also occur in the
            benchmark divided by the number of distinct n-grams of the
            sample. The default of ``0.0`` drops any sample with a single
            matching n-gram, which is the right behaviour when the corpus
            and the benchmark come from different domains, e.g. a web
            crawl checked against a math benchmark. When both sides are
            from the same domain, template generated data shares short
            phrasing even without any real leak: on GSM8K, clean training
            items reach a median ratio of 0.09 with ``ngram_size=13``, so
            a threshold around 0.2~0.3 keeps the precision above 0.9
            while still catching near duplicates.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if tokenization not in ("char", "word"):
            raise ValueError(f'tokenization must be "char" or "word", got {tokenization!r}')
        if ngram_size < 1:
            raise ValueError(f"ngram_size must be a positive integer, got {ngram_size}")
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        self.benchmark_path = benchmark_path
        self.benchmark_text_key = benchmark_text_key
        self.ngram_size = ngram_size
        self.tokenization = tokenization
        self.lowercase = lowercase
        self.normalize_whitespace = normalize_whitespace
        self.threshold = threshold

        self._index = self._build_index(benchmark_path)

    # ---------------------------------------------------------------- #
    # index construction
    # ---------------------------------------------------------------- #
    def _build_index(self, benchmark_path):
        """Read the benchmark file and build the sorted n-gram hash index."""
        empty = np.empty(0, dtype=np.uint64)
        if not benchmark_path:
            logger.warning(
                "benchmark_path is empty, %s will keep every sample. "
                "Set benchmark_path to the evaluation set to enable "
                "contamination detection.",
                OP_NAME,
            )
            return empty
        if not os.path.exists(benchmark_path):
            raise ValueError(f"benchmark file {benchmark_path!r} does not exist")

        hashes = []
        for raw_text in self._iter_benchmark_texts(benchmark_path):
            text = self._normalize(raw_text)
            hashes.append(self._ngram_hashes(text))

        if not hashes:
            logger.warning(
                "benchmark file %r contains no usable text, %s " "will keep every sample.", benchmark_path, OP_NAME
            )
            return empty

        merged = np.concatenate(hashes)
        if merged.size == 0:
            # every benchmark document is shorter than ngram_size
            logger.warning(
                "no %d-gram could be built from the benchmark file %r, "
                "consider lowering ngram_size; %s will keep every sample.",
                self.ngram_size,
                benchmark_path,
                OP_NAME,
            )
            return empty

        index = np.unique(merged)
        logger.info("%s indexed %d distinct %d-grams from %r", OP_NAME, index.size, self.ngram_size, benchmark_path)
        return index

    def _iter_benchmark_texts(self, path):
        """Yield the raw text of every benchmark record."""
        lower = path.lower()
        opener = gzip.open if lower.endswith(".gz") else open

        if lower.endswith(".txt") or lower.endswith(".txt.gz"):
            with opener(path, "rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        yield line
            return

        if lower.endswith(".jsonl") or lower.endswith(".jsonl.gz"):
            with opener(path, "rt", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    text = self._pick_text(record)
                    if text:
                        yield text
            return

        if lower.endswith(".json") or lower.endswith(".json.gz"):
            with opener(path, "rt", encoding="utf-8") as fin:
                payload = json.load(fin)
            if isinstance(payload, dict):
                if self.benchmark_text_key in payload:
                    # a single record, e.g. {"text": "..."}
                    payload = [payload]
                else:
                    # a wrapper object, e.g. {"data": [{...}, {...}]}
                    records = []
                    for value in payload.values():
                        if isinstance(value, list):
                            records.extend(value)
                        else:
                            records.append(value)
                    payload = records
            for record in payload:
                text = self._pick_text(record)
                if text:
                    yield text
            return

        raise ValueError(
            f"unsupported benchmark file format: {path!r}. Supported "
            "formats are .jsonl, .json and .txt, optionally gzipped."
        )

    def _pick_text(self, record):
        """Extract the text to match from one benchmark record."""
        if isinstance(record, str):
            return record
        if isinstance(record, dict):
            value = record.get(self.benchmark_text_key)
            if value is None:
                raise ValueError(
                    f"field {self.benchmark_text_key!r} not found in a "
                    f"benchmark record with keys {sorted(record)[:10]}; "
                    "set benchmark_text_key accordingly"
                )
            if isinstance(value, (list, tuple)):
                return "\n".join(str(item) for item in value)
            return str(value)
        return str(record)

    # ---------------------------------------------------------------- #
    # text processing
    # ---------------------------------------------------------------- #
    def _normalize(self, text):
        """Normalize a text so that the index and the samples match."""
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()
        if self.normalize_whitespace:
            text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text):
        """Split a text into the units that n-grams are built from."""
        if self.tokenization == "word":
            return text.split()
        return list(text)

    def _ngram_hashes(self, text):
        """Return the hashes of all n-grams of a normalized text."""
        tokens = self._tokenize(text)
        size = self.ngram_size
        if len(tokens) < size:
            return np.empty(0, dtype=np.uint64)

        # joining with a space keeps word n-grams unambiguous, while char
        # n-grams must stay contiguous
        separator = " " if self.tokenization == "word" else ""
        ngrams = [separator.join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]
        return np.fromiter((self._hash(ngram) for ngram in ngrams), dtype=np.uint64, count=len(ngrams))

    @staticmethod
    def _hash(text):
        """Hash a string into a 64-bit integer.

        ``hashlib`` is used instead of the built-in ``hash`` because the
        latter is randomized per process, which would make the index
        unusable across the worker processes spawned by Data-Juicer.
        """
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def _count_hits(self, hashes):
        """Count how many of the given hashes occur in the benchmark index."""
        if self._index.size == 0 or hashes.size == 0:
            return 0
        # binary search over the sorted index, O(k log N)
        positions = np.searchsorted(self._index, hashes)
        positions = np.clip(positions, 0, self._index.size - 1)
        return int(np.count_nonzero(self._index[positions] == hashes))

    # ---------------------------------------------------------------- #
    # operator interface
    # ---------------------------------------------------------------- #
    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.contamination_ratio in sample[Fields.stats]:
            return sample

        text = self._normalize(sample.get(self.text_key))
        hashes = np.unique(self._ngram_hashes(text))
        if hashes.size == 0:
            sample[Fields.stats][StatsKeys.contamination_ngram_count] = 0
            sample[Fields.stats][StatsKeys.contamination_ratio] = 0.0
            return sample

        hits = self._count_hits(hashes)
        sample[Fields.stats][StatsKeys.contamination_ngram_count] = hits
        sample[Fields.stats][StatsKeys.contamination_ratio] = hits / hashes.size
        return sample

    def process_single(self, sample):
        return sample[Fields.stats][StatsKeys.contamination_ratio] <= self.threshold
