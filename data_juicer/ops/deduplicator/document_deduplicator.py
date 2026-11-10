# Some code here has been modified from:
# https://github.com/bigscience-workshop/data-preparation/blob/main/preprocessing/training/01a_catalogue_cleaning_and_filtering/clean_helpers/deduplication.py
# --------------------------------------------------------

import hashlib
import string
from collections import defaultdict
from typing import Dict, Set

import regex as re

from data_juicer.utils.constant import HashKeys

from ..base_op import OPERATORS, Deduplicator


@OPERATORS.register_module("document_deduplicator")
class DocumentDeduplicator(Deduplicator):
    """Deduplicates samples at the document level using exact matching.

    This operator computes an MD5 hash for each sample's text. It can optionally convert the
    text to lowercase and ignore non-alphabet characters, including whitespaces, digits, and
    punctuation. The deduplication is based on the computed hash values, where samples with
    identical hashes are considered duplicates. The `compute_hash` method adds a 'hash' key
    to each sample, storing its MD5 hash. During processing, the first occurrence of each
    unique hash is kept, and subsequent duplicates are filtered out. If the `show_num`
    parameter is set, the operator also returns a specified number of duplicate pairs for
    inspection."""

    def __init__(
        self,
        lowercase: bool = False,
        ignore_non_character: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param lowercase: Whether to convert sample text to lower case
        :param ignore_non_character: Whether to ignore non-alphabet
            characters, including whitespaces, digits, and punctuations
        :param args: extra args
        :param kwargs: extra args.
        """
        super().__init__(*args, **kwargs)
        self.lowercase = lowercase
        self.remove_non_character_regex = (
            re.compile(f"\s+|\d+|[{re.escape(string.punctuation)}]")
            if ignore_non_character
            else None  # noqa: W605
        )

    def compute_hash(self, sample):
        """
        Compute md5 hash values for the sample.

        :param sample: input sample
        :return: sample with md5 hash value.
        """
        # check if it's computed already
        if HashKeys.hash in sample:
            return sample

        text = sample[self.text_key]
        if self.lowercase:
            text = text.lower()
        if self.remove_non_character_regex:
            text = self.remove_non_character_regex.sub("", text)

        def _get_hash(txt):
            return hashlib.md5(txt.strip().encode("utf-8")).hexdigest()

        sample[HashKeys.hash] = _get_hash(text)
        return sample

    def process(self, dataset, show_num=0):
        """
        For doc-level, dataset --> dataset.

        :param dataset: input dataset
        :param show_num: number of traced samples used when tracer is
            open.
        :return: deduplicated dataset and the sampled duplicate pairs.
        """
        # no need to deduplicate because too few samples
        if len(dataset) <= 1:
            return dataset, {}

        # keep the first sample of each hash; stream over the dataset instead
        # of materializing the entire hash column into a Python list first.
        # Peak memory drops from O(n) (full hash column) to O(unique_hashes)
        # (only the set of seen hashes is kept).
        hashes = set()
        keep = []
        dup_pairs = {}
        dup_hashes = None

        # First pass: if tracer is on, find the most duplicated hashes
        if show_num > 0:
            hash2ids: Dict[str, int] = {}
            for idx, sample in enumerate(dataset):
                h = sample[HashKeys.hash]
                hash2ids[h] = hash2ids.get(h, 0) + 1
            dup_samples = sorted(hash2ids.items(), key=lambda x: x[1], reverse=True)
            dup_hashes = set(
                [item[0] for item in dup_samples if item[1] > 1][:show_num]
            )
            dup_pairs = {hash_v: [] for hash_v in dup_hashes}

        # Second pass: collect keep indices and duplicate pairs
        for idx, sample in enumerate(dataset):
            h = sample[HashKeys.hash]
            if show_num > 0 and h in dup_hashes and len(dup_pairs[h]) < 2:
                # tracer is open and not enough duplicate sample pairs
                dup_pairs[h].append(sample)
            if h not in hashes:
                hashes.add(h)
                keep.append(idx)
        dataset = dataset.select(keep)
        return dataset, dup_pairs
