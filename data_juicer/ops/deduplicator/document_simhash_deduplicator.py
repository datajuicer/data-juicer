# Some code here has been modified from:
# https://github.com/bigscience-workshop/data-preparation
# --------------------------------------------------------

from collections import defaultdict, deque
from typing import Dict, Optional, Set

import numpy as np
import regex
from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import OPERATORS, Deduplicator
from ..common.helper_func import split_on_whitespace

simhash = LazyLoader("simhash", "simhash-pybind")

OP_NAME = "document_simhash_deduplicator"


@OPERATORS.register_module(OP_NAME)
class DocumentSimhashDeduplicator(Deduplicator):
    """Deduplicates samples at the document level using SimHash.

    This operator computes SimHash values for each sample and removes duplicates based on a
    specified Hamming distance threshold. It supports different tokenization methods:
    'space', 'punctuation', and 'character'. The SimHash is computed over shingles of a
    given window size, and the deduplication process clusters similar documents and retains
    only one from each cluster. The default mode converts text to lowercase and can ignore
    specific patterns. The key metric, Hamming distance, is used to determine similarity
    between SimHash values. Important notes:
    - The `ignore_pattern` parameter can be used to exclude certain substrings during
      SimHash computation.
    - For punctuation-based tokenization, the `ignore_pattern` should not include
      punctuations to avoid conflicts.
    - The `hamming_distance` must be less than the number of blocks (`num_blocks`).
    - Only the first sample in each cluster is retained by default."""

    def __init__(
        self,
        tokenization: str = "space",
        window_size: PositiveInt = 6,
        lowercase: bool = True,
        ignore_pattern: Optional[str] = None,
        num_blocks: PositiveInt = 6,
        hamming_distance: PositiveInt = 4,
        *args,
        **kwargs,
    ):
        """
        Initialization method :param tokenization: tokenization method for
        sample texts.

        It should be one of [space, punctuation, character]. For
        English-like languages, we recommend to use 'space'. And for
        Chinese-like languages, we recommend to use 'character'

        :param tokenization: tokenization method for sample texts
        :param window_size: window size of shingling
        :param lowercase: whether to convert text to lower case first
        :param ignore_pattern: whether to ignore sub-strings with
            specific pattern when computing simhash
        :param num_blocks: number of blocks in simhash computing
        :param hamming_distance: the max hamming distance threshold in
            near-duplicate detection. When the hamming distance of two
            sample texts is <= this threshold, they are regarded as
            similar samples and this op will only keep one of them after
            deduplication. This threshold should be always less than
            num_blocks
        """
        # about simhash computation
        super().__init__(*args, **kwargs)
        self.tokenization = tokenization
        self.window_size = window_size
        self.lowercase = lowercase
        self.ignore_pattern = ignore_pattern
        if self.ignore_pattern:
            self.ignore_pattern = regex.compile(self.ignore_pattern)

        # check parameters
        if self.ignore_pattern and self.tokenization == "punctuation":
            logger.warning(
                "Be careful that tokenization with punctuations "
                "won't work if the ignore pattern includes "
                "punctuations."
            )
        self.punctuation_pattern = regex.compile(r"\p{P}")

        # about deduplication
        self.num_blocks = num_blocks
        self.hamming_distance = hamming_distance

    def compute_hash(self, sample):
        """
        Compute simhash values for the sample.

        :param sample: input sample
        :return: sample with simhash value.
        """
        # check if it's computed already
        if HashKeys.simhash in sample:
            return sample

        text = sample[self.text_key]

        if self.lowercase:
            text = text.lower()
        if self.ignore_pattern:
            text = self.ignore_pattern.sub("", text)

        # get tokens for different tokenization method
        tokens = []
        if self.tokenization == "character":
            tokens = [str.encode(text[i : i + self.window_size]) for i in range(len(text) - self.window_size + 1)]
        elif self.tokenization == "punctuation":
            tokens = self.punctuation_pattern.split(text)
            tokens = [
                str.encode(" ".join(tokens[i : i + self.window_size]))
                for i in range(len(tokens) - self.window_size + 1)
            ]
        elif self.tokenization == "space":
            tokens = split_on_whitespace(text)
            tokens = [
                str.encode(" ".join(tokens[i : i + self.window_size]))
                for i in range(len(tokens) - self.window_size + 1)
            ]
        else:
            raise NotImplementedError(f"Unimplemented tokenization method [{self.tokenization}]")

        # compute simhash
        sample[HashKeys.simhash] = str(np.uint64(simhash.compute(map(simhash.unsigned_hash, tokens))))
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

        # find matches
        logger.info(f"Start querying {len(dataset)} samples.")
        matches = simhash.find_all(
            np.uint64(dataset[HashKeys.simhash]),
            self.num_blocks,
            self.hamming_distance,
        )
        logger.info(f"Querying done, found {len(matches)} matches.")

        # compute hash diff distribution
        graph = defaultdict(dict)
        for x, y in matches:
            x = str(x)
            y = str(y)
            graph[x][y] = graph[y][x] = True

        hash2ids: Dict[str, Set[str]] = defaultdict(set)
        hashes: Set[str] = set(dataset[HashKeys.simhash])
        hash2cluster: Dict[str, int] = {}
        visited: Set[str] = set()
        cluster_id: int = 0

        for sid, hash_val in enumerate(dataset[HashKeys.simhash]):
            hash2ids[hash_val].add(str(sid))

        # clustering
        dup_pairs = {}  # store duplicate pairs when show_num > 0
        while hashes:
            hash_val = hashes.pop()
            if hash_val in visited:
                continue

            # if this hash value is not in the matches list, it's regarded as a
            # single cluster
            if hash_val not in graph:
                continue

            # Otherwise, BFS to find the cluster
            q = deque([hash_val])
            visited.add(hash_val)
            hash2cluster[hash_val] = cluster_id
            if show_num > 0 and len(dup_pairs) < show_num:
                dup_pairs[cluster_id] = []

            while q:
                curr = q.popleft()
                for neighbor in graph[curr]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    q.append(neighbor)
                    hash2cluster[neighbor] = cluster_id

            cluster_id += 1
        logger.info(f"Found {cluster_id} clusters and {len(graph)} hashes.")

        # filter duplicated samples
        # NOTICE: For now, we only keep the first sample in a cluster. Maybe
        # there are some better strategies later.
        def _filter_simhash_dup_helper(sample, visited_clusters, visited_hashes):
            sample_hash_val = sample[HashKeys.simhash]
            if sample_hash_val not in hash2cluster:
                # single-sample cluster, we need to check hash value still.
                if sample_hash_val in visited_hashes:
                    return False
                else:
                    visited_hashes.add(sample_hash_val)
                    return True
            else:
                cluster_num = hash2cluster[sample_hash_val]
                if show_num > 0 and cluster_num in dup_pairs and len(dup_pairs[cluster_num]) < 2:
                    dup_pairs[cluster_num].append(sample)
                # regular cluster, check cluster number.
                if cluster_num in visited_clusters:
                    return False
                else:
                    visited_clusters.add(cluster_num)
                    return True

        cluster_record = set()
        hash_record = set()
        dataset = dataset.filter(
            _filter_simhash_dup_helper,
            fn_kwargs=dict(visited_clusters=cluster_record, visited_hashes=hash_record),
            load_from_cache_file=False if show_num > 0 else True,
        )
        logger.info(f"Keep {len(dataset)} samples after SimHash dedup.")

        return dataset, dup_pairs
