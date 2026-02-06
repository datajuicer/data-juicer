# Some code here has been modified from:
# https://github.com/bigcode-project/bigcode-dataset/blob/main/near_deduplication/minhash_deduplication.py
# --------------------------------------------------------

import hashlib
import struct
from collections import defaultdict
from typing import Optional

import numpy as np
import regex
from loguru import logger
from pydantic import Field, PositiveInt
from tqdm import tqdm
from typing_extensions import Annotated

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import prepare_sentencepiece_model

from ..base_op import OPERATORS, Deduplicator
from ..common.helper_func import UnionFind, split_on_whitespace

integrate = LazyLoader("scipy.integrate")

OP_NAME = "document_minhash_deduplicator"

MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)


def sha1_hash32(data):
    """
    Directly taken from datasketch package to avoid dependency.

    Parameters
    ----------
    data : bytes

    Returns
    -------
    int
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from
    datasketch.

    :param threshold: float. The threshold for similarity
    :param num_perm: int. The number of permutations
    :param false_positive_weight: float. The weight of false positive
    :param false_negative_weight: float. The weight of false negative
    :return: Tuple[int, int]. The optimal `b` and `r` parameters. The number of
        bands, and the number of rows per band respectively
    """

    def false_positive_probability(th: float, band: int, rows: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(rows)) ** float(band)

        a, _ = integrate.quad(proba, 0.0, th)
        return a

    def false_negative_probability(th: float, band: int, rows: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(rows)) ** float(band))

        a, _ = integrate.quad(proba, th, 1.0)
        return a

    # object: minimize the weighted FP and FN ratio
    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_probability(threshold, b, r)
            fn = false_negative_probability(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


@OPERATORS.register_module(OP_NAME)
class DocumentMinhashDeduplicator(Deduplicator):
    """Deduplicates samples at the document level using MinHash LSH.

    This operator computes MinHash values for each sample and uses Locality-Sensitive
    Hashing (LSH) to identify and remove near-duplicate documents. The Jaccard similarity
    threshold determines when two documents are considered duplicates. The tokenization
    method can be customized, and a Hugging Face tokenizer can be used for 'sentencepiece'
    tokenization. The minhash values are stored as bytes and are not kept in the final
    dataset. The number of bands and rows per band in LSH can be set manually or determined
    by an optimal parameter computation algorithm. Important notes:
    - If using 'punctuation' tokenization with an ignore pattern, ensure the pattern does
      not include punctuations.
    - For 'sentencepiece' tokenization, a tokenizer model path is required.
    - The deduplication process involves clustering and filtering, and only unique samples
      or the first sample in a cluster are retained."""

    def __init__(
        self,
        tokenization: str = "space",
        window_size: PositiveInt = 5,
        lowercase: bool = True,
        ignore_pattern: Optional[str] = None,
        num_permutations: PositiveInt = 256,
        jaccard_threshold: Annotated[float, Field(ge=0, le=1)] = 0.7,
        num_bands: Optional[PositiveInt] = None,
        num_rows_per_band: Optional[PositiveInt] = None,
        tokenizer_model: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param tokenization: tokenization method for sample texts. It
            should be one of [space, punctuation, character,
            sentencepiece]. For English-like languages, we recommend
            to use 'space', for Chinese-like languages, we recommend
            to use 'character', and for multiple languages, we recommend
            to use 'sentencepiece'. If using 'sentencepiece', please
            provided the model path in the 'tokenizer_model' field.
        :param window_size: window size of shingling
        :param lowercase: whether to convert text to lower case first
        :param ignore_pattern: whether to ignore sub-strings with
            specific pattern when computing minhash
        :param num_permutations: number of permutations in minhash
            computing
        :param jaccard_threshold: the min jaccard similarity threshold
            in near-duplicate detection. When the jaccard similarity of
            two sample texts is >= this threshold, they are regarded as
            similar samples and this op will only keep one of them after
            deduplication
        :param num_bands: number of bands in LSH. Default it's None, and
            it will be determined by an optimal params computation
            algorithm by minimize the weighted sum of probs of False
            Positives and False Negatives
        :param num_rows_per_band: number of rows in each band in LSH.
            Default it's None, and it will be determined by an optimal
            params computation algorithm
        :param tokenizer_model: path for the sentencepiece model, used for
            sentencepiece tokenization.
        """
        super().__init__(*args, **kwargs)
        # about minhash computation
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

        if self.tokenization == "sentencepiece":
            if tokenizer_model is None:
                raise ValueError("To use 'sentencepiece' tokenization, " "'tokenizer_model' is required.")
            self.tokenizer = prepare_sentencepiece_model(tokenizer_model)
        else:
            self.tokenizer = None

        # about deduplication
        self.num_permutation = num_permutations
        self.jaccard_threshold = jaccard_threshold
        self.num_bands = num_bands
        self.num_rows_per_band = num_rows_per_band

        # initialize deduplication parameters
        # check number of bands and rows
        if self.num_bands is None or self.num_rows_per_band is None:
            self.num_bands, self.num_rows_per_band = optimal_param(
                self.jaccard_threshold,
                self.num_permutation,
            )

        # compute hash ranges and create hash tables
        self.hash_ranges = [
            (i * self.num_rows_per_band, (i + 1) * self.num_rows_per_band) for i in range(self.num_bands)
        ]
        self.hash_tables = [defaultdict(set) for _ in range(self.num_bands)]

        # generate permutations
        gen = np.random.RandomState(seed=42)
        self.perm_a, self.perm_b = np.array(
            [
                (
                    gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                    gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                )
                for _ in range(self.num_permutation)
            ],
            dtype=np.uint64,
        ).T

    def compute_hash(self, sample):
        """
        Compute minhash values for the sample.

        :param sample: input sample
        :return: sample with minhash value.
        """
        # check if it's computed already
        if HashKeys.minhash in sample:
            return sample

        text = sample[self.text_key]

        if self.lowercase:
            text = text.lower()
        if self.ignore_pattern:
            text = self.ignore_pattern.sub("", text)

        # get tokens for different tokenization method
        tokens = set()
        if self.tokenization == "character":
            tokens = {str.encode(text[i : i + self.window_size]) for i in range(len(text) - self.window_size + 1)}
        elif self.tokenization == "punctuation":
            tokens = self.punctuation_pattern.split(text)
            tokens = {
                str.encode(" ".join(tokens[i : i + self.window_size]))
                for i in range(len(tokens) - self.window_size + 1)
            }
        elif self.tokenization == "space":
            tokens = split_on_whitespace(text)
            tokens = {
                str.encode(" ".join(tokens[i : i + self.window_size]))
                for i in range(len(tokens) - self.window_size + 1)
            }
        elif self.tokenization == "sentencepiece":
            tokens = self.tokenizer.encode(text, out_type=str)
            tokens = {
                str.encode("".join(tokens[i : i + self.window_size])) for i in range(len(tokens) - self.window_size + 1)
            }
        else:
            raise NotImplementedError(f"Unimplemented tokenization method [{self.tokenization}]")

        # compute minhash value
        hv = np.fromiter((sha1_hash32(token) for token in tokens), dtype=np.uint64, count=len(tokens))
        phv = np.bitwise_and((hv[:, None] * self.perm_a + self.perm_b) % MERSENNE_PRIME, MAX_HASH)
        hash_values = phv.min(axis=0)
        sample[HashKeys.minhash] = [bytes(hash_values[start:end].byteswap().data) for start, end in self.hash_ranges]
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

        minhashes = dataset[HashKeys.minhash]
        # remove bytes minhash column otherwise unexpected error would occur
        # when exporting the processed dataset
        dataset = dataset.remove_columns([HashKeys.minhash])

        # make clusters -- construct the minhash lookup tables of seg to ids
        logger.info(f"Start clustering for {len(dataset)} samples...")
        batch_size = 10000
        for i in tqdm(
            range(0, len(minhashes), batch_size), dynamic_ncols=True, desc="Iterating MinHashes of samples..."
        ):
            batch = minhashes[i : i + batch_size]
            for idx, hs in enumerate(batch):
                for h, hashtable in zip(hs, self.hash_tables):
                    hashtable[h].add(idx + i)

        # using UnionFind set to union samples within the same clusters
        union_find = UnionFind()
        for table in tqdm(self.hash_tables, dynamic_ncols=True, desc="Clustering"):
            for cluster in table.values():
                if len(cluster) <= 1:
                    continue
                idx = min(cluster)
                for x in cluster:
                    union_find.union(x, idx)
        logger.info(
            f"There are {len(set(union_find.parent.values()))} "
            f"clusters that includes multiple near-duplicate samples."
        )

        # record the duplicate sample pairs
        dup_pairs = {}
        if show_num > 0:
            for i in range(len(dataset)):
                cluster_idx = union_find.find(i)
                if cluster_idx != i:
                    if cluster_idx not in dup_pairs:
                        dup_pairs[cluster_idx] = [
                            dataset[cluster_idx],
                            dataset[i],
                        ]
                    else:
                        dup_pairs[cluster_idx].append(dataset[i])
                if len(dup_pairs) >= show_num:
                    break

        # filtering -- only keep those samples whose parent index is itself,
        # including:
        # 1. samples that form a cluster by themselves
        # 2. the first sample in a cluster that includes multiple samples
        def _filter_minhash_dup_helper(sample, index):
            return union_find.find(index) == index

        dataset = dataset.filter(
            _filter_minhash_dup_helper,
            with_indices=True,
        )
        logger.info(f"Keep {len(dataset)} samples after MinHash dedup.")

        return dataset, dup_pairs


@OPERATORS.register_module(f"{OP_NAME}_with_uid")
class DocumentMinhashDeduplicatorWithUid(DocumentMinhashDeduplicator):
    """
    A Deduplicator that performs document-level deduplication using MinHashLSH.

    Unlike `DocumentMinhashDeduplicator`, this class requires the dataset to include an additional column named
    '__dj__uid' of type int, with unique values for each sample. This column is essential for supporting
    incremental deduplication scenarios.

    For example, consider a scenario where you have an already deduplicated dataset A and a new dataset B that
    you wish to add. If you want to perform joint deduplication on both A and B while prioritizing the retention of
    data from A, you can ensure that all '__dj__uid' values in B are greater than those in A. Then, by applying
    this deduplicator to the combined dataset, duplicates will be resolved in favor of the entries from A.
    """

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

        minhashes = dataset[HashKeys.minhash]
        # remove bytes minhash column otherwise unexpected error would occur
        # when exporting the processed dataset
        dataset = dataset.remove_columns([HashKeys.minhash])
        uids = dataset[HashKeys.uid]
        uid2idx = {uid: idx for idx, uid in enumerate(uids)}

        # make clusters -- construct the minhash lookup tables of seg to ids
        logger.info(f"Start clustering for {len(dataset)} samples...")
        batch_size = 10000
        for i in tqdm(
            range(0, len(minhashes), batch_size), dynamic_ncols=True, desc="Iterating MinHashes of samples..."
        ):
            batch = minhashes[i : i + batch_size]
            batch_uid = uids[i : i + batch_size]
            for uid, hs in zip(batch_uid, batch):
                for h, hashtable in zip(hs, self.hash_tables):
                    hashtable[h].add(uid)

        # using UnionFind set to union samples within the same clusters
        union_find = UnionFind()
        for table in tqdm(self.hash_tables, dynamic_ncols=True, desc="Clustering"):
            for cluster in table.values():
                if len(cluster) <= 1:
                    continue
                uid = min(cluster)
                for x in cluster:
                    union_find.union(x, uid)
        logger.info(
            f"There are {len(set(union_find.parent.values()))} "
            f"clusters that includes multiple near-duplicate samples."
        )

        # record the duplicate sample pairs
        dup_pairs = {}
        if show_num > 0:
            for i in range(len(dataset)):
                uid = uids[i]
                cluster_uid = union_find.find(uid)
                cluster_idx = uid2idx[cluster_uid]
                if cluster_uid != uid and cluster_idx not in dup_pairs:
                    dup_pairs[cluster_idx] = [
                        dataset[cluster_idx],
                        dataset[i],
                    ]
                if len(dup_pairs) >= show_num:
                    break

        # filtering -- only keep those samples whose parent index is itself,
        # including:
        # 1. samples that form a cluster by themselves
        # 2. the first sample in a cluster that includes multiple samples
        def _filter_minhash_dup_helper(sample):
            uid = sample[HashKeys.uid]
            return union_find.find(uid) == uid

        dataset = dataset.filter(_filter_minhash_dup_helper)
        logger.info(f"Keep {len(dataset)} samples after MinHash dedup.")

        return dataset, dup_pairs
