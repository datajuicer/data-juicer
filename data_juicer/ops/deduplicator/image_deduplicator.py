from collections import defaultdict
from typing import Dict, Set, Tuple

import numpy as np

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image

from ..base_op import OPERATORS, Deduplicator
from ..op_fusion import LOADED_IMAGES
from .document_deduplicator import DocumentDeduplicator

imgdedup_methods = LazyLoader("imagededup.methods")

OP_NAME = "image_deduplicator"

HASH_METHOD = {"phash", "dhash", "whash", "ahash"}


def get_hash_method(method_name):
    mapping = {
        "phash": imgdedup_methods.PHash,
        "dhash": imgdedup_methods.DHash,
        "whash": imgdedup_methods.WHash,
        "ahash": imgdedup_methods.AHash,
    }

    return mapping[method_name]


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageDeduplicator(Deduplicator):
    """Deduplicates samples at the document level by exact matching of images.

    This operator compares images across documents to identify and remove duplicates.
    - It uses a specified hash method (default is 'phash') to compute image hashes.
    - If `consider_text` is set, it also considers text content for deduplication,
    using a text deduplicator in conjunction with the image hashes.
    - The key metric, `imagehash`, is computed for each sample. If `consider_text`
    is enabled, an additional `hash` field is used.
    - Duplicates are identified by comparing these hash values. Samples with
    identical hashes are considered duplicates.
    - When `show_num` is greater than 0, the operator also returns a subset of
    duplicate pairs for tracing purposes.
    - The operator caches the `imagehash` and, if applicable, the `hash` fields."""

    def __init__(
        self,
        method: str = "phash",
        consider_text: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param method: hash method for image
        :param consider_text: whether to consider text hash together with image
            hash when applying deduplication.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if method not in HASH_METHOD:
            raise ValueError(f"Keep strategy [{method}] is not supported. Can only be one of {HASH_METHOD}.")
        self.hasher = get_hash_method(method)()
        self.consider_text = consider_text
        self.text_dedup_op = None
        if self.consider_text:
            self.text_dedup_op = DocumentDeduplicator(**kwargs)

    def compute_hash(self, sample, context=False):
        # get hash of text first
        if self.consider_text:
            sample = self.text_dedup_op.compute_hash(sample)
        # check if it's computed already
        if HashKeys.imagehash in sample:
            return sample

        # there is no image in this sample
        sample[HashKeys.imagehash] = ""
        if self.image_key not in sample or not sample[self.image_key]:
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample,
            context,
            loaded_image_keys,
            load_image,
            mm_bytes_key=self.image_bytes_key,
        )

        # compute hash
        for key in images:
            sample[HashKeys.imagehash] += self.hasher.encode_image(image_array=np.array(images[key]))
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

        # keep the first sample of each hash, and every sample without one;
        # stream over the dataset instead of materializing the entire hash
        # column into a Python list first. Peak memory drops from O(n)
        # (full hash column) to O(unique_hashes) (only the set of seen hashes).
        hashes = set()
        keep = []
        dup_pairs = {}
        dup_hashes = None

        # First pass: if tracer is on, find the most duplicated hashes
        if show_num > 0:
            hash2ids = defaultdict(int)
            for sample in dataset:
                if self.consider_text:
                    h = (sample[HashKeys.imagehash], sample[HashKeys.hash])
                else:
                    h = sample[HashKeys.imagehash]
                if h:
                    hash2ids[h] += 1
            dup_samples = sorted(hash2ids.items(), key=lambda x: x[1], reverse=True)
            dup_hashes = set(
                [item[0] for item in dup_samples if item[1] > 1][:show_num]
            )
            dup_pairs = {hash_v: [] for hash_v in dup_hashes}

        # Second pass: collect keep indices and duplicate pairs
        for idx, sample in enumerate(dataset):
            if self.consider_text:
                h = (sample[HashKeys.imagehash], sample[HashKeys.hash])
            else:
                h = sample[HashKeys.imagehash]
            if not h:
                keep.append(idx)
                continue
            if show_num > 0 and h in dup_hashes and len(dup_pairs[h]) < 2:
                # tracer is open and not enough duplicate sample pairs
                dup_pairs[h].append(sample)
            if h not in hashes:
                hashes.add(h)
                keep.append(idx)
        dataset = dataset.select(keep)
        return dataset, dup_pairs
