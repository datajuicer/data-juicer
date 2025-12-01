import hashlib
from collections import defaultdict
from typing import Dict, Set, Tuple

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.mm_utils import close_video, load_data_with_context, load_video

from ..base_op import OPERATORS, Deduplicator
from ..op_fusion import LOADED_VIDEOS
from .document_deduplicator import DocumentDeduplicator

OP_NAME = "video_deduplicator"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoDeduplicator(Deduplicator):
    """Deduplicates samples at the document level using exact matching of videos.

    This operator computes a hash for each video in the sample and uses it to identify and
    remove duplicate documents. If `consider_text` is set to True, it also considers the
    text hash alongside the video hash for deduplication. The video hash is computed by
    hashing the video data, including all video streams in the container. The operator
    supports sampling and tracing of duplicate pairs when the `show_num` parameter is
    greater than 0. Important fields used for caching include 'videohash' and optionally
    'hash' if text is considered."""

    def __init__(self, consider_text: bool = False, *args, **kwargs):
        """
        Initialization.

        :param consider_text: whether to consider text hash together with video
            hash when applying deduplication.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.consider_text = consider_text
        self.text_dedup_op = None
        if self.consider_text:
            self.text_dedup_op = DocumentDeduplicator(**kwargs)

    def compute_hash(self, sample, context=False):
        # get hash of text first
        if self.consider_text:
            sample = self.text_dedup_op.compute_hash(sample)
        # check if it's computed already
        if HashKeys.videohash in sample:
            return sample

        # there is no video in this sample
        sample[HashKeys.videohash] = ""
        if self.video_key not in sample or not sample[self.video_key]:
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context, loaded_video_keys, load_video)

        # compute hash
        md5_hash = hashlib.md5()
        for key in videos:
            # consider the multi stream of video in one container
            for packet in videos[key].demux():
                if packet.stream.type == "video":
                    md5_hash.update(bytes(packet))

        for key in videos:
            close_video(videos[key])

        sample[HashKeys.videohash] = md5_hash.hexdigest()
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

        dup_hashes = None
        if show_num > 0:
            # sample duplicate pairs
            if self.consider_text:
                hash2ids: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
                hashes = zip(dataset[HashKeys.videohash], dataset[HashKeys.hash])
            else:
                hash2ids: Dict[int, Set[int]] = defaultdict(set)
                hashes = dataset[HashKeys.videohash]
            for sid, hash_val in enumerate(hashes):
                if hash_val:
                    hash2ids[hash_val].add(sid)
            dup_samples = sorted(list(hash2ids.items()), key=lambda x: len(x[1]), reverse=True)
            dup_hashes = set([item[0] for item in dup_samples if len(item[1]) > 1][:show_num])

        def _filter_dup_helper(sample, hashes):
            if self.consider_text:
                hash = (sample[HashKeys.videohash], sample[HashKeys.hash])
            else:
                hash = sample[HashKeys.videohash]
            if not hash:
                return True
            if show_num > 0 and hash in dup_hashes and len(dup_pairs[hash]) < 2:
                # tracer is open and not enough duplicate sample pairs
                dup_pairs[hash].append(sample)
            if hash in hashes:
                return False
            else:
                hashes.add(hash)
                return True

        hashes = set()
        dup_pairs = {hash_v: [] for hash_v in dup_hashes} if dup_hashes else {}
        dataset = dataset.filter(
            _filter_dup_helper, fn_kwargs=dict(hashes=hashes), load_from_cache_file=False if show_num > 0 else True
        )  # num_proc=1
        return dataset, dup_pairs
