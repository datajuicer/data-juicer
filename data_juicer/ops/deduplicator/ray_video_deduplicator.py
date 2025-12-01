import hashlib

from data_juicer.utils.mm_utils import close_video, load_data_with_context, load_video

from ..base_op import OPERATORS
from ..op_fusion import LOADED_VIDEOS
from .ray_basic_deduplicator import RayBasicDeduplicator

OP_NAME = "ray_video_deduplicator"


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class RayVideoDeduplicator(RayBasicDeduplicator):
    """Deduplicates samples at document-level using exact matching of videos in Ray distributed mode.

    This operator computes the MD5 hash of video streams in each sample and compares them to
    identify duplicates. It uses Ray distributed mode for parallel processing. The hash is
    computed by demuxing the video streams and updating the MD5 hash with each video packet.
    If a sample does not contain a valid video, it is assigned an empty hash value. The
    operator supports 'ray_actor' or 'redis' backends for deduplication."""

    def __init__(self, backend: str = "ray_actor", redis_address: str = "redis://localhost:6379", *args, **kwargs):
        """
        Initialization.
        :param backend: the backend for dedup, either 'ray_actor' or 'redis'
        :param redis_address: the address of redis server
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(backend=backend, redis_address=redis_address, *args, **kwargs)

    def calculate_hash(self, sample, context=False):
        if self.video_key not in sample or not sample[self.video_key]:
            return RayBasicDeduplicator.EMPTY_HASH_VALUE

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

        return md5_hash.hexdigest()
