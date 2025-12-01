from abc import ABC, abstractmethod

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader

from ..base_op import Filter

ray = LazyLoader("ray")
redis = LazyLoader("redis")

MERSENNE_PRIME = (1 << 61) - 1


class DedupSet:
    def __init__(self):
        self.hash_record = set()

    def is_unique(self, key):
        if key not in self.hash_record:
            self.hash_record.add(key)
            return True
        else:
            return False


def get_remote_dedup_set():
    """Get the remote version of DedupSet with Ray decorator applied at runtime."""
    return ray.remote(scheduling_strategy="SPREAD")(DedupSet)


class Backend(ABC):
    """
    Backend for deduplicator.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def is_unique(self, md5_value: str):
        pass


class ActorBackend(Backend):
    """
    Ray actor backend for deduplicator.
    """

    def __init__(self, dedup_set_num: int, RemoteDedupSet=None):
        self.dedup_set_num = dedup_set_num
        if RemoteDedupSet is None:
            RemoteDedupSet = get_remote_dedup_set()
        self.dedup_sets = [RemoteDedupSet.remote() for _ in range(self.dedup_set_num)]

    def is_unique(self, md5_value: str):
        dedup_set_id = int.from_bytes(md5_value.encode(), byteorder="little") % MERSENNE_PRIME % self.dedup_set_num
        return ray.get(self.dedup_sets[dedup_set_id].is_unique.remote(md5_value))


class RedisBackend(Backend):
    """
    Redis backend for deduplicator.
    """

    def __init__(self, redis_address: str):
        self.redis_address = redis_address
        self.redis_client = redis.from_url(url=self.redis_address)
        self.redis_client.flushdb(0)

    def is_unique(self, md5_value: str):
        return self.redis_client.setnx(md5_value, 1)


class RayBasicDeduplicator(Filter):
    """
    A basic exact matching deduplicator for RAY.
    Although its functionality is deduplication,
    it is implemented as Filter sub-class.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = "EMPTY"

    def __init__(self, backend: str = "ray_actor", redis_address: str = "redis://localhost:6379", *args, **kwargs):
        """
        Initialization.
        :param backend: the backend for dedup, either 'ray_actor' or 'redis'
        :param redis_address: the address of redis server
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.redis_address = redis_address
        self.backend = backend
        if backend == "ray_actor":
            dedup_set_num = int(ray.cluster_resources().get("CPU") / 2)
            self.backend = ActorBackend(dedup_set_num)
        elif backend == "redis":
            # TODO: add a barrier to ensure that flushdb is performed before
            # the operator is called
            self.backend = RedisBackend(redis_address)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def calculate_hash(self, sample, context=False):
        """Calculate hash value for the sample."""
        raise NotImplementedError

    def compute_stats_single(self, sample, context=False):
        # compute hash
        md5_value = self.calculate_hash(sample, context)
        # check existing
        sample[HashKeys.is_unique] = self.backend.is_unique(md5_value)
        return sample

    def process_single(self, sample):
        return sample[HashKeys.is_unique]
