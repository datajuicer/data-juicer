from abc import ABC, abstractmethod
from typing import Union

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
    Uses lazy initialization to defer actor creation until first use,
    allowing the cluster to autoscale before actors consume resources.
    """

    def __init__(self, dedup_set_num: Union[int, str], RemoteDedupSet=None):
        # Store config but don't create actors yet
        # dedup_set_num can be int or "auto"
        self._dedup_set_num_config = dedup_set_num
        self._RemoteDedupSet = RemoteDedupSet
        self._dedup_sets = None  # Lazy - created on first use
        self._actual_dedup_set_num = None

    @property
    def dedup_set_num(self):
        """Get actual dedup_set_num, calculating from cluster resources if 'auto'."""
        if self._actual_dedup_set_num is None:
            if self._dedup_set_num_config == "auto":
                self._actual_dedup_set_num = max(1, int(ray.cluster_resources().get("CPU", 1) / 2))
            else:
                self._actual_dedup_set_num = int(self._dedup_set_num_config)
        return self._actual_dedup_set_num

    def _ensure_actors(self):
        """Create actors on first use when cluster has scaled."""
        if self._dedup_sets is None:
            RemoteDedupSet = self._RemoteDedupSet or get_remote_dedup_set()
            self._dedup_sets = [RemoteDedupSet.remote() for _ in range(self.dedup_set_num)]

    def is_unique(self, md5_value: str):
        self._ensure_actors()
        dedup_set_id = int.from_bytes(md5_value.encode(), byteorder="little") % MERSENNE_PRIME % self.dedup_set_num
        return ray.get(self._dedup_sets[dedup_set_id].is_unique.remote(md5_value))


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

    def __init__(
        self,
        backend: str = "ray_actor",
        redis_address: str = "redis://localhost:6379",
        dedup_set_num: Union[int, str] = "auto",
        *args,
        **kwargs,
    ):
        """
        Initialization.
        :param backend: the backend for dedup, either 'ray_actor' or 'redis'
        :param redis_address: the address of redis server
        :param dedup_set_num: number of dedup set actors, or 'auto' to use CPU/2
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.redis_address = redis_address
        self.backend = backend
        if backend == "ray_actor":
            # Pass dedup_set_num directly - ActorBackend handles "auto" lazily
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
