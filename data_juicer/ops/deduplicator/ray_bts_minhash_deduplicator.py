import os
import time
from typing import List, Optional, Union

import numpy as np
import pyarrow as pa
import regex
from loguru import logger
from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import prepare_sentencepiece_model
from data_juicer.utils.ray_utils import ray_available_gpu_memories, ray_gpu_count

from ..base_op import OPERATORS, Deduplicator
from ..common.helper_func import split_on_whitespace
from .document_minhash_deduplicator import (
    MAX_HASH,
    MERSENNE_PRIME,
    optimal_param,
    sha1_hash32,
)

ray = LazyLoader("ray")

BATCH_SIZE = 1000


class IdGenerator:

    def __init__(self, start_id=0):
        self.next_id = start_id

    def get_next_id(self, count):
        current_id = self.next_id
        self.next_id += count
        return (current_id, self.next_id)


class EdgeBuffer:

    def __init__(self):
        self.edge_dict = {}

    def clear(self):
        self.edge_dict = {}

    def set_edges(self, edge_dict):
        self.edge_dict = edge_dict

    def get_edges(self, key):
        return self.edge_dict.pop(key, [])


class BTSUnionFind:
    """
    A distributed implementation of Union-Find with load balancing.

    The original paper on BTS Union-Find is available at:
    https://ieeexplore.ieee.org/document/10598116
    """

    def __init__(
        self,
        union_threshold,
        parallel_num,
        parallel_id,
        remote_edge_buffers,
        max_pending_edge_buffer_task,
        num_edge_buffer_task_returns,
    ):
        self.union_threshold = union_threshold
        self.parallel_num = parallel_num
        self.parallel_id = parallel_id
        self.hash_table = {}
        self.parent = {}
        self.old_parent = {}
        self.remote_edge_buffers = remote_edge_buffers
        self.edge_buffer = []
        self.edge_list_dict = {}
        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns

    def add_key_value_pairs(self, pairs):
        for key, value in pairs:
            if key not in self.hash_table:
                self.hash_table[key] = []
            self.hash_table[key].append(value)
            if len(self.hash_table[key]) > self.union_threshold:
                self.hash_table[key] = [self.union_list(self.hash_table[key])]

    def flush_key_value_pairs(self):
        for value in self.hash_table.values():
            if len(value) > 1:
                self.union_list(value)
        del self.hash_table

    def balanced_union_find(self):
        for x, y in self.edge_buffer:
            self.union(x, y)
        self.edge_buffer = []
        result_refs = []
        for remote_edge_buffer in self.remote_edge_buffers:
            if len(result_refs) > self.max_pending_edge_buffer_task:
                ready_refs, result_refs = ray.wait(result_refs, num_returns=self.num_edge_buffer_task_returns)
                edge_list = ray.get(ready_refs)
                for edges in edge_list:
                    for x, y in edges:
                        self.union(x, y)
                del ready_refs
            result_refs.append(remote_edge_buffer.get_edges.remote(self.parallel_id))
        edge_list = ray.get(result_refs)
        for edges in edge_list:
            for x, y in edges:
                self.union(x, y)
        del edge_list, result_refs
        self.rebalancing()
        return self.old_parent != self.parent

    def distribute_edge(self, u, v):
        hash_u = u // BATCH_SIZE % self.parallel_num
        hash_v = v // BATCH_SIZE % self.parallel_num
        if hash_u not in self.edge_list_dict:
            self.edge_list_dict[hash_u] = []
        self.edge_list_dict[hash_u].append((u, v))
        if hash_u != hash_v:
            if hash_v not in self.edge_list_dict:
                self.edge_list_dict[hash_v] = []
            self.edge_list_dict[hash_v].append((u, v))

    def set_edge_buffer(self):
        if self.parallel_id in self.edge_list_dict:
            self.edge_buffer = self.edge_list_dict[self.parallel_id]
            del self.edge_list_dict[self.parallel_id]
        else:
            self.edge_buffer = []
        ray.get(self.remote_edge_buffers[self.parallel_id].set_edges.remote(self.edge_list_dict))
        self.edge_list_dict = {}

    def edge_redistribution(self):
        self.flush_key_value_pairs()
        self.rebalancing()
        self.edge_list_dict = {}
        for u, v in self.parent.items():
            self.distribute_edge(u, v)
        self.parent = {}
        self.set_edge_buffer()

    def communication(self):
        self.edge_list_dict = {}
        del_list = []
        for u, v in self.parent.items():
            hash_u = u // BATCH_SIZE % self.parallel_num
            if self.parent[u] != self.old_parent.get(u, u) or (hash_u != self.parallel_id and v not in self.parent):
                self.distribute_edge(u, v)
            if hash_u != self.parallel_id:
                del_list.append(u)
        self.old_parent = self.parent.copy()
        for u in del_list:
            del self.parent[u]
        self.set_edge_buffer()

    def find(self, x):
        if x not in self.parent:
            return x
        else:
            self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)
        if px == py:
            return
        if px > py:
            px, py = py, px
        self.parent[py] = px

    def union_list(self, x_list):
        px_list = [self.find(x) for x in x_list]
        p = min(px_list)
        for px in px_list:
            if p != px:
                self.parent[px] = p
        return p

    def rebalancing(self):
        new_px_dict = {}
        for x in self.parent:
            hash_x = x // BATCH_SIZE % self.parallel_num
            px = self.find(x)
            key = (px, hash_x)
            if key not in new_px_dict:
                new_px_dict[key] = x
            else:
                new_px_dict[key] = min(new_px_dict[key], x)
        px_set = set(px for px, _ in new_px_dict)
        for px in px_set:
            hash_px = px // BATCH_SIZE % self.parallel_num
            key = (px, hash_px)
            if key not in new_px_dict:
                new_px_dict[key] = px
            else:
                new_px_dict[key] = min(new_px_dict[key], px)

        for x in self.parent:
            hash_x = x // BATCH_SIZE % self.parallel_num
            px = self.find(x)
            key = (px, hash_x)
            if x == new_px_dict[key]:
                continue
            self.parent[x] = new_px_dict[key]

    def squeeze(self):
        dup_keys = {x for x in self.parent if x // BATCH_SIZE % self.parallel_num == self.parallel_id}
        self.parent = dup_keys
        self.old_parent = {}
        self.edge_buffer = []
        ray.get(self.remote_edge_buffers[self.parallel_id].clear.remote())

    def dup_idx(self, queries):
        return [idx for uid, idx in queries if uid in self.parent]


def get_remote_classes():
    """Get remote versions of classes with Ray decorators applied at runtime."""
    # Apply ray.method decorator to get_next_id at runtime
    IdGenerator.get_next_id = ray.method(num_returns=2)(IdGenerator.get_next_id)

    return {
        "IdGenerator": ray.remote(IdGenerator),
        "EdgeBuffer": ray.remote(scheduling_strategy="SPREAD")(EdgeBuffer),
        "BTSUnionFind": ray.remote(scheduling_strategy="SPREAD")(BTSUnionFind),
    }


OP_NAME = "ray_bts_minhash_deduplicator"


class GPUMinHashActor:

    def __init__(self, width: int = 5, perm_a: np.ndarray = None, perm_b: np.ndarray = None, lowercase: bool = True):
        import cudf
        import rmm

        rmm.reinitialize(pool_allocator=True)
        self.width = width
        gen = np.random.RandomState(seed=42)
        if perm_a is None or perm_b is None:
            perm_a, perm_b = np.array(
                [
                    (
                        gen.randint(1, MERSENNE_PRIME, dtype=np.uint64),
                        gen.randint(0, MERSENNE_PRIME, dtype=np.uint64),
                    )
                    for _ in range(256)
                ],
                dtype=np.uint64,
            ).T
        self.perm_a = cudf.Series(perm_a).astype("uint32")
        self.perm_b = cudf.Series(perm_b).astype("uint32")
        self.lowercase = lowercase

    def compute_minhash(self, text_arr: pa.Array) -> pa.Array:
        """
        Compute MinHash signatures for texts in a table
        """
        import cudf

        text_df = cudf.Series.from_arrow(text_arr)
        if self.lowercase:
            text_df = text_df.str.lower()
        minhashes = text_df.str.minhash(seed=0, a=self.perm_a, b=self.perm_b, width=self.width)
        del text_df
        arrow_minhashes = minhashes.to_arrow()
        del minhashes
        return arrow_minhashes

    def __call__(self, table: pa.Table, text_key: str = "text") -> pa.Table:
        minhashes = self.compute_minhash(table[text_key])
        new_table = table.append_column("_minhash", minhashes)
        return new_table


@OPERATORS.register_module(OP_NAME)
class RayBTSMinhashDeduplicator(Deduplicator):
    """A MinhashLSH deduplicator that operates in Ray distributed mode.

    This operator uses the MinHash LSH technique to identify and remove near-duplicate
    samples from a dataset. It supports various tokenization methods, including space,
    punctuation, character, and sentencepiece. The Jaccard similarity threshold is used to
    determine if two samples are considered duplicates. If the Jaccard similarity of two
    samples is greater than or equal to the specified threshold, one of the samples is
    filtered out. The operator computes the MinHash values for each sample and uses a union-
    find algorithm to group similar samples. The key metric, Jaccard similarity, is computed
    based on the shingling of the text. The operator can run on both CPU and GPU, with
    specific batch size and memory configurations for each."""

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = "EMPTY"
    _batched_op = True

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
        union_find_parallel_num: Union[int, str] = "auto",
        union_threshold: Optional[int] = 256,
        max_pending_edge_buffer_task: Optional[int] = 20,
        num_edge_buffer_task_returns: Optional[int] = 10,
        max_pending_filter_tasks: Optional[int] = 20,
        num_filter_task_returns: Optional[int] = 10,
        merge_batch_size: Optional[int] = 1000,
        minhash_batch_size: Optional[Union[int, str]] = "auto",
        memory_per_sample: Optional[float] = 0.1,  # MB per sample
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
        :param union_find_parallel_num: number of parallel workers for
            union-find algorithm. Default it's 'auto', and it will be
            determined by half of the number of CPUs.
        :param union_threshold: threshold for minhash values group to
            perform union-find algorithm. Default it's 256.
        :param max_pending_edge_buffer_task: max number of pending edge buffer
            ray tasks. Default it's 20.
        :param num_edge_buffer_task_returns: number of edge buffer tasks for
            `ray.wait` to return. Default it's 10.
        :param max_pending_filter_tasks: max number of pending filter ray
            tasks. Default it's 20.
        :param num_filter_task_returns: number of filter tasks for `ray.wait`
            to return. Default it's 10.
        :param merge_batch_size: batch size for BTS operations. Default
            it's 1000.
        :param minhash_batch_size: batch size for MinHash computation. If "auto",
            it will be set to default value on CPU(1024), or auto calculated per
            available GPU memory and memory_per_sample setting for GPU.
        :param memory_per_sample: estimated memory needed per sample in MB.
            Used to calculate batch size based on available GPU memory.
            Default is 0.1 MB per sample.
        """

        super().__init__(*args, **kwargs)

        self.tokenization = tokenization
        self.window_size = window_size
        self.lowercase = lowercase
        self.ignore_pattern = ignore_pattern
        self.memory_per_sample = memory_per_sample
        if minhash_batch_size == "auto":
            if self.use_cuda():
                self.minhash_batch_size = 200_000
            else:
                self.minhash_batch_size = 1024
        else:
            self.minhash_batch_size = minhash_batch_size
        if self.ignore_pattern:
            self.ignore_pattern = regex.compile(self.ignore_pattern)
        if self.use_cuda() and self.tokenization != "character":
            raise ValueError("GPU MinHash computation is only supported for character tokenization")

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

        if self.tokenization == "character":

            def tokenization_func(text):
                return {str.encode(text[i : i + self.window_size]) for i in range(len(text) - self.window_size + 1)}

        elif self.tokenization == "punctuation":

            def tokenization_func(text):
                tokens = self.punctuation_pattern.split(text)
                return {
                    str.encode(" ".join(tokens[i : i + self.window_size]))
                    for i in range(len(tokens) - self.window_size + 1)
                }

        elif self.tokenization == "space":

            def tokenization_func(text):
                tokens = split_on_whitespace(text)
                return {
                    str.encode(" ".join(tokens[i : i + self.window_size]))
                    for i in range(len(tokens) - self.window_size + 1)
                }

        elif self.tokenization == "sentencepiece":

            def tokenization_func(text):
                tokens = self.tokenizer.encode(text, out_type=str)
                return {
                    str.encode("".join(tokens[i : i + self.window_size]))
                    for i in range(len(tokens) - self.window_size + 1)
                }

        else:
            raise NotImplementedError(f"Unimplemented tokenization method [{self.tokenization}]")
        self.tokenization_func = tokenization_func

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

        # Store config for lazy initialization - don't create actors yet
        self._union_find_parallel_num_config = union_find_parallel_num
        self._merge_batch_size_config = merge_batch_size

        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns
        self.max_pending_filter_tasks = max_pending_filter_tasks
        self.num_filter_task_returns = num_filter_task_returns
        self.union_threshold = union_threshold

        # Lazy initialization - actors created in _ensure_actors()
        self._actors_initialized = False
        self.union_find_parallel_num = None
        self.merge_batch_size = None
        self.remote_edge_buffers = None
        self.union_find_list = None
        self.empty_hash_value = None
        self.empty_hash_table_id = None

    def _ensure_actors(self):
        """Create actors lazily on first use, when cluster has autoscaled."""
        if self._actors_initialized:
            return

        # Calculate union_find_parallel_num NOW when cluster has scaled
        if self._union_find_parallel_num_config == "auto":
            self.union_find_parallel_num = max(1, int(ray.cluster_resources().get("CPU", 1) / 2))
        else:
            self.union_find_parallel_num = int(self._union_find_parallel_num_config)

        self.merge_batch_size = min(self._merge_batch_size_config, self.union_find_parallel_num)

        logger.info(f"union_find_parallel_num = {self.union_find_parallel_num}")

        # Create actors NOW when cluster has resources
        remote_classes = get_remote_classes()
        self.remote_edge_buffers = [remote_classes["EdgeBuffer"].remote() for _ in range(self.union_find_parallel_num)]
        self.union_find_list = [
            remote_classes["BTSUnionFind"].remote(
                self.union_threshold,
                self.union_find_parallel_num,
                i,
                self.remote_edge_buffers,
                self.max_pending_edge_buffer_task,
                self.num_edge_buffer_task_returns,
            )
            for i in range(self.union_find_parallel_num)
        ]

        empty_hash_value = np.full((self.num_rows_per_band,), MAX_HASH, dtype=np.uint32)
        self.empty_hash_value = b"\x00\x00\x00\x00" + empty_hash_value.tobytes()
        self.empty_hash_table_id = int(MAX_HASH % self.union_find_parallel_num)

        self._actors_initialized = True

    def band_minhash(self, minhash_list, uid_list):
        """
        Logic for creating and pusing LSH bands to the union find list
        """
        pairs = {}
        minhash_list = minhash_list.to_numpy(zero_copy_only=False)
        for minhash, uid in zip(minhash_list, uid_list):
            for i, (start, end) in enumerate(self.hash_ranges):
                hash_value = i.to_bytes(4, "big") + minhash[start:end].tobytes()
                hash_table_id = minhash[start] % self.union_find_parallel_num
                if hash_table_id not in pairs:
                    pairs[hash_table_id] = []
                pairs[hash_table_id].append((hash_value, uid))
        result_refs = []
        for i, p in pairs.items():
            if len(result_refs) > self.max_pending_filter_tasks:
                ready_refs, result_refs = ray.wait(result_refs, num_returns=self.num_filter_task_returns)
                ray.get(ready_refs)
            result_refs.append(self.union_find_list[i].add_key_value_pairs.remote(p))
        ray.get(result_refs)

    def calc_minhash(self, text_list: pa.Array, uid_list: List) -> pa.Table:
        """
        Logic for computing minhash values for each text in the input table
        """
        pairs = {}
        for text, uid in zip(text_list, uid_list):
            text = text.as_py()
            if self.lowercase:
                text = text.lower()
            if self.ignore_pattern:
                text = self.ignore_pattern.sub("", text)

            tokens = self.tokenization_func(text)
            if len(tokens) > 0:
                hv = np.array([sha1_hash32(token) for token in tokens], dtype=np.uint64)
                phv = ((hv[:, None] * self.perm_a[None, :] + self.perm_b) % MERSENNE_PRIME).astype(np.uint32)
                hash_values = phv.min(axis=0)
                for i, (start, end) in enumerate(self.hash_ranges):
                    hash_value = i.to_bytes(4, "big") + hash_values[start:end].tobytes()
                    hash_table_id = hash_values[start] % self.union_find_parallel_num
                    if hash_table_id not in pairs:
                        pairs[hash_table_id] = []
                    pairs[hash_table_id].append((hash_value, uid))
            else:
                if self.empty_hash_table_id not in pairs:
                    pairs[self.empty_hash_table_id] = []
                pairs[self.empty_hash_table_id].append((self.empty_hash_value, uid))
        result_refs = []
        for i, p in pairs.items():
            if len(result_refs) > self.max_pending_filter_tasks:
                ready_refs, result_refs = ray.wait(result_refs, num_returns=self.num_filter_task_returns)
                ray.get(ready_refs)
            result_refs.append(self.union_find_list[i].add_key_value_pairs.remote(p))
        ray.get(result_refs)

    def merge_op_batch(self, object_refs):
        results = []
        while object_refs:
            ready_refs, object_refs = ray.wait(object_refs, num_returns=min(self.merge_batch_size, len(object_refs)))
            results.extend(ray.get(ready_refs))
        return results

    def merge(self):
        self.merge_op_batch([union_find.edge_redistribution.remote() for union_find in self.union_find_list])
        while any(
            self.merge_op_batch([union_find.balanced_union_find.remote() for union_find in self.union_find_list])
        ):
            self.merge_op_batch([union_find.communication.remote() for union_find in self.union_find_list])
        self.merge_op_batch([union_find.squeeze.remote() for union_find in self.union_find_list])

    def filter_with_union_find(self, samples: pa.Table) -> pa.Table:
        query_dict = {}
        for idx, uid in enumerate(samples[HashKeys.uid]):
            uid = uid.as_py()
            hash_id = uid // BATCH_SIZE % self.union_find_parallel_num
            if hash_id not in query_dict:
                query_dict[hash_id] = []
            query_dict[hash_id].append((uid, idx))
        mask = np.ones(len(samples), dtype=np.bool_)
        result_refs = []
        for hash_id, query in query_dict.items():
            if len(result_refs) > self.max_pending_filter_tasks:
                ready_refs, result_refs = ray.wait(result_refs, num_returns=self.num_filter_task_returns)
                results = ray.get(ready_refs)
                for result in results:
                    mask[result] = False
                del ready_refs
            result_refs.append(self.union_find_list[hash_id].dup_idx.remote(query))
        results = ray.get(result_refs)
        for result in results:
            mask[result] = False
        del query_dict, results
        columns_to_keep = [name for name in samples.column_names if name != HashKeys.uid]
        return samples.select(columns_to_keep).filter(mask)

    def run(self, dataset, **kwargs):
        # Ignore additional parameters like exporter, tracer, etc.
        # Initialize actors lazily - now cluster has had time to autoscale
        self._ensure_actors()

        start_time = time.time()
        # Get remote IdGenerator only when needed
        remote_classes = get_remote_classes()
        id_generator = remote_classes["IdGenerator"].remote()

        def band_with_uid(table: pa.Table) -> pa.Table:
            num_rows = len(table)
            min_id, max_id = ray.get(id_generator.get_next_id.remote(num_rows))
            uid_list = range(min_id, max_id)
            self.band_minhash(table["_minhash"], uid_list)
            new_table = table.append_column(HashKeys.uid, pa.array(list(uid_list)))
            new_table = new_table.drop_columns(["_minhash"])
            return new_table

        def minhash_with_uid(table: pa.Table) -> pa.Table:
            num_rows = len(table)
            min_id, max_id = ray.get(id_generator.get_next_id.remote(num_rows))
            uid_list = range(min_id, max_id)
            self.calc_minhash(table[self.text_key], uid_list)
            new_table = table.append_column(HashKeys.uid, pa.array(list(uid_list)))
            return new_table

        tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())
        if self.use_cuda():
            logger.info("Using GPU for MinHash computation")
            # Get available GPU count and set concurrency
            gpu_count = ray_gpu_count()
            if gpu_count == 0:
                logger.error("No GPUs available in Ray cluster")
                raise RuntimeError("No GPUs available in Ray cluster")

            concurrency = max(1, gpu_count)  # Ensure at least 1 concurrent task
            logger.info(f"Setting GPU concurrency to {concurrency} based on available GPUs")

            # Get available GPU memory and set batch size
            gpu_memory = ray_available_gpu_memories()
            if len(gpu_memory):
                min_memory = min(gpu_memory)
                # Use 80% of available memory to leave room for overhead
                safe_memory = min_memory * 0.8
                estimated_batch_size = int(safe_memory / self.memory_per_sample)

                # For very large GPUs, cap at 2M samples to maintain reasonable processing time
                # This is a soft cap - can be adjusted based on performance testing
                max_reasonable_batch = 2_000_000
                batch_size = max(10_000, min(estimated_batch_size, max_reasonable_batch))

                logger.info(
                    f"Setting batch size to {batch_size} based on available GPU memory "
                    f"({min_memory}MB), memory per sample ({self.memory_per_sample}MB), "
                    f"and safe memory limit ({safe_memory}MB)"
                )
            else:
                batch_size = self.minhash_batch_size
                logger.info(f"Using default batch size of {batch_size}")

            dataset = dataset.map_batches(
                GPUMinHashActor,
                batch_format="pyarrow",
                zero_copy_batch=True,
                num_gpus=1,
                concurrency=ray.data.ActorPoolStrategy(size=concurrency),
                batch_size=batch_size,
            )
            dataset.map_batches(
                band_with_uid,
                batch_format="pyarrow",
                zero_copy_batch=True,
            ).write_parquet(tmp_dir)
            del dataset
        else:
            logger.info("Using CPU for MinHash computation")
            dataset.map_batches(
                minhash_with_uid,
                batch_format="pyarrow",
                zero_copy_batch=True,
            ).write_parquet(tmp_dir)
        end_time = time.time()
        logger.info(f"MinHash time = {end_time - start_time}")
        new_dataset = ray.data.read_parquet(tmp_dir)
        start_time = time.time()
        self.merge()
        end_time = time.time()
        logger.info(f"merge time = {end_time - start_time}")
        start_time = time.time()
        result = new_dataset.map_batches(
            self.filter_with_union_find,
            batch_format="pyarrow",
            zero_copy_batch=True,
        )
        end_time = time.time()
        logger.info(f"filter time = {end_time - start_time}")
        return result


@OPERATORS.register_module(f"{OP_NAME}_with_uid")
class RayBTSMinhashDeduplicatorWithUid(RayBTSMinhashDeduplicator):
    """
    A MinhashLSH deduplicator based on RAY.

    Unlike `RayBTSMinhashDeduplicator`, this class requires the input dataset to contain an additional column
    named '__dj__uid' of type int, where each value is unique across samples. This column serves two main purposes:

    1. **Reduced I/O Overhead**: Compared to RayBTSMinhashDeduplicator, this class does not persist intermediate
    results, thereby reducing disk read and write operations.

    2. **Support for Incremental Deduplication**: The '__dj__uid' column enables the deduplicator to perform
    incremental deduplication. This is particularly useful in scenarios where you already have a deduplicated dataset
    (e.g., dataset A) and want to add a new dataset (e.g., dataset B) while ensuring that duplicates are resolved
    in favor of the original data.

        For example, consider a scenario where you have an already deduplicated dataset A and a new dataset B that
        you wish to add. If you want to perform joint deduplication on both A and B while prioritizing the retention
        of data from A, you can ensure that all '__dj__uid' values in B are greater than those in A. Then, by applying
        this deduplicator to the combined dataset, duplicates will be resolved in favor of the entries from A.
    """

    def run(self, dataset, **kwargs):
        # Ignore additional parameters like exporter, tracer, etc.
        # Initialize actors lazily - now cluster has had time to autoscale
        self._ensure_actors()

        start_time = time.time()

        def minhash_with_uid(table: pa.Table) -> pa.Table:
            uid_list = table[HashKeys.uid].to_pylist()
            self.calc_minhash(table[self.text_key], uid_list)
            return table

        dataset.map_batches(
            minhash_with_uid,
            batch_format="pyarrow",
            zero_copy_batch=True,
        ).materialize()
        end_time = time.time()
        logger.info(f"MinHash time = {end_time - start_time}")

        start_time = time.time()
        self.merge()
        end_time = time.time()
        logger.info(f"merge time = {end_time - start_time}")
        result = dataset.map_batches(
            self.filter_with_union_find,
            batch_format="pyarrow",
            zero_copy_batch=True,
        )
        return result
