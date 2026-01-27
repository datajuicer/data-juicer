import os
import time
from typing import Optional, Union

import numpy as np
import pyarrow as pa
import regex
from loguru import logger
from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import prepare_sentencepiece_model

from ..base_op import OPERATORS, Deduplicator
from .document_minhash_deduplicator import MAX_HASH, optimal_param

ray = LazyLoader("ray")

BATCH_SIZE = 2048
MAX_DATA_NUM = 2**63


@ray.remote
class IdGenerator:
    def __init__(self, start_id=0):
        self.next_id = start_id

    @ray.method(num_returns=2)
    def get_next_id(self, count):
        current_id = self.next_id
        self.next_id += count
        return (current_id, self.next_id)


@ray.remote
class HashAggregator:
    def __init__(self, parallel_num):
        self.hash_table = [{} for _ in range(parallel_num)]

    def set_hash_pairs(self, pairs):
        for hash_table_id, hash_value, uid in pairs:
            self.hash_table[hash_table_id].setdefault(hash_value, []).append(uid)

    def get_hash_table(self, hash_table_id):
        result = self.hash_table[hash_table_id]
        self.hash_table[hash_table_id] = {}
        return result


@ray.remote
class EdgeBuffer:
    def __init__(self):
        self.edge_buffer = []

    def set_edges(self, edge_buffer):
        self.edge_buffer = edge_buffer

    def get_edges(self, parallel_id):
        result = self.edge_buffer[parallel_id]
        self.edge_buffer[parallel_id] = []
        return result


@ray.remote
class BTSUnionFind:
    def __init__(
        self,
        union_threshold,
        parallel_num,
        parallel_id,
        max_pending_edge_buffer_task,
        num_edge_buffer_task_returns,
        worker_node_ids,
        num_hash_aggregators,
        remote_edge_buffers_ref,
    ):
        self.union_threshold = union_threshold
        self.parallel_num = parallel_num
        self.parallel_id = parallel_id
        self.hash_table = {}
        self.parent = {}
        self.old_parent = {}
        self.remote_edge_buffers = remote_edge_buffers_ref
        self.remote_buffer = None
        self.edge_buffer = []
        self.remote_edges = [[] for _ in range(parallel_num)]
        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns
        self.worker_node_ids = worker_node_ids
        self.num_nodes = len(worker_node_ids)
        self.num_hash_aggregators = num_hash_aggregators

    def get_hash_table(self, hash_aggregators):
        result_refs = []
        for hash_aggregator in hash_aggregators:
            if len(result_refs) > self.max_pending_edge_buffer_task:
                ready_refs, result_refs = ray.wait(result_refs, num_returns=self.num_edge_buffer_task_returns)
                hash_table_list = ray.get(ready_refs)
                for hash_table in hash_table_list:
                    for key, value in hash_table.items():
                        self.hash_table.setdefault(key, []).extend(value)
                del ready_refs
            result_refs.append(hash_aggregator.get_hash_table.remote(self.parallel_id))
        hash_table_list = ray.get(result_refs)
        for hash_table in hash_table_list:
            for key, value in hash_table.items():
                self.hash_table.setdefault(key, []).extend(value)
        key_cnt = len(self.hash_table)
        value_cnt = 0
        for value in self.hash_table.values():
            value_cnt += len(value)
            if len(value) > 1:
                self.union_list(value)
        del self.hash_table
        return key_cnt, value_cnt

    def balanced_union_find(self):
        for x, y in self.edge_buffer:
            self.union(x, y)
        self.edge_buffer = []

        result_refs = []
        for i, remote_edge_buffer in enumerate(self.remote_edge_buffers):
            if i == self.parallel_id:
                continue
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
        self.remote_edges[hash_u].append((u, v))
        if hash_u != hash_v:
            self.remote_edges[hash_v].append((u, v))

    def set_edge_buffer(self):
        self.edge_buffer = self.remote_edges[self.parallel_id]
        self.remote_edges[self.parallel_id] = []
        ray.get(self.remote_edge_buffers[self.parallel_id].set_edges.remote(self.remote_edges))
        self.remote_edges = [[] for _ in range(self.parallel_num)]

    def edge_redistribution(self, hash_aggregators):
        key_cnt, value_cnt = self.get_hash_table(hash_aggregators)
        self.rebalancing()
        for u, v in self.parent.items():
            self.distribute_edge(u, v)
        self.parent = {}
        self.set_edge_buffer()
        return key_cnt, value_cnt

    def communication(self):
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
        self.dup_uids = {}
        for x in self.parent:
            x_div = x // BATCH_SIZE
            if x_div % self.parallel_num != self.parallel_id:
                continue
            if x_div not in self.dup_uids:
                self.dup_uids[x_div] = np.ones(BATCH_SIZE, dtype=np.bool_)
            self.dup_uids[x_div][x % BATCH_SIZE] = 0
        self.parent.clear()
        self.old_parent.clear()
        self.edge_buffer.clear()

    def remain_mask(self, uid_div, cmp_uid_mods):
        from bitarray import bitarray

        if uid_div in self.dup_uids:
            mask = self.dup_uids[uid_div][np.array(cmp_uid_mods.tolist(), dtype=np.bool_)]
            cmp_mask = bitarray()
            cmp_mask.extend(mask.tolist())
            return cmp_mask
        else:
            # return np.ones(uid_mods.sum(), dtype=np.bool_)
            cmp_mask = bitarray()
            cmp_mask.extend([1] * cmp_uid_mods.count(1))
            return cmp_mask


class MinhashCalculator:
    def __init__(
        self,
        num_hash_aggregators_per_node,
        num_permutation,
        num_bands,
        num_rows_per_band,
        union_find_parallel_num,
        text_key,
        tokenization: str = "space",
        window_size: PositiveInt = 5,
        lowercase: bool = True,
        ignore_pattern: Optional[str] = None,
        tokenizer_model: Optional[str] = None,
    ):
        node_id = ray.get_runtime_context().get_node_id()
        self.id_generator = ray.get_actor(name=f"id_generators_{node_id}")
        self.hash_aggregators = [
            ray.get_actor(name=f"hash_aggregator_{node_id}_{i}") for i in range(num_hash_aggregators_per_node)
        ]
        # about minhash computation
        self.text_key = text_key
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

        if self.tokenization == "character":
            self.tokenization_func = lambda x: [s.encode("utf-8") for s in list(x)]
        elif self.tokenization == "punctuation":
            self.tokenization_func = lambda x: [s.encode("utf-8") for s in self.punctuation_pattern.split(x)]
        elif self.tokenization == "space":
            from .tokenize import split_on_whitespace

            self.tokenization_func = split_on_whitespace
        elif self.tokenization == "sentencepiece":
            self.tokenization_func = lambda x: self.tokenizer.encode(x, out_type=str)
        else:
            raise NotImplementedError(f"Unimplemented tokenization method [{self.tokenization}]")
        self.num_permutation = num_permutation
        self.num_bands = num_bands
        self.num_rows_per_band = num_rows_per_band
        # compute hash ranges and create hash tables
        self.hash_ranges = [
            (i * self.num_rows_per_band, (i + 1) * self.num_rows_per_band) for i in range(self.num_bands)
        ]

        # generate permutations
        gen = np.random.RandomState(seed=42)
        max_int64 = np.iinfo(np.int64).max
        self.perm_a, self.perm_b = np.array(
            [
                (
                    gen.randint(1, max_int64, dtype=np.uint64),
                    gen.randint(0, max_int64, dtype=np.uint64),
                )
                for _ in range(self.num_permutation)
            ],
            dtype=np.uint64,
        ).T

        self.union_find_parallel_num = union_find_parallel_num

        empty_hash_value = np.full((self.num_rows_per_band,), MAX_HASH, dtype=np.uint32)
        self.empty_hash_value = b"\x00\x00\x00\x00" + empty_hash_value.tobytes()
        self.empty_hash_table_id = int(MAX_HASH % self.union_find_parallel_num)

    def calc_minhash(self, text_list: pa.Array, uid_begin: int, thread_num: int = 4) -> pa.Table:
        from .minhash import calc_minhash_batch_c
        from .tokenize import n_grams

        tokens = [n_grams(self.tokenization_func(text.as_py()), self.window_size) for text in text_list]
        pairs = calc_minhash_batch_c(
            tokens,
            uid_begin,
            self.perm_a,
            self.perm_b,
            self.empty_hash_value,
            self.hash_ranges,
            self.union_find_parallel_num,
            thread_num,
        )
        idx = np.random.randint(len(self.hash_aggregators))
        ray.get(self.hash_aggregators[idx].set_hash_pairs.remote(pairs))

    def __call__(self, table: pa.Table) -> pa.Table:
        num_rows = len(table)
        min_id, max_id = ray.get(self.id_generator.get_next_id.remote(num_rows))
        uid_list = range(min_id, max_id)
        self.calc_minhash(table[self.text_key], min_id)
        new_table = table.append_column(HashKeys.uid, pa.array(list(uid_list)))
        return new_table


class MinhashFilter:
    def __init__(
        self,
        num_nodes,
        union_find_parallel_num,
        max_pending_filter_tasks,
        num_filter_task_returns,
    ):
        self.num_nodes = num_nodes
        self.union_find_parallel_num = union_find_parallel_num
        self.max_pending_filter_tasks = max_pending_filter_tasks
        self.num_filter_task_returns = num_filter_task_returns

    def __call__(self, samples: pa.Table) -> pa.Table:
        from bitarray import bitarray

        uids = samples[HashKeys.uid].to_numpy()
        uids_div, uids_mod = np.divmod(uids, BATCH_SIZE)
        diff_idx = np.where(uids_div[1:] != uids_div[:-1])[0] + 1
        len_diff_idx = len(diff_idx)
        result_refs = []
        for i in range(len_diff_idx + 1):
            start = diff_idx[i - 1] if i > 0 else 0
            end = diff_idx[i] if i < len_diff_idx else len(uids)
            uid_mask = np.zeros(BATCH_SIZE, dtype=np.bool_)
            uid_mask[uids_mod[start:end]] = True
            compressed_uid_mask = bitarray()
            compressed_uid_mask.extend(uid_mask.tolist())
            uid_div = uids_div[start]
            hash_id = uid_div % self.union_find_parallel_num
            union_find = ray.get_actor(f"union_find_{hash_id}")
            result_refs.append(union_find.remain_mask.remote(uid_div, compressed_uid_mask))
        mask = np.concatenate([np.array(cmp_mask.tolist(), dtype=np.bool_) for cmp_mask in ray.get(result_refs)])
        columns_to_keep = [name for name in samples.column_names if name != HashKeys.uid]
        return samples.select(columns_to_keep).filter(mask)


OP_NAME = "ray_bts_minhash_cpp_deduplicator"


@OPERATORS.register_module(OP_NAME)
class RayBTSMinhashCppDeduplicator(Deduplicator):
    """
    A basic exact matching deduplicator for RAY.
    Although its functionality is deduplication,
    it is implemented as Filter sub-class.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = "EMPTY"
    _batched_op = True

    def __init__(
        self,
        tokenization: str = "space",
        window_size: PositiveInt = 5,
        lowercase: bool = True,
        ignore_pattern: Optional[str] = None,
        tokenizer_model: Optional[str] = None,
        num_permutations: PositiveInt = 256,
        jaccard_threshold: Annotated[float, Field(ge=0, le=1)] = 0.7,
        num_bands: Optional[PositiveInt] = None,
        num_rows_per_band: Optional[PositiveInt] = None,
        union_find_parallel_num: Union[int, str] = "auto",
        union_threshold: Optional[int] = 256,
        max_pending_edge_buffer_task: Optional[int] = 20,
        num_edge_buffer_task_returns: Optional[int] = 10,
        max_pending_filter_tasks: Optional[int] = 20,
        num_filter_task_returns: Optional[int] = 10,
        merge_batch_size: Optional[int] = 1000,
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
        self.tokenizer_model = tokenizer_model

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

        cpu_num = int(ray.cluster_resources().get("CPU"))
        if union_find_parallel_num == "auto":
            union_find_parallel_num = cpu_num // 2
        else:
            union_find_parallel_num = int(union_find_parallel_num)

        worker_node_ids = [node["NodeID"] for node in ray.nodes() if node["Resources"].get("CPU", 0)]
        self.num_nodes = num_nodes = len(worker_node_ids)
        logger.info(f"Total number of nodes: {num_nodes}")
        for node in ray.nodes():
            logger.info(node)

        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        num_ids_per_node = MAX_DATA_NUM // num_nodes
        self.id_generators = [
            IdGenerator.options(
                name=f"id_generators_{worker_node_ids[i]}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=worker_node_ids[i],
                    soft=False,
                ),
            ).remote(num_ids_per_node * i)
            for i in range(num_nodes)
        ]

        # hash_aggregators_map
        self.num_hash_aggregators_per_node = int(np.ceil(cpu_num / num_nodes * 0.2))
        num_hash_aggregators = self.num_hash_aggregators_per_node * num_nodes
        self.hash_aggregators = [
            HashAggregator.options(
                name=f"hash_aggregator_{worker_node_ids[i % num_nodes]}_{i // num_nodes}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=worker_node_ids[i % num_nodes],
                    soft=False,
                ),
            ).remote(parallel_num=union_find_parallel_num)
            for i in range(num_hash_aggregators)
        ]

        self.min_hash_concurrency = int(cpu_num - 4 * num_nodes)
        self.filter_concurrency = int(cpu_num * 0.75)
        self.max_pending_edge_buffer_task = max_pending_edge_buffer_task
        self.num_edge_buffer_task_returns = num_edge_buffer_task_returns
        self.max_pending_filter_tasks = max_pending_filter_tasks
        self.num_filter_task_returns = num_filter_task_returns
        self.merge_batch_size = min(merge_batch_size, union_find_parallel_num)

        logger.info(f"union_find_parallel_num = {union_find_parallel_num}")
        self.union_find_parallel_num = union_find_parallel_num
        self.union_threshold = union_threshold
        self.remote_edge_buffers = [
            EdgeBuffer.options(
                name=f"edge_buffer_{i}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=worker_node_ids[i % num_nodes],
                    soft=False,
                ),
            ).remote()
            for i in range(self.union_find_parallel_num)
        ]
        remote_edge_buffers_ref = ray.put(self.remote_edge_buffers)
        self.union_find_list = [
            BTSUnionFind.options(
                name=f"union_find_{i}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=worker_node_ids[i % num_nodes],
                    soft=False,
                ),
            ).remote(
                self.union_threshold,
                self.union_find_parallel_num,
                i,
                self.max_pending_edge_buffer_task,
                self.num_edge_buffer_task_returns,
                worker_node_ids,
                num_hash_aggregators,
                remote_edge_buffers_ref,
            )
            for i in range(self.union_find_parallel_num)
        ]
        self.tmp_dir = os.path.join(self.work_dir, ".tmp", ray.get_runtime_context().get_job_id())
        os.makedirs(self.tmp_dir)
        logger.info("init finished!")
        ray.data._internal.execution.operators.actor_pool_map_operator.DEFAULT_MAX_TASKS_IN_FLIGHT = 2

    def merge_op_batch(self, object_refs):
        results = []
        while object_refs:
            ready_refs, object_refs = ray.wait(object_refs, num_returns=min(self.merge_batch_size, len(object_refs)))
            results.extend(ray.get(ready_refs))
        return results

    def merge(self):
        hash_aggregators_ref = ray.put(self.hash_aggregators)
        result = self.merge_op_batch(
            [union_find.edge_redistribution.remote(hash_aggregators_ref) for union_find in self.union_find_list]
        )
        logger.info("start merge loop")
        key_cnt = sum(r[0] for r in result)
        value_cnt = sum(r[1] for r in result)
        logger.info(f"key_cnt = {key_cnt}, value_cnt = {value_cnt}")
        idx = 0
        while any(
            self.merge_op_batch([union_find.balanced_union_find.remote() for union_find in self.union_find_list])
        ):
            logger.info(f"loop {idx} start")
            self.merge_op_batch([union_find.communication.remote() for union_find in self.union_find_list])
            logger.info(f"loop {idx} finished")
            idx += 1
        self.merge_op_batch([union_find.squeeze.remote() for union_find in self.union_find_list])
        logger.info("merge finished")

    def run(self, dataset):
        start_time = time.time()
        dataset.map_batches(
            MinhashCalculator,
            batch_format="pyarrow",
            zero_copy_batch=True,
            concurrency=self.min_hash_concurrency,
            fn_constructor_kwargs={
                "num_hash_aggregators_per_node": self.num_hash_aggregators_per_node,
                "num_permutation": self.num_permutation,
                "num_bands": self.num_bands,
                "num_rows_per_band": self.num_rows_per_band,
                "union_find_parallel_num": self.union_find_parallel_num,
                "text_key": self.text_key,
                "tokenization": self.tokenization,
                "window_size": self.window_size,
                "lowercase": self.lowercase,
                "ignore_pattern": self.ignore_pattern,
                "tokenizer_model": self.tokenizer_model,
            },
        ).write_parquet(self.tmp_dir, ray_remote_args={"num_cpus": 0.25})
        end_time = time.time()
        logger.info(f"MinHash time = {end_time - start_time}")

        start_time = time.time()
        self.merge()
        end_time = time.time()
        logger.info(f"merge time = {end_time - start_time}")
        dataset = ray.data.read_parquet(self.tmp_dir)
        result = dataset.map_batches(
            MinhashFilter,
            batch_format="pyarrow",
            zero_copy_batch=True,
            concurrency=self.filter_concurrency,
            fn_constructor_kwargs={
                "num_nodes": self.num_nodes,
                "union_find_parallel_num": self.union_find_parallel_num,
                "max_pending_filter_tasks": self.max_pending_filter_tasks,
                "num_filter_task_returns": self.num_filter_task_returns,
            },
        )
        # logger.info(f'origin count = {dataset.count()}, keep count = {result.count()}')
        return result
