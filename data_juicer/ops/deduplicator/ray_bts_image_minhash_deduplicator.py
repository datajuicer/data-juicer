import os
import time
from typing import Optional, Union

import numpy as np
import pyarrow as pa
from loguru import logger
from pydantic import Field, PositiveInt
from typing_extensions import Annotated

from data_juicer.utils.constant import HashKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model
from data_juicer.utils.ray_utils import ray_available_gpu_memories, ray_gpu_count

from ..base_op import OPERATORS, Deduplicator
from ..op_fusion import LOADED_IMAGES
from .document_minhash_deduplicator import MAX_HASH, MERSENNE_PRIME, optimal_param

ray = LazyLoader("ray")

BATCH_SIZE = 1000
OP_NAME = "ray_bts_image_minhash_deduplicator"


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


def get_remote_classes(actor_memory: Optional[int] = None):
    """Get remote versions of classes with Ray decorators applied at runtime.

    :param actor_memory: Memory reservation for EdgeBuffer and BTSUnionFind actors in bytes.
    """
    # Apply ray.method decorator to get_next_id at runtime
    IdGenerator.get_next_id = ray.method(num_returns=2)(IdGenerator.get_next_id)

    remote_args = {"scheduling_strategy": "SPREAD"}
    if actor_memory is not None:
        remote_args["memory"] = actor_memory
        remote_args["num_cpus"] = 0.1

    return {
        "IdGenerator": ray.remote(IdGenerator),
        "EdgeBuffer": ray.remote(**remote_args)(EdgeBuffer),
        "BTSUnionFind": ray.remote(**remote_args)(BTSUnionFind),
    }


class ImageMinHashActor:
    def __init__(
        self,
        model_key,
        use_cuda: bool = True,
        perm_a: np.ndarray = None,
        perm_b: np.ndarray = None,
        num_permutation: int = 256,
        batch_size=None,
    ):
        import torch

        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model_key = model_key
        self.use_cuda = use_cuda
        self.model, self.processor = get_model(self.model_key, use_cuda=self.use_cuda)
        logger.info(f"Loading Image Minhash model: {model_key}")
        self.perm_a = (
            torch.randint(0, MERSENNE_PRIME, (num_permutation,), device=self.device, dtype=torch.uint64)
            if perm_a is None
            else torch.from_numpy(perm_a).to(self.device)
        )
        self.perm_b = (
            torch.randint(0, MERSENNE_PRIME, (num_permutation,), device=self.device, dtype=torch.uint64)
            if perm_b is None
            else torch.from_numpy(perm_b).to(self.device)
        )
        self.prime = torch.tensor(int(MERSENNE_PRIME), device=self.device, dtype=torch.uint64)
        self.batch_size = batch_size

    def _decode_images(self, samples: dict, image_key: str = "images", image_bytes_key: str = "image_bytes"):
        """
        Batch decode images using DALI with chunking and resizing for uniform shapes.
        Uses DALI to read from image bytes if available, otherwise reads from image paths.

        Returns:
            torch.Tensor: [N, C, H, W] tensor on GPU with all decoded images
        """
        import io

        import nvidia.dali as dali
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        import torch
        from PIL import Image, ImageFile

        # Determine input source: prefer image_bytes_key, fallback to image_key
        use_bytes = (
            image_bytes_key in samples and samples[image_bytes_key] is not None and len(samples[image_bytes_key]) > 0
        )
        use_paths = (
            not use_bytes and image_key in samples and samples[image_key] is not None and len(samples[image_key]) > 0
        )

        if not use_bytes and not use_paths:
            raise ValueError(f"Neither {image_bytes_key} nor {image_key} found in samples or both are empty")

        resize_x, resize_y = 224, 224

        if use_bytes:
            image_bytes_list = samples[image_bytes_key]
            batch_data = [np.frombuffer(img_bytes, dtype=np.uint8) for img_bytes in image_bytes_list]
            logger.info(f"Loading with image bytes")
        else:
            batch_data = samples[image_key]
            if isinstance(batch_data[0], list):
                batch_data = [item[0] for item in batch_data]
            logger.info(f"Loading with image paths")

        def _decode_image_with_dali():
            @dali.pipeline_def(
                batch_size=len(batch_data),
                num_threads=8,
                device_id=0 if self.use_cuda else None,
                exec_async=False,
                exec_pipelined=False,
            )
            def decode_pipeline(use_bytes: bool = True, paths=None):
                if use_bytes:
                    data = fn.external_source(name="raw_data", batch=True, dtype=types.UINT8)
                else:
                    data, _ = fn.readers.file(files=paths, random_shuffle=False, name="Reader")
                images = fn.decoders.image(
                    data,
                    device="mixed" if self.use_cuda else "cpu",
                    output_type=types.RGB,
                    use_fast_idct=True,
                )
                images = fn.resize(images, resize_x=resize_x, resize_y=resize_y, interp_type=types.INTERP_LINEAR)
                return images

            if use_bytes:
                pipe = decode_pipeline(use_bytes=use_bytes)
                pipe.build()
                pipe.feed_input("raw_data", batch_data)
            else:
                pipe = decode_pipeline(use_bytes=use_bytes, paths=batch_data)
                pipe.build()

            try:
                outputs = pipe.run()
            except Exception as e:
                logger.error(f"DALI pipeline failed, {e}, batch_size: {self.batch_size}")
                raise

            # Convert DALI tensors to PyTorch tensors
            dali_tensors = outputs[0].as_tensor()
            del outputs
            torch_tensors = torch.as_tensor(dali_tensors, device="cuda" if self.use_cuda else "cpu")
            del dali_tensors

            return torch_tensors

        def decode_images_with_torch():

            ImageFile.LOAD_TRUNCATED_IMAGES = True

            fallback_tensors = []
            for item in batch_data:
                try:
                    if use_bytes:
                        img_bytes = item.tobytes() if isinstance(item, np.ndarray) else item
                        img = Image.open(io.BytesIO(img_bytes))
                    else:
                        path = item[0] if isinstance(item, list) else item
                        img = Image.open(path)

                    img = img.convert("RGB")
                    img = img.resize((resize_x, resize_y), Image.Resampling.BILINEAR)

                    img_tensor = torch.from_numpy(np.array(img))
                    fallback_tensors.append(img_tensor)

                except Exception as inner_e:
                    logger.debug(f"Disaster recovery failed for an image ({inner_e}). Using black placeholder.")
                    fallback_tensors.append(torch.zeros((resize_y, resize_x, 3), dtype=torch.uint8))

            torch_tensors = torch.stack(fallback_tensors).to(self.device)

            return torch_tensors

        try:
            logger.info(f"Decoding batch of {len(batch_data)} images with DALI")
            torch_tensors = _decode_image_with_dali()
        except Exception as e:
            logger.error(f"DALI decoding failed with error: {e}. Falling back to PyTorch decoding.")
            torch_tensors = decode_images_with_torch()

        return torch_tensors

    def compute_minhash(
        self, samples: dict, image_key: str = "image", image_bytes_key: str = "image_bytes"
    ) -> pa.Array:
        import torch

        images = self._decode_images(samples, image_key=image_key, image_bytes_key=image_bytes_key)

        inputs = self.processor(images=images, return_tensors="pt", do_resize=False).to(self.device)
        del images
        with torch.no_grad():
            logits = self.model(**inputs).logits
            del inputs
            visual_tokens = torch.argmax(logits, dim=-1).to(torch.int64)
        tokens_expanded = visual_tokens.unsqueeze(-1)
        del visual_tokens, logits
        hash_vals = (tokens_expanded * self.perm_a.to(torch.int64) + self.perm_b.to(torch.int64)) % self.prime.to(
            torch.int64
        )
        del tokens_expanded
        minhash, _ = hash_vals.min(dim=1)
        del hash_vals
        # Flatten and convert via cuDF
        if self.use_cuda:
            import cudf

            minhash_flat = cudf.core.column.as_column(minhash.flatten()).to_arrow()
        else:
            minhash_np = minhash.cpu().numpy().astype(np.uint32)
            minhash_flat = pa.array(minhash_np.flatten(), type=pa.uint32())
            del minhash_np
        minhash_arrow = pa.FixedSizeListArray.from_arrays(minhash_flat, minhash.shape[1])
        del minhash
        return minhash_arrow

    def __call__(self, table: pa.Table, image_key: str = "image", image_bytes_key: str = "image_bytes") -> dict:
        samples = table.to_pydict()
        minhash_arrow = self.compute_minhash(samples, image_key=image_key, image_bytes_key=image_bytes_key)
        new_table = table.add_column(table.num_columns, "_minhash", minhash_arrow)
        return new_table


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class RayImageBTSMinhashDeduplicator(Deduplicator):
    """A distributed image deduplicator using MinHash LSH and BTS Union-Find on Ray.

    This operator implements a scalable near-duplicate image detection algorithm designed
    for large-scale datasets. It leverages Ray for distributed processing and supports both
    CPU and GPU execution.

    The deduplication process consists of the following steps:
    1. Visual Tokenization: Images are processed by a pre-trained Vision Transformer
       model (e.g., BEiT) to extract discrete visual tokens. These tokens represent the
       semantic content of the image patches.
    2. MinHash Computation: The set of visual tokens is used to compute MinHash
       signatures. This transforms the high-dimensional image data into a compact
       signature that preserves Jaccard similarity.
    3. LSH Banding: MinHash signatures are divided into bands to perform Locality
       Sensitive Hashing (LSH), allowing for efficient retrieval of candidate duplicate pairs.
    4. Distributed Union-Find (BTS): A Balanced Tree Structure (BTS) Union-Find
       algorithm is employed to resolve connected components (clusters of duplicate images)
       across distributed workers.
    5. Filtering: Based on the connected components, duplicate images are filtered out,
       keeping only one representative image per cluster.

    Key Features:
    - Deep Semantic Hashing**: Uses visual tokens from models like BEiT rather than simple pixel-level heuristics.
    - Scalability: Utilizing Ray and the BTS algorithm allows it to scale to billion-level datasets.
    - Performance: Supports NVIDIA DALI for accelerated image decoding and batch inference on GPUs.
    """

    # TODO: Set a more reasonable value
    EMPTY_HASH_VALUE = "EMPTY"
    _batched_op = True
    _accelerator = "cuda"

    def __init__(
        self,
        model_name: str = "microsoft/beit-base-patch16-224-pt22k",
        trust_remote_code: bool = True,
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
        minhash_batch_size: Optional[Union[int, str]] = "auto",
        memory_per_sample: Optional[float] = 25,  # MB per sample
        actor_memory: Optional[int] = None,  # Memory per actor (bytes)
        task_memory: Optional[int] = None,  # Memory per map_batches task (bytes)
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param model_name: the name of the image model to use for hash computation
        :param num_permutations: number of permutations in minhash
            computing (default: 256)
        :param jaccard_threshold: the min jaccard similarity threshold
            in near-duplicate detection. When the jaccard similarity of
            two sample images is >= this threshold, they are regarded as
            similar samples and this op will only keep one of them after
            deduplication (default: 0.7)
        :param num_bands: number of bands in LSH. Default it's None, and
            it will be determined by an optimal params computation
            algorithm by minimize the weighted sum of probs of False
            Positives and False Negatives
        :param num_rows_per_band: number of rows in each band in LSH.
            Default it's None, and it will be determined by an optimal
            params computation algorithm
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
            Default is 25 MB per sample.
        """

        super().__init__(*args, **kwargs)
        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=model_name, trust_remote_code=trust_remote_code
        )

        self.memory_per_sample = memory_per_sample
        if minhash_batch_size == "auto":
            if self.use_cuda():
                self.minhash_batch_size = 2048
            else:
                self.minhash_batch_size = 32
        else:
            self.minhash_batch_size = minhash_batch_size

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
        self.actor_memory = actor_memory
        self.task_memory = task_memory

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
        remote_classes = get_remote_classes(actor_memory=self.actor_memory)
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

    def _get_map_batches_kwargs(self):
        kwargs = {"batch_format": "pyarrow", "zero_copy_batch": True}
        if self.task_memory is not None:
            kwargs["memory"] = self.task_memory
        return kwargs

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
                # Use 70% of available memory to leave room for overhead
                safe_memory = min_memory * 0.7
                estimated_batch_size = int(safe_memory / self.memory_per_sample)
                max_reasonable_batch = 2_048
                batch_size = max(1, min(max_reasonable_batch, estimated_batch_size))

                logger.info(
                    f"Setting batch size to {batch_size} based on available GPU memory "
                    f"({min_memory}MB), memory per sample ({self.memory_per_sample}MB), "
                    f"and safe memory limit ({safe_memory}MB)"
                )
            else:
                batch_size = self.minhash_batch_size
                logger.info(f"Using default batch size of {batch_size}")
        else:
            logger.info("Using CPU for MinHash computation")
            # Get CPU count for concurrency
            cpu_count = int(ray.cluster_resources().get("CPU", 1))
            total_cluster_memory = int(ray.cluster_resources().get("memory", 0))
            safe_memory_total = total_cluster_memory * 0.7

            concurrency = max(1, cpu_count // 2)  # Use half of CPUs for workers
            memory_budget_per_worker = safe_memory_total / concurrency
            logger.info(f"Setting CPU concurrency to {concurrency} based on available CPUs")
            bytes_per_sample = self.memory_per_sample * 1024 * 1024
            estimated_batch_size = int(memory_budget_per_worker / bytes_per_sample)
            batch_size = max(32, min(estimated_batch_size, 1024))

            batch_size = batch_size
            logger.info(f"Using batch size of {batch_size} for CPU MinHash computation")

        from ray.data._internal.util import get_compute_strategy

        compute = get_compute_strategy(ImageMinHashActor, concurrency=(int(concurrency) // 4, int(concurrency)))
        dataset = dataset.map_batches(
            ImageMinHashActor,
            fn_constructor_kwargs={
                "model_key": self.model_key,
                "use_cuda": self.use_cuda(),
                "perm_a": self.perm_a,
                "perm_b": self.perm_b,
                "num_permutation": self.num_permutation,
                "batch_size": batch_size,
            },
            fn_kwargs={"image_key": self.image_key, "image_bytes_key": self.image_bytes_key},
            batch_format="pyarrow",
            zero_copy_batch=True,
            compute=compute,
            num_gpus=1 if self.use_cuda() else 0,
            batch_size=batch_size,
        )
        dataset.map_batches(band_with_uid, **self._get_map_batches_kwargs()).write_parquet(tmp_dir)
        logger.info(f"Write to temporary dir: {tmp_dir}")
        del dataset
        end_time = time.time()
        logger.info(f"MinHash time = {end_time - start_time}")
        concurrency = int(ray.available_resources().get("CPU", 1) // 4)
        new_dataset = ray.data.read_parquet(tmp_dir, concurrency=concurrency)
        start_time = time.time()
        self.merge()
        end_time = time.time()
        logger.info(f"merge time = {end_time - start_time}")
        start_time = time.time()
        concurrency = int(ray.available_resources().get("CPU", 1) // 4)
        compute = get_compute_strategy(self.filter_with_union_find, concurrency=concurrency)
        result = new_dataset.map_batches(
            self.filter_with_union_find,
            **self._get_map_batches_kwargs(),
            compute=compute,
        )
        end_time = time.time()
        logger.info(f"filter time = {end_time - start_time}")
        return result


@OPERATORS.register_module(f"{OP_NAME}_with_uid")
@LOADED_IMAGES.register_module(f"{OP_NAME}_with_uid")
class RayImageBTSMinhashDeduplicatorWithUid(RayImageBTSMinhashDeduplicator):
    """
    A MinhashLSH deduplicator based on RAY.
    ... (docstring omitted) ...
    """

    def run(self, dataset, **kwargs):
        self._ensure_actors()

        start_time = time.time()
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
                # Use 70% of available memory to leave room for overhead
                safe_memory = min_memory * 0.7
                estimated_batch_size = int(safe_memory / self.memory_per_sample)
                max_reasonable_batch = 2_048
                batch_size = max(1, min(max_reasonable_batch, estimated_batch_size))

                logger.info(
                    f"Setting batch size to {batch_size} based on available GPU memory "
                    f"({min_memory}MB), memory per sample ({self.memory_per_sample}MB), "
                    f"and safe memory limit ({safe_memory}MB)"
                )
            else:
                batch_size = self.minhash_batch_size
                logger.info(f"Using default batch size of {batch_size}")
        else:
            logger.info("Using CPU for MinHash computation")
            # Get CPU count for concurrency
            cpu_count = int(ray.cluster_resources().get("CPU", 1))
            total_cluster_memory = int(ray.cluster_resources().get("memory", 0))
            safe_memory_total = total_cluster_memory * 0.7

            concurrency = max(1, cpu_count // 2)  # Use half of CPUs for workers
            memory_budget_per_worker = safe_memory_total / concurrency
            logger.info(f"Setting CPU concurrency to {concurrency} based on available CPUs")
            bytes_per_sample = self.memory_per_sample * 1024 * 1024
            estimated_batch_size = int(memory_budget_per_worker / bytes_per_sample)
            batch_size = max(32, min(estimated_batch_size, 1024))

            batch_size = batch_size
            logger.info(f"Using batch size of {batch_size} for CPU MinHash computation")

        from ray.data._internal.util import get_compute_strategy

        compute = get_compute_strategy(ImageMinHashActor, concurrency=(int(concurrency) // 4, int(concurrency)))
        dataset = dataset.map_batches(
            ImageMinHashActor,
            fn_constructor_kwargs={
                "model_key": self.model_key,
                "use_cuda": self.use_cuda(),
                "perm_a": self.perm_a,
                "perm_b": self.perm_b,
                "num_permutation": self.num_permutation,
                "batch_size": batch_size,
            },
            fn_kwargs={"image_key": self.image_key, "image_bytes_key": self.image_bytes_key},
            batch_format="pyarrow",
            zero_copy_batch=True,
            compute=compute,
            num_gpus=1 if self.use_cuda() else 0,
            batch_size=batch_size,
        )

        def band_existing_uid(table: pa.Table) -> pa.Table:
            if HashKeys.uid not in table.column_names:
                raise ValueError(f"Dataset missing required column: {HashKeys.uid} for {OP_NAME}_with_uid operator.")

            self.band_minhash(table["_minhash"], table[HashKeys.uid])

            return table.drop_columns(["_minhash"])

        dataset = dataset.map_batches(
            band_existing_uid,
            batch_format="pyarrow",
            zero_copy_batch=True,
        ).materialize()

        end_time = time.time()
        logger.info(f"MinHash calculation and banding time = {end_time - start_time}")

        start_time = time.time()
        self.merge()
        end_time = time.time()
        logger.info(f"Union-Find merge time = {end_time - start_time}")

        start_time = time.time()
        result = dataset.map_batches(
            self.filter_with_union_find,
            batch_format="pyarrow",
            zero_copy_batch=True,
        )

        end_time = time.time()
        logger.info(f"Filter graph construction time = {end_time - start_time}")

        return result
