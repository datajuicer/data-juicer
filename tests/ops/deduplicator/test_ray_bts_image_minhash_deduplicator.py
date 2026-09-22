import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyarrow as pa
import torch
from PIL import Image, ImageFile
from ray.data._internal import util as ray_data_util

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.deduplicator import (
    ray_bts_image_minhash_deduplicator as image_minhash,
)
from data_juicer.ops.deduplicator.ray_bts_image_minhash_deduplicator import (
    RayImageBTSMinhashDeduplicator,
)
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


class RayImageBTSMinhashDeduplicatorTest(DataJuicerTestCaseBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cur_dir = os.path.dirname(os.path.abspath(__file__))
        cls.work_dir = os.path.join(cls.cur_dir, "image_dedup_test")
        if not os.path.exists(cls.work_dir):
            os.makedirs(cls.work_dir)

        # 准备测试图片路径
        cls.img_paths = [
            os.path.join(cls.work_dir, "img1.jpg"),
            os.path.join(cls.work_dir, "img2.jpg"),
            os.path.join(cls.work_dir, "img3.jpg"),
            os.path.join(cls.work_dir, "img4.jpg"),
        ]
        cls._generate_test_images()

    @classmethod
    def _generate_test_images(cls):
        img1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
        img1.save(cls.img_paths[0])
        img1.save(cls.img_paths[1])
        img3_np = np.array(img1).copy()
        img3_np[0, 0] = [254, 1, 1]
        Image.fromarray(img3_np).save(cls.img_paths[2])
        img4 = Image.new("RGB", (224, 224), color=(0, 0, 255))
        img4.save(cls.img_paths[3])

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.work_dir):
            shutil.rmtree(cls.work_dir)
        super().tearDownClass()

    def _run_minhash_dedup(self, dataset: Dataset, target_list, op):
        check_keys = [op.image_key] if hasattr(op, "image_key") else ["images"]
        res_list = self.run_single_op(dataset, op, check_keys)
        self.assertEqual(len(res_list), len(target_list))

    @TEST_TAG("ray")
    def test_image_path_deduplication(self):
        ds_list = [{"images": [p]} for p in self.img_paths]
        tgt_list = [{"images": [self.img_paths[0]]}, {"images": [self.img_paths[3]]}]

        dataset = self.generate_dataset(ds_list)
        op = RayImageBTSMinhashDeduplicator(jaccard_threshold=0.85, work_dir=self.work_dir, minhash_batch_size=2)
        self._run_minhash_dedup(dataset, tgt_list, op)

    @TEST_TAG("ray")
    def test_image_bytes_deduplication(self):
        ds_list = []
        for p in self.img_paths:
            with open(p, "rb") as f:
                ds_list.append({"image_bytes": f.read()})

        tgt_list = ds_list[:1] + ds_list[3:4]

        dataset = self.generate_dataset(ds_list)
        op = RayImageBTSMinhashDeduplicator(jaccard_threshold=0.85, work_dir=self.work_dir)

        check_keys = [op.image_bytes_key] if hasattr(op, "image_bytes_key") else ["image_bytes"]
        res_list = self.run_single_op(dataset, op, check_keys)
        self.assertEqual(len(res_list), len(tgt_list))


class ImageMinHashRegressionTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work_dir = tmp.name
        rng = np.random.RandomState(42)
        self.prime = (1 << 61) - 1
        self.perm_a, self.perm_b = np.array(
            [
                (rng.randint(1, self.prime, dtype=np.uint64), rng.randint(0, self.prime, dtype=np.uint64))
                for _ in range(16)
            ],
            dtype=np.uint64,
        ).T
        self.original_permutations = np.stack((self.perm_a, self.perm_b))
        self.tokens = np.array([[0, 1, 63, 127, 1023, 8191], [9, 9, 51, 512, 4095, 8190]], dtype=np.int64)

    def _actor(self, use_cuda=False):
        logits = torch.full((*self.tokens.shape, 8192), -1.0, device="cuda" if use_cuda else "cpu")
        logits.scatter_(-1, torch.as_tensor(self.tokens, device=logits.device).unsqueeze(-1), 1.0)
        model, processor = Mock(return_value=SimpleNamespace(logits=logits)), Mock()
        processor.return_value.to.return_value = {}
        with patch.object(image_minhash, "get_model", return_value=(model, processor)) as load:
            actor = image_minhash.ImageMinHashActor(
                "offline-model", use_cuda=use_cuda, perm_a=self.perm_a, perm_b=self.perm_b, num_permutation=16
            )
        load.assert_called_once_with("offline-model", use_cuda=use_cuda)
        for name, expected in zip(("perm_a", "perm_b"), self.original_permutations):
            actual = getattr(actor, name)
            self.assertEqual(actual.dtype, torch.uint64)
            self.assertTrue(np.all(expected > np.iinfo(np.uint32).max))
            np.testing.assert_array_equal(getattr(self, name), expected)
            np.testing.assert_array_equal(actual.cpu().numpy(), expected)
        return actor

    def _op(self):
        with patch.object(image_minhash, "prepare_model", return_value="offline-model"):
            return RayImageBTSMinhashDeduplicator(
                num_permutations=16,
                num_bands=4,
                num_rows_per_band=4,
                work_dir=self.work_dir,
                accelerator="cpu",
                minhash_batch_size=2,
            )

    def _check_signatures(self, use_cuda=False, simulate_cuda=False):
        actor = self._actor(use_cuda)
        if simulate_cuda:
            actor.use_cuda = True  # Exercise only the output branch; all tensors remain on CPU.
            self.assertEqual(actor.device.type, "cpu")
        with np.errstate(over="ignore"):
            wrapped = self.tokens[..., None] * self.perm_a.astype(np.int64) + self.perm_b.astype(np.int64)
        self.assertTrue(np.any(wrapped < 0))
        hashes = wrapped % np.int64(self.prime)
        expected = hashes.min(axis=1)
        if not actor.use_cuda:
            expected = expected.astype(np.uint32)
            self.assertFalse(np.array_equal(expected, hashes.astype(np.uint32).min(axis=1)))
        samples, decoded = {"images": [["first"], ["second"]]}, object()
        cudf_modules = {} if use_cuda else {"cudf": None}
        if simulate_cuda:
            cudf = Mock()
            cudf.core.column.as_column.side_effect = lambda tensor: SimpleNamespace(
                to_arrow=lambda: pa.array(tensor.numpy())
            )
            cudf_modules["cudf"] = cudf
        with (
            patch.dict(sys.modules, cudf_modules),
            patch.object(actor, "_decode_images", return_value=decoded) as decode,
        ):
            signatures = actor.compute_minhash(samples)
        if simulate_cuda:
            cudf.core.column.as_column.assert_called_once()
            flattened = cudf.core.column.as_column.call_args.args[0]
            self.assertEqual(flattened.dtype, torch.int64)
            np.testing.assert_array_equal(flattened.numpy(), expected.flatten())
        decode.assert_called_once_with(samples, image_key="images", image_bytes_key="image_bytes")
        actor.processor.assert_called_once_with(images=decoded, return_tensors="pt", do_resize=False)
        actor.processor.return_value.to.assert_called_once_with(actor.device)
        actor.model.assert_called_once_with()
        self.assertIsInstance(signatures, pa.FixedSizeListArray)
        self.assertEqual(signatures.type, pa.list_(pa.int64() if actor.use_cuda else pa.uint32(), 16))
        self.assertEqual(len(signatures), len(self.tokens))
        self.assertEqual(signatures.null_count, 0)
        self.assertEqual(signatures.values.null_count, 0)
        np.testing.assert_array_equal(signatures.values.to_numpy().reshape(expected.shape), expected)
        op = self._op()
        op.union_find_parallel_num, op.union_find_list = 1, [Mock()]
        with patch.object(image_minhash, "ray"):
            op.band_minhash(signatures, [7, 11])
        expected_pairs = [
            (band.to_bytes(4, "big") + row[start:end].tobytes(), uid)
            for row, uid in zip(expected, [7, 11])
            for band, (start, end) in enumerate(op.hash_ranges)
        ]
        op.union_find_list[0].add_key_value_pairs.remote.assert_called_once_with(expected_pairs)

    def test_cpu_original_arithmetic_and_band_bytes(self):
        self._check_signatures()

    def test_cpu_hosted_mock_cudf_bridge_and_band_bytes(self):
        self._check_signatures(simulate_cuda=True)

    @unittest.skipUnless(torch.cuda.is_available(), "Physical CUDA device unavailable")
    def test_physical_cuda_original_arithmetic_and_band_bytes(self):
        try:
            __import__("cudf")
        except ImportError:
            self.skipTest("cuDF unavailable")
        self._check_signatures(use_cuda=True)

    def test_missing_shared_permutations_fail_before_model_loading(self):
        for perm_a, perm_b in ((None, None), (self.perm_a, None), (None, self.perm_b)):
            with self.subTest(missing_a=perm_a is None, missing_b=perm_b is None):
                with patch.object(image_minhash, "get_model") as load:
                    with self.assertRaisesRegex(ValueError, "perm_a and perm_b must be provided"):
                        image_minhash.ImageMinHashActor("offline-model", perm_a=perm_a, perm_b=perm_b)
                    load.assert_not_called()

    def test_missing_dali_decodes_bytes_and_nested_paths_in_order(self):
        actor = self._actor()
        colors = [(17, 53, 201), (222, 101, 7)]
        paths = [Path(self.work_dir, f"{i}.png") for i in range(2)]
        for path, color in zip(paths, colors):
            Image.new("RGB", (11, 7), color).save(path)
        order = [1, 0, 1]
        nested = [[str(paths[i]), str(paths[1 - i])] for i in order]
        cases = {
            "bytes": {"image_bytes": [paths[i].read_bytes() for i in order]},
            "lists": {"images": nested},
            "ndarray": {"images": np.asarray(nested)},
            "ndarray_rows": {"images": [np.asarray(row) for row in nested]},
        }
        expected = np.broadcast_to(
            np.array([colors[i] for i in order], dtype=np.uint8)[:, None, None, :], (3, 224, 224, 3)
        )
        with patch.dict(sys.modules, {"nvidia.dali": None}), patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", False):
            for name, samples in cases.items():
                with self.subTest(source=name):
                    decoded = actor._decode_images(samples)
                    self.assertEqual(tuple(decoded.shape), (3, 224, 224, 3))
                    self.assertEqual(decoded.dtype, torch.uint8)
                    np.testing.assert_array_equal(decoded.numpy(), expected)

    def test_compute_minhash_resource_floor_and_shared_permutations(self):
        for use_cuda in (False, True):
            with self.subTest(use_cuda=use_cuda):
                op, dataset = self._op(), Mock()
                self.assertEqual(op.image_key, "images")
                self.assertEqual(op._supported_exec_modes, ("ray", "ray_partitioned"))
                with (
                    patch.object(op, "use_cuda", return_value=use_cuda),
                    patch.object(image_minhash, "ray") as ray_mock,
                    patch.object(image_minhash, "ray_gpu_count", return_value=1),
                    patch.object(image_minhash, "ray_available_gpu_memories", return_value=[1024]),
                    patch.object(
                        ray_data_util, "get_compute_strategy", wraps=ray_data_util.get_compute_strategy
                    ) as strategy,
                ):
                    ray_mock.cluster_resources.return_value = {"CPU": 1, "GPU": int(use_cuda), "memory": 1024**3}
                    self.assertIs(op._compute_minhash(dataset), dataset.map_batches.return_value)
                strategy.assert_called_once_with(image_minhash.ImageMinHashActor, concurrency=(1, 1))
                self.assertEqual(dataset.map_batches.call_args.args, (image_minhash.ImageMinHashActor,))
                options = dataset.map_batches.call_args.kwargs
                self.assertEqual(options["num_gpus"], int(use_cuda))
                self.assertGreater(options["batch_size"], 0)
                self.assertEqual((options["compute"].min_size, options["compute"].max_size), (1, 1))
                self.assertEqual(options["fn_kwargs"], {"image_key": "images", "image_bytes_key": "image_bytes"})
                for name in ("perm_a", "perm_b"):
                    self.assertIs(options["fn_constructor_kwargs"][name], getattr(op, name))
                    np.testing.assert_array_equal(getattr(op, name), getattr(self, name))

    def test_independent_runs_keep_lazy_inputs_and_use_unique_scratch(self):
        job_dir = Path(self.work_dir, ".tmp", "same-job")
        job_dir.mkdir(parents=True)
        sentinel = job_dir / "lazy-input.parquet"
        sentinel.write_bytes(b"untouched")
        paths = []

        def write_parquet(path):
            self.assertEqual(sentinel.read_bytes(), b"untouched")
            destination = Path(path)
            self.assertEqual(destination.parent, job_dir)
            destination.mkdir()
            (destination / "part.parquet").write_bytes(b"current-run")
            paths.append(destination)

        with (
            patch.object(image_minhash, "ray") as ray_mock,
            patch.object(
                image_minhash,
                "get_remote_classes",
                return_value={name: Mock() for name in ("IdGenerator", "EdgeBuffer", "BTSUnionFind")},
            ),
            patch.object(RayImageBTSMinhashDeduplicator, "merge"),
            patch.object(ray_data_util, "get_compute_strategy", wraps=ray_data_util.get_compute_strategy) as strategy,
        ):
            ray_mock.cluster_resources.return_value = {"CPU": 1, "memory": 1024**3}
            ray_mock.get_runtime_context.return_value.get_job_id.return_value = "same-job"
            for available_cpus in (1, 0):
                op, dataset = self._op(), Mock()
                ray_mock.available_resources.return_value = {"CPU": available_cpus}
                writer = dataset.map_batches.return_value.map_batches.return_value.write_parquet
                writer.side_effect = write_parquet
                result = op.run(dataset)
                self.assertEqual(op.union_find_parallel_num, 1)
                writer.assert_called_once()
                ray_mock.data.read_parquet.assert_called_with(writer.call_args.args[0], concurrency=1)
                strategy.assert_any_call(op.filter_with_union_find, concurrency=1)
                filtered = ray_mock.data.read_parquet.return_value.map_batches
                self.assertEqual(filtered.call_args.args, (op.filter_with_union_find,))
                self.assertEqual(filtered.call_args.kwargs["compute"].size, 1)
                self.assertIs(result, filtered.return_value)
        self.assertEqual(len(set(paths)), 2)
        self.assertEqual(sentinel.read_bytes(), b"untouched")
        for path in paths:
            self.assertEqual((path / "part.parquet").read_bytes(), b"current-run")


if __name__ == "__main__":
    unittest.main()
