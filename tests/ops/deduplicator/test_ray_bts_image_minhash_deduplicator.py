import json
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

    def _op(self, op_class=RayImageBTSMinhashDeduplicator, **kwargs):
        options = dict(
            num_permutations=16,
            num_bands=4,
            num_rows_per_band=4,
            work_dir=self.work_dir,
            accelerator="cpu",
            minhash_batch_size=2,
        )
        options.update(kwargs)
        with patch.object(image_minhash, "prepare_model", return_value="offline-model"):
            return op_class(**options)

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
        nested = [[str(paths[i])] for i in order]
        cases = {
            "bytes": {"image_bytes": [paths[i].read_bytes() for i in order]},
            "byte_lists": pa.table({"image_bytes": [[paths[i].read_bytes()] for i in order]}).to_pydict(),
            "lists": {"images": nested},
            "ndarray": {"images": np.asarray(nested)},
            "ndarray_rows": {"images": [np.asarray(row) for row in nested]},
            "null_bytes": pa.table({"images": nested, "image_bytes": [None, None, None]}).to_pydict(),
            "mixed": pa.table(
                {
                    "images": [[str(paths[0])], [str(paths[0])], [str(paths[1])]],
                    "image_bytes": [[paths[1].read_bytes()], None, []],
                }
            ).to_pydict(),
            "null_byte_items": pa.table({"images": nested, "image_bytes": [[None], [], [None]]}).to_pydict(),
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

    def test_dali_receives_normalized_sources_in_order(self):
        actor = self._actor()
        paths = [Path(self.work_dir, f"{i}.png") for i in range(2)]
        for i, path in enumerate(paths):
            Image.new("RGB", (11, 7), (i * 100, 53, 201)).save(path)
        encoded = [path.read_bytes() for path in paths]
        cases = {
            "paths": {"images": [[str(paths[1])], [str(paths[0])]]},
            "bytes": {"image_bytes": [[encoded[1]], [encoded[0]]]},
            "mixed": {"images": [[str(paths[0])], [str(paths[0])]], "image_bytes": [[encoded[1]], None]},
        }
        for name, samples in cases.items():
            with self.subTest(source=name):
                dali, factory, pipe = Mock(), Mock(), Mock()
                dali.pipeline_def.return_value = lambda function: factory
                factory.return_value = pipe
                decoded = torch.zeros((2, 224, 224, 3), dtype=torch.uint8)
                pipe.run.return_value = [SimpleNamespace(as_tensor=lambda: decoded)]
                modules = {
                    "nvidia": SimpleNamespace(dali=dali),
                    "nvidia.dali": dali,
                    "nvidia.dali.fn": dali.fn,
                    "nvidia.dali.types": dali.types,
                }
                with patch.dict(sys.modules, modules), patch.object(Image, "open") as fallback:
                    self.assertIs(actor._decode_images(samples), decoded)
                fallback.assert_not_called()
                pipe.build.assert_called_once_with()
                pipe.run.assert_called_once_with()
                if name == "paths":
                    factory.assert_called_once_with(use_bytes=False, paths=[str(paths[1]), str(paths[0])])
                    pipe.feed_input.assert_not_called()
                else:
                    factory.assert_called_once_with(use_bytes=True)
                    pipe.feed_input.assert_called_once()
                    source, buffers = pipe.feed_input.call_args.args
                    self.assertEqual(source, "raw_data")
                    self.assertEqual([buffer.tobytes() for buffer in buffers], [encoded[1], encoded[0]])
                    self.assertTrue(all(buffer.dtype == np.uint8 for buffer in buffers))

    def test_multiple_images_are_rejected_before_decoding(self):
        actor = self._actor()
        cases = [
            {"images": [["first", "second"]]},
            {"images": np.asarray([["first", "second"]])},
            {"images": [["first"], ["first", "second"]]},
            {"images": [["first", "second"]], "image_bytes": [[b"image"]]},
            {"images": [["first"]], "image_bytes": [[b"first", b"second"]]},
            {"image_bytes": [[b"first", b"second"]]},
        ]
        with patch.object(Image, "open") as decode:
            for samples in cases:
                with self.subTest(samples=samples):
                    with self.assertRaisesRegex(ValueError, "exactly one image per sample"):
                        actor._decode_images(samples)
            decode.assert_not_called()

    def test_missing_image_is_rejected_before_decoding(self):
        actor = self._actor()
        cases = [{"images": [[]]}, {"image_bytes": [None]}, {"images": [None], "image_bytes": [[]]}]
        for samples in cases:
            with self.subTest(samples=samples):
                with self.assertRaisesRegex(ValueError, "Missing image bytes and path"):
                    actor._decode_images(samples)

    def test_decode_failures_do_not_become_valid_black_images(self):
        actor = self._actor()
        bad_path = Path(self.work_dir, "bad.png")
        bad_path.write_bytes(b"invalid image")
        black_path = Path(self.work_dir, "black.png")
        Image.new("RGB", (224, 224), (0, 0, 0)).save(black_path)
        with patch.dict(sys.modules, {"nvidia.dali": None}), patch.object(ImageFile, "LOAD_TRUNCATED_IMAGES", False):
            for samples in (
                {"images": [[str(bad_path)], [str(black_path)]]},
                {"images": [[str(black_path)], [str(bad_path)]]},
                {"image_bytes": [b"invalid image", black_path.read_bytes()]},
                {"images": [[str(Path(self.work_dir, "missing.png"))]]},
            ):
                with self.subTest(samples=list(samples)):
                    with self.assertRaisesRegex(ValueError, "could not decode image at row [01]") as raised:
                        actor.compute_minhash(samples)
                    self.assertIsNotNone(raised.exception.__cause__)
                    actor.processor.assert_not_called()
                    actor.model.assert_not_called()
                    self.assertFalse(ImageFile.LOAD_TRUNCATED_IMAGES)
            decoded = actor._decode_images({"images": [[str(black_path)]]})
            self.assertEqual(tuple(decoded.shape), (1, 224, 224, 3))
            self.assertEqual(int(torch.count_nonzero(decoded)), 0)
            self.assertFalse(ImageFile.LOAD_TRUNCATED_IMAGES)

    def test_find_compresses_a_deep_chain_without_recursion(self):
        graph = image_minhash.BTSUnionFind(256, 1, 0, [], 20, 10)
        graph.parent = {uid: uid - 1 for uid in range(1, 10000)}
        self.assertEqual(graph.find(9999), 0)
        self.assertEqual(set(graph.parent.values()), {0})
        self.assertEqual(graph.find(10000), 10000)

    def test_invalid_explicit_batch_sizes_are_rejected(self):
        for value in (None, True, 0, -1, "invalid"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                self._op(minhash_batch_size=value)

    def test_with_uid_converts_arrow_integers_before_union(self):
        op = self._op(image_minhash.RayImageBTSMinhashDeduplicatorWithUid)
        graph = image_minhash.BTSUnionFind(256, 1, 0, [], 20, 10)
        op.union_find_parallel_num = 1
        op.union_find_list = [SimpleNamespace(add_key_value_pairs=SimpleNamespace(remote=graph.add_key_value_pairs))]
        table = pa.table(
            {
                "_minhash": pa.array([list(range(16))] * 2, type=pa.list_(pa.uint32(), 16)),
                image_minhash.HashKeys.uid: pa.chunked_array([[11], [7]], type=pa.int64()),
            }
        )
        dataset = Mock()

        def band_batch(function, **kwargs):
            result = function(table)
            self.assertNotIn("_minhash", result.column_names)
            self.assertEqual(result[image_minhash.HashKeys.uid].to_pylist(), [11, 7])
            return dataset

        dataset.map_batches.side_effect = band_batch
        with (
            patch.object(op, "_ensure_actors"),
            patch.object(op, "_compute_minhash", return_value=dataset),
            patch.object(op, "merge", side_effect=graph.flush_key_value_pairs),
            patch.object(image_minhash, "ray"),
        ):
            op.run(dataset)
        self.assertEqual(graph.parent, {11: 7})

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
                self.assertEqual(options["batch_size"], 2)
                self.assertEqual(options["fn_constructor_kwargs"]["batch_size"], 2)
                self.assertEqual((options["compute"].min_size, options["compute"].max_size), (1, 1))
                self.assertEqual(options["fn_kwargs"], {"image_key": "images", "image_bytes_key": "image_bytes"})
                for name in ("perm_a", "perm_b"):
                    self.assertIs(options["fn_constructor_kwargs"][name], getattr(op, name))
                    np.testing.assert_array_equal(getattr(op, name), getattr(self, name))

    def test_auto_batch_sizes_still_follow_available_memory(self):
        for use_cuda, expected in ((False, 229), (True, 28)):
            with self.subTest(use_cuda=use_cuda):
                op, dataset = self._op(minhash_batch_size="auto"), Mock()
                with (
                    patch.object(op, "use_cuda", return_value=use_cuda),
                    patch.object(image_minhash, "ray") as ray_mock,
                    patch.object(image_minhash, "ray_gpu_count", return_value=1),
                    patch.object(image_minhash, "ray_available_gpu_memories", return_value=[1024]),
                ):
                    ray_mock.cluster_resources.return_value = {"CPU": 4, "memory": 16 * 1024**3}
                    op._compute_minhash(dataset)
                self.assertEqual(dataset.map_batches.call_args.kwargs["batch_size"], expected)

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
                self.assertIsNone(op.union_find_parallel_num)
                writer.assert_called_once()
                ray_mock.data.read_parquet.assert_called_with(writer.call_args.args[0], concurrency=1)
                filtered = ray_mock.data.read_parquet.return_value.map_batches
                filter_function = filtered.call_args.args[0]
                run_op = filter_function.__self__
                self.assertIsNot(run_op, op)
                self.assertTrue(run_op._actors_initialized)
                self.assertEqual(run_op.union_find_parallel_num, 1)
                strategy.assert_any_call(filter_function, concurrency=1)
                self.assertEqual(filtered.call_args.kwargs["compute"].size, 1)
                self.assertIs(result, filtered.return_value)
        self.assertEqual(len(set(paths)), 2)
        self.assertEqual(sentinel.read_bytes(), b"untouched")
        for path in paths:
            self.assertEqual((path / "part.parquet").read_bytes(), b"current-run")


class ImageMinHashRayRegressionTest(DataJuicerTestCaseBase):
    """Real Ray graph tests with precomputed signatures; no model or GPU required."""

    def setUp(self):
        super().setUp()
        # CI's Ray containers share the repository mount, but not /tmp.
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(tmp.cleanup)
        self.work_dir = tmp.name
        patcher = patch.object(RayImageBTSMinhashDeduplicator, "_compute_minhash", lambda self, dataset: dataset)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _op(self, cls):
        with patch.object(image_minhash, "prepare_model", return_value="offline-model"):
            return cls(
                accelerator="cpu",
                work_dir=self.work_dir,
                num_permutations=2,
                num_bands=2,
                num_rows_per_band=1,
                union_find_parallel_num=1,
                minhash_batch_size=2,
                actor_memory=1024**2,
            )

    def _dataset(self, labels, signatures, uids=None):
        import ray

        columns = {"label": labels, "_minhash": pa.array(signatures, type=pa.list_(pa.uint32(), 2))}
        if uids is not None:
            columns[image_minhash.HashKeys.uid] = pa.array(uids, type=pa.int64())
        return ray.data.from_arrow(pa.table(columns))

    def _export_labels(self, result, name):
        directory = Path(self.work_dir, name)
        result.write_json(str(directory))
        rows = [json.loads(line) for path in directory.glob("*.json") for line in path.read_text().splitlines()]
        self.assertTrue(rows)
        self.assertTrue(all(set(row) == {"label"} for row in rows))
        return {row["label"] for row in rows}

    @TEST_TAG("ray")
    def test_reusing_both_variants_preserves_previous_lazy_results(self):
        for cls in (RayImageBTSMinhashDeduplicator, image_minhash.RayImageBTSMinhashDeduplicatorWithUid):
            with self.subTest(variant=cls.__name__):
                op = self._op(cls)
                uids = [11, 12, 13] if cls is image_minhash.RayImageBTSMinhashDeduplicatorWithUid else None
                first = op.run(
                    self._dataset(["first", "first-duplicate", "first-unique"], [[7, 9], [7, 9], [1, 2]], uids)
                )
                second = op.run(
                    self._dataset(["second", "second-duplicate", "second-unique"], [[3, 4], [3, 4], [7, 9]], uids)
                )
                self.assertEqual(self._export_labels(second, f"{cls.__name__}-second"), {"second", "second-unique"})
                self.assertEqual(self._export_labels(first, f"{cls.__name__}-first"), {"first", "first-unique"})

    @TEST_TAG("ray")
    def test_empty_arrow_input_preserves_schema(self):
        import ray

        schema = pa.schema([("images", pa.list_(pa.string())), ("text", pa.string())])
        dataset = ray.data.from_arrow(pa.Table.from_batches([], schema=schema))
        op = self._op(RayImageBTSMinhashDeduplicator)
        result = op.run(dataset)
        self.assertEqual(result.take_all(), [])
        self.assertEqual(result.schema().base_schema, schema)

    @TEST_TAG("ray")
    def test_upstream_filter_can_remove_all_rows_in_one_pipeline(self):
        import ray

        from data_juicer.core.data.ray_dataset import RayDataset
        from data_juicer.ops.filter.text_length_filter import TextLengthFilter

        op = self._op(RayImageBTSMinhashDeduplicator)
        source = ray.data.from_arrow(pa.table({"images": [["unused.png"]], "text": ["short"]}))
        result = RayDataset(source).process([TextLengthFilter(min_len=100), op]).data
        self.assertEqual(result.take_all(), [])
        self.assertEqual(result.count(), 0)
        output_dir = Path(self.work_dir, "empty-output")
        result.write_json(str(output_dir))
        rows = [line for path in output_dir.glob("*.json") for line in path.read_text().splitlines()]
        self.assertEqual(rows, [])

    @TEST_TAG("ray")
    def test_with_uid_merges_deep_band_chain(self):
        op = self._op(image_minhash.RayImageBTSMinhashDeduplicatorWithUid)
        uids = list(range(2999, -1, -1))
        dataset = self._dataset([str(uid) for uid in uids], [[uid // 2, (uid + 1) // 2] for uid in uids], uids)
        self.assertEqual(self._export_labels(op.run(dataset), "deep-chain"), {"0"})


if __name__ == "__main__":
    unittest.main()
