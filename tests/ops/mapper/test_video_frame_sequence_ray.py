"""Opt-in real Ray task/actor regressions with offline model predictions."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ray
import test_video_frame_sequence_contract as contract
from PIL import Image

from data_juicer.core.data.ray_dataset import RayDataset
from data_juicer.ops.base_op import Mapper
from data_juicer.utils.constant import CameraCalibrationKeys, Fields, MetaKeys

animal, body, depth = contract.animal, contract.body, contract.depth
face, hawor, moge = contract.face, contract.hawor, contract.moge


class FrameContractRayMapper(Mapper):
    """Exercise the real Ray transport and original process_single methods."""

    def __init__(self, kind, mode, root, **kwargs):
        super().__init__(
            accelerator="cpu", num_proc=1, num_cpus=1, auto_op_parallelism=False, ray_execution_mode=mode, **kwargs
        )
        self.kind = kind
        self.helper = contract.VideoFrameSequenceContractTest()
        self.helper.root = Path(root)
        self.module = {"animal": animal, "body": body, "depth": depth, "face": face, "hawor": hawor, "moge": moge}[kind]
        self.operator = self.helper.op(self.module)

    def process_single(self, sample, rank=None):
        if self.module is animal:
            kwargs = {"side_effect": self.helper.animal_models(self.operator)}
        elif self.module is body:
            kwargs = {"return_value": self.helper.body_model}
        elif self.module is depth:
            kwargs = {"return_value": self.helper.depth_model()}
        elif self.module is face:
            kwargs = {"return_value": self.helper.face_model()}
        elif self.module is moge:
            kwargs = {"return_value": self.helper.moge_model()}
        else:
            kwargs = {"side_effect": [(object(), object(), None, None), object()]}
        with patch.object(self.module, "get_model", **kwargs):
            result = self.operator.process_single(sample, rank)
        result["worker_pid"] = os.getpid()
        return result


@unittest.skipUnless(os.environ.get("DJ_RUN_RAY_TESTS") == "1", "Set DJ_RUN_RAY_TESTS=1 for real Ray tests.")
class VideoFrameSequenceRayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        environment = patch.dict(os.environ, {"RAY_JOB": "1"})
        environment.start()
        cls.addClassCleanup(environment.stop)
        cls.started_ray = not ray.is_initialized()
        if cls.started_ray:
            ray.init(
                num_cpus=2,
                include_dashboard=False,
                object_store_memory=200 * 1024 * 1024,
                runtime_env={"env_vars": {"RAY_JOB": "1"}},
            )

    @classmethod
    def tearDownClass(cls):
        if cls.started_ray:
            ray.shutdown()

    def run_mode(self, mode):
        with tempfile.TemporaryDirectory() as root:
            paths = []
            for index in [0, 2, 10]:
                p = Path(root) / f"frame_{index}.png"
                Image.new("RGB", (32, 24), (index, index, index)).save(p)
                paths.append(str(p))
            for kind in ["animal", "face", "body", "depth", "moge", "hawor"]:
                with self.subTest(kind=kind, mode=mode):
                    frames = [[], [paths[2], paths[0]], [paths[1]]]
                    rows = [
                        {"id": 0, MetaKeys.video_frames: [], "fps": 24, Fields.meta: {"calibration": []}},
                        {
                            "id": 1,
                            MetaKeys.video_frames: frames,
                            "fps": 24,
                            Fields.meta: {
                                "calibration": [
                                    {},
                                    {CameraCalibrationKeys.hfov: [1.0, 1.1]},
                                    {CameraCalibrationKeys.hfov: [1.2]},
                                ]
                            },
                        },
                    ]
                    op = FrameContractRayMapper(kind, mode, root)
                    dataset = RayDataset(ray.data.from_items(rows, override_num_blocks=2), auto_op_parallelism=False)
                    result = dataset.process(op).get(2)
                    result.sort(key=lambda row: row["id"])
                    self.assertEqual([r["id"] for r in result], [0, 1])
                    self.assertTrue(all(r["worker_pid"] != os.getpid() for r in result))
                    tags = result[1][Fields.meta]["predictions"]
                    if kind in ["animal", "face", "body", "depth"]:
                        self.assertEqual(tags["frame_paths"], [paths[2], paths[0], paths[1]])
                        self.assertEqual(tags["video_indices"], [1, 1, 2])
                    elif kind == "moge":
                        self.assertEqual(len(tags), 3)
                        self.assertEqual(len(tags[1][CameraCalibrationKeys.hfov]), 2)
                        self.assertEqual(len(tags[2][CameraCalibrationKeys.hfov]), 1)
                    else:
                        self.assertEqual(len(tags), 3)
                        self.assertEqual(tags[0]["left"]["frame_ids"], [])
                        self.assertAlmostEqual(tags[1]["fov_x"], 1.05)
                        self.assertAlmostEqual(tags[2]["fov_x"], 1.2)
                    json.dumps(result)

    def test_task_workers(self):
        self.run_mode("task")

    def test_actor_workers(self):
        self.run_mode("actor")


if __name__ == "__main__":
    unittest.main()
