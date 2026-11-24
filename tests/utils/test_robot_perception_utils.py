import base64
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from data_juicer.utils import model_utils
from data_juicer.utils import robot_perception_utils as perception
from data_juicer.utils import sam3_utils as sam3


class RobotAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, payload, self.headers.get("Authorization")))
        response = self.server.responses.pop(0)
        data = json.dumps({"choices": [{"message": response}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class RobotPerceptionUtilsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def _server(self, responses):
        server = ThreadingHTTPServer(("127.0.0.1", 0), RobotAPIHandler)
        server.requests, server.responses = [], list(responses)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def close():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(close)
        return server, f"http://127.0.0.1:{server.server_port}/v1"

    def test_vlm_uses_shared_api_client_and_preserves_generation_and_reasoning(self):
        server, endpoint = self._server([{"content": '{"target_object":"cup"}', "reasoning_content": "visible"}])
        generation = perception.ChatGenerationConfig(max_tokens=123, top_k=7, min_p=0.1, repetition_penalty=1.1)
        client = perception.ChatJSONClient(perception.VLLMChatClient("test-vlm", endpoint, "test-key", generation, 5))
        parsed, raw, reasoning = client.chat_json("find the cup", [])
        self.assertEqual(parsed, {"target_object": "cup"})
        self.assertEqual(json.loads(raw), parsed)
        self.assertEqual(reasoning, "visible")
        self.assertEqual(len(server.requests), 1)
        path, body, auth = server.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(auth, "Bearer test-key")
        self.assertEqual(
            (body["max_tokens"], body["top_k"], body["min_p"], body["repetition_penalty"]), (123, 7, 0.1, 1.1)
        )

    def test_failed_api_response_is_not_silently_used(self):
        client = perception.VLLMChatClient(
            "test-vlm", "http://localhost/v1", "test-key", perception.ChatGenerationConfig(), 5
        )
        with patch.object(perception, "get_model", return_value=Mock(return_value="")):
            with self.assertRaisesRegex(RuntimeError, "VLM request failed"):
                client.chat([])

    def test_multi_image_request_has_numbered_contact_sheet(self):
        paths = []
        for i, color in enumerate(("red", "blue", "green")):
            path = self.root / f"{i}.png"
            Image.new("RGB", (16, 16), color).save(path)
            paths.append(path)
        client = Mock()
        client.chat.return_value = {"content": '{"selected_candidate":1}'}
        perception.ChatJSONClient(client).chat_json("select", paths)
        parts = client.chat.call_args.args[0][-1]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "select"})
        self.assertEqual(len(parts), 2)
        encoded = parts[1]["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as sheet:
            self.assertEqual(sheet.size, (1536, 924))

    def test_final_grounding_round_trip_and_cache(self):
        server, endpoint = self._server(
            [{"content": '{"image_index":2}'}, {"content": '{"point_2d":[250,750],"evidence":"cup"}'}]
        )
        views = {}
        for camera in ("left", "right"):
            path = self.root / camera / "last.png"
            path.parent.mkdir()
            Image.new("RGB", (16, 16), "red").save(path)
            views[camera] = {"images": [str(path)]}
        with patch.dict("os.environ", {"VLM_API_KEY": "test-key"}):
            result = perception.ground_final_target(self.root, views, "cup", "test-vlm", endpoint)
            cached = perception.ground_final_target(self.root, views, "cup", "test-vlm", endpoint)
        self.assertEqual(result, cached)
        self.assertEqual((result["view"], result["point"], result["sample_index"]), ("right", [250, 750], 0))
        self.assertEqual(len(server.requests), 2)
        self.assertTrue(
            server.requests[1][1]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png")
        )

    def test_final_grounding_rejects_no_visible_target(self):
        path = self.root / "last.png"
        Image.new("RGB", (16, 16), "red").save(path)
        with patch.object(perception.VLLMChatClient, "chat", return_value={"content": '{"image_index":0}'}):
            with self.assertRaisesRegex(ValueError, "not visible"):
                perception.ground_final_target(self.root, {"left": {"images": [str(path)]}}, "cup", "test-vlm")

    def test_sam3_candidates_keep_source_mask_and_score(self):
        masks = np.zeros((2, 8, 8), dtype=bool)
        masks[0, 1:3, 2:5] = True
        masks[1, 3:6, 4:7] = True
        candidates = sam3.processed_to_candidates(np, torch, {"masks": masks, "scores": torch.tensor([0.2, 0.9])})
        detections = sam3.save_candidates(Image, np, candidates, self.root, 1)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["source_index"], 1)
        self.assertEqual(detections[0]["bbox_from_mask"], [4.0, 3.0, 7.0, 6.0])
        with Image.open(detections[0]["mask_path"]) as mask:
            np.testing.assert_array_equal(np.array(mask) > 0, masks[1])

    def test_sam3_loaders_use_shared_model_management_and_rank(self):
        args = SimpleNamespace(
            sam3_video_model=str(self.root), sam3_video_cache_dir=None, sam3_video_torch_dtype="fp32"
        )
        for tracking in (False, True):
            with (
                self.subTest(tracking=tracking),
                patch.object(sam3, "prepare_model", return_value="model-key") as prepare,
                patch.object(sam3, "get_model", return_value=(Mock(), Mock())) as get,
            ):
                if tracking:
                    sam3.load_video_tracker(args, torch, "cuda:2")
                else:
                    sam3.load_sam3(str(self.root), None, torch, "cuda:2", "fp32")
                self.assertEqual(prepare.call_args.kwargs["model_type"], "robot_sam3")
                self.assertEqual(prepare.call_args.kwargs.get("tracking", False), tracking)
                get.assert_called_once_with("model-key", rank=2, use_cuda=True)

    def test_sam3_image_and_video_model_classes_remain_distinct(self):
        for tracking in (False, True):
            classes = SimpleNamespace(
                Sam3Model=Mock(), Sam3Processor=Mock(), Sam3TrackerVideoModel=Mock(), Sam3TrackerVideoProcessor=Mock()
            )
            model_cls = classes.Sam3TrackerVideoModel if tracking else classes.Sam3Model
            processor_cls = classes.Sam3TrackerVideoProcessor if tracking else classes.Sam3Processor
            model = Mock()
            model.to.return_value = model
            model.eval.return_value = model
            model_cls.from_pretrained.return_value = model
            with self.subTest(tracking=tracking), patch.object(model_utils, "transformers", classes):
                result, processor = model_utils.prepare_robot_sam3_model(
                    str(self.root), tracking=tracking, torch_dtype=torch.float32, device="cpu"
                )
            self.assertIs(result, model)
            self.assertIs(processor, processor_cls.from_pretrained.return_value)
            model_cls.from_pretrained.assert_called_once_with(
                str(self.root), local_files_only=True, torch_dtype=torch.float32
            )
            model.to.assert_called_once_with("cpu")

    def test_pointing_requests_normalized_2d_coordinates_and_converts_to_pixels(self):
        client = Mock()
        parsed = {"point_2d": [625, 250]}
        client.chat_json.return_value = (parsed, json.dumps(parsed), "")
        result = perception.run_vlm_pointing(
            SimpleNamespace(vlm_model="test-vlm"),
            SimpleNamespace(vlm_client=client),
            "green bottle",
            self.root / "frame.jpg",
            320,
            180,
        )
        prompt, paths = client.chat_json.call_args.args
        self.assertIn('"point_2d"', prompt)
        self.assertIn("1000", prompt)
        self.assertEqual(paths, [self.root / "frame.jpg"])
        self.assertEqual(result["point_1000"], [625.0, 250.0])
        self.assertEqual(result["point_xy"], [200.0, 45.0])
        self.assertEqual(result["raw_response"], parsed)

    def test_tracker_preserves_reverse_frame_indices_and_removes_duplicate_anchor(self):
        session = SimpleNamespace(video_height=8, video_width=8)
        output = lambda index: SimpleNamespace(frame_idx=index, pred_masks=torch.ones((1, 1, 8, 8)))
        processor = Mock()
        processor.init_video_session.return_value = session
        processor.post_process_masks.return_value = [torch.ones((1, 1, 8, 8))]
        model = Mock(return_value=output(7))
        model.propagate_in_video_iterator.return_value = iter([output(7), output(3), output(0)])
        with patch("transformers.video_utils.load_video", return_value=(np.zeros((8, 8, 8, 3)), {})):
            frames, metadata = sam3.track_video(
                np,
                torch,
                model,
                processor,
                self.root / "video.mp4",
                [0, 0, 8, 8],
                np.ones((8, 8)),
                [4, 4],
                8,
                8,
                0,
                "cpu",
                "fp32",
                True,
                "mask",
                prompt_frame_idx=7,
                reverse=True,
            )
        self.assertEqual([frame["frame_index"] for frame in frames], [0, 3, 7])
        self.assertTrue(all(frame["bbox_xyxy"] == [0.0, 0.0, 8.0, 8.0] for frame in frames))
        self.assertEqual(metadata["tracking_prompt_frame_idx"], 7)
        self.assertTrue(metadata["tracking_reverse"])
        model.propagate_in_video_iterator.assert_called_once_with(session, start_frame_idx=7, reverse=True)


if __name__ == "__main__":
    unittest.main()
