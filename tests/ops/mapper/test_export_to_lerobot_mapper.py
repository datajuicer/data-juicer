import json
import os
import shutil
import tempfile
import unittest
import numpy as np

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.export_to_lerobot_mapper import ExportToLeRobotMapper
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import load_file_byte
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class ExportToLeRobotMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid3_path = os.path.join(data_path, 'video3.mp4')
    vid4_path = os.path.join(data_path, 'video4.mp4')

    def setUp(self):
        self.output_dir = tempfile.mkdtemp(prefix='lerobot_test_')

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def _make_sample(self, video_source, num_frames=10, task_desc="pick up the cup",
                     valid_frame_ids=None):
        """Create a synthetic sample with hand action data."""
        states = (np.random.randn(num_frames, 8).astype(np.float32)).tolist()
        actions = (np.random.randn(num_frames, 7).astype(np.float32)).tolist()
        if valid_frame_ids is None:
            valid_frame_ids = list(range(num_frames))

        sample = {
            'videos': [video_source],
            'text': task_desc,
            Fields.meta: {
                'hand_action_tags': [{
                    'right': {
                        'states': states,
                        'actions': actions,
                        'valid_frame_ids': valid_frame_ids,
                        'hand_type': 'right',
                    },
                    'left': {
                        'states': [],
                        'actions': [],
                        'valid_frame_ids': [],
                        'hand_type': 'left',
                    }
                }],
            }
        }
        return sample

    def test_process_single(self):
        """Test processing a single sample."""
        sample = self._make_sample(self.vid3_path, num_frames=10)
        ds_list = [sample]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            fps=10,
            robot_type='egodex_hand',
        )

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        self.assertEqual(len(res_list), 1)
        export_info = res_list[0][Fields.meta].get('lerobot_export', [])
        self.assertGreater(len(export_info), 0)

        ep = export_info[0]
        self.assertIn('uuid', ep)
        self.assertIn('parquet_path', ep)
        self.assertEqual(ep['num_frames'], 10)

        # Verify staging files exist
        self.assertTrue(os.path.exists(ep['parquet_path']))

    def test_finalize_dataset(self):
        """Test the full pipeline: process + finalize."""
        samples = [
            self._make_sample(self.vid3_path, num_frames=10, task_desc="pick up cup"),
            self._make_sample(self.vid4_path, num_frames=8, task_desc="place cup"),
        ]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            fps=10,
            robot_type='egodex_hand',
        )

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)

        # Finalize
        ExportToLeRobotMapper.finalize_dataset(
            output_dir=self.output_dir,
            fps=10,
            robot_type='egodex_hand',
        )

        meta_dir = os.path.join(self.output_dir, 'meta')

        # Check info.json
        info_path = os.path.join(meta_dir, 'info.json')
        self.assertTrue(os.path.exists(info_path))
        with open(info_path, 'r') as f:
            info = json.load(f)
        self.assertEqual(info['codebase_version'], 'v2.0')
        self.assertEqual(info['robot_type'], 'egodex_hand')
        self.assertEqual(info['total_episodes'], 2)
        self.assertEqual(info['total_frames'], 18)  # 10 + 8
        self.assertEqual(info['fps'], 10)
        self.assertEqual(info['total_tasks'], 2)

        # Check features
        self.assertIn('observation.state', info['features'])
        self.assertEqual(info['features']['observation.state']['shape'], [8])
        self.assertIn('action', info['features'])
        self.assertEqual(info['features']['action']['shape'], [7])

        # Check episodes.jsonl
        episodes_path = os.path.join(meta_dir, 'episodes.jsonl')
        self.assertTrue(os.path.exists(episodes_path))
        with open(episodes_path, 'r') as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(episodes), 2)

        # Check tasks.jsonl
        tasks_path = os.path.join(meta_dir, 'tasks.jsonl')
        self.assertTrue(os.path.exists(tasks_path))
        with open(tasks_path, 'r') as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(tasks), 2)

        # Check modality.json
        modality_path = os.path.join(meta_dir, 'modality.json')
        self.assertTrue(os.path.exists(modality_path))
        with open(modality_path, 'r') as f:
            modality = json.load(f)
        self.assertIn('state', modality)
        self.assertIn('action', modality)

        # Check data directory
        data_dir = os.path.join(self.output_dir, 'data', 'chunk-000')
        self.assertTrue(os.path.exists(data_dir))
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        self.assertEqual(len(parquet_files), 2)

        # Check staging is cleaned up
        staging_dir = os.path.join(self.output_dir, 'staging')
        self.assertFalse(os.path.exists(staging_dir))

    def test_empty_action_data(self):
        """Test with empty action data - should not export anything."""
        sample = {
            'videos': [self.vid3_path],
            'text': 'test',
            Fields.meta: {
                'hand_action_tags': [],
            }
        }

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
        )

        dataset = Dataset.from_list([sample])
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        export_info = res_list[0][Fields.meta].get('lerobot_export', [])
        self.assertEqual(len(export_info), 0)

    def test_same_task_deduplication(self):
        """Test that episodes with the same task share a task_index."""
        samples = [
            self._make_sample(self.vid3_path, num_frames=5, task_desc="pick up cup"),
            self._make_sample(self.vid4_path, num_frames=5, task_desc="pick up cup"),
        ]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            fps=10,
        )

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)

        ExportToLeRobotMapper.finalize_dataset(
            output_dir=self.output_dir, fps=10,
        )

        with open(os.path.join(self.output_dir, 'meta', 'info.json'), 'r') as f:
            info = json.load(f)
        self.assertEqual(info['total_tasks'], 1)  # same task

    def test_mul_proc(self):
        """Test with multiple processes."""
        samples = [
            self._make_sample(self.vid3_path, num_frames=5),
            self._make_sample(self.vid4_path, num_frames=5),
        ]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            fps=10,
        )

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, num_proc=2, with_rank=True)
        res_list = dataset.to_list()

        for sample in res_list:
            export_info = sample[Fields.meta].get('lerobot_export', [])
            self.assertGreater(len(export_info), 0)

    def test_video_bytes_input(self):
        """Test processing with video bytes input instead of file paths."""
        video_bytes = load_file_byte(self.vid3_path)
        sample = self._make_sample(video_bytes, num_frames=10)
        ds_list = [sample]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            fps=10,
            robot_type='egodex_hand',
        )

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        self.assertEqual(len(res_list), 1)
        export_info = res_list[0][Fields.meta].get('lerobot_export', [])
        self.assertGreater(len(export_info), 0)

        ep = export_info[0]
        self.assertEqual(ep['num_frames'], 10)
        self.assertTrue(os.path.exists(ep['parquet_path']))
        # Verify the video was written to staging
        self.assertTrue(os.path.exists(ep['video_path']))
        self.assertTrue(ep['video_path'].endswith('.mp4'))

    def test_video_bytes_finalize(self):
        """Test full pipeline with video bytes: process + finalize."""
        vid3_bytes = load_file_byte(self.vid3_path)
        vid4_bytes = load_file_byte(self.vid4_path)
        samples = [
            self._make_sample(vid3_bytes, num_frames=10, task_desc="pick up cup"),
            self._make_sample(vid4_bytes, num_frames=8, task_desc="place cup"),
        ]

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            fps=10,
            robot_type='egodex_hand',
        )

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)

        ExportToLeRobotMapper.finalize_dataset(
            output_dir=self.output_dir, fps=10, robot_type='egodex_hand',
        )

        with open(os.path.join(self.output_dir, 'meta', 'info.json'), 'r') as f:
            info = json.load(f)
        self.assertEqual(info['total_episodes'], 2)
        self.assertEqual(info['total_frames'], 18)
        self.assertEqual(info['total_videos'], 2)

        # Check video files in final directory
        video_dir = os.path.join(self.output_dir, 'videos', 'chunk-000',
                                 'observation.images.image')
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        self.assertEqual(len(video_files), 2)

    # ------------------------------------------------------------------
    # Segment-based export tests
    # ------------------------------------------------------------------
    def _make_segment_sample(self, n_frames=30, n_segments=2):
        """Create a sample with atomic action segments for segment export."""
        # Create dummy frame images
        frame_dir = os.path.join(self.output_dir, 'frames')
        os.makedirs(frame_dir, exist_ok=True)
        frame_paths = []
        for i in range(n_frames):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            path = os.path.join(frame_dir, f'frame_{i:04d}.jpg')
            import cv2
            cv2.imwrite(path, img)
            frame_paths.append(path)

        frames_per_seg = n_frames // n_segments
        segments = []
        for s in range(n_segments):
            start = s * frames_per_seg
            end = min((s + 1) * frames_per_seg - 1, n_frames - 1)
            n = end - start + 1
            states = np.random.randn(n, 8).astype(np.float32).tolist()
            actions = np.random.randn(n, 7).astype(np.float32).tolist()
            segments.append({
                "hand_type": "right" if s % 2 == 0 else "left",
                "segment_id": s,
                "start_frame": start,
                "end_frame": end,
                "states": states,
                "actions": actions,
                "valid_frame_ids": list(range(start, end + 1)),
                "caption": {
                    "think": "test reasoning",
                    "action": f"Pick up object {s}",
                },
            })

        return {
            "video_frames": frame_paths,
            "text": "",
            Fields.meta: {
                "atomic_action_segments": segments,
            },
        }

    def test_segment_export_creates_episodes(self):
        """Each segment should become a separate episode."""
        sample = self._make_segment_sample(n_frames=30, n_segments=3)
        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            segment_field='atomic_action_segments',
            fps=10,
        )

        dataset = Dataset.from_list([sample])
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        export_info = res_list[0][Fields.meta].get('lerobot_export', [])
        self.assertEqual(len(export_info), 3)

        for ep in export_info:
            self.assertIn('segment_id', ep)
            self.assertIn('hand_type', ep)
            self.assertGreater(ep['num_frames'], 0)
            self.assertTrue(os.path.exists(ep['parquet_path']))

    def test_segment_export_finalize(self):
        """Full segment pipeline: process + finalize."""
        sample = self._make_segment_sample(n_frames=30, n_segments=2)
        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            segment_field='atomic_action_segments',
            fps=10,
            robot_type='egodex_hand',
        )

        dataset = Dataset.from_list([sample])
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)

        ExportToLeRobotMapper.finalize_dataset(
            output_dir=self.output_dir, fps=10, robot_type='egodex_hand',
        )

        with open(os.path.join(self.output_dir, 'meta', 'info.json')) as f:
            info = json.load(f)
        self.assertEqual(info['total_episodes'], 2)
        # Each segment has its own task description
        self.assertEqual(info['total_tasks'], 2)

        # Check episode parquet files
        data_dir = os.path.join(self.output_dir, 'data', 'chunk-000')
        parquets = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        self.assertEqual(len(parquets), 2)

        # Check tasks.jsonl has per-segment captions
        with open(os.path.join(self.output_dir, 'meta', 'tasks.jsonl')) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        task_names = {t['task'] for t in tasks}
        self.assertIn('Pick up object 0', task_names)
        self.assertIn('Pick up object 1', task_names)

    def test_segment_export_skips_na(self):
        """Segments with N/A caption should be skipped."""
        sample = self._make_segment_sample(n_frames=20, n_segments=2)
        # Mark first segment as N/A
        sample[Fields.meta]['atomic_action_segments'][0]['caption'] = {
            'think': '', 'action': 'N/A'}

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            segment_field='atomic_action_segments',
            fps=10,
        )

        dataset = Dataset.from_list([sample])
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        export_info = res_list[0][Fields.meta].get('lerobot_export', [])
        self.assertEqual(len(export_info), 1)  # Only seg1 exported
        self.assertEqual(export_info[0]['segment_id'], 1)

    def test_segment_export_frame_index_relative(self):
        """Frame indices in parquet should be segment-relative (0-based)."""
        import pyarrow.parquet as pq

        sample = self._make_segment_sample(n_frames=20, n_segments=1)
        seg = sample[Fields.meta]['atomic_action_segments'][0]
        # Segment covers frames 0-19
        self.assertEqual(seg['start_frame'], 0)

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            segment_field='atomic_action_segments',
            fps=10,
        )

        dataset = Dataset.from_list([sample])
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        ep = res_list[0][Fields.meta]['lerobot_export'][0]
        table = pq.read_table(ep['parquet_path'])
        df = table.to_pandas()

        # frame_index should be 0-based (segment-relative)
        self.assertEqual(df['frame_index'].tolist()[0], 0)
        self.assertEqual(df['frame_index'].tolist()[-1],
                         len(seg['states']) - 1)

    def test_valid_frame_ids_in_parquet(self):
        """Test that valid_frame_ids are used as frame_index in parquet."""
        import pyarrow.parquet as pq

        # Sparse frame IDs: hand detected at frames 0, 3, 7, 12, 15
        valid_frame_ids = [0, 3, 7, 12, 15]
        sample = self._make_sample(
            self.vid3_path, num_frames=5,
            valid_frame_ids=valid_frame_ids,
        )

        op = ExportToLeRobotMapper(
            output_dir=self.output_dir,
            hand_action_field='hand_action_tags',
            fps=10,
        )

        dataset = Dataset.from_list([sample])
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        res_list = dataset.to_list()

        ep = res_list[0][Fields.meta]['lerobot_export'][0]
        table = pq.read_table(ep['parquet_path'])
        df = table.to_pandas()

        # frame_index should match valid_frame_ids
        self.assertEqual(df['frame_index'].tolist(), valid_frame_ids)
        # timestamp should be frame_id / fps
        expected_timestamps = [float(fid) / 10 for fid in valid_frame_ids]
        for actual, expected in zip(df['timestamp'].tolist(), expected_timestamps):
            self.assertAlmostEqual(actual, expected, places=5)


if __name__ == '__main__':
    unittest.main()
