import os
import pickle
import tempfile
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_captioning_from_human_tracks_mapper import (
    VideoCaptioningFromHumanTracksMapper,
)
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

hf_model = 'DAMO-NLP-SG/VideoLLaMA3-7B'


def _make_dummy_track(video_path, temp_dir):
    """Create a real bbox pkl file for testing. The VLM model still runs
    on real cropped video frames — no mock is used."""
    bbox_path = os.path.join(temp_dir, os.path.basename(video_path) + '_0.pkl')
    bbox_dict = {
        'frame': [0, 5, 10, 15, 20],
        'xy_bbox': [[10, 10, 80, 80], [12, 12, 82, 82], [14, 14, 84, 84],
                    [16, 16, 86, 86], [18, 18, 88, 88]],
        'xys_bbox': {
            'x': [45.0, 47.0, 49.0, 51.0, 53.0],
            'y': [45.0, 47.0, 49.0, 51.0, 53.0],
            's': [35.0, 35.0, 35.0, 35.0, 35.0],
        },
        'xy_human_bbox': {
            'x1': [5, 7, 9, 11, 13],
            'y1': [5, 7, 9, 11, 13],
            'x2': [85, 87, 89, 91, 93],
            'y2': [85, 87, 89, 91, 93],
        },
    }
    with open(bbox_path, 'wb') as f:
        pickle.dump(bbox_dict, f)
    return {'bbox_path': bbox_path}


@unittest.skip('Requires ~40GB GPU memory for VideoLLaMA3-7B. '
               'Enable manually when resources are available.')
class VideoCaptioningFromHumanTracksMapperTest(DataJuicerTestCaseBase):
    """Test for video_captioning_from_human_tracks_mapper, which generates
    captions for each tracked person using VideoLLaMA3.

    No mock is used — the real VideoLLaMA3-7B model runs on actual cropped
    video frames. Skipped by default due to high GPU memory requirements.
    """

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    vid4_path = os.path.join(data_path, 'video4.mp4')

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._temp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls) -> None:
        import shutil
        shutil.rmtree(cls._temp_dir, ignore_errors=True)
        super().tearDownClass(hf_model)

    def _run_mapper(self, ds_list, op, num_proc=1):
        dataset = Dataset.from_list(ds_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        res = dataset.flatten().select_columns([
            f'{Fields.meta}.{MetaKeys.track_video_caption}',
            f'{Fields.meta}.{MetaKeys.video_track_is_child}',
        ])
        return res

    def test_single_video(self):
        """Captioning should produce a text description for each track."""
        track = _make_dummy_track(self.vid4_path, self._temp_dir)
        ds_list = [{
            'text': f'{SpecialTokens.video} A person is speaking. {SpecialTokens.eoc}',
            'videos': [self.vid4_path],
            Fields.meta: {MetaKeys.human_track_data_path: [[track]]},
        }]
        op = VideoCaptioningFromHumanTracksMapper(
            video_describe_model_path=hf_model,
            temp_video_path=self._temp_dir,
            trust_remote_code=True,
        )
        res = self._run_mapper(ds_list, op)
        captions = res[f'{Fields.meta}.{MetaKeys.track_video_caption}']
        is_child = res[f'{Fields.meta}.{MetaKeys.video_track_is_child}']
        self.assertEqual(len(captions), 1)
        self.assertEqual(len(is_child), 1)


if __name__ == '__main__':
    unittest.main()
