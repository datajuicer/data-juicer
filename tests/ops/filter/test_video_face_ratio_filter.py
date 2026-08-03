import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.filter.video_face_ratio_filter import VideoFaceRatioFilter
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

try:
    import dlib  # noqa: F401
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False


@unittest.skipUnless(_DLIB_AVAILABLE, 'dlib package not installed.')
class VideoFaceRatioFilterTest(DataJuicerTestCaseBase):
    """Test for video_face_ratio_filter, which uses dlib to detect faces
    in video frames and keeps samples whose face-to-frame ratio is within
    a specified range.

    No mock is used — dlib's real frontal face detector runs on actual
    video files.
    """

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data')
    vid1_path = os.path.join(data_path, 'video14.mp4')
    vid4_path = os.path.join(data_path, 'video15.mp4')

    def _run_helper(self, op, source_list, target_list, select_field=None):
        dataset = Dataset.from_list(source_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=1)
        dataset = dataset.filter(op.process, num_proc=1)
        dataset = dataset.remove_columns(Fields.stats)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test_keep_all(self):
        """With threshold=0.0, all videos should be kept."""
        ds_list = [{'videos': [self.vid1_path]}, {'videos': [self.vid4_path]}]
        tgt_list = [{'videos': [self.vid1_path]}, {'videos': [self.vid4_path]}]
        op = VideoFaceRatioFilter(threshold=0.0, detect_interval=10, any_or_all='all')
        self._run_helper(op, ds_list, tgt_list)

    def test_filter_strict(self):
        """With a high threshold, videos without enough faces are filtered out."""
        ds_list = [{'videos': [self.vid1_path]}, {'videos': [self.vid4_path]}]
        op = VideoFaceRatioFilter(threshold=0.99, detect_interval=10, any_or_all='all')
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=1)
        dataset = dataset.filter(op.process, num_proc=1)
        dataset = dataset.remove_columns(Fields.stats)
        res_list = dataset.to_list()
        # at least one video should be filtered out with a very high threshold
        self.assertLess(len(res_list), len(ds_list))

    def test_any_strategy(self):
        """With 'any' strategy, a sample is kept if any video meets the threshold."""
        ds_list = [{'videos': [self.vid1_path, self.vid4_path]}]
        op = VideoFaceRatioFilter(threshold=0.0, detect_interval=10, any_or_all='any')
        self._run_helper(op, ds_list, ds_list)

    def test_all_strategy(self):
        """With 'all' strategy, a sample is kept only if all videos meet the threshold."""
        ds_list = [{'videos': [self.vid1_path, self.vid4_path]}]
        op = VideoFaceRatioFilter(threshold=0.0, detect_interval=10, any_or_all='all')
        self._run_helper(op, ds_list, ds_list)

    def test_no_video(self):
        """Samples with empty video lists should be kept (no videos to filter)."""
        ds_list = [{'videos': []}, {'videos': [self.vid4_path]}]
        tgt_list = [{'videos': []}, {'videos': [self.vid4_path]}]
        op = VideoFaceRatioFilter(threshold=0.0, detect_interval=10, any_or_all='all')
        self._run_helper(op, ds_list, tgt_list)

    def test_detect_interval(self):
        """A larger detect_interval should still produce valid stats."""
        ds_list = [{'videos': [self.vid4_path]}]
        tgt_list = [{'videos': [self.vid4_path]}]
        op = VideoFaceRatioFilter(threshold=0.0, detect_interval=20, any_or_all='all')
        self._run_helper(op, ds_list, tgt_list)

    def test_invalid_strategy(self):
        """Invalid any_or_all value should raise ValueError."""
        with self.assertRaises(ValueError):
            VideoFaceRatioFilter(any_or_all='invalid')

    def test_multi_process(self):
        """Test with multiple processes."""
        import multiprocess as mp
        mp.set_start_method('forkserver', force=True)

        ds_list = [{'videos': [self.vid1_path]}, {'videos': [self.vid4_path]}]
        op = VideoFaceRatioFilter(threshold=0.0, detect_interval=10, any_or_all='all')
        dataset = Dataset.from_list(ds_list)
        if Fields.stats not in dataset.features:
            dataset = dataset.add_column(name=Fields.stats, column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.compute_stats, num_proc=2)
        dataset = dataset.filter(op.process, num_proc=2)
        dataset = dataset.remove_columns(Fields.stats)
        res_list = dataset.to_list()
        self.assertEqual(len(res_list), len(ds_list))


if __name__ == '__main__':
    unittest.main()
