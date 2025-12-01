import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_captioning_from_audio_mapper import \
    VideoCaptioningFromAudioMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoCaptioningFromAudioMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')

    @staticmethod
    def _count_generated_caption_num(text):
        chunks = text.split(SpecialTokens.eoc)
        vid_num = 0
        cap_num = 0
        for chunk in chunks:
            if chunk.strip() == '':
                continue
            vid_num += chunk.count(SpecialTokens.video)
            caps = [
                cap for cap in chunk.split(SpecialTokens.video) if cap.strip()
            ]
            cap_num += len(caps)
        return vid_num, cap_num

    def _run_op(self, dataset: Dataset, caption_num, op, np=1):
        dataset = dataset.map(op.process, num_proc=np)
        text_list = dataset.select_columns(column_names=['text']).to_list()
        for txt in text_list:
            vid_num, cap_num = self._count_generated_caption_num(txt['text'])
            self.assertEqual(vid_num, cap_num)
        self.assertEqual(len(dataset), caption_num)

    def test_default_params(self):

        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid2_path]
        }, {
            'text': f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。',
            'videos': [self.vid3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoCaptioningFromAudioMapper()
        self._run_op(dataset, len(dataset) * 2, op)

    def test_with_eoc(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。'
            f'{SpecialTokens.eoc}',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。 '
            f'{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoCaptioningFromAudioMapper()
        self._run_op(dataset, len(dataset) * 2, op)

    def test_no_original_samples(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。'
            f'{SpecialTokens.eoc}',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。 '
            f'{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoCaptioningFromAudioMapper(keep_original_sample=False)
        self._run_op(dataset, len(dataset), op)

    def test_multi_chunk_samples(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。'
            f'{SpecialTokens.eoc} {SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，'
            f'拍打自己的胃部。 {SpecialTokens.eoc}',
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc} '
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid3_path, self.vid1_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoCaptioningFromAudioMapper()
        self._run_op(dataset, len(dataset) * 2, op)

    def test_multi_video_samples(self):

        ds_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。 '
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。 '
            f'{SpecialTokens.eoc}',
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'text':
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} {SpecialTokens.eoc} '
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。 '
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。 '
            f'{SpecialTokens.eoc} {SpecialTokens.video} 白色的小羊站在一旁讲话。'
            f'旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos':
            [self.vid3_path, self.vid1_path, self.vid2_path, self.vid1_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoCaptioningFromAudioMapper()
        self._run_op(dataset, len(dataset) * 2, op)

    def test_parallel(self):

        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid2_path]
        }, {
            'text': f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。',
            'videos': [self.vid3_path]
        }]
        dataset = Dataset.from_list(ds_list)
        op = VideoCaptioningFromAudioMapper()
        self._run_op(dataset, len(dataset) * 2, op, np=2)


if __name__ == '__main__':
    unittest.main()
