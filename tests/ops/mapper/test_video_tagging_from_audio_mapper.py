import os
import unittest
from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_tagging_from_audio_mapper import \
    VideoTaggingFromAudioMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoTaggingFromAudioMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')  # Music
    vid2_path = os.path.join(data_path, 'video2.mp4')  # Music
    vid3_path = os.path.join(data_path, 'video3.mp4')  # Music
    vid4_path = os.path.join(data_path, 'video4.mp4')  # Speech
    vid5_path = os.path.join(data_path, 'video5.mp4')  # Speech
    vid3_no_aud_path = os.path.join(data_path, 'video3-no-audio.mp4')  # EMPTY

    hf_ast = 'MIT/ast-finetuned-audioset-10-10-0.4593'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_ast)

    def _run_video_tagging_from_audio_mapper(self,
                                             op,
                                             source_list,
                                             target_list,
                                             tag_field_name=MetaKeys.video_audio_tags,
                                             num_proc=1):
        dataset = Dataset.from_list(source_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc)
        res_list = dataset.flatten().select_columns([f'{Fields.meta}.{tag_field_name}'])[f'{Fields.meta}.{tag_field_name}']
        self.assertEqual(res_list, target_list)

    def test(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text': f'{SpecialTokens.video} 一个人在帮另一个人梳头发。 {SpecialTokens.eoc}',
            'videos': [self.vid4_path]
        }, {
            'text':
            f'{SpecialTokens.video} 一个穿着红色连衣裙的女人在试衣服。 {SpecialTokens.eoc}',
            'videos': [self.vid5_path]
        }]
        tgt_list = [['Music'], ['Music'], ['Speech'], ['Speech']]
        op = VideoTaggingFromAudioMapper(self.hf_ast)
        self._run_video_tagging_from_audio_mapper(op, ds_list, tgt_list)

    def test_specified_tag_field_name(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text': f'{SpecialTokens.video} 一个人在帮另一个人梳头发。 {SpecialTokens.eoc}',
            'videos': [self.vid4_path]
        }, {
            'text':
            f'{SpecialTokens.video} 一个穿着红色连衣裙的女人在试衣服。 {SpecialTokens.eoc}',
            'videos': [self.vid5_path]
        }]
        tgt_list = [['Music'], ['Music'], ['Speech'], ['Speech']]
        tag_name = 'audio_tags'
        op = VideoTaggingFromAudioMapper(self.hf_ast, tag_field_name=tag_name)
        self._run_video_tagging_from_audio_mapper(op, ds_list, tgt_list, tag_field_name=tag_name)

    def test_multi_chunk(self):
        ds_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。'
            f'{SpecialTokens.eoc}{SpecialTokens.video} '
            f'身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}{SpecialTokens.video} 一个人在帮另一个人梳头发。 '
            f'{SpecialTokens.eoc}',
            'videos': [self.vid2_path, self.vid4_path]
        }, {
            'text':
            f'一个穿着红色连衣裙的女人在试衣服。 {SpecialTokens.video} {SpecialTokens.eoc} '
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid5_path, self.vid1_path]
        }]
        tgt_list = [['Music', 'Music'], ['Music', 'Speech'],
                    ['Speech', 'Music']]
        op = VideoTaggingFromAudioMapper(self.hf_ast)
        self._run_video_tagging_from_audio_mapper(op, ds_list, tgt_list)

    def test_multi_video(self):
        ds_list = [{
            'text':
            f'{SpecialTokens.video} {SpecialTokens.video} 白色的小羊站在一旁讲话。'
            f'旁边还有两只灰色猫咪和一只拉着灰狼的猫咪; 一个人在帮另一个人梳头发 {SpecialTokens.eoc}'
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid1_path, self.vid4_path, self.vid2_path]
        }, {
            'text':
            f'一个穿着红色连衣裙的女人在试衣服。 {SpecialTokens.video} {SpecialTokens.video} '
            f'白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid5_path, self.vid1_path]
        }]
        tgt_list = [['Music', 'Speech', 'Music'], ['Speech', 'Music']]
        op = VideoTaggingFromAudioMapper(self.hf_ast)
        self._run_video_tagging_from_audio_mapper(op, ds_list, tgt_list)

    def test_no_video(self):
        ds_list = [{
            'text': '白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': []
        }, {
            'text': f'身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}'
            f'{SpecialTokens.video} 一个人在帮另一个人梳头发。 {SpecialTokens.eoc}',
            'videos': [self.vid4_path]
        }]
        tgt_list = [[], ['Speech']]
        op = VideoTaggingFromAudioMapper(self.hf_ast)
        self._run_video_tagging_from_audio_mapper(op, ds_list, tgt_list)

    def test_no_audio(self):
        ds_list = [{
            'text':
            f'{SpecialTokens.video} {SpecialTokens.video} 白色的小羊站在一旁讲话。'
            f'旁边还有两只灰色猫咪和一只拉着灰狼的猫咪; 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}'
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid1_path, self.vid3_no_aud_path, self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} {SpecialTokens.video} '
            f'两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.video} 一个人在帮另一个人梳头发。',
            'videos': [self.vid3_path, self.vid3_no_aud_path, self.vid4_path]
        }]
        tgt_list = [['Music', 'EMPTY', 'Music'], ['Music', 'EMPTY', 'Speech']]
        op = VideoTaggingFromAudioMapper(self.hf_ast)
        self._run_video_tagging_from_audio_mapper(op, ds_list, tgt_list)

    def test_multi_process(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。'
            f'{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text': f'{SpecialTokens.video} 一个人在帮另一个人梳头发。 {SpecialTokens.eoc}',
            'videos': [self.vid4_path]
        }, {
            'text':
            f'{SpecialTokens.video} 一个穿着红色连衣裙的女人在试衣服。 {SpecialTokens.eoc}',
            'videos': [self.vid5_path]
        }]
        tgt_list = [['Music'], ['Music'], ['Speech'], ['Speech']]
        op = VideoTaggingFromAudioMapper(self.hf_ast)
        self._run_video_tagging_from_audio_mapper(op,
                                                  ds_list,
                                                  tgt_list,
                                                  num_proc=2)


if __name__ == '__main__':
    unittest.main()
