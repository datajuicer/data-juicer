# flake8: noqa: E501
import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_tagging_from_frames_mapper import \
    VideoTaggingFromFramesMapper
from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class VideoTaggingFromFramesMapperTest(DataJuicerTestCaseBase):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    vid3_path = os.path.join(data_path, 'video3.mp4')

    def _run_video_tagging_from_frames_mapper(self,
                                              op,
                                              source_list,
                                              target_list,
                                              num_proc=1):
        dataset = Dataset.from_list(source_list)
        if Fields.meta not in dataset.features:
            dataset = dataset.add_column(name=Fields.meta,
                                         column=[{}] * dataset.num_rows)
        dataset = dataset.map(op.process, num_proc=num_proc)
        res_list = dataset.to_list()
        self.assertEqual(res_list, target_list)

    def test(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'animal', 'ray', 'text', 'writing', 'yellow', 'game',
                    'screenshot', 'cartoon', 'cartoon character', 'person', 'robe',
                    'sky'
                ]]}
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'man', 'shirt', 't shirt', 't-shirt', 'wear', 'white', 'boy',
                    'catch', 'hand', 'blind', 'cotton candy', 'tennis racket',
                    'ball', 'person'
                ]]}
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'woman', 'table', 'sit', 'person', 'laptop', 'bookshelf',
                    'conversation', 'round table', 'closet', 'computer', 'girl',
                    'man', 'stool', 'computer screen', 'laugh', 'cabinet', 'hand',
                    'selfie', 'stand'
                ]]}
        }]
        op = VideoTaggingFromFramesMapper()
        self._run_video_tagging_from_frames_mapper(op, ds_list, tgt_list)

    def test_no_video(self):
        ds_list = [{
            'text': f'白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': []
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }]
        tgt_list = [{
            'text':
            f'白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[]]}
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'man', 'shirt', 't shirt', 't-shirt', 'wear', 'white', 'boy',
                    'catch', 'hand', 'blind', 'cotton candy', 'tennis racket',
                    'ball', 'person'
                ]]}
        }]
        op = VideoTaggingFromFramesMapper()
        self._run_video_tagging_from_frames_mapper(op, ds_list, tgt_list)

    def test_specified_tag_field_name(self):
        tag_field_name = 'my_tags'

        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path],
            Fields.meta: {
                tag_field_name: [[
                    'animal', 'ray', 'text', 'writing', 'yellow', 'game',
                    'screenshot', 'cartoon', 'cartoon character', 'person', 'robe',
                    'sky'
                ]]}
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path],
            Fields.meta: {
                tag_field_name: [[
                    'man', 'shirt', 't shirt', 't-shirt', 'wear', 'white', 'boy',
                    'catch', 'hand', 'blind', 'cotton candy', 'tennis racket',
                    'ball', 'person'
                ]]}
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path],
            Fields.meta: {
                tag_field_name: [[
                    'woman', 'table', 'sit', 'person', 'laptop', 'bookshelf',
                    'conversation', 'round table', 'closet', 'computer', 'girl',
                    'man', 'stool', 'computer screen', 'laugh', 'cabinet', 'hand',
                    'selfie', 'stand'
                ]]}
        }]
        op = VideoTaggingFromFramesMapper(tag_field_name=tag_field_name)
        self._run_video_tagging_from_frames_mapper(op, ds_list, tgt_list)

    def test_uniform(self):
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'cartoon', 'animal', 'anime', 'game', 'screenshot',
                    'video game', 'cartoon character', 'robe', 'ray', 'text',
                    'writing', 'yellow', 'doll', 'tail', 'sky', 'person']]}
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'man', 'shirt', 't shirt', 't-shirt', 'wear', 'white', 'boy',
                    'hand', 'catch', 'bulletin board', 'Wii', 'cotton candy',
                    'tennis racket', 'blind', 'game controller', 'remote', 'stand',
                    'video game', 'Wii controller', 'play', 'baseball uniform',
                    'toy', 'green']]}
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'table', 'sit', 'woman', 'bookshelf', 'conversation', 'person',
                    'round table', 'computer', 'girl', 'man', 'closet', 'laptop',
                    'stand', 'computer screen', 'talk', 'room', 'stool', 'hand',
                    'point'
                ]]}
        }]
        op = VideoTaggingFromFramesMapper(frame_sampling_method='uniform',
                                          frame_num=10)
        self._run_video_tagging_from_frames_mapper(op, ds_list, tgt_list)

    def test_multi_process(self):
        # WARNING: current parallel tests only work in spawn method
        import multiprocess
        original_method = multiprocess.get_start_method()
        multiprocess.set_start_method('spawn', force=True)
        # WARNING: current parallel tests only work in spawn method
        ds_list = [{
            'text': f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path]
        }]
        tgt_list = [{
            'text':
                f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。',
            'videos': [self.vid1_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'animal', 'ray', 'text', 'writing', 'yellow', 'game',
                    'screenshot', 'cartoon', 'cartoon character', 'person', 'robe',
                    'sky'
                ]]}
        }, {
            'text':
                f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}',
            'videos': [self.vid2_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'man', 'shirt', 't shirt', 't-shirt', 'wear', 'white', 'boy',
                    'catch', 'hand', 'blind', 'cotton candy', 'tennis racket',
                    'ball', 'person'
                ]]}
        }, {
            'text':
                f'{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid3_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'woman', 'table', 'sit', 'person', 'laptop', 'bookshelf',
                    'conversation', 'round table', 'closet', 'computer', 'girl',
                    'man', 'stool', 'computer screen', 'laugh', 'cabinet', 'hand',
                    'selfie', 'stand'
                ]]}
        }]
        op = VideoTaggingFromFramesMapper()
        self._run_video_tagging_from_frames_mapper(op,
                                                   ds_list,
                                                   tgt_list,
                                                   num_proc=2)
        # WARNING: current parallel tests only work in spawn method
        multiprocess.set_start_method(original_method, force=True)
        # WARNING: current parallel tests only work in spawn method

    def test_multi_chunk(self):
        ds_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid1_path, self.vid2_path]
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid2_path, self.vid3_path]
        }, {
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid1_path, self.vid3_path]
        }]
        tgt_list = [{
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。',
            'videos': [self.vid1_path, self.vid2_path],
            Fields.meta: {
                MetaKeys.video_frame_tags:
                [[
                    'animal', 'ray', 'text', 'writing', 'yellow', 'game',
                    'screenshot', 'cartoon', 'cartoon character', 'person', 'robe',
                    'sky'
                ], [
                    'man', 'shirt', 't shirt', 't-shirt', 'wear', 'white', 'boy',
                    'catch', 'hand', 'blind', 'cotton candy', 'tennis racket',
                    'ball', 'person'
                ]]}
        }, {
            'text':
            f'{SpecialTokens.video} 身穿白色上衣的男子，拿着一个东西，拍打自己的胃部。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid2_path, self.vid3_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'man', 'shirt', 't shirt', 't-shirt', 'wear', 'white', 'boy',
                    'catch', 'hand', 'blind', 'cotton candy', 'tennis racket',
                    'ball', 'person'
                ], [
                    'woman', 'table', 'sit', 'person', 'laptop', 'bookshelf',
                    'conversation', 'round table', 'closet', 'computer', 'girl',
                    'man', 'stool', 'computer screen', 'laugh', 'cabinet', 'hand',
                    'selfie', 'stand'
                ]]}
        }, {
            'text':
            f'{SpecialTokens.video} 白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪。{SpecialTokens.eoc}{SpecialTokens.video} 两个长头发的女子正坐在一张圆桌前讲话互动。 {SpecialTokens.eoc}',
            'videos': [self.vid1_path, self.vid3_path],
            Fields.meta: {
                MetaKeys.video_frame_tags: [[
                    'animal', 'ray', 'text', 'writing', 'yellow', 'game',
                    'screenshot', 'cartoon', 'cartoon character', 'person', 'robe',
                    'sky'
                ], [
                    'woman', 'table', 'sit', 'person', 'laptop', 'bookshelf',
                    'conversation', 'round table', 'closet', 'computer', 'girl',
                    'man', 'stool', 'computer screen', 'laugh', 'cabinet', 'hand',
                    'selfie', 'stand'
                ]]}
        }]
        op = VideoTaggingFromFramesMapper()
        self._run_video_tagging_from_frames_mapper(op, ds_list, tgt_list)


if __name__ == '__main__':
    unittest.main()
