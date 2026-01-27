import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_captioning_from_frames_mapper import \
    VideoCaptioningFromFramesMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class VideoCaptioningFromFramesMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')

    img1_path = os.path.join(data_path, 'img6.jpg')
    img2_path = os.path.join(data_path, 'img7.jpg')
    img3_path = os.path.join(data_path, 'img8.jpg')

    hf_img2seq = 'Salesforce/blip2-opt-2.7b'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_img2seq)

    def _run_mapper(self, ds_list, op, num_proc=1, caption_num=0):

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        dataset_list = dataset.select_columns(column_names=['text']).to_list()
        # assert the caption is generated successfully in terms of not_none
        # as the generated content is not deterministic
        self.assertEqual(len(dataset_list), caption_num)

    def test_default_params_no_eoc(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq)
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_default_params_with_eoc(self):

        ds_list = [
            {
                'text':
                f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪'
                f'{SpecialTokens.eoc}',
                'videos': [self.vid1_path]
            },
            {
                'text':
                f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃{SpecialTokens.eoc}',  # noqa: E501
                'videos': [self.vid2_path]
            }
        ]
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq)
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_multi_candidate_keep_random_any(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        caption_num = 4
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq,
                                             caption_num=caption_num,
                                             keep_candidate_mode='random_any')
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_multi_candidate_keep_all(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        caption_num = 4
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq,
                                             caption_num=caption_num,
                                             keep_candidate_mode='all')
        self._run_mapper(ds_list,
                         op,
                         caption_num=(1 + caption_num) * len(ds_list))

    def test_multi_candidate_keep_similar_one(self):
        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        caption_num = 4
        op = VideoCaptioningFromFramesMapper(
            hf_img2seq=self.hf_img2seq,
            caption_num=caption_num,
            keep_candidate_mode='similar_one_simhash')
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_remove_original_sample(self):

        ds_list = [
            {
                'text':
                f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
                'videos': [self.vid1_path]
            },
            {
                'text':
                f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',  # noqa: E501
                'videos': [self.vid2_path]
            }
        ]
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq,
                                             keep_original_sample=False)
        self._run_mapper(ds_list, op, caption_num=len(ds_list))

    def test_multi_candidate_remove_original_sample(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        caption_num = 4
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq,
                                             caption_num=caption_num,
                                             keep_original_sample=False)
        self._run_mapper(ds_list, op, caption_num=len(ds_list))

    def test_multi_process(self):
        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }] * 10
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq)
        self._run_mapper(ds_list, op, num_proc=2, caption_num=len(ds_list) * 2)

    def test_multi_process_remove_original_sample(self):
        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }] * 10

        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq,
                                             keep_original_sample=False)
        self._run_mapper(ds_list, op, num_proc=2, caption_num=len(ds_list))

    def test_frame_sampling_method(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq,
                                             frame_sampling_method='uniform')
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_frame_num(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq,
                                             frame_sampling_method='uniform',
                                             frame_num=5)
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_horizontal_flip(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq,
                                             horizontal_flip=True)
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_vertical_flip(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq,
                                             vertical_flip=True)
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_multi_tag(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}{SpecialTokens.video}'
            '白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [
                self.vid1_path,
                self.vid1_path,
            ]
        }]
        op = VideoCaptioningFromFramesMapper(hf_img2seq=self.hf_img2seq)
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_text_update_strategy_keep_origin(self):
        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。',
            'videos': [self.vid1_path]
        }]
        op = VideoCaptioningFromFramesMapper(
            hf_img2seq=self.hf_img2seq,
            legacy_split_by_text_token=False,
            text_update_strategy='keep_origin',
            caption_field='generated_captions',
            keep_original_sample=False,
            caption_num=2
        )
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        result = dataset.to_list()

        self.assertEqual(result[0]['text'], ds_list[0]['text'])
        self.assertIn('generated_captions', result[0])
        self.assertEqual(len(result), 1)

    def test_text_update_strategy_rewrite(self):
        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃。',
            'videos': [self.vid1_path, self.vid2_path]
        },
        ]
        op = VideoCaptioningFromFramesMapper(
            hf_img2seq=self.hf_img2seq,
            text_update_strategy='rewrite',
            legacy_split_by_text_token=False,
            keep_original_sample=False,
        )
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        result = dataset.to_list()

        self.assertNotEqual(result[0]['text'], ds_list[0]['text'])
        self.assertTrue(result[0]['text'].startswith(SpecialTokens.video))
        self.assertTrue(result[0]['text'].endswith(SpecialTokens.eoc))
        self.assertEqual(result[0]['text'].count(SpecialTokens.video), 2)

    def test_caption_field(self):
        ds_list = [
            {
                'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。',
                'videos': [self.vid1_path]
            },
            {
                'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃。',
                'videos': [self.vid1_path, self.vid2_path]
            }
        ]
        op = VideoCaptioningFromFramesMapper(
            hf_img2seq=self.hf_img2seq,
            text_update_strategy='keep_origin',
            caption_field='custom_caption_field',
            keep_original_sample=False,
            legacy_split_by_text_token=False,
        )
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        result = dataset.to_list()

        self.assertEqual(len(result), len(ds_list))
        for i in range(len(ds_list)):
            self.assertEqual(result[i]['text'], ds_list[i]['text'])
            self.assertEqual(len(result[i]['custom_caption_field']), i+1)

    def test_frame_field(self):
        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。',
            'videos': [self.vid1_path, self.vid2_path],
            'frames': [[self.img1_path, self.img2_path, self.img3_path], [self.img1_path, self.img2_path]]
        }]
        op = VideoCaptioningFromFramesMapper(
            hf_img2seq=self.hf_img2seq,
            text_update_strategy='keep_origin',
            caption_field='custom_caption_field',
            frame_field='frames',
            legacy_split_by_text_token=False,
            keep_original_sample=False,
            frame_num=3,
        )
        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=1, with_rank=True)
        result = dataset.to_list()

        self.assertEqual(result[0]['text'], ds_list[0]['text'])
        self.assertEqual(len(result[0]['custom_caption_field']), 2)


if __name__ == '__main__':
    unittest.main()
