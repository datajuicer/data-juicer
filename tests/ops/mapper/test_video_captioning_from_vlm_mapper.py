import os
import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.video_captioning_from_vlm_mapper import \
    VideoCaptioningFromVLMMapper
from data_juicer.utils.mm_utils import SpecialTokens
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

@unittest.skip('Possible OOM. Requires about 70GB GPU memory, but only 44GB provided.')
class VideoCaptioningFromVLMMapperTest(DataJuicerTestCaseBase):

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..',
                             'data')
    vid1_path = os.path.join(data_path, 'video1.mp4')
    vid2_path = os.path.join(data_path, 'video2.mp4')
    hf_model = 'Qwen/Qwen3-VL-8B-Instruct'

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass(cls.hf_model)

    def _run_mapper(self, ds_list, op, num_proc=1, caption_num=0):

        dataset = Dataset.from_list(ds_list)
        dataset = dataset.map(op.process, num_proc=num_proc, with_rank=True)
        dataset_list = dataset.select_columns(column_names=['text']).to_list()
        # assert the caption is generated successfully in terms of not_none
        # as the generated content is not deterministic
        self.assertEqual(len(dataset_list), caption_num)

    def test_default_params(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        op = VideoCaptioningFromVLMMapper(hf_model=self.hf_model)
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_enable_vllm(self):

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
        op = VideoCaptioningFromVLMMapper(hf_model=self.hf_model, enable_vllm=True)
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)

    def test_multi_candidate_keep_similar_one(self):
        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }, {
            'text': f'{SpecialTokens.video}身穿白色上衣的男子，拿着一个东西，拍打自己的胃',
            'videos': [self.vid2_path]
        }]
        caption_num = 4
        op = VideoCaptioningFromVLMMapper(
            hf_model=self.hf_model,
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
        op = VideoCaptioningFromVLMMapper(hf_model=self.hf_model,
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
        op = VideoCaptioningFromVLMMapper(hf_model=self.hf_model,
                                          caption_num=caption_num,
                                          keep_original_sample=False)
        self._run_mapper(ds_list, op, caption_num=len(ds_list))

    def test_multi_process(self):
        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }] * 10
        op = VideoCaptioningFromVLMMapper(hf_model=self.hf_model)
        self._run_mapper(ds_list, op, num_proc=2, caption_num=len(ds_list) * 2)

    def test_multi_process_remove_original_sample(self):
        ds_list = [{
            'text': f'{SpecialTokens.video}白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [self.vid1_path]
        }] * 10

        op = VideoCaptioningFromVLMMapper(hf_model=self.hf_model,
                                          keep_original_sample=False)
        self._run_mapper(ds_list, op, num_proc=2, caption_num=len(ds_list))

    def test_multi_tag(self):

        ds_list = [{
            'text': f'{SpecialTokens.video}{SpecialTokens.video}'
            '白色的小羊站在一旁讲话。旁边还有两只灰色猫咪和一只拉着灰狼的猫咪',
            'videos': [
                self.vid1_path,
                self.vid1_path,
            ]
        }]
        op = VideoCaptioningFromVLMMapper(hf_model=self.hf_model)
        self._run_mapper(ds_list, op, caption_num=len(ds_list) * 2)


if __name__ == '__main__':
    unittest.main()
