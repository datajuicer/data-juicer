import os
import unittest
from unittest.mock import Mock, patch, sentinel

from transformers import BatchEncoding

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


class VideoCaptioningFromVLMMapperRegressionTest(unittest.TestCase):

    def _make_mapper(self):
        op = object.__new__(VideoCaptioningFromVLMMapper)
        op.text_key = 'text'
        op.video_key = 'videos'
        op.keep_original_sample = False
        return op

    def test_hf_generation_uses_configured_token_limit(self):
        op = self._make_mapper()
        op.accelerator = 'cpu'
        op.num_newly_generated_samples = 1
        op.model_key = sentinel.model_key
        op.caption_num = 1
        op.prompt_key = None
        op.prompt = None
        op.system_prompt = None
        op.enable_vllm = False
        op.keep_candidate_mode = 'random_any'
        op.sampling_params = {'do_sample': False, 'max_new_tokens': 64}
        sample = {'text': SpecialTokens.video, 'videos': ['video.mp4']}
        inputs = BatchEncoding({'input_ids': [[1, 2]]})
        model = Mock(device='cpu')
        model.generate.return_value = [[1, 2, 3]]
        processor = Mock()
        processor.apply_chat_template.return_value = inputs
        processor.batch_decode.return_value = ['generated caption']
        module = 'data_juicer.ops.mapper.video_captioning_from_vlm_mapper'

        with patch(f'{module}.get_model', return_value=(model, processor)), \
                patch(f'{module}.load_data_with_context',
                      return_value=(sample, {'video.mp4': sentinel.video})), \
                patch(f'{module}.close_video') as close_video:
            output = op._process_single_sample(sample)

        model.generate.assert_called_once_with(input_ids=[[1, 2]],
                                               do_sample=False,
                                               max_new_tokens=64)
        self.assertIn('generated caption', output[0]['text'])
        close_video.assert_called_once_with(sentinel.video)

    def test_batch_preserves_generated_samples(self):
        sample = {'text': 'original', 'videos': ['video.mp4'], 'id': 7}
        generated = {**sample, 'text': 'generated'}
        samples = {key: [value] for key, value in sample.items()}
        for keep_original in (False, True):
            with self.subTest(keep_original=keep_original):
                op = self._make_mapper()
                op.keep_original_sample = keep_original
                op._process_single_sample = Mock(return_value=[generated])
                expected = [sample, generated] if keep_original else [generated]
                self.assertEqual(
                    op.process_batched(samples),
                    {key: [row[key] for row in expected] for key in sample})

    def test_empty_generation_returns_empty_batch(self):
        for samples in ({'text': [], 'videos': []},
                        {'text': ['no video'], 'videos': [[]]}):
            with self.subTest(samples=samples):
                self.assertEqual(self._make_mapper().process_batched(samples), {})


if __name__ == '__main__':
    unittest.main()
