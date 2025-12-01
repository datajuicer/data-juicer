import unittest

from data_juicer.core.data import NestedDataset as Dataset
from data_juicer.ops.mapper.text_chunk_mapper import TextChunkMapper
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class TextChunkMapperTest(DataJuicerTestCaseBase):

    def _run_helper(self, op, samples, target):
        dataset = Dataset.from_list(samples)
        dataset = dataset.map(op.process, batch_size=2)

        for d, t in zip(dataset, target):
            self.assertEqual(d['text'], t['text'])

    def test_naive_text_chunk(self):

        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à \n"
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        tgt_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à "
            },
            {
                'text':
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        op = TextChunkMapper(split_pattern='\n')
        self._run_helper(op, ds_list, tgt_list)
    
    def test_max_len_text_chunk(self):
        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à "
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        tgt_list = [
            {
                'text': "Today is Sunday and "
            },
            {
                'text': "it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT"
            },
            {
                'text':
                '4, plusieurs manière'
            },
            {
                'text':
                "s d'accéder à ces fo"
            },
            {
                'text':
                'nctionnalités sont c'
            },
            {
                'text':
                'onçues simultanément'
            },
            {
                'text':
                '.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        op = TextChunkMapper(max_len=20, split_pattern=None)
        self._run_helper(op, ds_list, tgt_list)
    
    def test_max_len_text_chunk_overlap_len(self):
        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à "
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        tgt_list = [
            {
                'text': "Today is Sunday and "
            },
            {
                'text': "d it's a happy day!"
            },
            {
                'text': "Sur la plateforme MT"
            },
            {
                'text': 'MT4, plusieurs maniè'
            },
            {
                'text': "ières d'accéder à ce"
            },
            {
                'text': 'ces fonctionnalités '
            },
            {
                'text': 's sont conçues simul'
            },
            {
                'text': 'ultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        op = TextChunkMapper(max_len=20, overlap_len=2)
        self._run_helper(op, ds_list, tgt_list)
    
    def test_max_len_and_split_pattern_text_chunk(self):
        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à \n"
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        tgt_list = [
            {
                'text': "Today is Sunday and "
            },
            {
                'text': "d it's a happy day!"
            },
            {
                'text': "Sur la plateforme MT"
            },
            {
                'text': 'MT4, plusieurs maniè'
            },
            {
                'text': "ières d'accéder à "
            },
            {
                'text': 'ces fonctionnalités '
            },
            {
                'text': 's sont conçues simul'
            },
            {
                'text': 'ultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        op = TextChunkMapper(
            max_len=20,
            overlap_len=2,
            split_pattern='\n'
        )
        self._run_helper(op, ds_list, tgt_list)

    def test_tokenizer_text_chunk(self):
        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à "
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        tgt_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text': "Sur la plateforme MT4, plusieurs manières"
            },
            {
                'text': "ières d'accéder à ces fonctionnalités"
            },
            {
                'text': "ités sont conçues simultanément."
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        op = TextChunkMapper(
            max_len=10,
            overlap_len=1,
            split_pattern=None,
            tokenizer='Qwen/Qwen-7B-Chat',
            trust_remote_code=True
        )
        self._run_helper(op, ds_list, tgt_list)

    def test_tiktoken_tokenizer_text_chunk(self):
        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à "
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        tgt_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text': "Sur la plateforme MT4, plusieurs manières d"
            },
            {
                'text': " d'accéder à ces fonctionnalités sont conçues simult"
            },
            {
                'text': " simultanément."
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        op = TextChunkMapper(
            max_len=10,
            overlap_len=1,
            split_pattern=None,
            tokenizer='gpt-4o',
            trust_remote_code=True
        )
        self._run_helper(op, ds_list, tgt_list)

    def test_dashscope_tokenizer_text_chunk(self):
        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à "
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        tgt_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text': "Sur la plateforme MT4, plusieurs manières"
            },
            {
                'text': "ières d'accéder à ces fonctionnalités"
            },
            {
                'text': "ités sont conçues simultanément."
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        op = TextChunkMapper(
            max_len=10,
            overlap_len=1,
            split_pattern=None,
            tokenizer='qwen2.5-72b-instruct',
            trust_remote_code=True
        )
        self._run_helper(op, ds_list, tgt_list)

    def test_all_text_chunk(self):
        ds_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text':
                "Sur la plateforme MT4, plusieurs manières d'accéder à \n"
                'ces fonctionnalités sont conçues simultanément.'
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        tgt_list = [
            {
                'text': "Today is Sunday and it's a happy day!"
            },
            {
                'text': "Sur la plateforme MT4, plusieurs manières"
            },
            {
                'text': "ières d'accéder à "
            },
            {
                'text': "ces fonctionnalités sont conçues simultan"
            },
            {
                'text': "anément."
            },
            {
                'text': '欢迎来到阿里巴巴！'
            },
        ]
        op = TextChunkMapper(
            max_len=10,
            overlap_len=1,
            split_pattern='\n',
            tokenizer='Qwen/Qwen-7B-Chat',
            trust_remote_code=True
        )
        self._run_helper(op, ds_list, tgt_list)


if __name__ == '__main__':
    unittest.main()
