import os
import unittest
import jsonlines as jl
from datasets import Dataset
from data_juicer.core import Tracer
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase

class TracerTest(DataJuicerTestCaseBase):

    def setUp(self) -> None:
        super().setUp()
        self.work_dir = 'tmp/test_tracer/'
        os.makedirs(self.work_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.work_dir):
            os.system(f'rm -rf {self.work_dir}')
        super().tearDown()

    def test_trace_mapper(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'processed text 2'},
            {'text': 'text 3'},
        ])
        dif_list = [
            {
                'original text': 'text 2',
                'processed_text': 'processed text 2',
            }
        ]
        tracer = Tracer(self.work_dir)
        tracer.trace_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_mapper_less_show_num(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'processed text 2'},
            {'text': 'processed text 3'},
        ])
        dif_list = [
            {
                'original text': 'text 2',
                'processed_text': 'processed text 2',
            }
        ]
        tracer = Tracer(self.work_dir, show_num=1)
        tracer.trace_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_mapper_same(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        tracer = Tracer(self.work_dir)
        tracer.trace_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    def test_trace_batched_mapper(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'augmented text 1-1'},
            {'text': 'augmented text 1-2'},
            {'text': 'text 2'},
            {'text': 'augmented text 2-1'},
            {'text': 'augmented text 2-2'},
            {'text': 'text 3'},
            {'text': 'augmented text 3-1'},
            {'text': 'augmented text 3-2'},
        ])
        dif_list = [
            {'text': 'text 1'},
            {'text': 'augmented text 1-1'},
            {'text': 'augmented text 1-2'},
            {'text': 'text 2'},
            {'text': 'augmented text 2-1'},
            {'text': 'augmented text 2-2'},
            {'text': 'text 3'},
            {'text': 'augmented text 3-1'},
            {'text': 'augmented text 3-2'},
        ]
        tracer = Tracer(self.work_dir)
        tracer.trace_batch_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_batched_mapper_less_show_num(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'augmented text 1-1'},
            {'text': 'augmented text 1-2'},
            {'text': 'text 2'},
            {'text': 'augmented text 2-1'},
            {'text': 'augmented text 2-2'},
            {'text': 'text 3'},
            {'text': 'augmented text 3-1'},
            {'text': 'augmented text 3-2'},
        ])
        dif_list = [
            {'text': 'text 1'},
            {'text': 'augmented text 1-1'},
            {'text': 'augmented text 1-2'},
            {'text': 'text 2'},
        ]
        tracer = Tracer(self.work_dir, show_num=4)
        tracer.trace_batch_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_filter(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 3'},
        ])
        dif_list = [
            {'text': 'text 2'},
        ]
        tracer = Tracer(self.work_dir)
        tracer.trace_filter('alphanumeric_filter', prev_ds, done_ds)
        trace_file_path = os.path.join(self.work_dir, 'trace', 'filter-alphanumeric_filter.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_filter_less_show_num(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1'},
        ])
        dif_list = [
            {'text': 'text 2'},
        ]
        tracer = Tracer(self.work_dir, show_num=1)
        tracer.trace_filter('alphanumeric_filter', prev_ds, done_ds)
        trace_file_path = os.path.join(self.work_dir, 'trace', 'filter-alphanumeric_filter.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_filter_same(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        tracer = Tracer(self.work_dir)
        tracer.trace_filter('alphanumeric_filter', prev_ds, done_ds)
        trace_file_path = os.path.join(self.work_dir, 'trace', 'filter-alphanumeric_filter.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    def test_trace_filter_empty(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([])
        dif_list = [
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ]
        tracer = Tracer(self.work_dir)
        tracer.trace_filter('alphanumeric_filter', prev_ds, done_ds)
        trace_file_path = os.path.join(self.work_dir, 'trace', 'filter-alphanumeric_filter.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_deduplicator(self):
        dup_pairs = {
            'hash1': ['text 1', 'text 1'],
            'hash2': ['text 2', 'text 2'],
            'hash3': ['text 3', 'text 3-1'],
        }
        dif_list = [
            {'dup1': 'text 1', 'dup2': 'text 1'},
            {'dup1': 'text 2', 'dup2': 'text 2'},
            {'dup1': 'text 3', 'dup2': 'text 3-1'},
        ]
        tracer = Tracer(self.work_dir)
        tracer.trace_deduplicator('document_deduplicator', dup_pairs)
        trace_file_path = os.path.join(self.work_dir, 'trace', 'duplicate-document_deduplicator.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_deduplicator_None(self):
        tracer = Tracer(self.work_dir)
        tracer.trace_deduplicator('document_deduplicator', None)
        trace_file_path = os.path.join(self.work_dir, 'trace', 'duplicate-document_deduplicator.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    def test_trace_deduplicator_empty(self):
        tracer = Tracer(self.work_dir)
        tracer.trace_deduplicator('document_deduplicator', {})
        trace_file_path = os.path.join(self.work_dir, 'trace', 'duplicate-document_deduplicator.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    def test_op_list_to_trace(self):
        prev_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'text 2'},
            {'text': 'text 3'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1'},
            {'text': 'processed text 2'},
            {'text': 'text 3'},
        ])
        tracer = Tracer(self.work_dir, op_list_to_trace=['non_existing_mapper'])
        tracer.trace_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))


if __name__ == '__main__':
    unittest.main()
