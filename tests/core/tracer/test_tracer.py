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
                'original_text': 'text 2',
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
                'original_text': 'text 2',
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

    def test_trace_mapper_with_trace_keys_single(self):
        """Test that trace_keys includes specified fields in trace output."""
        prev_ds = Dataset.from_list([
            {'text': 'text 1', 'sample_id': 'id-001'},
            {'text': 'text 2', 'sample_id': 'id-002'},
            {'text': 'text 3', 'sample_id': 'id-003'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1', 'sample_id': 'id-001'},
            {'text': 'processed text 2', 'sample_id': 'id-002'},
            {'text': 'text 3', 'sample_id': 'id-003'},
        ])
        dif_list = [
            {
                'original_text': 'text 2',
                'processed_text': 'processed text 2',
                'sample_id': 'id-002',
            }
        ]
        tracer = Tracer(self.work_dir, trace_keys=['sample_id'])
        tracer.trace_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_mapper_with_trace_keys_multiple(self):
        """Test that trace_keys includes multiple fields in trace output."""
        prev_ds = Dataset.from_list([
            {'text': 'text 1', 'sample_id': 'id-001', 'source': 'file1.jsonl'},
            {'text': 'text 2', 'sample_id': 'id-002', 'source': 'file1.jsonl'},
            {'text': 'text 3', 'sample_id': 'id-003', 'source': 'file2.jsonl'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1', 'sample_id': 'id-001', 'source': 'file1.jsonl'},
            {'text': 'processed text 2', 'sample_id': 'id-002', 'source': 'file1.jsonl'},
            {'text': 'text 3', 'sample_id': 'id-003', 'source': 'file2.jsonl'},
        ])
        dif_list = [
            {
                'original_text': 'text 2',
                'processed_text': 'processed text 2',
                'sample_id': 'id-002',
                'source': 'file1.jsonl',
            }
        ]
        tracer = Tracer(self.work_dir, trace_keys=['sample_id', 'source'])
        tracer.trace_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_mapper_with_trace_keys_missing_field(self):
        """Test that trace_keys handles missing field gracefully."""
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
                'original_text': 'text 2',
                'processed_text': 'processed text 2',
                'sample_id': None,
            }
        ]
        tracer = Tracer(self.work_dir, trace_keys=['sample_id'])
        tracer.trace_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_trace_mapper_without_trace_keys(self):
        """Test that without trace_keys, output is unchanged (default behavior)."""
        prev_ds = Dataset.from_list([
            {'text': 'text 1', 'sample_id': 'id-001'},
            {'text': 'text 2', 'sample_id': 'id-002'},
            {'text': 'text 3', 'sample_id': 'id-003'},
        ])
        done_ds = Dataset.from_list([
            {'text': 'text 1', 'sample_id': 'id-001'},
            {'text': 'processed text 2', 'sample_id': 'id-002'},
            {'text': 'text 3', 'sample_id': 'id-003'},
        ])
        # Without trace_keys, only original_text and processed_text are included
        dif_list = [
            {
                'original_text': 'text 2',
                'processed_text': 'processed text 2',
            }
        ]
        tracer = Tracer(self.work_dir)  # No trace_keys
        tracer.trace_mapper('clean_email_mapper', prev_ds, done_ds, 'text')
        trace_file_path = os.path.join(self.work_dir, 'trace', 'mapper-clean_email_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        self.assertEqual(dif_list, trace_records)

    def test_collect_mapper_sample_basic(self):
        """Test basic functionality of collect_mapper_sample method."""
        tracer = Tracer(self.work_dir, op_list_to_trace=['test_mapper'])
        
        original_sample = {'text': 'original text'}
        processed_sample = {'text': 'processed text'}
        
        # Should collect the sample since text differs
        result = tracer.collect_mapper_sample('test_mapper', original_sample, processed_sample, 'text')
        self.assertTrue(result)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        
        expected = [{
            'original_text': 'original text',
            'processed_text': 'processed text'
        }]
        self.assertEqual(expected, trace_records)

    def test_collect_mapper_sample_no_change(self):
        """Test collect_mapper_sample when original and processed texts are the same."""
        tracer = Tracer(self.work_dir)
        
        original_sample = {'text': 'same text'}
        processed_sample = {'text': 'same text'}
        
        # Should not collect the sample since text is the same
        result = tracer.collect_mapper_sample('test_mapper', original_sample, processed_sample, 'text')
        self.assertFalse(result)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_mapper.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    def test_collect_mapper_sample_with_trace_keys(self):
        """Test collect_mapper_sample with trace_keys functionality."""
        tracer = Tracer(self.work_dir, trace_keys=['sample_id', 'source'], op_list_to_trace=['test_mapper'])
        
        original_sample = {'text': 'original text', 'sample_id': 'id-123', 'source': 'source_file.txt'}
        processed_sample = {'text': 'processed text', 'sample_id': 'id-123', 'source': 'source_file.txt'}
        
        result = tracer.collect_mapper_sample('test_mapper', original_sample, processed_sample, 'text')
        self.assertTrue(result)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        
        expected = [{
            'sample_id': 'id-123',
            'source': 'source_file.txt',
            'original_text': 'original text',
            'processed_text': 'processed text'
        }]
        self.assertEqual(expected, trace_records)

    def test_collect_mapper_sample_with_missing_trace_keys(self):
        """Test collect_mapper_sample when trace keys are missing from sample."""
        tracer = Tracer(self.work_dir, trace_keys=['missing_key', 'existing_key'], op_list_to_trace=['test_mapper'])
        
        original_sample = {'text': 'original text', 'existing_key': 'value'}
        processed_sample = {'text': 'processed text', 'existing_key': 'value'}
        
        result = tracer.collect_mapper_sample('test_mapper', original_sample, processed_sample, 'text')
        self.assertTrue(result)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        
        expected = [{
            'missing_key': None,
            'existing_key': 'value',
            'original_text': 'original text',
            'processed_text': 'processed text'
        }]
        self.assertEqual(expected, trace_records)

    def test_collect_mapper_sample_not_in_op_list(self):
        """Test collect_mapper_sample when op is not in the trace list."""
        tracer = Tracer(self.work_dir, op_list_to_trace=['other_mapper'])
        
        original_sample = {'text': 'original text'}
        processed_sample = {'text': 'processed text'}
        
        # Should not collect since op is not in the trace list
        result = tracer.collect_mapper_sample('test_mapper', original_sample, processed_sample, 'text')
        self.assertFalse(result)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_mapper.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    def test_collect_filter_sample_basic(self):
        """Test basic functionality of collect_filter_sample method."""
        tracer = Tracer(self.work_dir, op_list_to_trace=['test_filter'])
        
        sample = {'text': 'filtered text', 'sample_id': 'id-456'}
        
        # Should collect the sample since it should be filtered out (keep=False)
        result = tracer.collect_filter_sample('test_filter', sample, False)
        self.assertTrue(result)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_filter.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        
        expected = [{'text': 'filtered text', 'sample_id': 'id-456'}]
        self.assertEqual(expected, trace_records)

    def test_collect_filter_sample_should_keep(self):
        """Test collect_filter_sample when sample should be kept."""
        tracer = Tracer(self.work_dir)
        
        sample = {'text': 'kept text'}
        
        # Should not collect the sample since it should be kept (keep=True)
        result = tracer.collect_filter_sample('test_filter', sample, True)
        self.assertFalse(result)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_filter.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    def test_collect_filter_sample_not_in_op_list(self):
        """Test collect_filter_sample when op is not in the trace list."""
        tracer = Tracer(self.work_dir, op_list_to_trace=['other_filter'])
        
        sample = {'text': 'filtered text'}
        
        # Should not collect since op is not in the trace list
        result = tracer.collect_filter_sample('test_filter', sample, False)
        self.assertFalse(result)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_filter.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    def test_collect_mapper_sample_show_num_limit(self):
        """Test collect_mapper_sample respects show_num limit."""
        tracer = Tracer(self.work_dir, show_num=2, op_list_to_trace=['limited_mapper'])
        
        # First sample should be collected
        original1 = {'text': 'original text 1'}
        processed1 = {'text': 'processed text 1'}
        result1 = tracer.collect_mapper_sample('limited_mapper', original1, processed1, 'text')
        self.assertTrue(result1)
        
        # Second sample should be collected
        original2 = {'text': 'original text 2'}
        processed2 = {'text': 'processed text 2'}
        result2 = tracer.collect_mapper_sample('limited_mapper', original2, processed2, 'text')
        self.assertTrue(result2)
        
        # Third sample should not be collected due to limit
        original3 = {'text': 'original text 3'}
        processed3 = {'text': 'processed text 3'}
        result3 = tracer.collect_mapper_sample('limited_mapper', original3, processed3, 'text')
        self.assertFalse(result3)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-limited_mapper.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        
        # Should only have 2 records despite 3 attempts
        self.assertEqual(2, len(trace_records))
        expected = [
            {'original_text': 'original text 1', 'processed_text': 'processed text 1'},
            {'original_text': 'original text 2', 'processed_text': 'processed text 2'}
        ]
        self.assertEqual(expected, trace_records)

    def test_collect_filter_sample_show_num_limit(self):
        """Test collect_filter_sample respects show_num limit."""
        tracer = Tracer(self.work_dir, show_num=1, op_list_to_trace=['limited_filter'])
        
        # First sample should be collected
        sample1 = {'text': 'filtered text 1'}
        result1 = tracer.collect_filter_sample('limited_filter', sample1, False)
        self.assertTrue(result1)
        
        # Second sample should not be collected due to limit
        sample2 = {'text': 'filtered text 2'}
        result2 = tracer.collect_filter_sample('limited_filter', sample2, False)
        self.assertFalse(result2)
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-limited_filter.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        
        # Should only have 1 record despite 2 attempts
        self.assertEqual(1, len(trace_records))
        expected = [{'text': 'filtered text 1'}]
        self.assertEqual(expected, trace_records)


if __name__ == '__main__':
    unittest.main()
