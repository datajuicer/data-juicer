import os
import unittest
import tempfile
import shutil
import jsonlines as jl
from data_juicer.core.tracer.ray_tracer import RayTracer
from data_juicer.utils.unittest_utils import TEST_TAG
import ray

class RayTracerTest(unittest.TestCase):

    def setUp(self) -> None:
        if not ray.is_initialized():
            ray.init(local_mode=True)  # Use local mode for testing
        self.work_dir = tempfile.mkdtemp(prefix='test_ray_tracer_')

    def tearDown(self):
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)
        if ray.is_initialized():
            ray.shutdown()

    @TEST_TAG("ray")
    def test_ray_tracer_initialization(self):
        """Test RayTracer initialization."""
        tracer = RayTracer.remote(self.work_dir)
        
        # Check that the tracer was created
        self.assertIsNotNone(tracer)
        
        # Test basic functionality
        should_trace = ray.get(tracer.should_trace_op.remote('perplexity_filter'))
        # By default, should trace all ops if no op_list_to_trace provided
        self.assertTrue(should_trace)

    @TEST_TAG("ray")
    def test_ray_tracer_with_op_list(self):
        """Test RayTracer with specific op list to trace."""
        tracer = RayTracer.remote(self.work_dir, op_list_to_trace=['specific_op'])
        
        should_trace = ray.get(tracer.should_trace_op.remote('specific_op'))
        self.assertTrue(should_trace)
        
        should_not_trace = ray.get(tracer.should_trace_op.remote('other_op'))
        self.assertFalse(should_not_trace)

    @TEST_TAG("ray")
    def test_collect_mapper_sample_basic(self):
        """Test basic functionality of collect_mapper_sample method in RayTracer."""
        tracer = RayTracer.remote(self.work_dir, op_list_to_trace=['test_mapper'])
        
        original_sample = {'text': 'original text'}
        processed_sample = {'text': 'processed text'}
        
        # Should collect the sample since text differs
        result = ray.get(tracer.collect_mapper_sample.remote('test_mapper', original_sample, processed_sample, 'text'))
        self.assertTrue(result)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
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

    @TEST_TAG("ray")
    def test_collect_mapper_sample_no_change(self):
        """Test collect_mapper_sample when original and processed texts are the same."""
        tracer = RayTracer.remote(self.work_dir)
        
        original_sample = {'text': 'same text'}
        processed_sample = {'text': 'same text'}
        
        # Should not collect the sample since text is the same
        result = ray.get(tracer.collect_mapper_sample.remote('test_mapper', original_sample, processed_sample, 'text'))
        self.assertFalse(result)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_mapper.jsonl')
        # File should not exist since no samples were collected
        self.assertFalse(os.path.exists(trace_file_path))

    @TEST_TAG("ray")
    def test_collect_mapper_sample_with_trace_keys(self):
        """Test collect_mapper_sample with trace_keys functionality."""
        tracer = RayTracer.remote(self.work_dir, trace_keys=['sample_id', 'source'], op_list_to_trace=['test_mapper'])
        
        original_sample = {'text': 'original text', 'sample_id': 'id-123', 'source': 'source_file.txt'}
        processed_sample = {'text': 'processed text', 'sample_id': 'id-123', 'source': 'source_file.txt'}
        
        result = ray.get(tracer.collect_mapper_sample.remote('test_mapper', original_sample, processed_sample, 'text'))
        self.assertTrue(result)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
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

    @TEST_TAG("ray")
    def test_collect_mapper_sample_with_missing_trace_keys(self):
        """Test collect_mapper_sample when trace keys are missing from sample."""
        tracer = RayTracer.remote(self.work_dir, trace_keys=['missing_key', 'existing_key'], op_list_to_trace=['test_mapper'])
        
        original_sample = {'text': 'original text', 'existing_key': 'value'}
        processed_sample = {'text': 'processed text', 'existing_key': 'value'}
        
        result = ray.get(tracer.collect_mapper_sample.remote('test_mapper', original_sample, processed_sample, 'text'))
        self.assertTrue(result)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
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

    @TEST_TAG("ray")
    def test_collect_mapper_sample_not_in_op_list(self):
        """Test collect_mapper_sample when op is not in the trace list."""
        tracer = RayTracer.remote(self.work_dir, op_list_to_trace=['other_mapper'])
        
        original_sample = {'text': 'original text'}
        processed_sample = {'text': 'processed text'}
        
        # Should not collect since op is not in the trace list
        result = ray.get(tracer.collect_mapper_sample.remote('test_mapper', original_sample, processed_sample, 'text'))
        self.assertFalse(result)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_mapper.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    @TEST_TAG("ray")
    def test_collect_filter_sample_basic(self):
        """Test basic functionality of collect_filter_sample method."""
        tracer = RayTracer.remote(self.work_dir, op_list_to_trace=['test_filter'])
        
        sample = {'text': 'filtered text', 'sample_id': 'id-456'}
        
        # Should collect the sample since it should be filtered out (keep=False)
        result = ray.get(tracer.collect_filter_sample.remote('test_filter', sample, False))
        self.assertTrue(result)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_filter.jsonl')
        self.assertTrue(os.path.exists(trace_file_path))
        
        trace_records = []
        with jl.open(trace_file_path, 'r') as reader:
            for s in reader:
                trace_records.append(s)
        
        expected = [{'text': 'filtered text', 'sample_id': 'id-456'}]
        self.assertEqual(expected, trace_records)

    @TEST_TAG("ray")
    def test_collect_filter_sample_should_keep(self):
        """Test collect_filter_sample when sample should be kept."""
        tracer = RayTracer.remote(self.work_dir)
        
        sample = {'text': 'kept text'}
        
        # Should not collect the sample since it should be kept (keep=True)
        result = ray.get(tracer.collect_filter_sample.remote('test_filter', sample, True))
        self.assertFalse(result)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_filter.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    @TEST_TAG("ray")
    def test_collect_filter_sample_not_in_op_list(self):
        """Test collect_filter_sample when op is not in the trace list."""
        tracer = RayTracer.remote(self.work_dir, op_list_to_trace=['other_filter'])
        
        sample = {'text': 'filtered text'}
        
        # Should not collect since op is not in the trace list
        result = ray.get(tracer.collect_filter_sample.remote('test_filter', sample, False))
        self.assertFalse(result)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
        trace_file_path = os.path.join(self.work_dir, 'trace', 'sample_trace-test_filter.jsonl')
        self.assertFalse(os.path.exists(trace_file_path))

    @TEST_TAG("ray")
    def test_collect_mapper_sample_show_num_limit(self):
        """Test collect_mapper_sample respects show_num limit."""
        tracer = RayTracer.remote(self.work_dir, show_num=2, op_list_to_trace=['limited_mapper'])
        
        # First sample should be collected
        original1 = {'text': 'original text 1'}
        processed1 = {'text': 'processed text 1'}
        result1 = ray.get(tracer.collect_mapper_sample.remote('limited_mapper', original1, processed1, 'text'))
        self.assertTrue(result1)
        
        # Second sample should be collected
        original2 = {'text': 'original text 2'}
        processed2 = {'text': 'processed text 2'}
        result2 = ray.get(tracer.collect_mapper_sample.remote('limited_mapper', original2, processed2, 'text'))
        self.assertTrue(result2)
        
        # Third sample should not be collected due to limit
        original3 = {'text': 'original text 3'}
        processed3 = {'text': 'processed text 3'}
        result3 = ray.get(tracer.collect_mapper_sample.remote('limited_mapper', original3, processed3, 'text'))
        self.assertFalse(result3)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
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

    @TEST_TAG("ray")
    def test_collect_filter_sample_show_num_limit(self):
        """Test collect_filter_sample respects show_num limit."""
        tracer = RayTracer.remote(self.work_dir, show_num=1, op_list_to_trace=['limited_filter'])
        
        # First sample should be collected
        sample1 = {'text': 'filtered text 1'}
        result1 = ray.get(tracer.collect_filter_sample.remote('limited_filter', sample1, False))
        self.assertTrue(result1)
        
        # Second sample should not be collected due to limit
        sample2 = {'text': 'filtered text 2'}
        result2 = ray.get(tracer.collect_filter_sample.remote('limited_filter', sample2, False))
        self.assertFalse(result2)
        
        # Finalize traces to write to file
        ray.get(tracer.finalize_traces.remote())
        
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

    @TEST_TAG("ray")
    def test_is_collection_complete(self):
        """Test is_collection_complete method."""
        tracer = RayTracer.remote(self.work_dir, show_num=1, op_list_to_trace=['test_op'])
        
        # Initially should not be complete
        is_complete = ray.get(tracer.is_collection_complete.remote('test_op'))
        self.assertFalse(is_complete)
        
        # Collect one sample
        original_sample = {'text': 'original text'}
        processed_sample = {'text': 'processed text'}
        ray.get(tracer.collect_mapper_sample.remote('test_op', original_sample, processed_sample, 'text'))
        
        # Now should be complete since we reached the limit of 1
        is_complete = ray.get(tracer.is_collection_complete.remote('test_op'))
        self.assertTrue(is_complete)

    @TEST_TAG("ray")
    def test_finalize_traces_empty(self):
        """Test finalize_traces when no traces were collected."""
        tracer = RayTracer.remote(self.work_dir)
        
        # Don't collect anything, just finalize
        ray.get(tracer.finalize_traces.remote())
        
        # No trace files should exist
        trace_dir = os.path.join(self.work_dir, 'trace')
        if os.path.exists(trace_dir):
            files = os.listdir(trace_dir)
            # Should have no trace files or only empty ones
            self.assertEqual(len(files), 0)


if __name__ == '__main__':
    unittest.main()