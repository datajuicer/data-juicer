import unittest

import numpy as np
import pyarrow as pa

from data_juicer.ops.base_op import (
    OPERATORS,
    TAGGING_OPS,
    OP,
    Filter,
    Mapper,
    Deduplicator,
    Selector,
    Grouper,
    Aggregator,

    convert_dict_list_to_list_dict,
    convert_list_dict_to_dict_list,
    convert_arrow_to_python,
    catch_map_batches_exception,
    catch_map_single_exception,
    sample_to_dict,
)
from data_juicer.utils.constant import Fields
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


# ---------------------------------------------------------------------------
# Concrete subclasses for testing abstract base classes
# ---------------------------------------------------------------------------

class IdentityMapper(Mapper):
    _batched_op = False

    def process_single(self, sample):
        return sample


class UpperMapper(Mapper):
    _batched_op = False

    def process_single(self, sample):
        sample['text'] = sample['text'].upper()
        return sample


class BatchedUpperMapper(Mapper):
    _batched_op = True

    def process_batched(self, samples):
        samples['text'] = [t.upper() for t in samples['text']]
        return samples


class LengthFilter(Filter):
    _batched_op = False

    def compute_stats_single(self, sample, context=False):
        sample[Fields.stats] = {
            **sample.get(Fields.stats, {}),
            'text_len': len(sample['text']),
        }
        return sample

    def process_single(self, sample):
        return sample[Fields.stats]['text_len'] >= 5


class ErrorMapper(Mapper):
    _batched_op = False

    def process_single(self, sample):
        raise ValueError('intentional error')



# ---------------------------------------------------------------------------
# Tests for standalone conversion functions
# ---------------------------------------------------------------------------

class ConversionFunctionsTest(DataJuicerTestCaseBase):

    def test_list_dict_to_dict_list(self):
        samples = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
        result = convert_list_dict_to_dict_list(samples)
        self.assertEqual(result, {'a': [1, 3], 'b': [2, 4]})

    def test_list_dict_to_dict_list_single(self):
        samples = [{'x': 'hello'}]
        result = convert_list_dict_to_dict_list(samples)
        self.assertEqual(result, {'x': ['hello']})

    def test_dict_list_to_list_dict(self):
        samples = {'a': [1, 3], 'b': [2, 4]}
        result = convert_dict_list_to_list_dict(samples)
        self.assertEqual(result, [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])

    def test_dict_list_to_list_dict_single(self):
        samples = {'x': ['hello']}
        result = convert_dict_list_to_list_dict(samples)
        self.assertEqual(result, [{'x': 'hello'}])

    def test_roundtrip_conversion(self):
        original = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]
        roundtripped = convert_dict_list_to_list_dict(
            convert_list_dict_to_dict_list(original))
        self.assertEqual(roundtripped, original)


class ConvertArrowToPythonTest(DataJuicerTestCaseBase):

    def test_passes_dict_through(self):
        @convert_arrow_to_python
        def fn(sample):
            return sample

        d = {'text': ['hello']}
        self.assertEqual(fn(d), d)

    def test_converts_arrow_table(self):
        @convert_arrow_to_python
        def fn(sample):
            return sample

        table = pa.table({'text': ['hello', 'world']})
        result = fn(table)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['text'], ['hello', 'world'])


class SampleToDictTest(DataJuicerTestCaseBase):

    def test_dict_passthrough(self):
        d = {'text': 'hello'}
        self.assertIs(sample_to_dict(d), d)

    def test_arrow_table(self):
        table = pa.table({'text': ['hello']})
        result = sample_to_dict(table)
        self.assertIsInstance(result, dict)
        self.assertEqual(result['text'], ['hello'])

    def test_unsupported_type_raises(self):
        with self.assertRaises(ValueError):
            sample_to_dict([1, 2, 3])


# ---------------------------------------------------------------------------
# Tests for exception-catching wrappers
# ---------------------------------------------------------------------------

class CatchMapBatchesExceptionTest(DataJuicerTestCaseBase):

    def test_normal_execution(self):
        def fn(samples):
            samples['text'] = [t.upper() for t in samples['text']]
            return samples

        wrapped = catch_map_batches_exception(fn)
        result = wrapped({'text': ['hello'], Fields.stats: [{}],
                          Fields.source_file: ['a.txt']})
        self.assertEqual(result['text'], ['HELLO'])

    def test_error_propagates_without_skip(self):
        def fn(samples):
            raise ValueError('boom')

        wrapped = catch_map_batches_exception(fn, skip_op_error=False)
        with self.assertRaises(ValueError):
            wrapped({'text': ['hello']})

    def test_error_skipped_returns_empty(self):
        def fn(samples):
            raise ValueError('boom')

        wrapped = catch_map_batches_exception(fn, skip_op_error=True,
                                              op_name='test_op')
        result = wrapped({'text': ['hello'], 'other': [1]})
        self.assertEqual(result['text'], [])
        self.assertEqual(result[Fields.stats], [])
        self.assertEqual(result[Fields.source_file], [])

    def test_arrow_input_converted(self):
        def fn(samples):
            return samples

        wrapped = catch_map_batches_exception(fn)
        table = pa.table({'text': ['hello'], Fields.stats: [None],
                          Fields.source_file: ['f']})
        result = wrapped(table)
        self.assertIsInstance(result, dict)


class CatchMapSingleExceptionTest(DataJuicerTestCaseBase):

    def test_unbatched_passthrough(self):
        def fn(sample):
            sample['text'] = sample['text'].upper()
            return sample

        wrapped = catch_map_single_exception(fn)
        result = wrapped({'text': 'hello', 'n': 1})
        self.assertEqual(result['text'], 'HELLO')

    def test_batched_input_unwrapped_and_rewrapped(self):
        def fn(sample):
            sample['text'] = sample['text'].upper()
            return sample

        wrapped = catch_map_single_exception(fn, return_sample=True)
        result = wrapped({'text': ['hello'], 'n': [1]})
        self.assertEqual(result['text'], ['HELLO'])
        self.assertEqual(result['n'], [1])

    def test_batched_error_skipped(self):
        def fn(sample):
            raise ValueError('boom')

        wrapped = catch_map_single_exception(fn, skip_op_error=True,
                                             op_name='test')
        result = wrapped({'text': ['hello'], Fields.stats: [{}],
                          Fields.source_file: ['f']})
        self.assertEqual(result['text'], [])

    def test_batched_error_propagates_without_skip(self):
        def fn(sample):
            raise ValueError('boom')

        wrapped = catch_map_single_exception(fn, skip_op_error=False)
        with self.assertRaises(ValueError):
            wrapped({'text': ['hello'], 'n': [1]})

    def test_return_sample_false(self):
        def fn(sample):
            return len(sample['text'])

        wrapped = catch_map_single_exception(fn, return_sample=False)
        result = wrapped({'text': ['hello'], 'n': [1]})
        self.assertEqual(result, [5])


# ---------------------------------------------------------------------------
# Tests for OPMetaClass
# ---------------------------------------------------------------------------

class OPMetaClassTest(DataJuicerTestCaseBase):

    def test_init_args_stashed(self):
        op = IdentityMapper(text_key='content', batch_size=5)
        self.assertEqual(op._init_kwargs['text_key'], 'content')
        self.assertEqual(op._init_kwargs['batch_size'], 5)
        self.assertEqual(op._init_args, ())


# ---------------------------------------------------------------------------
# Tests for OP base class
# ---------------------------------------------------------------------------

class OPInitTest(DataJuicerTestCaseBase):

    def test_default_keys(self):
        op = IdentityMapper()
        self.assertEqual(op.text_key, 'text')
        self.assertEqual(op.image_key, 'images')
        self.assertEqual(op.audio_key, 'audios')
        self.assertEqual(op.video_key, 'videos')
        self.assertEqual(op.query_key, 'query')
        self.assertEqual(op.response_key, 'response')
        self.assertEqual(op.history_key, 'history')

    def test_custom_keys(self):
        op = IdentityMapper(text_key='content', image_key='imgs')
        self.assertEqual(op.text_key, 'content')
        self.assertEqual(op.image_key, 'imgs')

    def test_default_batch_size_cpu(self):
        op = IdentityMapper()
        self.assertEqual(op.batch_size, 1000)

    def test_custom_batch_size(self):
        op = IdentityMapper(batch_size=32)
        self.assertEqual(op.batch_size, 32)

    def test_accelerator_default(self):
        op = IdentityMapper()
        self.assertEqual(op.accelerator, 'cpu')

    def test_accelerator_override(self):
        op = IdentityMapper(accelerator='cuda')
        self.assertEqual(op.accelerator, 'cuda')
        self.assertEqual(op.batch_size, 10)

    def test_skip_op_error_default_false(self):
        op = IdentityMapper()
        self.assertFalse(op.skip_op_error)

    def test_memory_string_parsed(self):
        op = IdentityMapper(memory='2GB')
        self.assertAlmostEqual(op.memory, 2.0, places=1)

    def test_ray_execution_mode_valid(self):
        for mode in [None, 'actor', 'task']:
            op = IdentityMapper(ray_execution_mode=mode)
            self.assertEqual(op.ray_execution_mode, mode)

    def test_ray_execution_mode_invalid(self):
        with self.assertRaises(AssertionError):
            IdentityMapper(ray_execution_mode='invalid')

    def test_index_key_default_none(self):
        op = IdentityMapper()
        self.assertIsNone(op.index_key)

    def test_work_dir(self):
        op = IdentityMapper(work_dir='/tmp/test')
        self.assertEqual(op.work_dir, '/tmp/test')

    def test_turbo_default_false(self):
        op = IdentityMapper()
        self.assertFalse(op.turbo)


class OPFingerprintTest(DataJuicerTestCaseBase):

    def test_deterministic(self):
        op1 = IdentityMapper(batch_size=10)
        op2 = IdentityMapper(batch_size=10)
        self.assertEqual(op1._fingerprint_bytes(), op2._fingerprint_bytes())

    def test_different_params_different_fingerprint(self):
        op1 = IdentityMapper(batch_size=10)
        op2 = IdentityMapper(batch_size=20)
        self.assertNotEqual(op1._fingerprint_bytes(),
                            op2._fingerprint_bytes())

    def test_work_dir_excluded(self):
        op1 = IdentityMapper(work_dir='/tmp/a')
        op2 = IdentityMapper(work_dir='/tmp/b')
        self.assertEqual(op1._fingerprint_bytes(), op2._fingerprint_bytes())

    def test_init_args_excluded(self):
        op1 = IdentityMapper(batch_size=10)
        op2 = IdentityMapper(batch_size=10)
        op1._init_kwargs = {'work_dir': '/different'}
        self.assertEqual(op1._fingerprint_bytes(), op2._fingerprint_bytes())


class OPPropertyTest(DataJuicerTestCaseBase):

    def test_is_batched_op_default(self):
        op = IdentityMapper()
        self.assertFalse(op.is_batched_op())

    def test_is_batched_op_class_attr(self):
        op = BatchedUpperMapper()
        self.assertTrue(op.is_batched_op())

    def test_is_batched_op_batch_mode_override(self):
        op = IdentityMapper(batch_mode=True)
        self.assertTrue(op.is_batched_op())

    def test_is_batched_op_conflict_raises(self):
        class ForcedBatched(Mapper):
            _batched_op = True
            def process_batched(self, samples):
                return samples
        with self.assertRaises(ValueError):
            ForcedBatched(batch_mode=False).is_batched_op()

    def test_use_cuda_cpu(self):
        op = IdentityMapper(accelerator='cpu')
        self.assertFalse(op.use_cuda())

    def test_use_ray_actor_explicit_actor(self):
        op = IdentityMapper(ray_execution_mode='actor')
        self.assertTrue(op.use_ray_actor())

    def test_use_ray_actor_explicit_task(self):
        op = IdentityMapper(ray_execution_mode='task')
        self.assertFalse(op.use_ray_actor())

    def test_empty_history_standalone(self):
        op = IdentityMapper()
        h = op.empty_history()
        self.assertIsInstance(h, np.ndarray)
        self.assertEqual(h.shape, (0, 0))


class OPParameterMethodsTest(DataJuicerTestCaseBase):

    def test_remove_extra_parameters_default(self):
        params = {'self': None, 'x': 1, 'y': 2, '_private': 3}
        op = IdentityMapper()
        result = op.remove_extra_parameters(params)
        self.assertNotIn('self', result)
        self.assertNotIn('_private', result)
        self.assertEqual(result, {'x': 1, 'y': 2})

    def test_remove_extra_parameters_with_keys(self):
        params = {'a': 1, 'b': 2, 'c': 3}
        op = IdentityMapper()
        result = op.remove_extra_parameters(params, keys=['b'])
        self.assertEqual(result, {'a': 1, 'c': 3})

    def test_add_parameters_deep_copy(self):
        op = IdentityMapper()
        init_dict = {'a': [1, 2]}
        result = op.add_parameters(init_dict, b=3)
        self.assertEqual(result, {'a': [1, 2], 'b': 3})
        result['a'].append(99)
        self.assertEqual(init_dict['a'], [1, 2])


# ---------------------------------------------------------------------------
# Tests for OP.run (column injection)
# ---------------------------------------------------------------------------

class OPRunTest(DataJuicerTestCaseBase):

    def test_run_adds_index_column(self):
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset.from_list([{'text': 'a'}, {'text': 'b'}])
        op = IdentityMapper(index_key='idx')
        result = op.run(ds)
        self.assertIn('idx', result.features)
        self.assertEqual(list(result['idx']), [0, 1])


# ---------------------------------------------------------------------------
# Tests for Mapper
# ---------------------------------------------------------------------------

class MapperInitSubclassTest(DataJuicerTestCaseBase):

    def test_override_process_raises(self):
        with self.assertRaises(TypeError):
            class BadMapper(Mapper):
                def process(self, sample):
                    return sample

    def test_override_process_single_ok(self):
        class GoodMapper(Mapper):
            def process_single(self, sample):
                return sample
        op = GoodMapper()
        self.assertIsNotNone(op)


class MapperProcessTest(DataJuicerTestCaseBase):

    def test_single_process(self):
        op = UpperMapper()
        result = op.process({'text': 'hello'})
        self.assertEqual(result['text'], 'HELLO')

    def test_batched_from_single_fallback(self):
        op = IdentityMapper()
        samples = {'text': ['a', 'b'], 'n': [1, 2]}
        result = op.process_batched(samples)
        self.assertEqual(result['text'], ['a', 'b'])
        self.assertEqual(result['n'], [1, 2])

    def test_batched_process_adds_new_keys(self):
        class AddKeyMapper(Mapper):
            _batched_op = False
            def process_single(self, sample):
                sample['length'] = len(sample['text'])
                return sample

        op = AddKeyMapper()
        samples = {'text': ['hi', 'hello']}
        result = op.process_batched(samples)
        self.assertEqual(result['length'], [2, 5])

    def test_mapper_run_maps_dataset(self):
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset.from_list([
            {'text': 'hello'},
            {'text': 'world'},
        ])
        op = UpperMapper()
        result = op.run(ds)
        texts = list(result['text'])
        self.assertEqual(texts, ['HELLO', 'WORLD'])

    def test_mapper_callable(self):
        op = UpperMapper()
        result = op({'text': 'hello'})
        self.assertEqual(result['text'], 'HELLO')


class MapperErrorHandlingTest(DataJuicerTestCaseBase):

    def test_error_propagates_by_default(self):
        op = ErrorMapper(skip_op_error=False)
        with self.assertRaises(ValueError):
            op.process({'text': 'hello'})

    def test_error_skipped_when_configured(self):
        op = ErrorMapper(skip_op_error=True)
        result = op.process({'text': ['hello'], Fields.stats: [{}],
                             Fields.source_file: ['f'], 'n': [1]})
        self.assertEqual(result['text'], [])


# ---------------------------------------------------------------------------
# Tests for Filter
# ---------------------------------------------------------------------------

class FilterInitSubclassTest(DataJuicerTestCaseBase):

    def test_override_process_raises(self):
        with self.assertRaises(TypeError):
            class BadFilter(Filter):
                def process(self, sample):
                    return True

    def test_override_compute_stats_raises(self):
        with self.assertRaises(TypeError):
            class BadFilter(Filter):
                def compute_stats(self, sample):
                    return sample


class FilterGetKeepBooleanTest(DataJuicerTestCaseBase):

    def test_normal_ranges(self):
        test_cases = [
            (True, True, False, 5, 1, 10, True),
            (True, True, False, 5, None, 10, True),
            (True, True, False, 5, 1, None, True),
            (True, True, False, 5, None, None, True),
            (True, True, False, 5, 1, 5, True),
            (True, True, False, 5, 5, 10, True),
            (True, True, False, 5, 5, 5, True),
            (True, True, False, 5, 1, 4, False),
            (True, True, False, 5, 6, 10, False),
            (True, False, False, 5, 1, 10, True),
            (True, False, False, 5, 5, 10, True),
            (True, False, False, 5, 1, 5, False),
            (False, True, False, 5, 1, 10, True),
            (False, True, False, 5, 5, 10, False),
            (False, True, False, 5, 1, 5, True),
            (True, True, True, 5, 1, 10, False),
            (True, True, True, 5, None, 10, False),
            (True, True, True, 5, 1, None, False),
            (True, True, True, 5, None, None, False),
            (True, True, True, 5, 1, 5, True),
            (True, True, True, 5, 5, 10, True),
            (True, True, True, 5, 5, 5, True),
            (False, True, True, 5, 1, 5, True),
            (False, True, True, 5, 5, 10, False),
            (False, True, True, 5, 5, 5, True),
            (True, False, True, 5, 1, 5, False),
            (True, False, True, 5, 5, 10, True),
            (True, False, True, 5, 5, 5, True),
            (False, False, True, 5, 1, 5, False),
            (False, False, True, 5, 5, 10, False),
            (False, False, True, 5, 5, 5, False),
        ]
        for tc in test_cases:
            min_ci, max_ci, rev, val, mn, mx, tgt = tc
            op = LengthFilter(min_closed_interval=min_ci,
                              max_closed_interval=max_ci,
                              reversed_range=rev)
            self.assertEqual(op.get_keep_boolean(val, mn, mx), tgt,
                             msg=f'Failed for {tc}')


class FilterProcessTest(DataJuicerTestCaseBase):

    def test_compute_stats_batched_default(self):
        op = LengthFilter()
        samples = {
            'text': ['hi', 'hello world'],
            Fields.stats: [{}, {}],
        }
        result = op.compute_stats_batched(samples)
        self.assertEqual(result[Fields.stats][0]['text_len'], 2)
        self.assertEqual(result[Fields.stats][1]['text_len'], 11)

    def test_process_batched_default(self):
        op = LengthFilter()
        samples = {
            Fields.stats: [
                {'text_len': 3},
                {'text_len': 10},
            ],
        }
        result = list(op.process_batched(samples))
        self.assertEqual(result, [False, True])

    def test_filter_run_filters_dataset(self):
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset.from_list([
            {'text': 'hi'},
            {'text': 'hello world'},
            {'text': 'ab'},
        ])
        op = LengthFilter()
        result = op.run(ds)
        texts = sorted(result['text'])
        self.assertEqual(texts, ['hello world'])

    def test_filter_callable_calls_compute_stats(self):
        op = LengthFilter()
        sample = {'text': 'hello', Fields.stats: {}}
        result = op(sample)
        self.assertIn('text_len', result[Fields.stats])

    def test_filter_reversed_range(self):
        op = LengthFilter(reversed_range=True)
        self.assertTrue(op.get_keep_boolean(3, 5, 10))
        self.assertFalse(op.get_keep_boolean(7, 5, 10))


class FilterRunOptionsTest(DataJuicerTestCaseBase):

    def test_run_reduce_false(self):
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset.from_list([
            {'text': 'hi'},
            {'text': 'hello world'},
        ])
        op = LengthFilter()
        result = op.run(ds, reduce=False)
        self.assertEqual(len(result), 2)
        self.assertIn(Fields.stats, result.features)


# ---------------------------------------------------------------------------
# Tests for Deduplicator, Selector, Grouper, Aggregator, Pipeline
# ---------------------------------------------------------------------------

class DeduplicatorTest(DataJuicerTestCaseBase):

    def _make_dedup(self, **kwargs):
        import hashlib

        class HashDedup(Deduplicator):
            def compute_hash(self, sample):
                sample['__dj__hash'] = hashlib.md5(
                    sample['text'].encode()).hexdigest()
                return sample

            def process(self, dataset, show_num=0):
                seen = set()
                keep = []
                dups = []
                for i, h in enumerate(dataset['__dj__hash']):
                    if h not in seen:
                        seen.add(h)
                        keep.append(i)
                    else:
                        dups.append((i, h))
                return dataset.select(keep).remove_columns(
                    ['__dj__hash']), dups

        return HashDedup(**kwargs)

    def test_run_reduce_true_deduplicates(self):
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset.from_list([
            {'text': 'hello'}, {'text': 'world'}, {'text': 'hello'},
        ])
        op = self._make_dedup()
        result = op.run(ds, reduce=True)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result['text']), ['hello', 'world'])

    def test_run_reduce_false_keeps_all_with_hash(self):
        from data_juicer.core.data import NestedDataset
        ds = NestedDataset.from_list([
            {'text': 'hello'}, {'text': 'world'}, {'text': 'hello'},
        ])
        op = self._make_dedup()
        result = op.run(ds, reduce=False)
        self.assertEqual(len(result), 3)
        self.assertIn('__dj__hash', result.features)
        self.assertEqual(result['__dj__hash'][0], result['__dj__hash'][2])
        self.assertNotEqual(result['__dj__hash'][0], result['__dj__hash'][1])

    def test_compute_hash_wrapped_skip_op_error(self):
        from data_juicer.core.data import NestedDataset

        class FailDedup(Deduplicator):
            def compute_hash(self, sample):
                raise RuntimeError('hash failed')
            def process(self, dataset, show_num=0):
                return dataset, []

        op = FailDedup(skip_op_error=True)
        ds = NestedDataset.from_list([{'text': 'a'}])
        result = op.run(ds, reduce=False)
        self.assertEqual(len(result), 0)

    def test_compute_hash_wrapped_error_propagates(self):
        from data_juicer.core.data import NestedDataset

        class FailDedup(Deduplicator):
            def compute_hash(self, sample):
                raise RuntimeError('hash failed')
            def process(self, dataset, show_num=0):
                return dataset, []

        op = FailDedup(skip_op_error=False)
        ds = NestedDataset.from_list([{'text': 'a'}])
        with self.assertRaises(RuntimeError):
            op.run(ds, reduce=False)


class SelectorTest(DataJuicerTestCaseBase):

    def test_run_selects_at_dataset_level(self):
        from data_juicer.core.data import NestedDataset

        class TopNSelector(Selector):
            def process(self, dataset):
                return dataset.select(range(min(2, len(dataset))))

        ds = NestedDataset.from_list([
            {'text': 'a'}, {'text': 'b'}, {'text': 'c'},
        ])
        op = TopNSelector()
        result = op.run(ds)
        self.assertEqual(len(result), 2)
        self.assertEqual(list(result['text']), ['a', 'b'])


class GrouperRunTest(DataJuicerTestCaseBase):

    def test_run_creates_nested_dataset(self):
        from data_juicer.core.data import NestedDataset

        class SimpleGrouper(Grouper):
            def process(self, dataset):
                return [{'text': ['a', 'b']}, {'text': ['c']}]

        ds = NestedDataset.from_list([
            {'text': 'a'}, {'text': 'b'}, {'text': 'c'},
        ])
        op = SimpleGrouper()
        result = op.run(ds)
        self.assertEqual(len(result), 2)


class AggregatorRunTest(DataJuicerTestCaseBase):

    def test_run_adds_batch_meta(self):
        from data_juicer.core.data import NestedDataset

        class SimpleAgg(Aggregator):
            def process_single(self, sample):
                sample['summary'] = 'done'
                return sample

        ds = NestedDataset.from_list([
            {'text': ['a', 'b']},
            {'text': ['c']},
        ])
        op = SimpleAgg()
        result = op.run(ds)
        self.assertIn(Fields.batch_meta, result.features)
        self.assertEqual(list(result['summary']), ['done', 'done'])

    def test_run_preserves_existing_batch_meta(self):
        from data_juicer.core.data import NestedDataset

        class SimpleAgg(Aggregator):
            def process_single(self, sample):
                sample['summary'] = 'done'
                return sample

        ds = NestedDataset.from_list([
            {'text': ['a'], Fields.batch_meta: {'existing': True}},
        ])
        op = SimpleAgg()
        result = op.run(ds)
        self.assertEqual(result[Fields.batch_meta][0], {'existing': True})


if __name__ == '__main__':
    unittest.main()
