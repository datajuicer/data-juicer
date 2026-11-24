import unittest

from data_juicer.ops.deduplicator.ray_basic_deduplicator import ActorBackend, RedisBackend
from data_juicer.ops.deduplicator.ray_document_deduplicator import RayDocumentDeduplicator
from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase


class RayBasicDeduplicatorTest(DataJuicerTestCaseBase):

    def test_prepare_for_ray_execution_reuses_existing_actor_handles(self):
        class RemoteDedupSet:
            calls = 0

            @classmethod
            def remote(cls):
                cls.calls += 1
                return object()

        backend = ActorBackend(dedup_set_num=2, RemoteDedupSet=RemoteDedupSet)

        backend.prepare_for_ray_execution()
        dedup_sets = backend._dedup_sets
        backend.prepare_for_ray_execution()

        self.assertIs(backend._dedup_sets, dedup_sets)
        self.assertEqual(RemoteDedupSet.calls, 2)

    def test_redis_backend_requests_ray_materialization(self):
        op = RayDocumentDeduplicator(
            lowercase=False,
            ignore_non_character=False,
            dedup_set_num=1,
            batch_size=1,
            num_proc=4,
            auto_op_parallelism=False,
        )
        op.backend = RedisBackend.__new__(RedisBackend)

        self.assertTrue(op._prepare_for_ray_map_batches())


if __name__ == '__main__':
    unittest.main()
