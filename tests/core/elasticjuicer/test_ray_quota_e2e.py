import unittest

from data_juicer.core.elasticjuicer.quota import BatchSizeQuota
from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor
from data_juicer.utils.unittest_utils import TEST_TAG, DataJuicerTestCaseBase


def _real_ray_available():
    try:
        import ray
    except ImportError:
        return False
    return bool(getattr(ray, "__version__", None)) and callable(getattr(ray, "get", None))


class IdentityMapper:
    def process(self, batch):
        return batch


@unittest.skipUnless(_real_ray_available(), "real Ray is required for unittest-dist quota E2E")
class ElasticJuicerRayQuotaE2ETest(DataJuicerTestCaseBase):
    @TEST_TAG("ray")
    def test_quota_applies_over_explicit_remote_actor_handle(self):
        import ray

        remote_actor = ray.remote(num_cpus=0)(RayAdaptiveMapperActor)
        actor = remote_actor.remote(
            IdentityMapper,
            initial_batch_size=32,
            max_batch_size=32,
            job_id="quota-ray-job",
            actor_id="quota-ray-actor",
        )
        try:
            application = ray.get(
                actor.apply_quota.remote(
                    BatchSizeQuota(
                        job_id="quota-ray-job",
                        actor_id="quota-ray-actor",
                        revision=1,
                        max_batch_size=7,
                    )
                )
            )
            state = ray.get(actor.get_quota_state.remote())

            self.assertTrue(application.applied)
            self.assertEqual(application.effective_max_batch_size, 7)
            self.assertEqual(state.hard_limit, 7)
            self.assertEqual(state.current_batch_size, 7)
            self.assertEqual(state.last_revision, 1)
            self.assertIsNone(state.local_oom_upper_bound)
        finally:
            ray.kill(actor, no_restart=True)
