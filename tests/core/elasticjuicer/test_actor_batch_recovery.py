from core.elasticjuicer.test_ray_adaptive_mapper import make_actor, make_batch
from core.elasticjuicer.test_stage_profile import profile_batch, profile_op

from data_juicer.core.elasticjuicer.adaptive_mapper import AdaptiveBatchContractError
from data_juicer.core.elasticjuicer.ray_adaptive_mapper import RayAdaptiveMapperActor


def recovery_actor():
    op = profile_op(batch_size=32, threshold=32)
    return RayAdaptiveMapperActor(type(op), op._init_args, op._init_kwargs, 32)


def learn_cost_change(actor):
    actor(profile_batch(512, "cheap"))
    actor.op.threshold = 4
    actor(profile_batch(4096, "expensive"))
    assert actor.controller.current_batch_size == 4
    assert actor.controller.oom_reprobe_events == 4


def test_generic_actor_recovers_without_profile_contract():
    actor = make_actor()
    actor(make_batch(80))
    actor.op.threshold = 8
    actor.op.calls.clear()
    result = actor(make_batch(1024))
    assert result["id"] == list(range(1024))
    assert max(map(len, actor.op.calls)) == 8


def test_stable_capacity_bounds_failed_reprobes():
    actor = make_actor()
    actor(make_batch(10000))
    assert actor.controller.oom_reprobe_events == 4
    before = actor.controller.oom_events
    actor(make_batch(10000))
    assert actor.controller.oom_events == before


def test_same_actor_restores_known_cheap_batch_without_stageprofile():
    actor = recovery_actor()
    learn_cost_change(actor)
    actor.op.threshold = 32
    actor.op.calls.clear()
    result = actor(profile_batch(512, "cheap"))
    assert result["id"] == list(range(512))
    assert max(map(len, actor.op.calls)) == 32
    assert actor.controller.capacity_recovery_hints == 1


def test_resource_change_does_not_reuse_old_capacity():
    actor = recovery_actor()
    learn_cost_change(actor)
    actor.op.memory_budget_bytes = 32
    actor.op.threshold = 32
    actor(profile_batch(512, "cheap"))
    assert actor.controller.current_batch_size == 4
    assert actor.controller.capacity_recovery_hints == 0


def test_smaller_success_does_not_renew_larger_proof(monkeypatch):
    actor = recovery_actor()
    actor(profile_batch(32, "cheap"))
    context, proof = next(iter(actor._recovery_history.items()))
    actor.controller.current_batch_size = 4
    actor.controller.oom_upper_bound = 5
    actor.controller.max_oom_reprobes = 0
    monkeypatch.setattr("data_juicer.core.elasticjuicer.ray_adaptive_mapper.time.time_ns", lambda: 10**12)
    actor(profile_batch(4, "cheap"))
    assert actor._recovery_history[context] == proof


def test_fresh_oom_consumes_and_invalidates_old_proof():
    actor = recovery_actor()
    learn_cost_change(actor)
    before = actor.controller.capacity_recovery_hints
    actor(profile_batch(1024, "cheap"))  # Capacity did not actually recover.
    assert actor.controller.capacity_recovery_hints == before + 1
    actor(profile_batch(4096, "cheap"))
    assert actor.controller.capacity_recovery_hints == before + 1


def test_contract_failure_cannot_add_local_recovery_proof():
    import pytest

    actor = recovery_actor()
    actor.op.failure = "rows"
    with pytest.raises(AdaptiveBatchContractError):
        actor(profile_batch(32, "cheap"))
    assert not actor._recovery_history


def test_expired_or_mixed_cost_proof_does_not_rearm():
    actor = recovery_actor()
    learn_cost_change(actor)
    context, proof = next((key, value) for key, value in actor._recovery_history.items() if value[0] == 32)
    actor._recovery_history[context] = (32, 0, False)
    actor.op.threshold = 32
    batch = profile_batch(512, "cheap")
    batch["cost"][0] = "other"
    actor(batch)
    actor(profile_batch(512, "cheap"))
    assert actor.controller.current_batch_size == 4
    assert actor.controller.capacity_recovery_hints == 0


def test_input_change_revokes_hint_before_fresh_success_window():
    actor = recovery_actor()
    learn_cost_change(actor)
    actor(profile_batch(1, "cheap"))
    assert actor.controller.state.capacity_recovery_hint_pending
    actor(profile_batch(8, "expensive"))
    assert not actor.controller.state.capacity_recovery_hint_pending
    assert actor.controller.oom_upper_bound == 5


def test_local_history_is_bounded_without_profile_service():
    actor = recovery_actor()
    for index in range(100):
        actor(profile_batch(1, f"cost-{index}"))
    assert len(actor._recovery_history) == 64
