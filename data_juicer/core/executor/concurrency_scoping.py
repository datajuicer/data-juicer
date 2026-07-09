"""Utility for scoping op concurrency when running partitions concurrently."""


def scope_op_concurrency(op, max_concurrent_partitions: int) -> int:
    """Returns the concurrency a single partition should use for this op.

    When multiple partitions run concurrently, each partition should use a
    fraction of the total GPU/actor resources to avoid over-subscription.

    Args:
        op: An operator instance with ``use_ray_actor()`` and ``num_proc``.
        max_concurrent_partitions: How many partitions will run in parallel.

    Returns:
        The concurrency value the partition should pass through to
        ``map_batches``.
    """
    if not op.use_ray_actor() or not op.num_proc or op.num_proc <= 0:
        return op.num_proc  # CPU ops or auto-mode unchanged
    return max(1, op.num_proc // max_concurrent_partitions)
