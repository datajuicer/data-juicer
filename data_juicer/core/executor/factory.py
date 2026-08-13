from .base import ExecutorBase
from .default_executor import DefaultExecutor


class ExecutorFactory:
    @staticmethod
    def create_executor(executor_type: str) -> ExecutorBase:
        if executor_type in ("local", "default"):
            return DefaultExecutor
        elif executor_type == "ray":
            from .ray_executor import RayExecutor

            return RayExecutor
        elif executor_type == "ray_partitioned":
            from .ray_executor_partitioned import PartitionedRayExecutor

            return PartitionedRayExecutor
        # TODO: add nemo support
        #  elif executor_type == "nemo":
        #    return NemoExecutor
        # TODO: add dask support
        #  elif executor_type == "dask":
        #    return DaskExecutor
        else:
            raise ValueError("Unsupported executor type")

    @staticmethod
    def create_executor_from_config(cfg) -> ExecutorBase:
        """Create the configured executor, adding auto sharding when eligible."""
        from .elastic_sharding.context import should_wrap_executor

        if should_wrap_executor(cfg):
            from .elastic_sharding.executor import ElasticShardingExecutor

            return ElasticShardingExecutor(cfg)
        executor_class = ExecutorFactory.create_executor(cfg.executor_type)
        return executor_class(cfg)
