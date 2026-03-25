from loguru import logger

from .base import ExecutorBase
from .default_executor import DefaultExecutor


class ExecutorFactory:
    @staticmethod
    def create_executor(executor_type: str) -> ExecutorBase:
        if executor_type in ("local", "default"):
            return DefaultExecutor
        elif executor_type == "ray":
            from .elastic_ray_executor import ElasticRayExecutor
            logger.info('Using ElasticRayExecutor (adaptive scheduling enabled for GPU operators)')
            return ElasticRayExecutor
        elif executor_type == "ray_partitioned":
            from .ray_executor_partitioned import PartitionedRayExecutor

            return PartitionedRayExecutor
        elif executor_type == "elastic_ray":
            from .elastic_ray_executor import ElasticRayExecutor

            return ElasticRayExecutor
        # TODO: add nemo support
        #  elif executor_type == "nemo":
        #    return NemoExecutor
        # TODO: add dask support
        #  elif executor_type == "dask":
        #    return DaskExecutor
        else:
            raise ValueError("Unsupported executor type")
