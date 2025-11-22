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
