class ExecutorFactory:
    @staticmethod
    def create_executor(executor_type: str):
        if executor_type in ("local", "default"):
            from .default_executor import DefaultExecutor

            return DefaultExecutor
        elif executor_type == "ray":
            from .ray_executor import RayExecutor

            return RayExecutor
        # TODO: add nemo support
        #  elif executor_type == "nemo":
        #    return NemoExecutor()
        # TODO: add dask support
        #  elif executor_type == "dask":
        #    return DaskExecutor()
        else:
            raise ValueError("Unsupported executor type")
