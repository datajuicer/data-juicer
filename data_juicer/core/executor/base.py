from abc import ABC, abstractmethod
from typing import Optional

from jsonargparse import Namespace
from pydantic import PositiveInt

from data_juicer.config import init_configs


class ExecutorBase(ABC):
    @abstractmethod
    def __init__(self, cfg: Optional[Namespace] = None):
        self.cfg = init_configs() if cfg is None else cfg
        self.executor_type = "base"

    @abstractmethod
    def run(self, load_data_np: Optional[PositiveInt] = None, skip_return=False):
        """Abstract method for ExecutorBase.run

        :param load_data_np: deprecated; number of workers when loading the
            dataset. Still applied by the default executor and the Analyzer,
            which is where `np` and `load_dataset_kwargs.num_proc` replace it.
            The Ray executors have never applied it; their read parallelism
            comes from `override_num_blocks` and the cluster.
        """
