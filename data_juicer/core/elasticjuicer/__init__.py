"""
ElasticJuicer: Adaptive Resource Scheduling for Data-Juicer

A system that provides dynamic resource management and OOM prevention for
multimodal data processing pipelines.
"""

__version__ = "0.1.0"

# Core ElasticJuicer classes
from .elastic_juicer import ElasticJuicer
from .scheduler.scheduler_config import SchedulerConfig
from .scheduler.tower import Tower
from .scheduler.captain import Captain, CaptainPool
from .scheduler.micro_scheduler import MicroScheduler


# Lazy import for tuner (requires ray dependency)
def get_pbt_tuner():
    from .tuner.pbt_tuner import PBTTuner
    return PBTTuner


__all__ = [
    "profiler",
    "ElasticJuicer",
    "SchedulerConfig",
    "Tower",
    "Captain",
    "CaptainPool",
    "MicroScheduler",
    "get_pbt_tuner",
]
