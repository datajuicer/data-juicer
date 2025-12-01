import time
from contextlib import contextmanager

from loguru import logger


@contextmanager
def timing_context(description):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logger.info(f"{description} took {elapsed_time:.2f} seconds")


# yapf: disable
with timing_context('Importing operator modules'):
    from . import aggregator, deduplicator, filter, grouper, mapper, selector
    from .base_op import (
        ATTRIBUTION_FILTERS,
        NON_STATS_FILTERS,
        OPERATORS,
        TAGGING_OPS,
        UNFORKABLE,
        Aggregator,
        Deduplicator,
        Filter,
        Grouper,
        Mapper,
        Selector,
    )
    from .load import load_ops

__all__ = [
    'load_ops',
    'Filter',
    'Mapper',
    'Deduplicator',
    'Selector',
    'Grouper',
    'Aggregator',
    'UNFORKABLE',
    'NON_STATS_FILTERS',
    'OPERATORS',
    'TAGGING_OPS',
]
