import time
from contextlib import contextmanager
from importlib.metadata import entry_points

from loguru import logger


@contextmanager
def timing_context(description):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logger.debug(f"{description} took {elapsed_time:.2f} seconds")


# yapf: disable
with timing_context('Importing operator modules'):
    from . import aggregator, deduplicator, filter, grouper, mapper, pipeline, selector
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
        Pipeline,
        Selector,
    )
    from .fused_sequential_batch_op import FusedSequentialBatchOp  # noqa: F401
    from .load import load_ops
    from .op_env import (
        OPEnvManager,
        OPEnvSpec,
        analyze_lazy_loaded_requirements,
        analyze_lazy_loaded_requirements_for_code_file,
        op_requirements_to_op_env_spec,
    )

# Entry-point group name that external operator plugin packages must declare
# under [project.entry-points] in their pyproject.toml to be auto-discovered.
OP_PLUGIN_ENTRY_POINT_GROUP = 'data_juicer.ops'


def load_op_plugins(group: str = OP_PLUGIN_ENTRY_POINT_GROUP):
    """Discover and import external operator plugin packages.

    Any installed distribution that declares an entry point under ``group``
    (default ``data_juicer.ops``) is imported here, which triggers the module
    level ``OPERATORS.register_module(...)`` calls in those packages so that
    their operators become visible in the global ``OPERATORS`` registry just
    like the built-in ones.

    A failing plugin (e.g. broken import, missing dependency) only emits a
    warning and is skipped, so a single bad plugin never breaks the whole
    pipeline.

    :param group: the entry point group to scan.
    :return: the list of successfully loaded plugin entry point names.
    """
    loaded = []
    # importlib.metadata.entry_points has two API shapes across Python
    # versions: 3.10+ supports the ``group=`` selection keyword, while older
    # ones return a dict keyed by group name.
    try:
        eps = entry_points(group=group)
    except TypeError:  # pragma: no cover - only on very old importlib
        eps = entry_points().get(group, [])

    for ep in eps:
        try:
            ep.load()
            loaded.append(ep.name)
            logger.debug(f'Loaded operator plugin [{ep.name}] from entry point group [{group}].')
        except Exception as e:  # noqa: BLE001 - a bad plugin must not crash dj
            logger.warning(
                f'Failed to load operator plugin [{ep.name}] from entry point '
                f'group [{group}]: {e}. '
                f'Operators registered by this plugin before the error '
                f'remain available, but any operators defined after the '
                f'failing one in the same module will not be loaded.'
            )
    if loaded:
        logger.info(f'Discovered {len(loaded)} operator plugin(s) via entry points: {loaded}')
    return loaded


# eagerly discover external operator plugins at import time so that the
# registry is fully populated before init_configs() reads OPERATORS.modules.
with timing_context('Discovering operator plugins'):
    load_op_plugins()

__all__ = [
    'load_ops',
    'load_op_plugins',
    'OP_PLUGIN_ENTRY_POINT_GROUP',
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
    'Pipeline',
    'OPEnvSpec',
    'op_requirements_to_op_env_spec',
    'OPEnvManager',
    'analyze_lazy_loaded_requirements',
    'analyze_lazy_loaded_requirements_for_code_file',
]
