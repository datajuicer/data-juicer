import time
from contextlib import contextmanager

from loguru import logger

from data_juicer.config import init_configs


@contextmanager
def timing_context(description):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    logger.info(f"{description} took {elapsed_time:.2f} seconds")


@logger.catch(reraise=True)
def main():
    with timing_context("Loading configuration"):
        cfg = init_configs()

    with timing_context("Initializing executor"):
        from data_juicer.core.executor import ExecutorFactory

        executor = ExecutorFactory.create_executor_from_config(cfg)

    with timing_context("Running executor"):
        executor.run()


if __name__ == "__main__":
    main()
