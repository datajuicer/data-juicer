import os
from functools import wraps

from datasets import disable_caching, enable_caching, is_caching_enabled

# Default cache location
DEFAULT_CACHE_HOME = "~/.cache"
CACHE_HOME = os.getenv("CACHE_HOME", DEFAULT_CACHE_HOME)

# Default data_juicer cache location
DEFAULT_DATA_JUICER_CACHE_HOME = os.path.join(CACHE_HOME, "data_juicer")
DATA_JUICER_CACHE_HOME = os.path.expanduser(os.getenv("DATA_JUICER_CACHE_HOME", DEFAULT_DATA_JUICER_CACHE_HOME))

# Default assets cache location
DEFAULT_DATA_JUICER_ASSETS_CACHE = os.path.join(DATA_JUICER_CACHE_HOME, "assets")
DATA_JUICER_ASSETS_CACHE = os.getenv("DATA_JUICER_ASSETS_CACHE", DEFAULT_DATA_JUICER_ASSETS_CACHE)
# Default models cache location
DEFAULT_DATA_JUICER_MODELS_CACHE = os.path.join(DATA_JUICER_CACHE_HOME, "models")
DATA_JUICER_MODELS_CACHE = os.getenv("DATA_JUICER_MODELS_CACHE", DEFAULT_DATA_JUICER_MODELS_CACHE)
DATA_JUICER_EXTERNAL_MODELS_HOME = os.getenv("DATA_JUICER_EXTERNAL_MODELS_HOME", None)

CACHE_COMPRESS = None


class DatasetCacheControl:
    """Define a range that change the cache state temporarily."""

    def __init__(self, on: bool = False):
        self.on = on

    def __enter__(self):
        """
        Record the original cache state and turn it to the target state.
        """
        self.previous_state = is_caching_enabled()
        if self.on:
            enable_caching()
        else:
            disable_caching()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore the original cache state.
        """
        if self.previous_state:
            enable_caching()
        else:
            disable_caching()


def dataset_cache_control(on):
    """
    A more easy-to-use decorator for functions that need to control the cache
    state temporarily.
    """

    def dataset_cache_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            with DatasetCacheControl(on=on):
                return func(*args, **kwargs)

        return wrapped_function

    return dataset_cache_decorator
