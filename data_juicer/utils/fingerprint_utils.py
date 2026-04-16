from typing import Any, Dict, List, Union

import dill
import xxhash
from datasets.fingerprint import (
    _CACHING_ENABLED,
    fingerprint_warnings,
    format_kwargs_for_fingerprint,
    format_transform_for_fingerprint,
    generate_random_fingerprint,
    validate_fingerprint,
)
from loguru import logger


class Hasher:
    """Hasher that accepts python objects as inputs."""

    dispatch: Dict = {}

    def __init__(self):
        self.m = xxhash.xxh64()

    @classmethod
    def hash_bytes(cls, value: Union[bytes, List[bytes]]) -> str:
        value = [value] if isinstance(value, bytes) else value
        m = xxhash.xxh64()
        for x in value:
            m.update(x)
        return m.hexdigest()

    @classmethod
    def _find_op_owner(cls, value):
        """Walk the ``__self__`` / ``__wrapped__`` chain to find an object
        that exposes ``_fingerprint_bytes``.  Returns ``(obj, func_name)``
        or ``(None, None)``."""
        # Direct bound method
        obj = getattr(value, "__self__", None)
        if obj is not None:
            if callable(getattr(obj, "_fingerprint_bytes", None)):
                func_name = getattr(value, "__name__", getattr(value, "__qualname__", ""))
                return obj, func_name
        # Walk the full __wrapped__ chain (handles multiple decorator
        # layers such as wrap_func_with_nested_access → @wraps → bound
        # method).
        cur = value
        for _ in range(10):  # guard against infinite loops
            cur = getattr(cur, "__wrapped__", None)
            if cur is None:
                break
            obj = getattr(cur, "__self__", None)
            if obj is not None and callable(getattr(obj, "_fingerprint_bytes", None)):
                func_name = getattr(cur, "__name__", getattr(cur, "__qualname__", ""))
                return obj, func_name
        return None, None

    @classmethod
    def hash_default(cls, value: Any) -> str:
        """
        Use dill to serialize objects to avoid serialization failures.

        If the object exposes a ``_fingerprint_bytes()`` method (e.g. OP
        subclasses), use it so that execution-only attributes like
        ``work_dir`` are excluded from the cache key.
        """
        fingerprint_bytes = getattr(value, "_fingerprint_bytes", None)
        if callable(fingerprint_bytes):
            return cls.hash_bytes(fingerprint_bytes())
        # For bound methods / wrapped functions whose __self__ supports
        # _fingerprint_bytes, hash the (fingerprint, method_name) pair
        # instead of dill-dumping the bound method (which would
        # re-serialize the full object including excluded attrs).
        obj, func_name = cls._find_op_owner(value)
        if obj is not None:
            return cls.hash_bytes(obj._fingerprint_bytes() + dill.dumps(func_name))
        return cls.hash_bytes(dill.dumps(value))

    @classmethod
    def hash(cls, value: Any) -> str:
        if type(value) in cls.dispatch:
            return cls.dispatch[type(value)](cls, value)
        else:
            return cls.hash_default(value)

    def update(self, value: Any) -> None:
        header_for_update = f"=={type(value)}=="
        value_for_update = self.hash(value)
        self.m.update(header_for_update.encode("utf8"))
        self.m.update(value_for_update.encode("utf-8"))

    def hexdigest(self) -> str:
        return self.m.hexdigest()


def update_fingerprint(fingerprint, transform, transform_args):
    """
    Combining various objects to update the fingerprint.
    """

    hasher = Hasher()
    hasher.update(fingerprint)
    try:
        hasher.update(transform)
    except:  # noqa various errors might raise here from pickle or dill
        if _CACHING_ENABLED:
            if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                logger.warning(
                    f"Transform {transform} couldn't be hashed properly, \
                     a random hash was used instead. Make sure your \
                     transforms and parameters are serializable with \
                     pickle or dill for the dataset fingerprinting and \
                     caching to work. If you reuse this transform, the \
                     caching mechanism will consider it to be different \
                     from the previous calls and recompute everything. \
                     This warning is only showed once. Subsequent hashing \
                     failures won't be showed."
                )
                fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
            else:
                logger.info(
                    f"Transform {transform} couldn't be hashed properly, \
                     a random hash was used instead."
                )
        else:
            logger.info(
                f"Transform {transform} couldn't be hashed properly, a \
                 random hash was used instead. This doesn't affect caching \
                 since it's disabled."
            )

        return generate_random_fingerprint()
    for key in sorted(transform_args):
        hasher.update(key)
        try:
            hasher.update(transform_args[key])
        except:  # noqa various errors might raise here from pickle or dill
            if _CACHING_ENABLED:
                if not fingerprint_warnings.get("update_fingerprint_transform_hash_failed", False):
                    logger.warning(
                        f"Parameter '{key}'={transform_args[key]} of the \
                         transform {transform} couldn't be hashed properly, \
                         a random hash was used instead. Make sure your \
                         transforms and parameters are serializable with \
                         pickle or dill for the dataset fingerprinting and \
                         caching to work. If you reuse this transform, the \
                         caching mechanism will consider it to be different \
                         from the previous calls and recompute everything. \
                         This warning is only showed once. Subsequent hashing \
                         failures won't be showed."
                    )
                    fingerprint_warnings["update_fingerprint_transform_hash_failed"] = True
                else:
                    logger.info(
                        f"Parameter '{key}'={transform_args[key]} of the \
                         transform {transform} couldn't be hashed properly, \
                         a random hash was used instead."
                    )
            else:
                logger.info(
                    f"Parameter '{key}'={transform_args[key]} of the transform \
                     {transform} couldn't be hashed properly, a random hash \
                     was used instead. This doesn't affect caching since it's \
                     disabled."
                )
            return generate_random_fingerprint()
    return hasher.hexdigest()


def generate_fingerprint(ds, *args, **kwargs):
    """
    Generate new fingerprints by using various kwargs of the dataset.
    """
    if args:
        args = list(args)
        dataset_kwargs = {"shard": ds, "function": args[0]}
    else:
        dataset_kwargs = {"shard": ds}
    dataset_kwargs.update(kwargs)

    # we create a unique hash from the function,
    # current dataset file and the mapping args
    transform = format_transform_for_fingerprint(ds._map_single)
    kwargs_for_fingerprint = format_kwargs_for_fingerprint(ds._map_single, (), dataset_kwargs)
    kwargs_for_fingerprint["fingerprint_name"] = "new_fingerprint"
    new_fingerprint = update_fingerprint(ds._fingerprint, transform, kwargs_for_fingerprint)
    validate_fingerprint(new_fingerprint)
    return new_fingerprint
