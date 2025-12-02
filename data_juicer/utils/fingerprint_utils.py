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
    def hash_default(cls, value: Any) -> str:
        """
        Use dill to serialize objects to avoid serialization failures.
        """
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
