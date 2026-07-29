"""
Unified filesystem utilities for Data-Juicer.

Provides a single dispatch entry to create a PyArrow FileSystem
according to the path scheme (s3://, hdfs://), consuming the
backend-specific keys from an extra-args dict without mutating it.

Convention: this module is the central dispatch extension point for
remote filesystems. When adding support for a new backend (e.g.
Iceberg/Delta/Hudi), prefer extending the scheme branches in
``create_filesystem_for_path`` instead of re-implementing per-callsite
prefix-detection logic in exporters or load strategies.
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple
from urllib.parse import urlparse

from loguru import logger

if TYPE_CHECKING:
    import pyarrow.fs

# config keys consumed when building a PyArrow S3 filesystem,
# accepted by data_juicer.utils.s3_utils.create_pyarrow_s3_filesystem
S3_FS_KEYS = (
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region",
    "endpoint_url",
)

# config keys consumed when building a PyArrow HDFS filesystem,
# accepted by data_juicer.utils.hdfs_utils.create_pyarrow_hdfs_filesystem
HDFS_FS_KEYS = (
    "hdfs_host",
    "hdfs_port",
    "hdfs_user",
    "hdfs_kerb_ticket",
    "hdfs_extra_conf",
)


def _split_args(extra_args: Dict, consumed_keys: Tuple[str, ...]) -> Tuple[Dict, Dict]:
    """
    Split ``extra_args`` into (consumed conf, remaining args) according to
    ``consumed_keys``, without mutating the input dict.

    :param extra_args: the original extra-args dict.
    :param consumed_keys: keys to be consumed by the filesystem backend.
    :return: a tuple of (conf, remaining_args), both are new dicts.
    """
    conf = {}
    remaining_args = {}
    for key in extra_args:
        if key in consumed_keys:
            conf[key] = extra_args[key]
        else:
            remaining_args[key] = extra_args[key]
    return conf, remaining_args


def create_filesystem_for_path(
    path: str,
    extra_args: Optional[Dict] = None,
) -> Tuple[Optional["pyarrow.fs.FileSystem"], Dict]:
    """
    Create a PyArrow FileSystem for ``path`` according to its scheme.

    - 's3://...': delegate to ``create_pyarrow_s3_filesystem``.
    - 'hdfs://...': delegate to ``create_pyarrow_hdfs_filesystem``. The
      config passed to it includes ``path`` so that host/port can be
      inferred from the path when not explicitly provided.
    - other (local) paths: no filesystem is created.

    The input ``extra_args`` is never mutated; the backend-specific keys
    (see ``S3_FS_KEYS`` / ``HDFS_FS_KEYS``) are consumed and removed from
    the returned ``remaining_args`` copy instead.

    :param path: the target path to create a filesystem for.
    :param extra_args: extra config dict that may contain backend-specific
        keys along with other unrelated keys.
    :return: a tuple of (fs, remaining_args). ``fs`` is a PyArrow
        FileSystem instance for remote paths, or None for local paths;
        ``remaining_args`` is a new dict without the consumed keys.
    """
    if not path:
        raise ValueError(f"Invalid path for filesystem creation: {path!r}")
    if extra_args is None:
        extra_args = {}

    # urlparse already lowercases the scheme; keep .lower() as a defensive
    # guarantee for case-insensitive dispatch
    scheme = urlparse(path).scheme.lower()
    if scheme == "s3":
        # s3_utils imports pyarrow at module level, so import it lazily
        from data_juicer.utils.s3_utils import (
            create_pyarrow_s3_filesystem,
            validate_s3_path,
        )

        validate_s3_path(path)
        logger.info(f"Detected S3 path: {path}. Creating PyArrow S3 filesystem.")
        s3_conf, remaining_args = _split_args(extra_args, S3_FS_KEYS)
        return create_pyarrow_s3_filesystem(s3_conf), remaining_args

    if scheme == "hdfs":
        # import at call time so the pyarrow-dependent construction stays lazy
        from data_juicer.utils.hdfs_utils import (
            create_pyarrow_hdfs_filesystem,
            validate_hdfs_path,
        )

        validate_hdfs_path(path)
        logger.info(f"Detected HDFS path: {path}. Creating PyArrow HDFS filesystem.")
        hdfs_conf, remaining_args = _split_args(extra_args, HDFS_FS_KEYS)
        # include the original path so host/port can be inferred from it
        hdfs_conf["path"] = path
        return create_pyarrow_hdfs_filesystem(hdfs_conf), remaining_args

    return None, dict(extra_args)
