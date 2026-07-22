"""
HDFS utilities for Data-Juicer.

Provides unified HDFS filesystem creation for the default executor
and Ray executor (PyArrow backend). PyArrow's HadoopFileSystem relies
on libhdfs and a working Hadoop/JVM environment on every node, so make
sure the following environment variables are configured on all nodes
before use:

    export HADOOP_HOME=/path/to/hadoop
    export JAVA_HOME=/path/to/java
    export CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob)
    export ARROW_LIBHDFS_DIR=$HADOOP_HOME/lib/native   # depends on env
"""

from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import pyarrow.fs
from loguru import logger


def parse_hdfs_path(path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse the host and port from an HDFS URI.

    e.g. 'hdfs://namenode:8020/user/data' -> ('namenode', 8020)
         'hdfs://namenode/user/data'      -> ('namenode', None)
         'hdfs:///user/data'              -> (None, None)  # rely on default fs

    Args:
        path: HDFS path to parse.

    Returns:
        A tuple of (host, port). host/port may be None, in which case
        PyArrow will fall back to the default filesystem configured via
        Hadoop conf (fs.defaultFS).
    """
    parsed = urlparse(path)
    host = parsed.hostname
    port = parsed.port
    return host, port


def strip_hdfs_scheme(path: str) -> str:
    """
    Strip the hdfs:// scheme/authority from a path, returning the bare
    filesystem path that PyArrow HadoopFileSystem expects.

    e.g. 'hdfs://namenode:8020/user/data/file.jsonl' -> '/user/data/file.jsonl'

    Args:
        path: HDFS path to strip.

    Returns:
        Bare filesystem path without the hdfs:// scheme.
    """
    parsed = urlparse(path)
    return parsed.path or path


def create_pyarrow_hdfs_filesystem(ds_config: Dict = {}) -> "pyarrow.fs.HadoopFileSystem":
    """
    Create a PyArrow HadoopFileSystem for reading/writing HDFS.

    Configuration priority for host/port:
    1. Explicit fields in ``ds_config`` ('hdfs_host', 'hdfs_port').
    2. Parsed from the 'path' field in ``ds_config``.
    3. Default ('default'), letting PyArrow use Hadoop conf (fs.defaultFS).

    Optional ``ds_config`` fields:
      - hdfs_host: namenode host, or 'default'.
      - hdfs_port: namenode port (int).
      - hdfs_user: user name for HDFS access.
      - hdfs_kerb_ticket: path to the Kerberos ticket cache.
      - hdfs_extra_conf: dict of extra Hadoop configurations.

    Args:
        ds_config: Dataset/export configuration dictionary.

    Returns:
        A configured pyarrow.fs.HadoopFileSystem instance.
    """
    host = ds_config.get("hdfs_host")
    port = ds_config.get("hdfs_port")

    # fall back to parsing from path if host is not explicitly provided
    if host is None and ds_config.get("path"):
        parsed_host, parsed_port = parse_hdfs_path(ds_config["path"])
        host = parsed_host
        if port is None:
            port = parsed_port

    # PyArrow uses 'default' to indicate reliance on Hadoop conf (fs.defaultFS)
    if not host:
        host = "default"

    fs_kwargs = {"host": host}
    if port is not None:
        fs_kwargs["port"] = int(port)
    if "hdfs_user" in ds_config:
        fs_kwargs["user"] = ds_config["hdfs_user"]
    if "hdfs_kerb_ticket" in ds_config:
        fs_kwargs["kerb_ticket"] = ds_config["hdfs_kerb_ticket"]
    if "hdfs_extra_conf" in ds_config:
        fs_kwargs["extra_conf"] = ds_config["hdfs_extra_conf"]

    logger.info(f"Creating PyArrow HadoopFileSystem with host={host}, port={port}")
    return pyarrow.fs.HadoopFileSystem(**fs_kwargs)


def validate_hdfs_path(path: str) -> None:
    """
    Validate that a path is a valid HDFS path.

    Args:
        path: Path to validate.

    Raises:
        ValueError: If path doesn't start with 'hdfs://'.
    """
    if not path.startswith("hdfs://"):
        raise ValueError(f"HDFS path must start with 'hdfs://', got: {path}")
