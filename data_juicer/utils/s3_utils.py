"""
S3 utilities for Data-Juicer.

Provides unified S3 authentication and filesystem creation for both
s3fs (default executor) and PyArrow (Ray executor) backends.
"""

import os
from typing import Dict, Tuple

import pyarrow.fs
import s3fs
from loguru import logger

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv

    # Load environment variables from .env file if it exists
    load_dotenv(override=True)  # override=True means .env file takes precedence over environment variables
except ImportError:
    # python-dotenv not installed, .env files won't be automatically loaded
    pass


def get_aws_credentials(ds_config: Dict = {}) -> Tuple[str, str, str]:
    """
    Get AWS credentials with priority order:
    1. Environment variables (most secure, recommended for production)
    2. Explicit config parameters (for development/testing)
    3. Default AWS credential chain (boto3-style: env vars, ~/.aws/credentials, IAM roles)

    Args:
        ds_config: Dataset configuration dictionary containing optional AWS credentials.
                  If not provided, an empty dict is used.

    Returns:
        Tuple of (access_key_id, secret_access_key, session_token)
    """

    # Try environment variables first (most secure)
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")

    # Fall back to config if not in environment
    if not aws_access_key_id and "aws_access_key_id" in ds_config:
        aws_access_key_id = ds_config["aws_access_key_id"]
        logger.warning(
            "AWS credentials found in config file. For better security, "
            "consider using environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)"
        )
    if not aws_secret_access_key and "aws_secret_access_key" in ds_config:
        aws_secret_access_key = ds_config["aws_secret_access_key"]
    if not aws_session_token and "aws_session_token" in ds_config:
        aws_session_token = ds_config["aws_session_token"]

    return aws_access_key_id, aws_secret_access_key, aws_session_token


def create_s3fs_filesystem(ds_config: Dict) -> "s3fs.S3FileSystem":
    """
    Create an s3fs S3FileSystem with proper authentication.

    Authentication priority:
    1. Environment variables (most secure, recommended for production)
    2. Explicit config parameters (for development/testing)
    3. Default AWS credential chain (boto3-style: env vars, ~/.aws/credentials, IAM roles)

    Args:
        ds_config: Dataset configuration dictionary containing optional AWS credentials

    Returns:
        s3fs.S3FileSystem instance configured with credentials
    """

    # Get credentials with priority order
    aws_access_key_id, aws_secret_access_key, aws_session_token = get_aws_credentials(ds_config)

    s3_config = {}
    client_kwargs = {}

    # Set credentials if provided
    if aws_access_key_id:
        s3_config["key"] = aws_access_key_id
    if aws_secret_access_key:
        s3_config["secret"] = aws_secret_access_key
    if aws_session_token:
        s3_config["token"] = aws_session_token

    # Endpoint URL (for S3-compatible services like MinIO)
    if "endpoint_url" in ds_config:
        client_kwargs["endpoint_url"] = ds_config["endpoint_url"]

    if client_kwargs:
        s3_config["client_kwargs"] = client_kwargs

    # Create S3 filesystem
    # If no explicit credentials, s3fs will use default AWS credential chain
    if s3_config:
        fs = s3fs.S3FileSystem(**s3_config)
        logger.info("Using explicit AWS credentials for S3 access")
    else:
        fs = s3fs.S3FileSystem()
        logger.info("Using default AWS credential chain for S3 access")

    return fs


def create_pyarrow_s3_filesystem(ds_config: Dict) -> "pyarrow.fs.S3FileSystem":
    """
    Create a PyArrow S3FileSystem with proper authentication.

    Authentication priority:
    1. Environment variables (most secure, recommended for production)
    2. Explicit config parameters (for development/testing)
    3. Default AWS credential chain (boto3-style: env vars, ~/.aws/credentials, IAM roles)

    Args:
        ds_config: Dataset configuration dictionary containing optional AWS credentials

    Returns:
        pyarrow.fs.S3FileSystem instance configured with credentials
    """

    # Get credentials with priority order
    aws_access_key_id, aws_secret_access_key, aws_session_token = get_aws_credentials(ds_config)

    s3_options = {}

    # Set credentials if provided
    if aws_access_key_id:
        s3_options["access_key"] = aws_access_key_id
    if aws_secret_access_key:
        s3_options["secret_key"] = aws_secret_access_key
    if aws_session_token:
        s3_options["session_token"] = aws_session_token
    if "endpoint_url" in ds_config:
        s3_options["endpoint_override"] = ds_config["endpoint_url"]

    # Create S3 filesystem
    # If no explicit credentials, PyArrow will use default AWS credential chain
    if s3_options:
        s3_fs = pyarrow.fs.S3FileSystem(**s3_options)
        logger.info("Using explicit AWS credentials for S3 access")
    else:
        s3_fs = pyarrow.fs.S3FileSystem()
        logger.info("Using default AWS credential chain for S3 access")

    return s3_fs


def validate_s3_path(path: str) -> None:
    """
    Validate that a path is a valid S3 path.

    Args:
        path: Path to validate

    Raises:
        ValueError: If path doesn't start with 's3://'
    """
    if not path.startswith("s3://"):
        raise ValueError(f"S3 path must start with 's3://', got: {path}")
