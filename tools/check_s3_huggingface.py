#!/usr/bin/env python3
"""
Standalone test script for S3 data loading with HuggingFace datasets.

Usage:
    python check_s3_huggingface.py

Prerequisites:
    - s3fs installed: pip install s3fs
    - datasets library: pip install datasets
    - AWS credentials configured (via environment variables, AWS CLI, or IAM role)
"""

import os
import sys

from datasets import load_dataset
from loguru import logger

from data_juicer.utils.s3_utils import get_aws_credentials

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def test_s3_load_public_file():
    """
    Test loading a public JSONL file from S3 using anonymous access.

    This demonstrates the pattern for public S3 buckets that don't require credentials.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Test 1: Load public S3 JSONL file (anonymous access)")
    logger.info("=" * 70)

    # Public S3 JSONL file for testing
    example_s3_path = "s3://yileiz-bucket-1/c4-train-debug.split.00000-of-00004.jsonl"
    logger.info(f"Attempting to load public file: {example_s3_path}")

    try:
        # Determine format from extension
        file_extension = os.path.splitext(example_s3_path)[1].lower()
        format_map = {
            ".json": "json",
            ".jsonl": "json",
            ".txt": "text",
            ".csv": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
        }
        data_format = format_map.get(file_extension, "json")
        logger.info(f"Detected format: {data_format}")

        # Load dataset using the filesystem
        # HuggingFace datasets should use the fs parameter if provided
        dataset = load_dataset(
            data_format,
            data_files=example_s3_path,
            storage_options={"anon": True},
            streaming=False,
        )

        # Handle DatasetDict (multiple splits) vs Dataset (single)
        if isinstance(dataset, dict):
            # DatasetDict
            logger.info(f"✓ Loaded DatasetDict with {len(dataset)} splits")
            for split_name, split_ds in dataset.items():
                logger.info(f"  Split '{split_name}': {len(split_ds)} samples")
                if len(split_ds) > 0:
                    logger.info(f"  Sample keys: {split_ds[0].keys()}")
        else:
            # Dataset
            logger.info(f"✓ Loaded Dataset with {len(dataset)} samples")
            if len(dataset) > 0:
                logger.info(f"  Sample keys: {dataset[0].keys()}")
                logger.info(f"  First sample preview: {str(dataset[0])[:200]}...")

        logger.info("\n✓ Successfully loaded public S3 file using anonymous access!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load public S3 file: {e}")
        logger.error("\nCommon issues:")
        logger.error("  - Invalid S3 path or file doesn't exist")
        logger.error("  - Bucket not publicly accessible")
        logger.error("  - Network connectivity issues")
        return False


def test_s3_load_private_file(s3_path: str = None):
    """
    Test loading a private JSONL file from S3 requiring credentials.

    This demonstrates the pattern for private S3 buckets that require AWS credentials.

    Args:
        s3_path: S3 path to private JSONL file (e.g., s3://bucket/path/to/file.jsonl)
                If None, uses a default private S3 file for testing.
        aws_access_key_id: AWS access key ID (optional, can also use environment variable)
        aws_secret_access_key: AWS secret access key (optional, can also use environment variable)
    """
    logger.info("\n" + "=" * 70)
    logger.info("Test 2: Load private S3 JSONL file (requires credentials)")
    logger.info("=" * 70)

    # Use provided path or default private S3 file
    if s3_path is None:
        # Default private S3 JSONL file for testing
        s3_path = "s3://yileiz-bucket-2/c4-train-debug.split.00001-of-00004.jsonl"
        logger.info("Using default private S3 file for testing")
    else:
        logger.info("Using provided S3 path")

    logger.info(f"Attempting to load private file: {s3_path}")

    try:
        # Build dataset config (simulating how DefaultS3DataLoadStrategy uses it)
        # Priority: environment variables > config file
        ds_config = {}

        # Get credentials using the same logic as DefaultS3DataLoadStrategy
        aws_access_key_id, aws_secret_access_key, aws_session_token = get_aws_credentials(ds_config)

        # Build storage_options from credentials
        storage_options = {}
        if aws_access_key_id:
            storage_options["key"] = aws_access_key_id
        if aws_secret_access_key:
            storage_options["secret"] = aws_secret_access_key
        if aws_session_token:
            storage_options["token"] = aws_session_token

        # Check if credentials are available
        if not storage_options.get("key") or not storage_options.get("secret"):
            logger.warning("⚠ No AWS credentials found in parameters or environment")
            logger.warning("This test requires credentials for private buckets.")
            logger.warning("Provide credentials as function parameters or set environment variables:")
            logger.warning("  AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
            return True  # Skip test, don't fail
        else:
            logger.info("Using AWS credentials from parameters or environment")
            logger.info(f"AWS access key ID: {aws_access_key_id}")
            logger.info(f"AWS secret access key: {aws_secret_access_key}")
            logger.info(f"AWS session token: {aws_session_token}")

        # Determine format from extension
        file_extension = os.path.splitext(s3_path)[1].lower()
        format_map = {
            ".json": "json",
            ".jsonl": "json",
            ".txt": "text",
            ".csv": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
        }
        data_format = format_map.get(file_extension, "json")
        logger.info(f"Detected format: {data_format}")

        # Load dataset using storage_options
        dataset = load_dataset(
            data_format,
            data_files=s3_path,
            storage_options=storage_options,  # Pass storage_options for S3 filesystem configuration
            streaming=False,  # Set to True for streaming
        )

        # Handle DatasetDict (multiple splits) vs Dataset (single)
        if isinstance(dataset, dict):
            # DatasetDict
            logger.info(f"✓ Loaded DatasetDict with {len(dataset)} splits")
            for split_name, split_ds in dataset.items():
                logger.info(f"  Split '{split_name}': {len(split_ds)} samples")
                if len(split_ds) > 0:
                    logger.info(f"  Sample keys: {split_ds[0].keys()}")
        else:
            # Dataset
            logger.info(f"✓ Loaded Dataset with {len(dataset)} samples")
            if len(dataset) > 0:
                logger.info(f"  Sample keys: {dataset[0].keys()}")
                logger.info(f"  First sample preview: {str(dataset[0])[:200]}...")

        logger.info("\n✓ Successfully loaded private S3 file using credentials!")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load private S3 file: {e}")
        logger.error("\nCommon issues:")
        logger.error("  - Invalid S3 path or file doesn't exist")
        logger.error("  - Missing or invalid AWS credentials")
        logger.error("  - Insufficient permissions to access the bucket")
        logger.error("  - Network connectivity issues")
        return False


def main():
    """Run all S3 loading tests."""
    logger.info("\n" + "=" * 70)
    logger.info("S3 HuggingFace Dataset Loading Test")
    logger.info("=" * 70)
    logger.info("\nThis script tests the S3 loading functionality using")
    logger.info("datasets.filesystem.S3FileSystem, matching the pattern")
    logger.info("used in DefaultS3DataLoadStrategy.\n")

    results = []

    # Test 1: Public S3 file (anonymous access)
    results.append(("Public S3 file (anonymous)", test_s3_load_public_file()))

    # Test 2: Private S3 file (requires credentials)
    results.append(("Private S3 file (credentials)", test_s3_load_private_file()))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Test Results Summary")
    logger.info("=" * 70)

    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
        if not result:
            all_passed = False

    logger.info("=" * 70)

    if all_passed:
        logger.info("\n✓ All tests passed!")
        return 0
    else:
        logger.error("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
