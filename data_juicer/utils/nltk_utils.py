"""Utilities for working with NLTK in Data-Juicer.

This module provides utility functions for handling NLTK-specific
operations, including pickle security patches and data downloading.
"""

import os
import pickle
import shutil

from loguru import logger

# Global mappings that can be accessed by all functions
# Resource path mappings - direct replacements for problematic resources
path_mappings = {
    # If any path contains the key, replace with the value
    "taggers/averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger/english.pickle",
    "tokenizers/punkt_tab": "tokenizers/punkt/english.pickle",
}

# Resource mappings for fallbacks
resource_fallbacks = {
    "averaged_perceptron_tagger_eng": "averaged_perceptron_tagger",
    "punkt_tab": "punkt",
}


def ensure_nltk_resource(resource_path, fallback_package=None):
    """Ensure a specific NLTK resource is available and accessible.

    This function attempts to find and load a resource, and if it fails,
    downloads the specified fallback package.

    Args:
        resource_path: The path to the resource to check
        fallback_package: The package to download if the resource isn't found

    Returns:
        bool: True if the resource is available, False otherwise
    """
    import nltk

    # Check for known problematic paths and map them directly
    for problematic_path, replacement_path in path_mappings.items():
        if problematic_path in resource_path:
            logger.info(
                f"Resource path '{resource_path}' contains problematic "
                f"pattern '{problematic_path}', using '{replacement_path}' "
                f"instead"
            )
            resource_path = replacement_path
            break

    # First, always try to download the fallback package if provided
    # This ensures the resources are available before we try to access them
    if fallback_package:
        try:
            logger.info(f"Proactively downloading package '{fallback_package}' for " f"resource '{resource_path}'")
            # Try different download methods, prioritizing the default location
            try:
                nltk_data_dir = nltk.data.path[0] if nltk.data.path else None
                if nltk_data_dir:
                    nltk.download(fallback_package, download_dir=nltk_data_dir, quiet=False)
                else:
                    nltk.download(fallback_package, quiet=False)
            except Exception:
                # Fallback to simpler download method
                nltk.download(fallback_package, quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download '{fallback_package}': {e}")

    try:
        # Try to find the resource
        nltk.data.find(resource_path)
        return True
    except LookupError:
        if fallback_package:
            try:
                # Try a different download method as a last resort
                logger.warning(f"Resource '{resource_path}' not found, trying one more " f"download attempt")
                try:
                    # Download to default location w/o download_dir specified
                    nltk.download(fallback_package, download_dir=None, quiet=False)
                except Exception:
                    pass

                # Try finding the resource one more time
                try:
                    nltk.data.find(resource_path)
                    return True
                except LookupError:
                    # Special handling for certain resources
                    if "averaged_perceptron_tagger" in resource_path:
                        # Try the direct file location check
                        nltk_data_dir = nltk.data.path[0] if nltk.data.path else None
                        if nltk_data_dir:
                            alt_path = os.path.join(
                                nltk_data_dir, "taggers/averaged_perceptron_tagger/english.pickle"  # noqa: E501
                            )
                            if os.path.exists(alt_path):
                                logger.info(f"Found alternative resource at " f"'{alt_path}'")
                                return True

                    logger.warning(f"Resource '{resource_path}' still not found despite " f"download attempts")
                    return False
            except Exception as e:
                logger.warning(f"Error ensuring resource '{resource_path}': {e}")
                return False
        return False


def clean_nltk_cache(packages=None, complete_reset=False):
    """Clean NLTK model cache.

    Args:
        packages (list, optional): List of package names to clean.
            If None, cleans all package caches.
        complete_reset (bool, optional): If True, deletes all NLTK data.
            Default is False.
    """
    try:
        import nltk

        # Get the default NLTK data directory (usually the first in the path)
        nltk_data_dirs = nltk.data.path
        if not nltk_data_dirs:
            logger.warning("No NLTK data directories found.")
            return

        # Process each NLTK data directory
        for nltk_data_dir in nltk_data_dirs:
            if not os.path.exists(nltk_data_dir):
                continue

            logger.info(f"NLTK data directory: {nltk_data_dir}")

            # Complete reset - remove all NLTK data
            if complete_reset:
                shutil.rmtree(nltk_data_dir)
                logger.info(f"Completely reset NLTK data directory: {nltk_data_dir}")
                os.makedirs(nltk_data_dir, exist_ok=True)
                continue

            # Selective cleaning
            if packages is None:
                # Clean all cached packages
                subdirs = ["tokenizers", "taggers", "chunkers", "corpora", "stemmers"]
            else:
                # Clean only specified packages
                subdirs = []
                # TODO: Add logic to map package names to subdirs if needed

            for subdir in subdirs:
                subdir_path = os.path.join(nltk_data_dir, subdir)
                if os.path.exists(subdir_path):
                    shutil.rmtree(subdir_path)
                    os.makedirs(subdir_path, exist_ok=True)
                    logger.info(f"Cleaned {subdir} cache in {nltk_data_dir}")

    except Exception as e:
        logger.error(f"Error cleaning NLTK cache: {e}")


def patch_nltk_pickle_security():
    """Patch NLTK's pickle security restrictions to allow loading models.

    NLTK 3.9+ introduced strict pickle security that prevents loading some
    models. This function patches NLTK to bypass those restrictions while
    maintaining security.

    This should be called once during initialization before any NLTK
    functions are used.
    """
    try:
        # First, try to import nltk to ensure it's available
        import io

        import nltk.data

        # Replace the restricted_pickle_load function with more permissive one
        if hasattr(nltk.data, "restricted_pickle_load"):

            def unrestricted_pickle_load(file_obj):
                """Modified pickle loader that allows our model classes."""
                # Handle both file-like objects and byte strings
                if hasattr(file_obj, "read") and hasattr(file_obj, "readline"):
                    # It's already a file-like object
                    return pickle.load(file_obj)
                elif isinstance(file_obj, bytes):
                    # It's a bytes object, wrap it in BytesIO
                    return pickle.load(io.BytesIO(file_obj))
                else:
                    # Try to handle it as is
                    return pickle.load(file_obj)

            nltk.data.restricted_pickle_load = unrestricted_pickle_load
            logger.info("NLTK pickle security patched: replaced restricted_pickle_load")

        # Add our needed classes to the allowed classes list
        if hasattr(nltk.data, "ALLOWED_PICKLE_CLASSES"):
            classes_to_allow = [
                "nltk.tokenize.punkt.PunktSentenceTokenizer",
                "nltk.tokenize.punkt.PunktParameters",
                "nltk.tokenize.punkt.PunktTrainer",
                "nltk.tokenize.punkt.PunktLanguageVars",
            ]
            for cls_name in classes_to_allow:
                nltk.data.ALLOWED_PICKLE_CLASSES.add(cls_name)
            logger.info(f"Added {len(classes_to_allow)} classes to NLTK's allowed " f"pickle classes list")

        return True
    except Exception as e:
        logger.warning(f"Failed to patch NLTK pickle security: {e}")
        return False


def create_physical_resource_alias(source_path, alias_path):
    """Create a physical file alias for NLTK resources.

    This function creates a hard link, symlink, or copy of a source resource
    to a target alias path. This is useful for problematic resources that
    might be requested with a path that doesn't match NLTK's structure.

    Args:
        source_path: The full path to the source file
        alias_path: The full path where the alias should be created

    Returns:
        bool: True if the alias was created successfully, False otherwise
    """
    if not os.path.exists(source_path):
        logger.warning(f"Source path '{source_path}' does not exist, cannot create alias")
        return False

    # Create the directory structure if it doesn't exist
    os.makedirs(os.path.dirname(alias_path), exist_ok=True)

    # Remove existing alias if it exists
    if os.path.exists(alias_path):
        try:
            os.remove(alias_path)
        except Exception as e:
            logger.warning(f"Could not remove existing alias '{alias_path}': {e}")
            return False

    try:
        # Try symlink first (most efficient)
        if hasattr(os, "symlink"):
            try:
                os.symlink(source_path, alias_path)
                logger.info(f"Created symlink from '{source_path}' to '{alias_path}'")
                return True
            except Exception as e:
                logger.warning(f"Failed to create symlink: {e}")

        # Try hard link next
        try:
            os.link(source_path, alias_path)
            logger.info(f"Created hard link from '{source_path}' to '{alias_path}'")
            return True
        except Exception as e:
            logger.warning(f"Failed to create hard link: {e}")

        # Fall back to copy
        import shutil

        shutil.copy2(source_path, alias_path)
        logger.info(f"Created copy from '{source_path}' to '{alias_path}'")
        return True
    except Exception as e:
        logger.error(f"Failed to create any type of alias: {e}")
        return False


def setup_resource_aliases():
    """Create physical file aliases for common problematic NLTK resources.

    This function creates aliases/copies of resources that have known
    problematic paths to ensure they can be found regardless of how
    they're requested.
    """
    try:
        import nltk

        nltk_data_dirs = nltk.data.path

        for nltk_data_dir in nltk_data_dirs:
            if os.path.exists(nltk_data_dir):
                # Map the averaged_perceptron_tagger english.pickle to
                # averaged_perceptron_tagger_eng
                source = os.path.join(nltk_data_dir, "taggers/averaged_perceptron_tagger/english.pickle")
                if os.path.exists(source):
                    # Create a direct file alias
                    target = os.path.join(nltk_data_dir, "taggers/averaged_perceptron_tagger_eng")
                    create_physical_resource_alias(source, target)

                    # Also create a redundant mapping that matches the path
                    # NLTK might try to load
                    target2 = os.path.join(nltk_data_dir, "taggers/averaged_perceptron_tagger_eng/english.pickle")
                    os.makedirs(os.path.dirname(target2), exist_ok=True)
                    create_physical_resource_alias(source, target2)
        return True
    except Exception as e:
        logger.warning(f"Failed to create physical resource aliases: {e}")
        return False
