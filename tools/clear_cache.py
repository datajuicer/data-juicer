#!/usr/bin/env python3
"""
Cache clearing script for Data-Juicer.
Clears all types of caches to ensure fresh model loading.
"""

import gc
import os
import shutil


def clear_data_juicer_cache():
    """Clear all Data-Juicer related caches."""
    print("🧹 Clearing Data-Juicer caches...")

    # Clear model cache from memory
    try:
        from data_juicer.utils.model_utils import free_models

        free_models(clear_model_zoo=True)
        print("✅ Cleared model cache from memory")
    except Exception as e:
        print(f"⚠️  Could not clear model cache from memory: {e}")

    # Clear downloaded model files
    try:
        from data_juicer.utils.cache_utils import DATA_JUICER_MODELS_CACHE

        if os.path.exists(DATA_JUICER_MODELS_CACHE):
            shutil.rmtree(DATA_JUICER_MODELS_CACHE)
            print(f"✅ Cleared downloaded models: {DATA_JUICER_MODELS_CACHE}")
        else:
            print("ℹ️  No downloaded models cache found")
    except Exception as e:
        print(f"⚠️  Could not clear downloaded models: {e}")

    # Clear assets cache
    try:
        from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE

        if os.path.exists(DATA_JUICER_ASSETS_CACHE):
            shutil.rmtree(DATA_JUICER_ASSETS_CACHE)
            print(f"✅ Cleared assets cache: {DATA_JUICER_ASSETS_CACHE}")
        else:
            print("ℹ️  No assets cache found")
    except Exception as e:
        print(f"⚠️  Could not clear assets cache: {e}")


def clear_huggingface_cache():
    """Clear HuggingFace cache."""
    print("🤗 Clearing HuggingFace cache...")

    try:
        from transformers import TRANSFORMERS_CACHE

        if os.path.exists(TRANSFORMERS_CACHE):
            shutil.rmtree(TRANSFORMERS_CACHE)
            print(f"✅ Cleared HuggingFace cache: {TRANSFORMERS_CACHE}")
        else:
            print("ℹ️  No HuggingFace cache found")
    except Exception as e:
        print(f"⚠️  Could not clear HuggingFace cache: {e}")


def clear_nltk_cache():
    """Clear NLTK cache."""
    print("📚 Clearing NLTK cache...")

    try:
        from data_juicer.utils.nltk_utils import clean_nltk_cache

        clean_nltk_cache(complete_reset=True)
        print("✅ Cleared NLTK cache")
    except Exception as e:
        print(f"⚠️  Could not clear NLTK cache: {e}")


def clear_python_cache():
    """Clear Python cache files."""
    print("🐍 Clearing Python cache...")

    # Clear __pycache__ directories
    cache_dirs = []
    for root, dirs, files in os.walk("."):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                cache_path = os.path.join(root, dir_name)
                cache_dirs.append(cache_path)
                try:
                    shutil.rmtree(cache_path)
                except Exception as e:
                    print(f"⚠️  Could not clear {cache_path}: {e}")

    if cache_dirs:
        print(f"✅ Cleared {len(cache_dirs)} Python cache directories")
    else:
        print("ℹ️  No Python cache directories found")


def clear_system_cache():
    """Clear system-level caches."""
    print("💻 Clearing system caches...")

    # Clear macOS system cache (if on macOS)
    if os.uname().sysname == "Darwin":
        try:
            # Clear various macOS caches
            cache_paths = [
                os.path.expanduser("~/Library/Caches"),
                "/System/Library/Caches",
            ]

            for cache_path in cache_paths:
                if os.path.exists(cache_path):
                    # Only clear specific subdirectories to avoid system issues
                    for item in os.listdir(cache_path):
                        item_path = os.path.join(cache_path, item)
                        if os.path.isdir(item_path) and "python" in item.lower():
                            try:
                                shutil.rmtree(item_path)
                                print(f"✅ Cleared system cache: {item_path}")
                            except Exception:
                                pass  # Skip if we can't clear it
        except Exception as e:
            print(f"⚠️  Could not clear system cache: {e}")


def force_garbage_collection():
    """Force garbage collection to free memory."""
    print("🗑️  Running garbage collection...")

    # Force garbage collection
    gc.collect()

    # Clear any remaining references
    import sys

    for module_name in list(sys.modules.keys()):
        if module_name.startswith("data_juicer") or "transformers" in module_name:
            try:
                del sys.modules[module_name]
            except Exception:
                pass

    # Force another garbage collection
    gc.collect()
    print("✅ Garbage collection completed")


def main():
    """Main function to clear all caches."""
    print("🚀 Starting comprehensive cache clearing...")
    print("=" * 50)

    # Clear all types of caches
    clear_data_juicer_cache()
    print()

    clear_huggingface_cache()
    print()

    clear_nltk_cache()
    print()

    clear_python_cache()
    print()

    clear_system_cache()
    print()

    force_garbage_collection()
    print()

    print("=" * 50)
    print("✅ Cache clearing completed!")
    print("\n💡 Next time you run the benchmark, models will be loaded fresh from disk.")
    print("   This should eliminate the caching speed difference between runs.")


if __name__ == "__main__":
    main()
