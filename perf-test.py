#!/usr/bin/env python3
"""
Simple single-operator benchmark to test data loading and Ray Data parallelism.
Enhanced for debugging Ray/DataJuicer GPU actor initialization issues.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from loguru import logger

# ── Paths ─────────────────────────────────────────────────────────────────────
DJ_CODE_PATH = "/mnt/workspace/yileiz/data-juicer"
OUTPUT_DIR = "/mnt/workspace/yileiz/outputs/partitioned_ray/simple_workdir"
MODEL_PATH = "/mnt/workspace/miaoxiang.zfr/models/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE"
DEFAULT_CAPTION_JSONL = "/mnt/workspace/miaoxiang.zfr/data/Youku-AliceMind/caption_val_abs_6k.jsonl"
DEFAULT_VIDEO_DIR = "/mnt/workspace/shurui.ksr/Project/data/modelscope/Youku-AliceMind/videos/caption"
# ──────────────────────────────────────────────────────────────────────────────

if os.path.exists(DJ_CODE_PATH):
    sys.path.insert(0, DJ_CODE_PATH)


def setup_logging(log_dir=None):
    """Setup logging to file and console."""
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "benchmark.log")

    logger.remove()

    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )

    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="100 MB",
    )

    logger.info(f"Log file: {log_file}")
    return log_dir, log_file


def monitor_gpu():
    """Print GPU utilization."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        logger.info(f"GPU Status:\n{result.stdout}")
    except Exception as e:
        logger.warning(f"Failed to query GPU status: {e}")


def log_ray_paths():
    """Print likely Ray log locations for easier debugging."""
    ray_tmp = "/tmp/ray"
    if os.path.exists(ray_tmp):
        logger.info(f"Ray temp dir exists: {ray_tmp}")
        logger.info("Check Ray logs under: /tmp/ray/session_latest/logs/")
        logger.info("Ray Data logs often under: /tmp/ray/session_latest/logs/ray-data/")
    else:
        logger.warning("Ray temp dir /tmp/ray not found yet")


def prepare_jsonl_from_caption(jsonl_path, video_base_dir, num_samples=None, output_path=None):
    """Prepare JSONL with absolute video paths."""
    if output_path is None:
        output_path = jsonl_path.replace(".jsonl", "_abs.jsonl")

    if os.path.exists(output_path):
        logger.info(f"Output already exists: {output_path}")
        return output_path

    count = 0
    missing = 0
    with open(jsonl_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if num_samples and count >= num_samples:
                break
            sample = json.loads(line)
            videos = sample.get("videos", [])
            abs_videos = [os.path.join(video_base_dir, os.path.basename(v)) for v in videos]
            if all(os.path.exists(v) for v in abs_videos):
                out_sample = {"videos": abs_videos, "text": sample.get("caption", "")}
                f_out.write(json.dumps(out_sample, ensure_ascii=False) + "\n")
                count += 1
            else:
                missing += 1

    logger.info(f"Created {output_path} with {count} samples, skipped {missing} missing-video samples")
    return output_path


def split_jsonl(jsonl_path, num_shards=96):
    """Split JSONL into shards."""
    shard_dir = jsonl_path.replace(".jsonl", f"_sharded_{num_shards}")
    marker = os.path.join(shard_dir, "_DONE")

    if os.path.exists(marker):
        logger.info(f"Sharded data exists: {shard_dir}")
        return shard_dir

    os.makedirs(shard_dir, exist_ok=True)

    writers = [open(os.path.join(shard_dir, f"shard_{i:04d}.jsonl"), "w") for i in range(num_shards)]

    count = 0
    try:
        with open(jsonl_path, "r") as f_in:
            for line in f_in:
                writers[count % num_shards].write(line)
                count += 1
    finally:
        for w in writers:
            w.close()

    with open(marker, "w") as f:
        f.write(f"{count} samples\n")

    logger.info(f"Split {count} samples into {num_shards} shards")
    return shard_dir


def require_module(module_name, pip_hint=None):
    """Fail fast if module is missing."""
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        hint = f" Please install it first: {pip_hint}" if pip_hint else ""
        raise RuntimeError(f"Missing required module [{module_name}].{hint}\nOriginal error: {e}") from e


def precheck_environment(fail_fast=True):
    """
    Precheck environment in driver process to avoid hanging inside Ray actors.
    """
    logger.info("=" * 80)
    logger.info("Prechecking environment before starting Ray actors")
    logger.info("=" * 80)

    # Basic env
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f'HF_ENDPOINT={os.environ.get("HF_ENDPOINT")}')

    # Model path
    if not os.path.exists(MODEL_PATH):
        msg = f"Model path does not exist: {MODEL_PATH}"
        if fail_fast:
            raise FileNotFoundError(msg)
        logger.warning(msg)
    else:
        logger.info(f"Model path exists: {MODEL_PATH}")

    # Required modules
    require_module("torch", "pip install torch")
    require_module("transformers", "pip install transformers")
    require_module("ray", "pip install ray")
    require_module("pyarrow", "pip install pyarrow")

    # Torch / CUDA visibility
    import torch

    logger.info(f"torch version: {torch.__version__}")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            except Exception:
                pass

    logger.info("Environment precheck passed.")


def init_ray(object_store_gb=300, num_gpus=8):
    """Initialize Ray with better defaults."""
    # Pre-import to avoid circular import issues in Ray workers
    logger.info("Pre-importing modules to avoid fsspec issues in Ray workers...")
    import fsspec
    import fsspec.spec
    import fsspec.utils  # noqa: F401

    try:
        from huggingface_hub import HfFileSystem  # noqa: F401
    except ImportError:
        pass  # OK if not available

    import ray

    if ray.is_initialized():
        logger.info("Ray already initialized")
        return

    # Check if there's a running Ray cluster
    ray_address = os.environ.get("RAY_ADDRESS")

    if ray_address:
        # Connect to specified cluster
        logger.info(f"Connecting to Ray cluster at {ray_address}...")
        ray.init(address=ray_address)
        logger.info("Connected to existing Ray cluster")
    else:
        # Start a new local Ray instance
        logger.info(f"Starting new Ray instance with {num_gpus} GPUs, {object_store_gb}GB object store...")
        ray.init(
            num_gpus=num_gpus,
            object_store_memory=object_store_gb * 1024**3,
        )
        logger.info(f"Ray initialized successfully")

    log_ray_paths()


def run_simple_benchmark(
    data_path,
    num_shards=96,
    num_partitions=8,
    fail_fast=True,
    executor_type="ray",
):
    """Run benchmark with DataJuicer + video_aesthetics_filter.

    Args:
        executor_type: 'ray' (standard, uses all GPUs) or 'ray_partitioned' (partitioned).
            ray_partitioned auto-detects GPU count and runs partitions concurrently.
    """
    import ray  # noqa: F401
    import yaml

    from data_juicer.config import init_configs
    from data_juicer.core.executor.ray_executor import RayExecutor
    from data_juicer.core.executor.ray_executor_partitioned import (
        PartitionedRayExecutor,
    )

    # Environment
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Fail fast before actors
    precheck_environment(fail_fast=fail_fast)

    # Initialize Ray
    init_ray(object_store_gb=300)

    # Shard data
    if os.path.isfile(data_path):
        data_path = split_jsonl(data_path, num_shards)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = os.path.join(OUTPUT_DIR, f"dj_run_{timestamp}")
    os.makedirs(work_dir, exist_ok=True)

    logger.info(f"Using executor type: {executor_type}")

    # Detect available GPUs from Ray cluster
    import ray as _ray

    num_gpus = int(_ray.cluster_resources().get("GPU", 0))
    if num_gpus <= 0:
        raise RuntimeError("No GPUs available in Ray cluster")
    logger.info(f"Detected {num_gpus} GPUs in Ray cluster")

    # Base config
    cfg_dict = {
        "project_name": "simple-benchmark",
        "executor_type": executor_type,
        "dataset_path": data_path,
        "export_path": os.path.join(work_dir, "result.jsonl"),
        "work_dir": work_dir,
        "video_key": "videos",
        "skip_op_error": False,  # fail loudly
        "use_cache": False,
        "open_monitor": True,
        "debug": False,
        "auto_op_parallelism": False,  # Disable auto calculation to use explicit num_proc
        "process": [
            {
                "video_aesthetics_filter": {
                    "hf_scorer_model": MODEL_PATH,
                    "trust_remote_code": True,
                    "min_score": 0.4,
                    "max_score": 1.0,
                    "frame_num": 9223372036854775807,  # sys.maxsize - use all frames
                    "reduce_mode": "avg",
                    "skip_op_error": False,  # fail loudly during debugging
                    "batch_mode": True,
                    "num_gpus": 1,
                    "num_proc": num_gpus,
                },
            },
        ],
    }

    # Add partition config only for ray_partitioned executor
    if executor_type == "ray_partitioned":
        cfg_dict["partition"] = {
            "mode": "manual",
            "num_of_partitions": num_partitions,
        }
        cfg_dict["checkpoint"] = {
            "enabled": False,
        }

    config_path = os.path.join(work_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg_dict, f, allow_unicode=True, sort_keys=False)

    logger.info(f"Config saved to {config_path}")
    logger.info(f"Work dir: {work_dir}")
    logger.info(f"Data path: {data_path}")
    if executor_type == "ray_partitioned":
        logger.info(f"Num partitions: {num_partitions}")

    monitor_gpu()

    cfg = init_configs(args=["--config", config_path])

    t0 = time.time()
    if executor_type == "ray":
        executor = RayExecutor(cfg)
    else:
        executor = PartitionedRayExecutor(cfg)
    logger.info(f"Executor init ({executor_type}): {time.time() - t0:.2f}s")

    t1 = time.time()
    try:
        executor.run()
    except Exception:
        logger.exception("DataJuicer execution failed")
        logger.error(f"Please inspect Ray logs under /tmp/ray/session_latest/logs/")
        raise

    logger.info(f"Processing: {time.time() - t1:.2f}s")
    monitor_gpu()
    logger.info(f"Total: {time.time() - t0:.2f}s")
    logger.info(f"Output dir: {work_dir}")


def run_ray_data_test(data_path, num_shards=96):
    """Test raw Ray Data parallelism without DataJuicer."""
    import ray

    if os.path.isfile(data_path):
        data_path = split_jsonl(data_path, num_shards)

    init_ray(object_store_gb=100)

    logger.info(f"Reading data from {data_path}")

    t0 = time.time()
    ds = ray.data.read_json(data_path)
    count = ds.count()
    try:
        num_blocks = ds.num_blocks()
    except Exception:
        num_blocks = "unknown_before_materialize"
    logger.info(f"Loaded dataset: {count} rows, {num_blocks} blocks")

    def count_videos(row):
        return {"video_count": len(row.get("videos", [])), "text_len": len(row.get("text", ""))}

    t1 = time.time()
    ds = ds.map(count_videos)
    result = ds.take(5)
    logger.info(f"Map result: {result}")
    logger.info(f"Map time: {time.time() - t1:.2f}s")

    t2 = time.time()
    total = ds.count()
    logger.info(f"Total rows: {total}, count time: {time.time() - t2:.2f}s")

    logger.info(f"Total time: {time.time() - t0:.2f}s")


def run_direct_gpu_test(
    data_path,
    num_shards=96,
    batch_size=8,
    gpu_concurrency=8,
    fail_fast=True,
):
    """
    Direct GPU test bypassing PartitionedRayExecutor.
    This tests if Ray Data GPU actors work correctly.
    """
    import pyarrow
    import ray

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # Precheck before actor creation
    precheck_environment(fail_fast=fail_fast)

    init_ray(object_store_gb=300)

    logger.info("Direct GPU Test - bypassing PartitionedRayExecutor")
    monitor_gpu()

    t0 = time.time()
    if os.path.isfile(data_path):
        data_path = split_jsonl(data_path, num_shards)

    ds = ray.data.read_json(data_path)
    row_count = ds.count()
    logger.info(f"Loaded {row_count} rows in {time.time() - t0:.2f}s")

    def add_stats_column(table: pyarrow.Table):
        new_column_data = [{} for _ in range(len(table))]
        return table.append_column("__dj__stats__", [new_column_data])

    ds = ds.map_batches(add_stats_column, batch_format="pyarrow")
    logger.info("Added __dj__stats__ column")

    from data_juicer.ops.filter.video_aesthetics_filter import VideoAestheticsFilter

    # Create operator on driver for validation only
    op_t0 = time.time()
    op = VideoAestheticsFilter(
        hf_scorer_model=MODEL_PATH,
        trust_remote_code=True,
        min_score=0.4,
        max_score=1.0,
        frame_num=9223372036854775807,  # sys.maxsize - use all frames
        reduce_mode="avg",
        num_gpus=1,
    )
    logger.info(f"Operator init on driver: {time.time() - op_t0:.2f}s")
    logger.info(f"Operator: {op._name}")
    logger.info(f"  use_cuda: {op.use_cuda()}")
    logger.info(f"  use_ray_actor: {op.use_ray_actor()}")
    logger.info(f"  num_gpus: {op.num_gpus}")
    logger.info(f"  num_proc: {op.num_proc}")

    # Restrict concurrency to available GPUs
    import torch

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpus <= 0:
        raise RuntimeError("No CUDA GPUs visible, cannot run direct GPU test")

    gpu_concurrency = min(gpu_concurrency, available_gpus)
    logger.info(f"Using gpu_concurrency={gpu_concurrency}, batch_size={batch_size}")

    # Prefer new API style: concurrency=
    t1 = time.time()
    logger.info("Creating Ray Data GPU actor pipeline...")

    try:
        ds = ds.map_batches(
            VideoAestheticsFilter,
            fn_constructor_args=op._init_args,
            fn_constructor_kwargs=op._init_kwargs,
            batch_size=batch_size,
            num_cpus=1,
            num_gpus=1,
            concurrency=gpu_concurrency,
            batch_format="pyarrow",
        )
        logger.info("Using map_batches(..., concurrency=...)")
    except TypeError:
        # Fallback for older Ray versions
        from ray.data import ActorPoolStrategy

        logger.warning("Ray version does not support concurrency= here, fallback to ActorPoolStrategy")
        ds = ds.map_batches(
            VideoAestheticsFilter,
            fn_constructor_args=op._init_args,
            fn_constructor_kwargs=op._init_kwargs,
            batch_size=batch_size,
            num_cpus=1,
            num_gpus=1,
            compute=ActorPoolStrategy(size=gpu_concurrency),
            batch_format="pyarrow",
        )

    logger.info("Executing pipeline...")
    t2 = time.time()
    try:
        result = ds.materialize()
    except Exception:
        logger.exception("Direct GPU pipeline execution failed")
        logger.error("Please inspect /tmp/ray/session_latest/logs/")
        raise

    logger.info(f"Pipeline execution: {time.time() - t2:.2f}s")

    count = result.count()
    logger.info(f"Result: {count} rows")

    monitor_gpu()
    logger.info(f"Total time: {time.time() - t0:.2f}s")
    logger.info(f"Pipeline setup time: {time.time() - t1:.2f}s")


def run_direct_gpu_test_dj_match(
    data_path,
    num_shards=96,
    batch_size=10,  # DJ CUDA default
    gpu_concurrency=8,
    fail_fast=True,
):
    """
    Direct GPU test that matches the DJ pipeline as closely as possible.
    Adds: convert_to_absolute_paths, count(), columns(), filter step.
    """
    from functools import partial

    import pyarrow
    import ray

    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    precheck_environment(fail_fast=fail_fast)
    init_ray(object_store_gb=300)

    logger.info("Direct GPU Test (DJ-matched pipeline)")
    monitor_gpu()

    t0 = time.time()
    if os.path.isfile(data_path):
        data_path = split_jsonl(data_path, num_shards)

    ds = ray.data.read_json(data_path)

    # --- Match DJ: count() before processing ---
    t_count = time.time()
    row_count = ds.count()
    logger.info(f"count(): {row_count} rows in {time.time() - t_count:.2f}s")

    # --- Match DJ: columns() ---
    t_cols = time.time()
    cols = ds.columns()
    logger.info(f"columns(): {cols} in {time.time() - t_cols:.2f}s")

    # --- Match DJ: convert_to_absolute_paths ---
    dataset_dir = os.path.dirname(data_path)

    def convert_to_absolute_paths(batch, dataset_dir, path_keys):
        for key in path_keys:
            if key in batch.column_names:
                col = batch.column(key)
                new_col = []
                for val in col.to_pylist():
                    if isinstance(val, list):
                        new_col.append([os.path.join(dataset_dir, p) if not os.path.isabs(p) else p for p in val])
                    elif isinstance(val, str):
                        new_col.append(os.path.join(dataset_dir, val) if not os.path.isabs(val) else val)
                    else:
                        new_col.append(val)
                idx = batch.column_names.index(key)
                batch = batch.set_column(idx, key, [new_col])
        return batch

    path_keys = [k for k in ["videos", "images", "audios"] if k in cols]
    if path_keys:
        ds = ds.map_batches(
            partial(convert_to_absolute_paths, dataset_dir=dataset_dir, path_keys=path_keys),
            batch_format="pyarrow",
            zero_copy_batch=True,
            batch_size=1000,
        )
        logger.info(f"Added convert_to_absolute_paths for keys: {path_keys}")

    # --- Match DJ: add __dj__stats__ column ---
    def add_stats_column(table: pyarrow.Table):
        new_column_data = [{} for _ in range(len(table))]
        return table.append_column("__dj__stats__", [new_column_data])

    ds = ds.map_batches(add_stats_column, batch_format="pyarrow", batch_size=1000)
    logger.info("Added __dj__stats__ column")

    # --- Match DJ: compute_stats via actor ---
    from data_juicer.ops.filter.video_aesthetics_filter import VideoAestheticsFilter

    op = VideoAestheticsFilter(
        hf_scorer_model=MODEL_PATH,
        trust_remote_code=True,
        min_score=0.4,
        max_score=1.0,
        frame_num=9223372036854775807,
        reduce_mode="avg",
        num_gpus=1,
        batch_mode=True,
    )
    logger.info(f"Op: {op._name}, batch_size={batch_size}, is_batched={op.is_batched_op()}")

    import torch

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpus <= 0:
        raise RuntimeError("No CUDA GPUs visible")
    gpu_concurrency = min(gpu_concurrency, available_gpus)
    logger.info(f"gpu_concurrency={gpu_concurrency}, batch_size={batch_size}")

    t1 = time.time()
    ds = ds.map_batches(
        VideoAestheticsFilter,
        fn_constructor_args=op._init_args,
        fn_constructor_kwargs=op._init_kwargs,
        batch_size=batch_size,
        num_gpus=1,
        concurrency=gpu_concurrency,
        batch_format="pyarrow",
    )
    logger.info("Added compute_stats map_batches (actor mode)")

    # --- Match DJ: filter step ---
    def filter_batch(batch, filter_func):
        mask = pyarrow.array(filter_func(batch.to_pydict()))
        return batch.filter(mask)

    ds = ds.map_batches(
        partial(filter_batch, filter_func=op.process),
        batch_format="pyarrow",
        zero_copy_batch=True,
        batch_size=1000,
    )
    logger.info("Added filter_batch step")

    # --- Execute ---
    logger.info("Executing full DJ-matched pipeline...")
    t2 = time.time()
    try:
        result = ds.materialize()
    except Exception:
        logger.exception("Pipeline execution failed")
        raise

    logger.info(f"Pipeline execution: {time.time() - t2:.2f}s")
    count = result.count()
    logger.info(f"Result: {count} rows (filtered from {row_count})")
    monitor_gpu()
    logger.info(f"Total time: {time.time() - t0:.2f}s")
    logger.info(f"Pipeline time (from first map_batches): {time.time() - t1:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Simple benchmark")
    parser.add_argument(
        "--caption-jsonl",
        type=str,
        default=DEFAULT_CAPTION_JSONL,
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=DEFAULT_VIDEO_DIR,
    )
    parser.add_argument("--num-samples", type=int, default=6000)
    parser.add_argument("--num-shards", type=int, default=96)
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-concurrency", type=int, default=8)
    parser.add_argument("--fail-fast", action="store_true", default=True)
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.add_argument("--mode", type=str, choices=["ray", "dj", "gpu", "gpu-dj", "both"], default="gpu")
    parser.add_argument(
        "--executor",
        type=str,
        choices=["ray", "ray_partitioned"],
        default="ray",
        help='Executor type: "ray" (standard, parallel GPUs) or "ray_partitioned" (partitioned)',
    )
    args = parser.parse_args()

    log_dir, log_file = setup_logging()
    logger.info(f"Arguments: {args}")

    jsonl_path = prepare_jsonl_from_caption(args.caption_jsonl, args.video_dir, args.num_samples)

    if args.mode in ["ray", "both"]:
        logger.info("\n" + "=" * 60)
        logger.info("Testing Ray Data parallelism")
        logger.info("=" * 60)
        run_ray_data_test(jsonl_path, args.num_shards)

    if args.mode in ["dj", "both"]:
        logger.info("\n" + "=" * 60)
        logger.info(f"Testing DataJuicer with single operator (executor={args.executor})")
        logger.info("=" * 60)
        run_simple_benchmark(
            jsonl_path,
            num_shards=args.num_shards,
            num_partitions=args.partitions,
            fail_fast=args.fail_fast,
            executor_type=args.executor,
        )

    if args.mode == "gpu":
        logger.info("\n" + "=" * 60)
        logger.info("Testing Direct GPU (bypass PartitionedRayExecutor)")
        logger.info("=" * 60)
        run_direct_gpu_test(
            jsonl_path,
            num_shards=args.num_shards,
            batch_size=args.batch_size,
            gpu_concurrency=args.gpu_concurrency,
            fail_fast=args.fail_fast,
        )

    if args.mode == "gpu-dj":
        logger.info("\n" + "=" * 60)
        logger.info("Testing Direct GPU (DJ-matched pipeline)")
        logger.info("=" * 60)
        run_direct_gpu_test_dj_match(
            jsonl_path,
            num_shards=args.num_shards,
            batch_size=10,  # DJ CUDA default
            gpu_concurrency=args.gpu_concurrency,
            fail_fast=args.fail_fast,
        )


if __name__ == "__main__":
    main()
