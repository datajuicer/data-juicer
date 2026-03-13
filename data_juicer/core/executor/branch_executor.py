"""
Branch Executor: multi-branch execution from a common checkpoint (WIP).

Architecture:
- Common process: on driver (default) or on Ray when branch.run_common_on_ray and backend=ray (pure Ray).
- Thread backend: branches run in a thread pool, each with DefaultExecutor(dataset=common_dataset);
  no extra I/O, shared memory.
- Ray backend: common either runs on driver then we ray.put/save_to_disk for branches, or common runs as
  a Ray task and returns common_ds_spec (ref or path); branches use that spec. Branch configs are minimal
  serializable dicts. fail_fast: cancel remaining on first error. Retries: only failed branches resubmitted.

Example config:
    executor_type: 'branch'
    branch:
      backend: ray   # or "thread"
    common_process: [...]
    branches:
      - name: "branch1"
        process: [...]
        export_path: "./out/branch1.jsonl"

Future TODOs (Issue #915):
- DAGExecutionMixin integration for branch executor (visualize / track branch DAG).
- Per-branch Ray resource hints (num_cpus, num_gpus) and scheduling options.
- Optional common process as Ray actor for very large datasets (streaming / chunked).
- Schema validation for branch config (common_process, branches[*].process, export_path).
- Unit and integration tests for thread/ray/run_common_on_ray paths.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from data_juicer.core.data import DatasetBuilder, NestedDataset
from data_juicer.core.executor.base import ExecutorBase
from data_juicer.core.executor.default_executor import DefaultExecutor
from data_juicer.core.executor.event_logging_mixin import EventLoggingMixin, EventType
from data_juicer.core.executor.pipeline_dag import PipelineDAG


def _get_branch_opts(cfg: Any) -> Dict[str, Any]:
    """Read branch-related options from config (all optional)."""
    opts = getattr(cfg, "branch", None) or {}
    if not isinstance(opts, dict):
        opts = {}
    return {
        "backend": opts.get("backend", "thread"),  # "thread" | "ray"
        "run_common_on_ray": opts.get("run_common_on_ray", False),  # when backend=ray, run common as Ray task
        "parallel": opts.get("parallel", True),
        "fail_fast": opts.get("fail_fast", True),
        "max_workers": opts.get("max_workers"),  # None => len(branches)
        "retries": opts.get("retries", 0),
    }


def _run_common_on_ray(common_cfg_dict: dict) -> dict:
    """
    Run the common process on a Ray worker (module-level for serialization).
    Loads dataset from config (dataset_path must be worker-accessible, e.g. NFS/S3),
    runs DefaultExecutor with common_process, returns common_ds_spec: {"ref": ref} or {"path": path}.
    """
    import ray
    from jsonargparse import dict_to_namespace

    cfg = dict_to_namespace(common_cfg_dict)
    executor = DefaultExecutor(cfg)
    dataset = executor.run(load_data_np=cfg.np, skip_export=True, skip_return=True)
    try:
        ref = ray.put(dataset)
        return {"ref": ref}
    except Exception:
        import os
        path = os.path.join(cfg.checkpoint_dir, "dataset")
        dataset.save_to_disk(path)
        return {"path": path}


def _run_one_branch_ray(common_ds_spec: dict, branch_cfg_dict: dict) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Run a single branch on a Ray worker (module-level for serialization).
    common_ds_spec: {"path": str} to load from disk, or {"ref": ray.ObjectRef} to get from object store.
    branch_cfg_dict: minimal serializable config for DefaultExecutor (no full cfg).
    Returns (branch_name, export_path, error_msg).
    """
    import ray
    from jsonargparse import dict_to_namespace

    from data_juicer.core.data import NestedDataset

    branch_name = branch_cfg_dict.get("name", "unknown")
    try:
        if "ref" in common_ds_spec:
            ds = ray.get(common_ds_spec["ref"])
        else:
            ds = NestedDataset.load_from_disk(common_ds_spec["path"])
    except Exception as e:
        return (branch_name, None, str(e))
    try:
        cfg = dict_to_namespace(branch_cfg_dict)
        executor = DefaultExecutor(cfg)
        executor.run(dataset=ds, skip_export=False, skip_return=True)
        return (branch_name, getattr(cfg, "export_path", None), None)
    except Exception as e:
        return (branch_name, None, str(e))


class BranchExecutor(ExecutorBase, EventLoggingMixin):
    """
    Executes common_process once, then runs each branch (parallel or sequential).
    Optional: event logging, minimal branch DAG, per-branch retry and fail_fast/continue.
    """

    def __init__(self, cfg: Optional[Any] = None):
        super().__init__(cfg)
        self.executor_type = "branch"
        self.work_dir = self.cfg.work_dir
        self.job_id = getattr(self.cfg, "job_id", None)

        self.common_process = getattr(self.cfg, "common_process", [])
        self.branches = getattr(self.cfg, "branches", [])

        if not self.common_process:
            raise ValueError("common_process must be specified for branch executor")
        if not self.branches:
            raise ValueError("branches must be specified for branch executor")

        self.common_ckpt_dir = os.path.join(self.work_dir, ".branch_ckpt", "common")
        os.makedirs(self.common_ckpt_dir, exist_ok=True)

        self.datasetbuilder = DatasetBuilder(self.cfg, executor_type="default")

        # Ensure job_id for event logging when not set by config pipeline
        if getattr(self.cfg, "job_id", None) is None:
            setattr(self.cfg, "job_id", f"branch-{int(time.time())}")
            self.job_id = self.cfg.job_id

        # Optional event logging (no-op if event_logging disabled in config)
        if not hasattr(self, "event_logger"):
            EventLoggingMixin.__init__(self)

        # Minimal branch DAG: common -> [branch_1, branch_2, ...]
        self.pipeline_dag: Optional[PipelineDAG] = self._build_branch_dag()

    def _build_branch_dag(self) -> PipelineDAG:
        """Build a minimal DAG for common + branches (for monitoring/visualization)."""
        dag = PipelineDAG(self.work_dir)
        branch_ids = []
        for i, b in enumerate(self.branches):
            nid = b.get("name", f"branch_{i}")
            branch_ids.append(nid)
            dag.nodes[nid] = {
                "node_id": nid,
                "operation_name": nid,
                "node_type": "branch",
                "dependencies": ["common"],
                "execution_order": 1,
                "metadata": {"branch_index": i},
            }
        dag.nodes["common"] = {
            "node_id": "common",
            "operation_name": "common",
            "node_type": "common",
            "dependencies": [],
            "execution_order": 0,
            "metadata": {},
        }
        dag.parallel_groups = [branch_ids]
        dag.execution_plan = ["common"] + branch_ids
        dag.edges = [("common", b) for b in branch_ids]
        return dag

    def run(
        self,
        load_data_np: Optional[int] = None,
        skip_export: bool = False,
        skip_return: bool = False,
    ):
        """Run common process then all branches; return dict[branch_name] -> dataset."""
        run_start = time.time()
        opts = _get_branch_opts(self.cfg)

        if hasattr(self, "_log_event"):
            self._log_event(EventType.JOB_START, "Branch job started", metadata={"branches": [b.get("name", "") for b in self.branches]})

        logger.info("Starting branch execution...")
        backend = opts.get("backend", "thread")

        if backend == "ray" and opts.get("run_common_on_ray"):
            common_ds_spec = self._get_common_ds_spec_via_ray(load_data_np)
        else:
            common_dataset = self._execute_common_process(load_data_np)
            if backend == "ray":
                common_ds_spec = self._build_common_ds_spec(common_dataset)
            else:
                common_ds_spec = None  # thread/sequential use in-memory common_dataset

        if backend == "ray":
            branch_results, errors = self._run_branches_ray(common_ds_spec, load_data_np, skip_export, opts)
        elif opts["parallel"] and len(self.branches) > 1:
            branch_results, errors = self._run_branches_parallel(
                common_dataset, load_data_np, skip_export, opts
            )
        else:
            branch_results, errors = self._run_branches_sequential(
                common_dataset, load_data_np, skip_export, opts
            )

        if errors and opts["fail_fast"]:
            first_err = next(iter(errors.values()))
            if hasattr(self, "_log_event"):
                self._log_event(EventType.JOB_FAILED, f"Branch job failed: {first_err}", error_message=str(first_err))
            raise RuntimeError(f"Branch execution failed: {first_err}")

        if hasattr(self, "_log_event"):
            self._log_event(
                EventType.JOB_COMPLETE,
                f"Branch job completed in {time.time() - run_start:.2f}s",
                duration=time.time() - run_start,
                metadata={"completed": list(branch_results.keys()), "failed": list(errors.keys())},
            )
        logger.info("Branch execution completed")
        return branch_results

    def _run_branches_sequential(
        self,
        common_dataset: Any,
        load_data_np: Optional[int],
        skip_export: bool,
        opts: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        results, errors = {}, {}
        for branch in self.branches:
            name = branch.get("name", f"branch_{len(results) + len(errors)}")
            ds, err = self._execute_branch_with_retry(branch, common_dataset, load_data_np, skip_export, opts, name)
            if err:
                errors[name] = err
                if opts["fail_fast"]:
                    return results, errors
            else:
                results[name] = ds
        return results, errors

    def _run_branches_parallel(
        self,
        common_dataset: Any,
        load_data_np: Optional[int],
        skip_export: bool,
        opts: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        results, errors = {}, {}
        max_workers = opts["max_workers"] or len(self.branches)
        if hasattr(self, "_log_event"):
            self._log_event(
                EventType.DAG_PARALLEL_GROUP_START,
                f"Branch parallel group started ({len(self.branches)} branches)",
                metadata={"node_count": len(self.branches)},
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_branch = {
                executor.submit(
                    self._execute_branch_with_retry,
                    branch,
                    common_dataset,
                    load_data_np,
                    skip_export,
                    opts,
                    branch.get("name", f"branch_{i}"),
                ): branch
                for i, branch in enumerate(self.branches)
            }
            for future in as_completed(future_to_branch):
                branch = future_to_branch[future]
                name = branch.get("name", "unknown")
                try:
                    ds, err = future.result()
                    if err:
                        errors[name] = err
                        if opts["fail_fast"]:
                            for f in future_to_branch:
                                f.cancel()
                            return results, errors
                    else:
                        results[name] = ds
                except Exception as e:
                    errors[name] = str(e)
                    if opts["fail_fast"]:
                        for f in future_to_branch:
                            f.cancel()
                        return results, errors
        if hasattr(self, "_log_event"):
            self._log_event(
                EventType.DAG_PARALLEL_GROUP_COMPLETE,
                f"Branch parallel group completed",
                metadata={"completed_nodes": len(results), "node_count": len(self.branches)},
            )
        return results, errors

    def _build_common_ds_spec(self, common_dataset: Any) -> dict:
        """Build common_ds_spec from in-memory dataset (object store first, else disk)."""
        try:
            import ray
        except ImportError:
            raise RuntimeError("Ray backend requires 'ray' to be installed. pip install ray")
        try:
            ref = ray.put(common_dataset)
            return {"ref": ref}
        except Exception:
            path = os.path.join(self.common_ckpt_dir, "dataset")
            try:
                common_dataset.save_to_disk(path)
            except Exception as e:
                raise RuntimeError(f"Failed to save common dataset for Ray branches: {e}") from e
            return {"path": path}

    def _get_common_ds_spec_via_ray(self, load_data_np: Optional[int]) -> dict:
        """Run common process as a Ray task; return common_ds_spec (ref or path)."""
        try:
            import ray
        except ImportError:
            raise RuntimeError("Ray backend requires 'ray' to be installed. pip install ray")
        if not ray.is_initialized():
            ray.init()
        if hasattr(self, "_log_event"):
            self._log_event(EventType.DAG_NODE_START, "Common process (Ray task) started", operation_name="common")
        common_cfg_dict = self._common_cfg_minimal_dict(load_data_np)
        ref = ray.remote(_run_common_on_ray).remote(common_cfg_dict)
        common_ds_spec = ray.get(ref)
        if hasattr(self, "_log_event"):
            self._log_event(EventType.DAG_NODE_COMPLETE, "Common process (Ray task) completed", operation_name="common")
        return common_ds_spec

    def _common_cfg_minimal_dict(self, load_data_np: Optional[int]) -> dict:
        """Minimal serializable config for the common process (Ray worker). dataset_path must be worker-accessible."""
        np_val = load_data_np if load_data_np is not None else (getattr(self.cfg, "np", None) or 1)
        minimal = {
            "process": self.common_process,
            "export_path": os.path.join(self.common_ckpt_dir, "_common_skip_export.jsonl"),
            "work_dir": self.work_dir,
            "checkpoint_dir": self.common_ckpt_dir,
            "np": np_val,
            "use_checkpoint": getattr(self.cfg, "use_checkpoint", False),
            "use_cache": getattr(self.cfg, "use_cache", False),
            "cache_compress": getattr(self.cfg, "cache_compress", "gzip"),
            "export_type": getattr(self.cfg, "export_type", "jsonl"),
            "export_shard_size": getattr(self.cfg, "export_shard_size", None),
            "export_in_parallel": getattr(self.cfg, "export_in_parallel", False),
            "keep_stats_in_res_ds": getattr(self.cfg, "keep_stats_in_res_ds", False),
            "keep_hashes_in_res_ds": getattr(self.cfg, "keep_hashes_in_res_ds", False),
            "open_tracer": False,
            "op_fusion": getattr(self.cfg, "op_fusion", False),
            "fusion_strategy": getattr(self.cfg, "fusion_strategy", "sequential"),
            "adaptive_batch_size": getattr(self.cfg, "adaptive_batch_size", False),
            "open_monitor": getattr(self.cfg, "open_monitor", False),
        }
        if hasattr(self.cfg, "dataset_path") and self.cfg.dataset_path:
            minimal["dataset_path"] = self.cfg.dataset_path
        if hasattr(self.cfg, "dataset") and self.cfg.dataset:
            minimal["dataset"] = self.cfg.dataset
        if hasattr(self.cfg, "export_extra_args") and self.cfg.export_extra_args:
            minimal["export_extra_args"] = dict(self.cfg.export_extra_args)
        return minimal

    def _run_branches_ray(
        self,
        common_ds_spec: dict,
        load_data_np: Optional[int],
        skip_export: bool,
        opts: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """Run branches as Ray remote tasks; common_ds_spec is from driver or from _run_common_on_ray."""
        try:
            import ray
        except ImportError:
            raise RuntimeError("Ray backend requires 'ray' to be installed. pip install ray")

        if not ray.is_initialized():
            ray.init()

        results, errors = {}, {}
        branch_cfg_dicts = [
            self._branch_cfg_minimal_dict(branch, skip_export) for branch in self.branches
        ]
        fail_fast = opts.get("fail_fast", True)
        retries = opts.get("retries", 0)

        if hasattr(self, "_log_event"):
            self._log_event(
                EventType.DAG_PARALLEL_GROUP_START,
                f"Branch Ray parallel group started ({len(self.branches)} branches)",
                metadata={"node_count": len(self.branches), "backend": "ray"},
            )

        remote_fn = ray.remote(_run_one_branch_ray)
        all_names = [b.get("name", f"branch_{i}") for i, b in enumerate(self.branches)]
        refs = [remote_fn.remote(common_ds_spec, d) for d in branch_cfg_dicts]

        for attempt in range(retries + 1):
            remaining = list(refs)
            round_results, round_errors = {}, {}
            while remaining:
                ready, remaining = ray.wait(remaining, num_returns=1)
                branch_name, _, err = ray.get(ready[0])
                if err:
                    round_errors[branch_name] = err
                    if fail_fast:
                        if remaining:
                            try:
                                ray.cancel(remaining)
                            except Exception:
                                pass
                        return results, {**errors, **round_errors}
                else:
                    round_results[branch_name] = None
            results.update(round_results)
            for k in round_results:
                errors.pop(k, None)
            errors.update(round_errors)
            if not round_errors or attempt >= retries:
                break
            failed_names = set(round_errors)
            refs = [
                remote_fn.remote(common_ds_spec, branch_cfg_dicts[i])
                for i in range(len(all_names))
                if all_names[i] in failed_names
            ]
            if not refs:
                break

        if hasattr(self, "_log_event"):
            self._log_event(
                EventType.DAG_PARALLEL_GROUP_COMPLETE,
                "Branch Ray parallel group completed",
                metadata={"completed_nodes": len(results), "node_count": len(self.branches)},
            )
        return results, errors

    def _branch_cfg_minimal_dict(self, branch: Dict[str, Any], skip_export: bool) -> dict:
        """
        Build a minimal serializable config for one branch (Ray workers).
        Only includes keys DefaultExecutor needs; avoids deepcopy(full cfg) and
        non-serializable / large payloads.
        """
        branch_name = branch.get("name", "unknown")
        export_path = branch.get("export_path") if not skip_export else os.path.join(
            self.work_dir, ".branch_ckpt", branch_name, "_skip_export.jsonl"
        )
        minimal = {
            "name": branch_name,
            "process": branch.get("process", []),
            "export_path": export_path,
            "work_dir": self.work_dir,
            "checkpoint_dir": os.path.join(self.work_dir, ".branch_ckpt", branch_name),
            "np": getattr(self.cfg, "np", None) or 1,
            "use_checkpoint": False,
            "use_cache": getattr(self.cfg, "use_cache", False),
            "cache_compress": getattr(self.cfg, "cache_compress", "gzip"),
            "export_type": getattr(self.cfg, "export_type", "jsonl"),
            "export_shard_size": getattr(self.cfg, "export_shard_size", None),
            "export_in_parallel": getattr(self.cfg, "export_in_parallel", False),
            "keep_stats_in_res_ds": getattr(self.cfg, "keep_stats_in_res_ds", False),
            "keep_hashes_in_res_ds": getattr(self.cfg, "keep_hashes_in_res_ds", False),
            "open_tracer": False,
            "op_fusion": getattr(self.cfg, "op_fusion", False),
            "fusion_strategy": getattr(self.cfg, "fusion_strategy", "sequential"),
            "adaptive_batch_size": getattr(self.cfg, "adaptive_batch_size", False),
            "open_monitor": getattr(self.cfg, "open_monitor", False),
        }
        if hasattr(self.cfg, "export_extra_args") and self.cfg.export_extra_args:
            minimal["export_extra_args"] = dict(self.cfg.export_extra_args)
        return minimal

    def _execute_branch_with_retry(
        self,
        branch: Dict[str, Any],
        common_dataset: Any,
        load_data_np: Optional[int],
        skip_export: bool,
        opts: Dict[str, Any],
        branch_name: str,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Run one branch with optional retries. Returns (dataset, error_message)."""
        retries = opts.get("retries", 0)
        last_err = None
        for attempt in range(retries + 1):
            if hasattr(self, "_log_event"):
                self._log_event(
                    EventType.DAG_NODE_START,
                    f"Branch {branch_name} started" + (f" (attempt {attempt + 1})" if retries else ""),
                    operation_name=branch_name,
                    metadata={"attempt": attempt + 1},
                )
            start = time.time()
            try:
                ds = self._execute_branch(branch, common_dataset, load_data_np, skip_export)
                if hasattr(self, "_log_event"):
                    self._log_event(
                        EventType.DAG_NODE_COMPLETE,
                        f"Branch {branch_name} completed in {time.time() - start:.2f}s",
                        operation_name=branch_name,
                        duration=time.time() - start,
                    )
                return ds, None
            except Exception as e:
                last_err = str(e)
                if hasattr(self, "_log_event"):
                    self._log_event(
                        EventType.DAG_NODE_FAILED,
                        f"Branch {branch_name} failed: {last_err}",
                        operation_name=branch_name,
                        error_message=last_err,
                    )
                if attempt < retries:
                    logger.warning(f"Branch {branch_name} attempt {attempt + 1} failed, retrying: {e}")
                else:
                    return None, last_err
        return None, last_err

    def _execute_common_process(self, load_data_np: Optional[int] = None):
        common_cfg = deepcopy(self.cfg)
        common_cfg.process = self.common_process
        common_cfg.export_path = None
        common_cfg.work_dir = self.work_dir
        common_cfg.checkpoint_dir = self.common_ckpt_dir

        common_executor = DefaultExecutor(common_cfg)
        common_executor.dataset_builder = self.datasetbuilder

        common_dataset = common_executor.run(load_data_np=load_data_np, skip_export=True, skip_return=True)
        logger.info(f"Common process completed. Dataset size: {len(common_dataset)}")
        return common_dataset

    def _execute_branch(
        self,
        branch: Dict[str, Any],
        common_dataset: Any,
        load_data_np: Optional[int] = None,
        skip_export: bool = False,
    ):
        branch_name = branch.get("name", "unknown")
        branch_process = branch.get("process", [])
        if not branch_process:
            logger.warning(f"Branch {branch_name} has no process, returning common dataset")
            return common_dataset

        branch_cfg = deepcopy(self.cfg)
        branch_cfg.process = branch_process
        branch_cfg.export_path = branch.get("export_path") if not skip_export else None
        branch_cfg.work_dir = self.work_dir
        branch_cfg.checkpoint_dir = os.path.join(self.work_dir, ".branch_ckpt", branch_name)

        branch_executor = DefaultExecutor(branch_cfg)
        return branch_executor.run(
            dataset=common_dataset,
            load_data_np=load_data_np,
            skip_export=skip_export,
            skip_return=True,
        )
