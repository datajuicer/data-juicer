import asyncio
import os
import re
import subprocess
import sys
import time

import jsonlines as jl
from jsonargparse import namespace_to_dict
from loguru import logger

from data_juicer.core.sandbox.data_pool_manipulators import check_io_paths
from data_juicer.core.sandbox.helper_funcs import ALL_FUNCS
from data_juicer.utils.file_utils import follow_read
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)


class BaseModelExecutor(object):
    """
    Base abstraction for model executor within the DataJuicer's sandbox
    """

    def __init__(self, model_config: dict, watcher=None):
        self.model_config = model_config
        self.executor = None
        self.watcher = watcher
        # log to tell the end of model execution
        self.END_OF_MODEL_EXEC = "<DJ-Sandbox> End of ModelExecutor's running <DJ-Sandbox>"

    async def run(self, run_type, run_obj=None, **kwargs):
        """
        conduct some model-related execution tasks
            given specified run_type and run_obj
        """
        watch_task = asyncio.create_task(self.watch_run(run_type, run_obj, **kwargs))
        if self.watcher is None:
            run_task = asyncio.create_task(self._run(run_type, run_obj, **kwargs))
            ret = await run_task
            return ret
        else:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
                log_f_name = os.path.join(self.watcher.sandbox_cfg.work_dir, f"model_exe_{run_type}_{timestamp}.log")
                self.watcher.model_exe_log_file = log_f_name
                with open(log_f_name, "w") as log_f:
                    sys.stdout = log_f
                    sys.stderr = log_f
                    run_task = asyncio.create_task(self._run(run_type, run_obj, **kwargs))
                    ret = await run_task
                    print(self.END_OF_MODEL_EXEC, flush=True)
                    await watch_task
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            return ret

    def run_subprocess(self, script_path, run_args, working_dir, cmd="bash"):
        run_args = [str(arg) for arg in run_args]
        args = [cmd, script_path] + run_args
        subprocess.run(args, cwd=working_dir)

    async def _run(self, run_type, run_obj=None, **kwargs):
        raise NotImplementedError

    async def watch_run(self, run_type, run_obj=None, **kwargs):
        """
        watch the running process in an online manner, and
            return the summarized results
        """
        met_eof = False
        while not met_eof:
            if os.path.exists(self.watcher.model_exe_log_file):
                async for line in follow_read(self.watcher.model_exe_log_file):
                    if self.END_OF_MODEL_EXEC in line:
                        met_eof = True
                        break
                    else:
                        self._watch_run(line, **kwargs)
            else:
                await asyncio.sleep(0.1)

    def _watch_run(self, line, **kwargs):
        """
        customized log watcher, depending on specific model executor
        """
        raise NotImplementedError

    def data_connector(self, input_data, **kwargs):
        """
        convert input_data (usually in Data-Juicer's Dataset format) into
            the appropriate format for specific model executor
        """
        raise NotImplementedError


class ModelScopeExecutor(BaseModelExecutor):
    def data_connector(self, input_data, split="train", key_remapping=None, **kwargs):
        try:
            from modelscope.msdatasets import MsDataset

            if isinstance(input_data, str):
                # dataset path
                ds = MsDataset.load(input_data, split=split).ds_instance
            else:
                ds = input_data
            if key_remapping:
                features = ds.features
                for key in key_remapping:
                    if key in features:
                        ds = ds.rename_column(key, key_remapping[key])
            return ds
        except ModuleNotFoundError:
            raise ModuleNotFoundError("modelscope package not installed")

    def _watch_run(self, line, **kwargs):
        # e.g., "2023-07-02 17:26:50,324 - modelscope - INFO - epoch
        # [1][100/4953]\tlr: 4.825e-05, memory: 8742, loss: 5.6125\n"
        pattern = r"loss:\s*([0-9.]+)"

        if self.watcher is not None:
            match = re.search(pattern, line)
            if match:
                loss_value = float(match.group(1))
                self.watcher.watch(loss_value, "loss")


class ModelscopeInferProbeExecutor(ModelScopeExecutor):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        try:
            from modelscope.pipelines import pipeline

            self.executor = pipeline(**model_config)
        except ModuleNotFoundError:
            raise ModuleNotFoundError("modelscope package not installed")

    async def _run(self, run_type, run_obj=None, **kwargs):
        if run_type == "infer_on_data":
            return self.executor(self.data_connector(run_obj), **kwargs)
        else:
            raise ValueError(f"run_type {run_type} not supported")


class ModelscopeTrainExecutor(ModelScopeExecutor):
    def __init__(self, model_config, watcher=None):
        super().__init__(model_config, watcher)
        self.model_config = namespace_to_dict(model_config)
        self.executor = None

    def cfg_modify_fn(self, cfg):
        cfg.merge_from_dict(self.model_config)
        return cfg

    def build_executor(self, model_name, trainer_name, work_dir, train_dataset=None, eval_dataset=None):
        try:
            from modelscope.trainers import build_trainer

            kwargs = dict(
                model=model_name,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                work_dir=work_dir,
                cfg_modify_fn=self.cfg_modify_fn,
            )
            self.executor = build_trainer(
                name=trainer_name,
                default_args=kwargs,
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError("modelscope package not installed")

    async def _run(self, run_type, run_obj=None, **kwargs):
        # training cfg updated, such as datasets and training parameters
        builder_kwargs = {
            "model_name": self.model_config["model_name"],
            "trainer_name": self.model_config["trainer_name"],
        }
        if "key_remapping" in self.model_config:
            key_remapping = self.model_config["key_remapping"]
        else:
            key_remapping = None
        if "train_dataset" in self.model_config:
            builder_kwargs["train_dataset"] = self.data_connector(
                self.model_config["train_dataset"], split="train", key_remapping=key_remapping
            )
        if "eval_dataset" in self.model_config:
            builder_kwargs["eval_dataset"] = self.data_connector(
                self.model_config["eval_dataset"], split="val", key_remapping=key_remapping
            )
        if "work_dir" in self.model_config:
            builder_kwargs["work_dir"] = self.model_config["work_dir"]
        self.work_dir = builder_kwargs["work_dir"]
        self.build_executor(**builder_kwargs)
        self.executor.train()
        return self.work_dir


class LLMInferExecutor(BaseModelExecutor):
    """
    A inference executor for LLM inference.
    The model preparation method should be implemented by the subclass for specific type of model.

    The config file for this type of executor should at least include the following items:
    1. `type`: model type.
    2. `build_messages_func`: the helper func to build the messages.
    3. `parse_output_func`: the helper func to build the messages.
    4. `dataset_path`: the input datasets or data pools use to construct the input messages for LLM inference.
        Only support jsonl files for now.
    5. `export_path`: the output dir to store the inference results.
    6. `infer_res_key`: the key name to store the inference results. It's "response" in default.
    """

    def __init__(self, model_config: dict, watcher=None):
        super().__init__(model_config, watcher)
        # model related params
        self.executor, self.sampling_params = self.prepare_executor()

        # inference format related params
        self.build_messages_func = model_config.get("build_messages_func", "build_input")
        self.parse_output_func = model_config.get("parse_output_func", "parse_output")
        self.func_kwargs = model_config.get("func_kwargs", {})

        self.build_messages_func = ALL_FUNCS.get(self.build_messages_func)
        self.parse_output_func = ALL_FUNCS.get(self.parse_output_func)

        # inference dataset related
        self.dataset_path = model_config.get("dataset_path", [])
        self.export_path = model_config.get("export_path", None)
        self.infer_res_key = model_config.get("infer_res_key", "response")
        if isinstance(self.dataset_path, str):
            self.dataset_path = [self.dataset_path]

    def prepare_executor(self):
        raise NotImplementedError

    def executor_infer(self, messages):
        raise NotImplementedError

    async def _run(self, run_type, run_obj=None, **kwargs):
        # check I/O paths
        existing_input_paths, export_path = check_io_paths(self.dataset_path, self.export_path)
        output_paths = []
        for input_path in existing_input_paths:
            input_basename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(export_path, f"{input_basename}_inference.jsonl")
            with jl.open(output_path, "w") as writer:
                with jl.open(input_path) as reader:
                    for item in reader:
                        non_batch = False
                        messages_list = self.build_messages_func(item, **self.func_kwargs)
                        if len(messages_list) > 0 and not isinstance(messages_list[0], list):
                            messages_list = [messages_list]
                            non_batch = True
                        results = []
                        for messages in messages_list:
                            output = self.executor_infer(messages)
                            results.append(self.parse_output_func(output, item, **self.func_kwargs))
                        if non_batch:
                            item[self.infer_res_key] = results[0]
                        else:
                            item[self.infer_res_key] = results
                        writer.write(item)
            output_paths.append(output_path)
        return output_paths


class HFTransformersInferExecutor(LLMInferExecutor):
    """
    A inference executor for model inference with Huggingface Transformers.

    The config file for this executor should at least include the following items:
    1. `type`: must be "huggingface".
    2. `model_path`: the path to the HF model.
    3. `model_params`: extra parameters for the model.
    4. `sampling_params`: extra sampling parameters for the model.
    """

    def __init__(self, model_config: dict, watcher=None):
        super().__init__(model_config, watcher)

    def prepare_executor(self):
        model = self.model_config.get("model", None)
        model_params = self.model_config.get("model_params", {})
        sampling_params = self.model_config.get("sampling_params", {})
        executor, _ = get_model(
            prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path=model,
                return_pipe=True,
                **model_params,
            ),
            use_cuda=True,
        )
        sampling_params = update_sampling_params(sampling_params, model, False)
        return executor, sampling_params

    def executor_infer(self, messages):
        response = self.executor(messages, return_full_text=False, **self.sampling_params)
        output = response[0]["generated_text"]
        return output


class VLLMInferExecutor(LLMInferExecutor):
    """
    A inference executor for model inference with vLLM.

    The config file for this executor should at least include the following items:
    1. `type`: must be "vllm".
    2. `model_path`: the path to the vLLM model.
    3. `model_params`: extra parameters for the model.
    4. `sampling_params`: extra sampling parameters for the model.
    5. other parameters can be referred to the class LLMInferExecutor
    """

    def __init__(self, model_config: dict, watcher=None):
        super().__init__(model_config, watcher)

    def prepare_executor(self):
        # model related params
        torch = LazyLoader("torch")
        vllm = LazyLoader("vllm")
        model = self.model_config.get("model", None)
        model_params = self.model_config.get("model_params", {})
        sampling_params = self.model_config.get("sampling_params", {})
        if model_params.get("tensor_parallel_size") is None:
            tensor_parallel_size = torch.cuda.device_count()
            logger.info(f"Set tensor_parallel_size to {tensor_parallel_size} for vllm.")
            model_params["tensor_parallel_size"] = tensor_parallel_size
        executor, _ = get_model(
            prepare_model(
                model_type="vllm",
                pretrained_model_name_or_path=model,
                **model_params,
            ),
            use_cuda=True,
        )
        sampling_params = vllm.SamplingParams(**update_sampling_params(sampling_params, model, False))
        return executor, sampling_params

    def executor_infer(self, messages):
        response = self.executor.chat(messages, self.sampling_params)
        output = response[0].outputs[0].text
        return output


class APIModelInferExecutor(LLMInferExecutor):
    """
    A inference executor for model inference with OpenAI API.

    The config file for this executor should at least include the following items:
    1. `type`: must be "api".
    2. `model`: the API model used to inference.
    3. `model_params`: extra parameters for the model.
    4. `sampling_params`: extra sampling parameters for the model.
    5. `api_endpoint`: URL endpoint for the API.
    6. `response_path`: Path to extract content from the API response. Defaults to 'choices.0.message.content'.
    7. `max_retry_num`: the max number of retries when the API request fails.
    8. other parameters can be referred to the class LLMInferExecutor
    """

    def __init__(self, model_config: dict, watcher=None):
        super().__init__(model_config, watcher)
        # model related params
        self.max_retry_num = self.model_config.get("max_retry_num", 5)

    def prepare_executor(self):
        # model related params
        api_endpoint = self.model_config.get("api_endpoint", None)
        response_path = self.model_config.get("response_path", None)
        model = self.model_config.get("model", None)
        model_params = self.model_config.get("model_params", {})
        sampling_params = self.model_config.get("sampling_params", {})
        executor = get_model(
            prepare_model(
                model_type="api",
                model=model,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )
        )
        sampling_params = sampling_params
        return executor, sampling_params

    def executor_infer(self, messages):
        try_count = 0
        while try_count <= self.max_retry_num:
            try:
                output = self.executor(messages, **self.sampling_params)
                return output
            except Exception as e:
                logger.warning(f"error: {e} -- retries: {try_count}")
                try_count += 1
                if try_count > self.max_retry_num:
                    logger.error("Retried too many times. Abort!")
                    raise
                time.sleep(try_count * 1)


class LLaVAExecutor(BaseModelExecutor):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError("To be refactored, from DJ's refinement for " "LLaVA related experiments.")


class LLaMAFactoryExecutor(BaseModelExecutor):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError("To be refactored, which is used in " "data-juicer competition.")


class MegatronExecutor(BaseModelExecutor):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        raise NotImplementedError("To be refactored from dj's `thirdparty`.")
