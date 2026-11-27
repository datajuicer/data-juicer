import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.ops.base_op import OPERATORS, Mapper
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)
from data_juicer.utils.ray_utils import is_ray_mode

vllm = LazyLoader("vllm")

OP_NAME = "generate_qa_from_text_mapper"


@OPERATORS.register_module(OP_NAME)
class GenerateQAFromTextMapper(Mapper):
    """Generates question and answer pairs from text using a specified model.

    Supports local Hugging Face and vLLM models and API models. Each generated QA pair
    becomes a row with question and answer text in the configured query and response
    columns. The system prompt, output pattern, and maximum number of pairs per input
    can be configured. By default, the parser extracts pairs marked with Human: and
    Assistant:, as used by the PAI Doc2QA models."""

    DEFAULT_SYSTEM_PROMPT = (
        "Generate question-answer pairs grounded in the source text. "
        "Write each question after 'Human:' and its answer after 'Assistant:'. "
        "Repeat this format for every pair. Use the same language as the source text."
    )

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        hf_model: str = "alibaba-pai/pai-qwen1_5-7b-doc2qa",
        max_num: Optional[PositiveInt] = None,
        *,
        output_pattern: Optional[str] = None,
        enable_vllm: bool = False,
        api_model: Optional[str] = None,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_params: Optional[Dict] = None,
        sampling_params: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_model: Huggingface model ID.
        :param max_num: The max num of returned QA sample for each text.
            Not limit if it is None.
        :param output_pattern: Regular expression pattern to extract
            questions and answers from model response.
        :param enable_vllm: Whether to use vllm for inference acceleration.
        :param api_model: API model name. Selects API inference when set.
        :param api_endpoint: API URL or path. Defaults to '/chat/completions'.
        :param response_path: Dot-separated path to text in the API response.
        :param system_prompt: Instructions for the model. API mode supplies a
            QA format prompt by default. Pass an empty string to omit it.
        :param model_params: Parameters for initializing the model.
        :param sampling_params: Sampling parameters for text generation,
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.

        The default data format parsed by this interface is as follows:
        Model Input:
            蒙古国的首都是乌兰巴托（Ulaanbaatar）
            冰岛的首都是雷克雅未克（Reykjavik）
        Model Output:
            蒙古国的首都是乌兰巴托（Ulaanbaatar）
            冰岛的首都是雷克雅未克（Reykjavik）
            Human: 请问蒙古国的首都是哪里？
            Assistant: 你好，根据提供的信息，蒙古国的首都是乌兰巴托（Ulaanbaatar）。
            Human: 冰岛的首都是哪里呢？
            Assistant: 冰岛的首都是雷克雅未克（Reykjavik）。
            ...
        """

        if api_model is not None:
            if enable_vllm:
                raise ValueError("api_model and enable_vllm cannot be used together.")
            kwargs["accelerator"] = "cpu"
        elif api_endpoint is not None or response_path is not None:
            raise ValueError("api_endpoint and response_path require api_model.")
        super().__init__(**kwargs)

        self.api_model = api_model
        self.max_num = max_num

        if output_pattern is None:
            self.output_pattern = r"Human:(.*?)Assistant:(.*?)(?=Human:|$)"
        else:
            self.output_pattern = output_pattern

        self.enable_vllm = enable_vllm
        model_params = model_params or {}
        sampling_params = sampling_params or {}
        self.system_prompt = system_prompt
        if api_model is not None:
            if self.system_prompt is None:
                self.system_prompt = self.DEFAULT_SYSTEM_PROMPT
            # Let the batch error handler distinguish request errors from empty completions.
            model_params = {**model_params, "raise_on_error": True}
            self.model_key = prepare_model(
                model_type="api", model=api_model, endpoint=api_endpoint, response_path=response_path, **model_params
            )
            self.sampling_params = sampling_params
            return

        sampling_params = update_sampling_params(sampling_params, hf_model, self.enable_vllm)

        if enable_vllm:
            if not is_ray_mode():
                # cannot initialize vllm replicas on different GPUs
                self.num_proc = 1
            self.model_key = prepare_model(model_type="vllm", pretrained_model_name_or_path=hf_model, **model_params)
            self.sampling_params = vllm.SamplingParams(**sampling_params)
        else:
            self.model_key = prepare_model(
                model_type="huggingface", pretrained_model_name_or_path=hf_model, return_pipe=True, **model_params
            )
            self.sampling_params = sampling_params

    def parse_output(self, raw_output):
        logger.debug(raw_output)
        qa_list = []
        matches = re.findall(self.output_pattern, raw_output, re.DOTALL)
        for match in matches:
            user, assistant = match
            qa_list.append((user.strip(), assistant.strip()))
        return qa_list

    def process_batched(self, samples, rank=None):
        model = get_model(self.model_key, rank, self.use_cuda())
        if self.api_model is None:
            model, _ = model

        input_keys = samples.keys()
        num_samples = len(samples[next(iter(input_keys))])
        qa_keys = {self.query_key, self.response_key}
        passthrough_keys = input_keys - qa_keys
        output_samples = {key: [] for key in input_keys | qa_keys}

        for i in range(num_samples):
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": samples[self.text_key][i]})

            if self.api_model is not None:
                output = model(messages, **self.sampling_params)
            elif self.enable_vllm:
                response = model.chat(messages, self.sampling_params)
                output = response[0].outputs[0].text
            else:
                # model is pipe
                response = model(messages, return_full_text=False, **self.sampling_params)
                output = response[0]["generated_text"]

            qa_list = self.parse_output(output)

            if self.max_num is not None:
                qa_list = qa_list[: self.max_num]

            if len(qa_list) > 0:
                for q, a in qa_list:
                    for input_k in passthrough_keys:
                        output_samples[input_k].append(samples[input_k][i])
                    output_samples[self.query_key].append(q)
                    output_samples[self.response_key].append(a)
            else:
                logger.warning("No question and answer was extracted from current sample!")

        return output_samples
