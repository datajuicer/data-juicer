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

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")

OP_NAME = "generate_qa_from_text_mapper"


# TODO: Extend LLM-based OPs into API-based implementation.
@OPERATORS.register_module(OP_NAME)
class GenerateQAFromTextMapper(Mapper):
    """Generates question and answer pairs from text using a specified model.

    This operator uses a Hugging Face model to generate QA pairs from the input text. It
    supports both Hugging Face and vLLM models for inference. The recommended models, such
    as 'alibaba-pai/pai-llama3-8b-doc2qa', are trained on Chinese data and are suitable for
    Chinese text. The operator can limit the number of generated QA pairs per text and
    allows custom output patterns for parsing the model's response. By default, it uses a
    regular expression to extract questions and answers from the model's output. If no QA
    pairs are extracted, a warning is logged."""

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        hf_model: str = "alibaba-pai/pai-qwen1_5-7b-doc2qa",
        max_num: Optional[PositiveInt] = None,
        *,
        output_pattern: Optional[str] = None,
        enable_vllm: bool = False,
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

        super().__init__(**kwargs)

        self.max_num = max_num

        if output_pattern is None:
            self.output_pattern = r"Human:(.*?)Assistant:(.*?)(?=Human|$)"  # noqa: E501
        else:
            self.output_pattern = output_pattern

        self.enable_vllm = enable_vllm
        model_params = model_params or {}
        sampling_params = sampling_params or {}

        sampling_params = update_sampling_params(sampling_params, hf_model, self.enable_vllm)

        if enable_vllm:
            tensor_parallel_size = model_params.get("tensor_parallel_size", 1)
            model_params["tensor_parallel_size"] = tensor_parallel_size
            logger.info(f"Set tensor_parallel_size to {tensor_parallel_size} for vllm.")
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
        model, _ = get_model(self.model_key, rank, self.use_cuda())

        input_keys = samples.keys()
        num_samples = len(samples[next(iter(input_keys))])
        output_keys = input_keys | {self.query_key, self.response_key}
        output_samples = {key: [] for key in output_keys}

        for i in range(num_samples):
            messages = [{"role": "user", "content": samples[self.text_key][i]}]

            if self.enable_vllm:
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
                    for input_k in input_keys:
                        output_samples[input_k].append(samples[input_k][i])
                    output_samples[self.query_key].append(q)
                    output_samples[self.response_key].append(a)
            else:
                logger.warning("No question and answer was extracted from current sample!")

        return output_samples
