import random
import re
from copy import deepcopy
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)
from data_juicer.utils.ray_utils import is_ray_mode

from ..base_op import OPERATORS, Mapper

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")

OP_NAME = "optimize_prompt_mapper"


@OPERATORS.register_module(OP_NAME)
class OptimizePromptMapper(Mapper):
    """Optimize prompts based on existing ones in the same batch.

    This operator uses the existing prompts and newly optimized prompts as examples to
    generate better prompts. It supports using a Hugging Face model or an API for text
    generation. The operator can be configured to keep the original samples or replace them
    with the generated ones. The optimization process involves multiple retries if the
    generated prompt is empty. The operator operates in batch mode and can leverage vLLM for
    inference acceleration on CUDA devices.

    - Uses existing and newly generated prompts to optimize future prompts.
    - Supports both Hugging Face models and API-based text generation.
    - Can keep or replace original samples with generated ones.
    - Retries up to a specified number of times if the generated prompt is empty.
    - Operates in batch mode and can use vLLM for acceleration on CUDA.
    - References: https://doc.agentscope.io/v0/en/build_tutorial/prompt_optimization.html"""

    DEFAULT_SYSTEM_PROMPT = (
        "请你仔细观察多个示例提示词，按照你的理解，总结出相应规矩，然后写出一个新的更好的提示词，以让模型更好地完成指定任务。"
        "注意，新生成的【提示词】需要满足如下要求：\n"
        "1. 生成的【提示词】不能与输入的【提示词】完全一致，但是需要保持格式类似。\n"
        "2. 生成的【提示词】相比于输入的【提示词】不能有很大的变化，更多应该是关键词、核心参数等方面的微调。\n"
        "3. 生成时只需生成带有【提示词】前缀的提示词，不需生成其他任何额外信息。\n"
    )

    DEFAULT_INPUT_TEMPLATE = "{}"
    DEFAULT_EXAMPLE_TEMPLATE = "\n如下是一条示例数据：\n{}"
    DEFAULT_PROMPT_TEMPLATE = "【提示词】\n{}\n"
    DEFAULT_OUTPUT_PATTERN = r"【提示词】(.*?)(?=【|$)"

    _batched_op = True
    _accelerator = "cuda"

    def __init__(
        self,
        api_or_hf_model: str = "Qwen/Qwen2.5-7B-Instruct",
        gen_num: PositiveInt = 3,
        max_example_num: PositiveInt = 3,
        keep_original_sample: bool = True,
        retry_num: int = 3,
        *,
        api_endpoint: Optional[str] = None,
        response_path: Optional[str] = None,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        example_template: Optional[str] = None,
        prompt_template: Optional[str] = None,
        output_pattern: Optional[str] = None,
        enable_vllm: bool = False,
        is_hf_model: bool = False,
        model_params: Optional[Dict] = None,
        sampling_params: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialization method.

        :param api_or_hf_model: API or huggingface model name.
        :param gen_num: The number of new prompts to generate.
        :param max_example_num: Maximum number of example prompts to include as
            context when generating new optimized prompts.
        :param keep_original_sample: whether to keep the original sample. If
            it's set to False, there will be only generated texts in the final
            datasets and the original texts will be removed. It's True in
            default.
        :param retry_num: how many times to retry to generate the prompt if the
            parsed generated prompt is empty. It's 3 in default.
        :param api_endpoint: URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'choices.0.message.content'.
        :param system_prompt: System prompt for guiding the generation task.
        :param input_template: Template for building the input prompt. It must
            include one placeholder '{}', which will be replaced by
            `example_num` formatted examples defined by `example_template`.
        :param example_template: Template for formatting one prompt example. It
            must include one placeholder '{}', which will be replaced by one
            formatted prompt.
        :param prompt_template: Template for formatting a single prompt
            within each example. Must include two placeholders '{}' for the
            question and answer.
        :param output_pattern: Regular expression pattern to extract questions
            and answers from model response.
        :param enable_vllm: Whether to use vllm for inference acceleration.
        :param is_hf_model:  If true, use Transformers for loading hugging face or
            local llm.
        :param model_params: Parameters for initializing the model.
        :param sampling_params: Sampling parameters for text generation.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        self.gen_num = gen_num
        self.max_example_num = max_example_num
        self.keep_original_sample = keep_original_sample
        self.retry_num = retry_num

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.example_template = example_template or self.DEFAULT_EXAMPLE_TEMPLATE  # noqa: E501
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

        self.enable_vllm = enable_vllm
        self.is_hf_model = is_hf_model
        model_params = model_params or {}
        sampling_params = sampling_params or {}

        sampling_params = update_sampling_params(sampling_params, api_or_hf_model, self.enable_vllm)

        if enable_vllm:
            if not is_ray_mode():
                # cannot initialize vllm replicas on different GPUs
                self.num_proc = 1
            self.model_key = prepare_model(
                model_type="vllm", pretrained_model_name_or_path=api_or_hf_model, **model_params
            )
            self.sampling_params = vllm.SamplingParams(**sampling_params)
        elif is_hf_model:
            self.model_key = prepare_model(
                model_type="huggingface",
                pretrained_model_name_or_path=api_or_hf_model,
                return_pipe=True,
                **model_params,
            )
            self.sampling_params = sampling_params
        else:
            self.sampling_params = sampling_params

            self.model_key = prepare_model(
                model_type="api",
                model=api_or_hf_model,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )

    def build_input(self, prompt_examples):
        formatted_examples = "".join(
            [self.example_template.format(self.prompt_template.format(p)) for p in prompt_examples]
        )
        input_prompt = self.input_template.format(formatted_examples)
        return input_prompt

    def parse_output(self, raw_output):
        logger.debug(raw_output)
        output_prompt = ""
        matches = re.findall(self.output_pattern, raw_output, re.DOTALL)
        if len(matches) > 0:
            output_prompt = matches[0].strip()
        return output_prompt

    def generate_one_prompt(self, model, input_prompt_samples):
        input_prompt = self.build_input(input_prompt_samples)
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]

        cnt = 0
        while True:
            if self.enable_vllm:
                response = model.chat(messages, self.sampling_params)
                output = response[0].outputs[0].text
            elif self.is_hf_model:
                # model is pipe
                response = model(messages, return_full_text=False, **self.sampling_params)
                output = response[0]["generated_text"]
            else:
                output = model(messages, **self.sampling_params)

            output_prompt = self.parse_output(output)
            if output_prompt == "":
                cnt += 1
                if cnt >= self.retry_num:
                    logger.warning("Retry to generate the prompt failed!")
                    break
                logger.warning(
                    f"Parse model response error! No data generated " f"for the current example! Retry for {cnt} time."
                )
            else:
                break

        return output_prompt

    def process_batched(self, samples, rank=None, *args, **kwargs):
        # init model
        if self.enable_vllm or self.is_hf_model:
            model, _ = get_model(self.model_key, rank, self.use_cuda())
        else:
            model = get_model(self.model_key, rank, self.use_cuda())

        # get the existing prompts and use the existing prompts as the examples
        if self.prompt_key not in samples:
            return samples
        prompt_batch = samples[self.prompt_key]
        batch_size = len(prompt_batch)
        max_example_num = min(self.max_example_num, batch_size)
        input_prompt_samples = random.sample(prompt_batch, max_example_num)

        output_prompts = []
        for _ in range(self.gen_num):
            output_prompt = self.generate_one_prompt(model, input_prompt_samples)
            if output_prompt:
                output_prompts.append(output_prompt)
                input_prompt_samples.append(output_prompt)
                if len(input_prompt_samples) > self.max_example_num:
                    input_prompt_samples.pop(0)

        # add the generated prompts to the samples
        res_samples = deepcopy(samples)
        if self.keep_original_sample:
            res_samples[self.prompt_key] += output_prompts
        else:
            res_samples[self.prompt_key] = output_prompts

        # add other replicate fields
        for key in res_samples:
            if key != self.prompt_key:
                new_values = [res_samples[key][0]] * len(output_prompts)
                if self.keep_original_sample:
                    # take the first original sample as the reference
                    res_samples[key] += new_values
                else:
                    res_samples[key] = new_values
        return res_samples
