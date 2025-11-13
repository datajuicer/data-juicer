import json
import random
import re
from typing import Dict, Optional

from loguru import logger
from pydantic import PositiveInt

from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)

from ..base_op import OPERATORS, Mapper

torch = LazyLoader("torch")
vllm = LazyLoader("vllm")
rouge = LazyLoader("rouge")

OP_NAME = "generate_qa_from_examples_mapper"


# TODO: Extend LLM-based OPs into API-based implementation.
@OPERATORS.register_module(OP_NAME)
class GenerateQAFromExamplesMapper(Mapper):
    """Generates question and answer pairs from examples using a Hugging Face model.

    This operator generates QA pairs based on provided seed examples. The number of
    generated samples is determined by the length of the empty dataset configured in the
    YAML file. The operator uses a Hugging Face model to generate new QA pairs, which are
    then filtered based on their similarity to the seed examples. Samples with a similarity
    score below the specified threshold are kept. The similarity is computed using the
    ROUGE-L metric. The operator requires a seed file in chatml format, which provides the
    initial QA examples. The generated QA pairs must follow specific formatting rules, such
    as maintaining the same format as the input examples and ensuring that questions and
    answers are paired correctly."""

    DEFAULT_SYSTEM_PROMPT = (
        "请你仔细观察多个示例数据的输入和输出，按照你的理解，总结出相应规矩，然后写出一个新的【问题】和【回答】。"
        "注意，新生成的【问题】和【回答】需要满足如下要求：\n"
        "1. 生成的【问题】和【回答】不能与输入的【问题】和【回答】一致，但是需要保持格式相同。\n"
        "2. 生成的【问题】不一定要局限于输入【问题】的话题或领域，生成的【回答】需要正确回答生成的【问题】。\n"
        "3. 提供的【问题】和【回答】可能是多轮对话，生成的【问题】和【回答】也可以是多轮，但是需要保持格式相同。\n"
        "4. 生成的【问题】和【回答】必须成对出现，而且【问题】需要在【回答】之前。\n"
    )

    DEFAULT_INPUT_TEMPLATE = "{}"
    DEFAULT_EXAMPLE_TEMPLATE = "\n如下是一条示例数据：\n{}"
    DEFAULT_QA_PAIR_TEMPLATE = "【问题】\n{}\n【回答】\n{}\n"
    DEFAULT_OUTPUT_PATTERN = r"【问题】(.*?)【回答】(.*?)(?=【问题】|$)"

    _accelerator = "cuda"

    def __init__(
        self,
        hf_model: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        seed_file: str = "",
        example_num: PositiveInt = 3,
        similarity_threshold: float = 0.7,
        system_prompt: Optional[str] = None,
        input_template: Optional[str] = None,
        example_template: Optional[str] = None,
        qa_pair_template: Optional[str] = None,
        output_pattern: Optional[str] = None,
        enable_vllm: bool = False,
        model_params: Optional[Dict] = None,
        sampling_params: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_model: Huggingface model ID.
        :param seed_file: Path to the seed file in chatml format.
        :param example_num: The number of selected examples.
            Randomly select N examples from "seed_file" and
            put them into prompt as QA examples.
        :param similarity_threshold: The similarity score threshold
            between the generated samples and the seed examples.
            Range from 0 to 1. Samples with similarity score less than
            this threshold will be kept.
        :param system_prompt: System prompt for guiding the generation task.
        :param input_template: Template for building the input prompt. It must
            include one placeholder '{}', which will be replaced by
            `example_num` formatted examples defined by `example_template`.
        :param example_template: Template for formatting one QA example. It
            must include one placeholder '{}', which will be replaced by one
            formatted qa_pair.
        :param qa_pair_template: Template for formatting a single QA pair
            within each example. Must include two placeholders '{}' for the
            question and answer.
        :param output_pattern: Regular expression pattern to extract questions
            and answers from model response.
        :param enable_vllm: Whether to use vllm for inference acceleration.
        :param model_params: Parameters for initializing the model.
        :param sampling_params: Sampling parameters for text generation.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(**kwargs)

        if not seed_file:
            raise ValueError(
                "Please provide `seed_file` in chatml format."
                "Example: data-juicer/demos/data/demo-dataset-chatml.jsonl"
            )

        self.seed_file = seed_file
        self.example_num = example_num
        self.similarity_threshold = similarity_threshold
        self.similarity_type = "rouge_l"

        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.input_template = input_template or self.DEFAULT_INPUT_TEMPLATE
        self.example_template = example_template or self.DEFAULT_EXAMPLE_TEMPLATE  # noqa: E501
        self.qa_pair_template = qa_pair_template or self.DEFAULT_QA_PAIR_TEMPLATE
        self.output_pattern = output_pattern or self.DEFAULT_OUTPUT_PATTERN

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

        self.seed_qa_samples = self._load_seed_qa_samples()
        if len(self.seed_qa_samples) == 0:
            raise ValueError("No QA data was parsed from the seed file!")

    def _load_seed_qa_samples(self):
        """Load QA pairs from chatml format file."""
        qa_samples = []
        with open(self.seed_file, encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                qa_pairs = self._parse_chatml_str(line)
                if len(qa_pairs) > 0:
                    qa_samples.append(qa_pairs)
        return qa_samples

    def _sample_to_str(self, qa_sample):
        return "\n".join(["\n".join(qa_pair) for qa_pair in qa_sample]) + "\n"

    def _max_rouge_l_score(self, hypothesis, references):
        r = rouge.Rouge()
        max_score = 0.0
        hyp_str = self._sample_to_str(hypothesis)
        for reference in references:
            ref_str = self._sample_to_str(reference)
            scores = r.get_scores(hyp_str, ref_str)
            rouge_l_score = scores[0]["rouge-l"]["f"]
            if rouge_l_score > max_score:
                max_score = rouge_l_score
        return max_score

    def _parse_chatml_str(self, sample_str):
        user_input = None
        assistant_output = None
        qa_pairs = []
        data = json.loads(sample_str)
        for message in data["messages"]:
            role = message["role"]
            content = message["content"]
            if role == "user":
                user_input = content
            elif role == "assistant":
                assistant_output = content
                qa_pairs.append((user_input, assistant_output))
        return qa_pairs

    def build_input(self, qa_examples):
        def format_qa_pairs(qa_example):
            return "".join([self.qa_pair_template.format(q, a) for q, a in qa_example if q and a])

        formatted_examples = "".join(
            [self.example_template.format(format_qa_pairs(qa_example)) for qa_example in qa_examples]
        )
        input_prompt = self.input_template.format(formatted_examples)
        return input_prompt

    def parse_output(self, raw_output):
        logger.debug(raw_output)
        output_qa_pairs = []
        matches = re.findall(self.output_pattern, raw_output, re.DOTALL)
        for match in matches:
            question, answer = match
            output_qa_pairs.append((question.strip(), answer.strip()))
        return output_qa_pairs

    def process_single(self, sample, rank=None):
        model, _ = get_model(self.model_key, rank, self.use_cuda())

        random_qa_samples = random.sample(self.seed_qa_samples, self.example_num)
        input_prompt = self.build_input(random_qa_samples)

        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": input_prompt}]

        if self.enable_vllm:
            response = model.chat(messages, self.sampling_params)
            output = response[0].outputs[0].text
        else:
            # model is pipe
            response = model(messages, return_full_text=False, **self.sampling_params)
            output = response[0]["generated_text"]

        output_qa_pairs = self.parse_output(output)
        if len(output_qa_pairs) == 0:
            logger.warning("Parse model response error! " "No data generated for the current response!")
            sample.update({self.query_key: "", self.response_key: "", self.history_key: self.empty_history()})
            return sample

        if self.similarity_type == "rouge_l":
            sim_score = self._max_rouge_l_score(output_qa_pairs, random_qa_samples)
        else:
            raise ValueError(f'Not support similarity type "{self.similarity_type}"!')

        if sim_score <= self.similarity_threshold:
            query, response = output_qa_pairs[-1]
            history = output_qa_pairs[:-1]
            if len(history) == 0:
                history = self.empty_history()
        else:
            query = response = ""
            history = self.empty_history()
            logger.info("Filter this generated sample due to similarity.")

        sample.update({self.query_key: query, self.response_key: response, self.history_key: history})
        return sample
