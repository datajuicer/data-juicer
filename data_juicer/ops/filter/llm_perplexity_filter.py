import logging
from typing import Dict, Optional

import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter

torch = LazyLoader("torch")
transformers = LazyLoader("transformers")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OP_NAME = "llm_perplexity_filter"


@OPERATORS.register_module(OP_NAME)
class LLMPerplexityFilter(Filter):
    """Filter to keep samples with perplexity scores within a specified range, computed using a
    specified LLM.

    This operator computes the perplexity score for each sample using a Hugging Face LLM. It
    then filters the samples based on whether their perplexity scores fall within the
    specified minimum and maximum score range. The perplexity score is calculated as the
    exponential of the loss value from the LLM. The operator uses a query and response
    template to format the input text for the LLM. If the perplexity score is not already
    cached in the sample's stats under the key 'llm_perplexity', it will be computed."""

    _accelerator = "cuda"

    def __init__(
        self,
        hf_model: str = "Qwen/Qwen2.5-0.5B",
        model_params: Optional[Dict] = None,
        min_score: float = 1.0,
        max_score: float = 100.0,
        query_template: Optional[str] = None,
        response_template: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_model: huggingface embedding model name.
        :param model_params: Parameters for initializing the API model.
        :param min_score: Minimum perplexity score.
        :param max_score: Maximum perplexity score.
        :param query_template: Template for building the query string.
        :param response_template: Template for building the response string.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        self.query_template = query_template or ""
        self.response_template = response_template or "{text}"
        if model_params is None:
            model_params = {}
        self.model_params = model_params
        self.model_key = prepare_model(model_type="huggingface", pretrained_model_name_or_path=hf_model, **model_params)

    @torch.no_grad()
    def _loss(self, example, pre_example=None, rank=None):
        model, tokenizer = get_model(self.model_key, rank, self.use_cuda())
        model.eval()
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

        pre_msgs = pre_example["messages"] if pre_example is not None else []
        msgs = pre_msgs + example["messages"]
        # TODO: chat template
        full_text = " ".join([msg["content"] for msg in msgs]).strip()
        response_text = msgs[-1]["content"].strip()
        max_length = self.model_params.get("max_length", None)
        full_tokenized = tokenizer(full_text, max_length=max_length, truncation=True, return_tensors="pt")
        input_ids = full_tokenized["input_ids"]
        response_ids = tokenizer(response_text, max_length=max_length, truncation=True, return_tensors="pt")[
            "input_ids"
        ][0]
        response_len = len(response_ids) - int(tokenizer.bos_token_id is not None)
        labels = input_ids.clone()
        labels[0, :-response_len] = -100

        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        loss = model(input_ids=input_ids, labels=labels).loss.item()

        return loss

    def sample_with_messages(self, sample, system_prompt=None):
        if "messages" in sample:
            return sample
        messages = [
            {"role": "user", "content": self.query_template.format(**sample)},
            {"role": "assistant", "content": self.response_template.format(**sample)},
        ]
        if system_prompt is not None:
            messages = [{"role": "system", "content": system_prompt}] + messages

        return {"messages": messages, **sample}

    def compute_stats_single(self, sample, rank=None):

        # check if it's computed already
        if StatsKeys.llm_perplexity in sample[Fields.stats]:
            return sample

        sample_w_msgs = self.sample_with_messages(sample)

        sample[Fields.stats][StatsKeys.llm_perplexity] = np.exp(self._loss(sample_w_msgs, rank))

        return sample

    def process_single(self, sample):
        ppl = sample[Fields.stats][StatsKeys.llm_perplexity]

        return self.get_keep_boolean(ppl, self.min_score, self.max_score)
