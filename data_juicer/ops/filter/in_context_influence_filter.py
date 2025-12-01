from typing import Dict, List, Optional

from datasets import Dataset
from loguru import logger

from data_juicer.ops.base_op import ATTRIBUTION_FILTERS, OPERATORS
from data_juicer.ops.filter.llm_perplexity_filter import LLMPerplexityFilter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader("torch")
transformers = LazyLoader("transformers")

OP_NAME = "in_context_influence_filter"


@OPERATORS.register_module(OP_NAME)
@ATTRIBUTION_FILTERS.register_module(OP_NAME)
class InContextInfluenceFilter(LLMPerplexityFilter):
    """Filter to keep texts based on their in-context influence on a validation set.

    This operator calculates the in-context influence of each sample by comparing
    perplexities with and without the sample as context. The influence score is computed as
    the ratio of these perplexities. If `valid_as_demo` is True, the score is L(A|Q) /
    L(A|task_desc, Q_v, A_v, Q). Otherwise, it is L(A_v|Q) / L(A_v|task_desc, Q, A, Q_v).
    The operator retains samples whose in-context influence score is within a specified
    range. The in-context influence score is stored in the 'in_context_influence' field of
    the sample's stats. The validation set must be prepared using the
    `prepare_valid_feature` method if not provided during initialization."""

    # This operator is currently under development and evaluation as part of an ongoing research project.
    # The Data-Juicer team retains full copyright over this operator.

    _accelerator = "cuda"

    def __init__(
        self,
        valid_dataset: Optional[List[Dict]] = None,
        task_desc: str = None,
        valid_as_demo: bool = False,
        n_shot: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param valid_dataset: The dataset to use for validation.
            If None, 'self.prepare_valid_feature' should be manually called before applying the filter.
        :param task_desc: The description of the validation task.
        :param valid_as_demo: If true, score =  L(A|Q) / L(A|task_desc, Q_v, A_v, Q);
                              If false, score = L(A_v|Q) L(A_v|task_desc, Q, A, Q_v).
        :param n_shot: The number of shots in validation.
        """
        super().__init__(*args, **kwargs)
        self.valid_as_demo = valid_as_demo
        self.task_desc = task_desc
        self.valid_feature = {}
        if valid_dataset is not None:
            self.prepare_valid_feature(Dataset.from_list(valid_dataset), task_desc, n_shot)
        else:
            logger.warning(
                f"valid_dataset and task_desc are both None when initializing {OP_NAME}. \
                'prepare_valid_feature' method should be manually called before applying the filter."
            )

    @property
    def valid_feature_ready(self):
        return "valid_samples" in self.valid_feature and "valid_losses" in self.valid_feature

    def prepare_valid_feature(self, dataset=None, task_desc=None, n_shot=None, *args, **kwargs):
        n_shot = n_shot or len(dataset)
        self.valid_feature["valid_samples"] = []
        self.valid_feature["valid_losses"] = []
        for i, sample in enumerate(dataset):
            if i >= n_shot:
                break
            sample_w_msgs = self.sample_with_messages(sample, system_prompt=task_desc)
            self.valid_feature["valid_samples"].append(sample_w_msgs)
            loss = self._loss(sample_w_msgs)
            self.valid_feature["valid_losses"].append(loss)

    def compute_stats_single(self, sample, rank=None):
        # check if it's computed already
        if StatsKeys.in_context_influence in sample[Fields.stats]:
            return sample

        assert self.valid_feature_ready, "Validation feature not ready yet. Call prepare_valid_feature first."

        sample_w_msgs = self.sample_with_messages(sample)

        scores = []
        if self.valid_as_demo:
            # L(A|Q) / L(A|Q_v, A_v, Q)
            loss_wo_demo = self._loss(sample_w_msgs, rank=rank)
            for valid_sample in self.valid_feature["valid_samples"]:
                loss_w_demo = self._loss(sample_w_msgs, pre_example=valid_sample, rank=rank)
                scores.append(loss_wo_demo / loss_w_demo)
        else:
            # L(A_v|Q_v) / L(A_v|Q, A, Q_v)
            for valid_sample, loss_wo_demo in zip(
                self.valid_feature["valid_samples"], self.valid_feature["valid_losses"]
            ):
                loss_w_demo = self._loss(valid_sample, pre_example=sample_w_msgs, rank=rank)
                scores.append(loss_wo_demo / loss_w_demo)

        # TODO: aggregation strategies
        in_context_influence = sum(scores) / len(scores)
        sample[Fields.stats][StatsKeys.in_context_influence] = in_context_influence

        return sample

    def process_single(self, sample):
        score = sample[Fields.stats][StatsKeys.in_context_influence]
        if score is None:
            return True

        return self.get_keep_boolean(score, self.min_score, self.max_score)
