import logging

import numpy as np
from jsonargparse.typing import ClosedUnitInterval

from data_juicer.ops.base_op import OPERATORS, Filter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

torch = LazyLoader("torch")
transformers = LazyLoader("transformers")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OP_NAME = "text_pair_similarity_filter"


@OPERATORS.register_module(OP_NAME)
class TextPairSimilarityFilter(Filter):
    """Filter to keep text pairs with similarities within a specific range.

    This operator computes the similarity between two texts in a pair using a Hugging Face
    CLIP model. It keeps samples where the similarity score falls within the specified min
    and max thresholds. The key metric, 'text_pair_similarity', is computed as the cosine
    similarity between the text embeddings. The operator supports two strategies for keeping
    samples: 'any' (keep if any pair meets the condition) and 'all' (keep only if all pairs
    meet the condition). If the second text key is not provided, the operator will raise an
    error. The similarity scores are cached under the 'text_pair_similarity' field in the
    sample's stats."""

    _accelerator = "cuda"

    def __init__(
        self,
        hf_clip="openai/clip-vit-base-patch32",
        trust_remote_code=False,
        min_score: ClosedUnitInterval = 0.1,
        max_score: ClosedUnitInterval = 1.0,
        text_key_second=None,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_clip: clip model name on huggingface to compute
            the similarity between image and text.
        :param trust_remote_code: whether to trust the remote code of HF models.
        :param min_score: The min similarity to keep samples.
        :param max_score: The max similarity to keep samples.
        :param text_key_second: used to store the other sentence
            in the text pair.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        torch.set_num_threads(1)

        kwargs["mem_required"] = "1500MB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"
        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=hf_clip, trust_remote_code=trust_remote_code
        )
        self.text_key_second = text_key_second

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.text_pair_similarity in sample[Fields.stats]:
            return sample

        # there is no target text
        if self.text_key_second is None:
            logger.error(
                "This OP (text_pair_similarity_filter) requires \
                processing multiple fields, and you need to specify \
                valid `text_key_second`"
            )

        # there is no text in this sample
        if (
            self.text_key not in sample
            or len(sample[self.text_key]) == 0
            or self.text_key_second not in sample
            or len(sample[self.text_key_second]) == 0
        ):
            sample[Fields.stats][StatsKeys.text_pair_similarity] = np.array([], dtype=np.float64)
            return sample

        model, processor = get_model(self.model_key, rank, self.use_cuda())

        text1 = sample[self.text_key]
        text2 = sample[self.text_key_second]

        text_tensors = processor(text=[text1, text2], padding=True, return_tensors="pt").to(model.device)
        text_features = model.get_text_features(**text_tensors)

        similarity = torch.cosine_similarity(text_features[0], text_features[1], dim=0)
        sample[Fields.stats][StatsKeys.text_pair_similarity] = [similarity]

        return sample

    def process_single(self, sample, rank=None):
        similarity = sample[Fields.stats][StatsKeys.text_pair_similarity]
        if len(similarity) <= 0:
            return True

        keep_bools = np.array(
            [self.get_keep_boolean(sim_value, self.min_score, self.max_score) for sim_value in similarity]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
