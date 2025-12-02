import numpy as np

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

torch = LazyLoader("torch")

OP_NAME = "image_nsfw_filter"


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageNSFWFilter(Filter):
    """Filter to keep samples whose images have nsfw scores in a specified range.

    This operator uses a Hugging Face model to compute the nsfw scores for each image in a
    sample. It keeps samples based on the specified `min_score` and `max_score` thresholds.
    The operator supports two strategies: 'any' (keep the sample if any image meets the
    condition) or 'all' (keep the sample only if all images meet the condition). The nsfw
    scores are cached in the 'image_nsfw_score' field of the sample's stats."""

    _accelerator = "cuda"

    def __init__(
        self,
        hf_nsfw_model: str = "Falconsai/nsfw_image_detection",
        trust_remote_code: bool = False,
        min_score: float = 0.0,
        max_score: float = 0.5,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_nsfw_model: nsfw detection model name on huggingface.
        :param trust_remote_code: whether to trust the remote code of HF models.
        :param min_score: the min nsfw score threshold for samples.
            range from 0 to 1.
        :param max_score: the max nsfw score threshold for samples.
            range from 0 to 1.
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["mem_required"] = "1GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        self.min_score = min_score
        self.max_score = max_score
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"
        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=hf_nsfw_model, trust_remote_code=trust_remote_code
        )

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.image_nsfw_score in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.image_nsfw_score] = np.array([], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        model, processor = get_model(self.model_key, rank, self.use_cuda())

        images = [images[key] for key in images]
        inputs = processor(images=images, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        nsfw_scores = [float(scores[1]) for scores in torch.softmax(logits, dim=-1)]

        sample[Fields.stats][StatsKeys.image_nsfw_score] = nsfw_scores

        return sample

    def process_single(self, sample, rank=None):
        itm_scores = sample[Fields.stats][StatsKeys.image_nsfw_score]
        if len(itm_scores) <= 0:
            return True

        keep_bools = np.array(
            [self.get_keep_boolean(itm_score, self.min_score, self.max_score) for itm_score in itm_scores]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
