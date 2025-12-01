from typing import List

import numpy as np
from loguru import logger
from PIL import ImageOps

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import (
    SpecialTokens,
    iou,
    load_data_with_context,
    load_image,
    remove_special_tokens,
)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter
from ..op_fusion import LOADED_IMAGES

torch = LazyLoader("torch")
nltk = LazyLoader("nltk")

OP_NAME = "phrase_grounding_recall_filter"


# NER algorithm adapted from GLIP starts
# https://github.com/microsoft/GLIP/blob/main/maskrcnn_benchmark/engine/predictor_glip.py#L107-L127
def find_noun_phrases(caption: str, pos_tagger=None) -> List[str]:
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)

    # Use the provided POS tagger if available, or fallback to the default
    if pos_tagger:
        pos_tags = pos_tagger(tokens)
    else:
        pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == "NP":
            noun_phrases.append(" ".join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    punct = [
        "|",
        ":",
        ";",
        "@",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "^",
        "'",
        '"',
        "’",
        "`",
        "?",
        "$",
        "%",
        "#",
        "!",
        "&",
        "*",
        "+",
        ",",
        ".",
    ]
    for p in punct:
        text = text.replace(p, "")
    return text.strip()


def run_ner(caption, pos_tagger=None):
    noun_phrases = find_noun_phrases(caption, pos_tagger)
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != ""]
    noun_phrases = list(set(noun_phrases))  # remove duplicate ners
    return noun_phrases


# NER algorithm adapted from GLIP ends


@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class PhraseGroundingRecallFilter(Filter):
    """Filter to keep samples based on the phrase grounding recall of phrases extracted from
    text in images.

    This operator uses a Hugging Face Owl-ViT model to locate phrases extracted from the
    text within the images. It keeps samples where the phrase grounding recall is within a
    specified range. The recall is computed by comparing the number of correctly located
    phrases to the total number of phrases. The operator can handle multiple images per text
    chunk and supports different strategies for reducing the recall values (e.g., average,
    max, min). It also allows for flipping images horizontally or vertically. The key metric
    'phrase_grounding_recall' is computed and stored in the sample's stats. If no images are
    present, the recall is set to an empty array."""

    _accelerator = "cuda"

    def __init__(
        self,
        hf_owlvit: str = "google/owlvit-base-patch32",
        trust_remote_code: bool = False,
        min_recall: float = 0.1,
        max_recall: float = 1.0,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        any_or_all: str = "any",
        reduce_mode: str = "avg",
        iou_thr: float = 0.5,
        large_area_ratio_thr: float = 0.95,
        conf_thr: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param hf_owlvit: Owl-ViT model name on huggingface to locate the
            phrases extracted from the text.
        :param trust_remote_code: whether to trust the remote code of HF models.
        :param min_recall: The min phrase grounding recall to keep samples.
        :param max_recall: The max phrase grounding recall to keep samples.
        :param horizontal_flip: Flip image horizontally (left to right).
        :param vertical_flip: Flip image vertically (top to bottom).
        :param any_or_all: keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param reduce_mode: reduce mode when one text corresponds to
            multiple images in a chunk.
            'avg': Take the average of multiple values
            'max': Take the max of multiple values
            'min': Take the min of multiple values
        :param iou_thr: the IoU threshold for NMS-like post-process. If two
            predicted bboxes are overlap with an IoU larger than this
            threshold, the bbox with less confidence will be removed. Default:
            0.5.
        :param large_area_ratio_thr: the area ratio threshold for filtering out
            those large predicted bboxes. If the area of a predicted bbox
            accounts for more than this ratio threshold of the whole image
            area, this bbox will be removed. Default: 0.95.
        :param conf_thr: the confidence score threshold for removing
            low-confidence bboxes. If the confidence score of a predicted bbox
            is lower than the threshold, this bbox will be removed. Default: 0.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs["mem_required"] = "1GB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)
        self.min_recall = min_recall
        self.max_recall = max_recall
        if reduce_mode not in ["avg", "max", "min"]:
            raise ValueError(
                f"Reduce mode [{reduce_mode}] is not supported. " f'Can only be one of ["avg", "max", "min"].'
            )
        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"
        self.model_key = prepare_model(
            model_type="huggingface", pretrained_model_name_or_path=hf_owlvit, trust_remote_code=trust_remote_code
        )
        self.reduce_mode = reduce_mode
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.iou_thr = iou_thr
        self.large_area_ratio_thr = large_area_ratio_thr
        self.conf_thr = conf_thr

        # Initialize NLTK resources needed for NER extraction
        logger.info("Loading NLTK resources for NER extraction...")
        self.nltk_tagger_key = prepare_model(model_type="nltk_pos_tagger")

        # Ensure NLTK resources are correctly downloaded and available
        try:
            # Import nltk here to ensure it's available
            import nltk

            from data_juicer.utils.nltk_utils import patch_nltk_pickle_security

            # Ensure pickle security patches are applied
            patch_nltk_pickle_security()

            # Download required resources if not already available
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
        except Exception as e:
            logger.warning(f"Error initializing NLTK resources: {e}. " "NER extraction may not work correctly.")

    def compute_stats_single(self, sample, rank=None, context=False):
        # check if it's computed already
        if StatsKeys.phrase_grounding_recall in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.phrase_grounding_recall] = np.array([], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        text = sample[self.text_key]
        offset = 0
        recalls = []
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        # Get the POS tagger if available
        pos_tagger = get_model(self.nltk_tagger_key) if hasattr(self, "nltk_tagger_key") else None

        for chunk in text.split(SpecialTokens.eoc):
            count = chunk.count(SpecialTokens.image)

            # no image or no text
            if count == 0 or len(chunk) == 0:
                continue
            else:
                text_this_chunk = remove_special_tokens(chunk)
                ners_this_chunk = run_ner(text_this_chunk, pos_tagger)
                num_ners = len(ners_this_chunk)
                if num_ners <= 0:
                    # no ners found, just skip this chunk
                    recalls.append(1.0)
                    continue
                images_this_chunk = []
                for image_key in loaded_image_keys[offset : offset + count]:
                    image = images[image_key]
                    if self.horizontal_flip:
                        image = ImageOps.mirror(image)
                    if self.vertical_flip:
                        image = ImageOps.flip(image)
                    images_this_chunk.append(image)

                ners_batch = [ners_this_chunk] * len(images_this_chunk)
                inputs = processor(
                    text=ners_batch, images=images_this_chunk, return_tensors="pt", padding=True, truncation=True
                ).to(model.device)

                with torch.no_grad():
                    outputs = model(**inputs)
                    target_sizes = torch.tensor([img.size[::-1] for img in images_this_chunk]).to(model.device)
                    results = processor.post_process_object_detection(
                        outputs, threshold=self.conf_thr, target_sizes=target_sizes
                    )

                image_recalls = []
                for idx, result in enumerate(results):
                    scores = result["scores"]
                    labels = result["labels"]
                    boxes = result["boxes"]

                    # sort by the confidence scores
                    # and only keep the first num_ners predictions
                    order_idx = scores.argsort(descending=True)
                    scores = scores[order_idx].tolist()[:num_ners]
                    labels = labels[order_idx].tolist()[:num_ners]
                    boxes = boxes[order_idx].tolist()[:num_ners]

                    image_area = target_sizes[idx].prod()
                    hit = {}
                    for box, label, score in zip(boxes, labels, scores):
                        # this ner is already hit
                        if ners_this_chunk[label] in hit:
                            continue
                        # skip boxes nearly cover the whole image
                        xmin, ymin, xmax, ymax = box
                        box_area = (xmax - xmin) * (ymax - ymin)
                        if 1.0 * box_area / image_area > self.large_area_ratio_thr:
                            continue
                        # skip overlapped boxes with nms-like method
                        suppressed = False
                        for ner in hit:
                            if iou(box, hit[ner][0]) > self.iou_thr:
                                suppressed = True
                                break
                        if suppressed:
                            continue

                        # record the new hit box
                        hit[ners_this_chunk[label]] = (box, score)

                    recall = 1.0 * len(hit) / num_ners
                    image_recalls.append(recall)

                if self.reduce_mode == "avg":
                    image_recall = sum(image_recalls) / len(image_recalls)
                elif self.reduce_mode == "max":
                    image_recall = max(image_recalls)
                else:
                    image_recall = min(image_recalls)

                recalls.append(image_recall)
            offset += count
        sample[Fields.stats][StatsKeys.phrase_grounding_recall] = recalls

        return sample

    def process_single(self, sample):
        recalls = sample[Fields.stats][StatsKeys.phrase_grounding_recall]
        if len(recalls) <= 0:
            return True

        keep_bools = np.array([self.get_keep_boolean(recall, self.min_recall, self.max_recall) for recall in recalls])

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
