import os

import numpy as np
from loguru import logger

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import detect_faces, load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, UNFORKABLE, Filter
from ..op_fusion import LOADED_IMAGES

cv2 = LazyLoader("cv2", "opencv-python")

OP_NAME = "image_face_ratio_filter"


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageFaceRatioFilter(Filter):
    """Filter to keep samples with face area ratios within a specific range.

    This operator filters samples based on the ratio of the largest face area to the total
    image area. It uses an OpenCV classifier for face detection. The key metric,
    'face_ratios', is computed for each image in the sample. Samples are kept if the face
    area ratios fall within the specified min and max ratio range. The filtering strategy
    can be set to 'any' (keep if any image meets the condition) or 'all' (keep only if all
    images meet the condition). If no images are present in the sample, the sample is
    retained."""

    _default_kwargs = {
        "scaleFactor": 1.1,
        "minNeighbors": 3,
        "minSize": None,
        "maxSize": None,
    }

    def __init__(
        self,
        cv_classifier: str = "",
        min_ratio: float = 0.0,
        max_ratio: float = 0.4,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param cv_classifier: OpenCV classifier path for face detection.
            By default, we will use 'haarcascade_frontalface_alt.xml'.
        :param min_ratio: Min ratio for the largest face area in an image.
        :param max_ratio: Max ratio for the largest face area in an image.
        :param any_or_all: Keep this sample with 'any' or 'all' strategy of
            all images. 'any': keep this sample if any images meet the
            condition. 'all': keep this sample only if all images meet the
            condition.
        :param args: Extra positional arguments.
        :param kwargs: Extra keyword arguments.
        """
        super().__init__(*args, **kwargs)

        if cv_classifier == "":
            cv_classifier = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_alt.xml")
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

        self.extra_kwargs = self._default_kwargs
        for key in kwargs:
            if key in self.extra_kwargs:
                self.extra_kwargs[key] = kwargs[key]

        if any_or_all not in ["any", "all"]:
            raise ValueError(f"Keep strategy [{any_or_all}] is not supported. " f'Can only be one of ["any", "all"].')
        self.any = any_or_all == "any"

        self.model_key = prepare_model(model_type="opencv_classifier", model_path=cv_classifier)

    def compute_stats_single(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.face_ratios in sample[Fields.stats]:
            return sample

        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            sample[Fields.stats][StatsKeys.face_ratios] = np.array([], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        model = get_model(self.model_key)

        # detect faces
        face_detections = {}
        for key, image in images.items():
            face_detections[key] = detect_faces(image, model, **self.extra_kwargs)
        logger.debug(f"detections: {face_detections}")

        # compute face area ratios for each image considering the largest face
        face_area_ratios = {}
        for key, dets in face_detections.items():
            image_area = images[key].width * images[key].height
            face_area_ratios[key] = max([w * h for _, _, w, h in dets], default=0.0) / image_area
        logger.debug(f"ratios: {face_area_ratios}")

        sample[Fields.stats][StatsKeys.face_ratios] = [face_area_ratios[key] for key in loaded_image_keys]
        return sample

    def process_single(self, sample):
        face_ratios = sample[Fields.stats][StatsKeys.face_ratios]
        if len(face_ratios) <= 0:
            return True

        keep_bools = np.array(
            [self.get_keep_boolean(face_ratio, self.min_ratio, self.max_ratio) for face_ratio in face_ratios]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
