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

OP_NAME = "image_face_count_filter"


@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageFaceCountFilter(Filter):
    """Filter to keep samples with the number of faces within a specific range.

    This operator uses an OpenCV classifier for face detection. It filters samples based on
    the number of faces detected in the images, keeping only those with a face count within
    the specified range. The operator supports two strategies: 'any' (keep if any image
    meets the condition) and 'all' (keep only if all images meet the condition). The face
    counts are cached in the 'face_counts' field. If no images are present in the sample,
    the face count is set to an empty array."""

    _default_kwargs = {
        "scaleFactor": 1.1,
        "minNeighbors": 3,
        "minSize": None,
        "maxSize": None,
    }

    def __init__(
        self,
        cv_classifier: str = "",
        min_face_count: int = 1,
        max_face_count: int = 1,
        any_or_all: str = "any",
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param cv_classifier: OpenCV classifier path for face detection.
            By default, we will use 'haarcascade_frontalface_alt.xml'.
        :param min_face_count: Minimum number of faces required for samples.
        :param max_face_count: Maximum number of faces required for samples.
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

        self.min_face_count = min_face_count
        self.max_face_count = max_face_count

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
            sample[Fields.stats][StatsKeys.face_counts] = np.array([], dtype=np.float64)
            return sample

        # load images
        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        model = get_model(self.model_key)

        # count the number of detected faces in each image
        face_counts = {}
        try:
            for key, image in images.items():
                dets = detect_faces(image, model, **self.extra_kwargs)
                face_counts[key] = len(dets)
            logger.debug(f"face counts: {face_counts}")
        except Exception as e:
            logger.exception(e)

        sample[Fields.stats][StatsKeys.face_counts] = [face_counts[key] for key in loaded_image_keys]
        return sample

    def process_single(self, sample):
        face_counts = sample[Fields.stats][StatsKeys.face_counts]
        if len(face_counts) <= 0:
            return True

        keep_bools = np.array(
            [self.get_keep_boolean(face_count, self.min_face_count, self.max_face_count) for face_count in face_counts]
        )

        # different strategies
        if self.any:
            return keep_bools.any()
        else:
            return keep_bools.all()
