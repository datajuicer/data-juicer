import numpy as np

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = "image_segment_mapper"

torch = LazyLoader("torch")
ultralytics = LazyLoader("ultralytics")


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageSegmentMapper(Mapper):
    """Perform segment-anything on images and return the bounding boxes.

    This operator uses a FastSAM model to detect and segment objects in images, returning
    their bounding boxes. It processes each image in the sample, and stores the bounding
    boxes in the 'bbox_tag' field under the 'meta' key. If no images are present in the
    sample, an empty array is stored instead. The operator allows setting the image
    resolution, confidence threshold, and IoU (Intersection over Union) score threshold for
    the segmentation process. Bounding boxes are represented as N x M x 4 arrays, where N is
    the number of images, M is the number of detected boxes, and 4 represents the
    coordinates."""

    _accelerator = "cuda"

    def __init__(self, imgsz=1024, conf=0.05, iou=0.5, model_path="FastSAM-x.pt", *args, **kwargs):
        """
        Initialization method.

        :param imgsz: resolution for image resizing
        :param conf: confidence score threshold
        :param iou: IoU (Intersection over Union) score threshold
        :param model_path: the path to the FastSAM model. Model name should be
            one of ['FastSAM-x.pt', 'FastSAM-s.pt'].

        """
        kwargs["mem_required"] = "800MB" if kwargs.get("mem_required", 0) == 0 else kwargs["mem_required"]
        super().__init__(*args, **kwargs)

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        self.model_key = prepare_model(model_type="fastsam", model_path=model_path)

    def process_single(self, sample, rank=None, context=False):
        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            # N x M x 4 for N images, M boxes, 4 coords
            sample[Fields.meta][MetaKeys.bbox_tag] = np.empty((0, 0, 4), dtype=np.float32)
            return sample

        if MetaKeys.bbox_tag in sample[Fields.meta]:
            return sample

        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        model = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
        sample[Fields.meta][MetaKeys.bbox_tag] = []

        for image in images:
            masks = model(image, retina_masks=True, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)[0]
            sample[Fields.meta][MetaKeys.bbox_tag].append(masks.boxes.xywh.cpu().numpy())

        # match schema
        if len(sample[Fields.meta][MetaKeys.bbox_tag]) == 0:
            sample[Fields.meta][MetaKeys.bbox_tag] = np.empty((0, 0, 4), dtype=np.float32)
        return sample
