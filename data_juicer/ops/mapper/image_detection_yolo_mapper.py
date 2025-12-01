import numpy as np

from data_juicer.utils.constant import Fields, MetaKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.mm_utils import load_data_with_context, load_image
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, TAGGING_OPS, UNFORKABLE, Mapper
from ..op_fusion import LOADED_IMAGES

OP_NAME = "image_detection_yolo_mapper"

torch = LazyLoader("torch")
ultralytics = LazyLoader("ultralytics")


@UNFORKABLE.register_module(OP_NAME)
@TAGGING_OPS.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
@LOADED_IMAGES.register_module(OP_NAME)
class ImageDetectionYoloMapper(Mapper):
    """Perform object detection using YOLO on images and return bounding boxes and class
    labels.

    This operator uses a YOLO model to detect objects in images. It processes each image in
    the sample, returning the bounding boxes and class labels for detected objects. The
    operator sets the `bbox_tag` and `class_label_tag` fields in the sample's metadata. If
    no image is present or no objects are detected, it sets `bbox_tag` to an empty array and
    `class_label_tag` to -1. The operator uses a confidence score threshold and IoU
    (Intersection over Union) score threshold to filter detections."""

    _accelerator = "cuda"

    def __init__(self, imgsz=640, conf=0.05, iou=0.5, model_path="yolo11n.pt", *args, **kwargs):
        """
        Initialization method.

        :param imgsz: resolution for image resizing
        :param conf: confidence score threshold
        :param iou: IoU (Intersection over Union) score threshold
        :param model_path: the path to the YOLO model.

        """
        kwargs.setdefault("mem_required", "800MB")
        super().__init__(*args, **kwargs)

        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou

        self.model_key = prepare_model(model_type="yolo", model_path=model_path)

    def process_single(self, sample, rank=None, context=False):
        # there is no image in this sample
        if self.image_key not in sample or not sample[self.image_key]:
            # N x M x 4 for N images, M boxes, 4 coords
            sample[Fields.meta][MetaKeys.bbox_tag] = np.empty((0, 0, 4), dtype=np.float32)
            sample[Fields.meta][MetaKeys.class_label_tag] = -1
            return sample

        if MetaKeys.bbox_tag in sample[Fields.meta]:
            return sample

        loaded_image_keys = sample[self.image_key]
        sample, images = load_data_with_context(
            sample, context, loaded_image_keys, load_image, mm_bytes_key=self.image_bytes_key
        )

        model = get_model(self.model_key, rank=rank, use_cuda=self.use_cuda())
        sample[Fields.meta][MetaKeys.bbox_tag] = []
        sample[Fields.meta][MetaKeys.class_label_tag] = []

        for image in images:
            targets = model(image, imgsz=self.imgsz, conf=self.conf, iou=self.iou, verbose=False)[0]
            sample[Fields.meta][MetaKeys.bbox_tag].append(targets.boxes.xywh.cpu().numpy())
            sample[Fields.meta][MetaKeys.class_label_tag].append(targets.boxes.cls.cpu().numpy())

        # match schema
        if len(sample[Fields.meta][MetaKeys.bbox_tag]) == 0:
            sample[Fields.meta][MetaKeys.bbox_tag] = np.empty((0, 0, 4), dtype=np.float32)
            sample[Fields.meta][MetaKeys.class_label_tag] = -1
        return sample
