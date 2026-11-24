# image_detection_yolo_mapper

Perform object detection using YOLO on images and return bounding boxes and class labels.

This operator uses a YOLO model to detect objects in images. It processes each image in the sample, returning the bounding boxes and class labels for detected objects. The operator sets the `bbox_tag` and `class_label_tag` fields in the sample's metadata. If no image is present or no objects are detected, it sets `bbox_tag` to an empty array and `class_label_tag` to -1. The operator uses a confidence score threshold and IoU (Intersection over Union) score threshold to filter detections.

使用 YOLO 对图像进行目标检测并返回边界框和类别标签。

该算子使用 YOLO 模型检测图像中的目标。它处理样本中的每张图像，返回检测到的目标的边界框和类别标签。该算子在样本的元数据中设置 `bbox_tag` 和 `class_label_tag` 字段。如果没有图像或没有检测到目标，它将 `bbox_tag` 设置为空数组并将 `class_label_tag` 设置为 -1。该算子使用置信度分数阈值和 IoU（交并比）分数阈值来过滤检测结果。

Type 算子类型: **mapper**

Tags 标签: gpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `imgsz` |  | `640` | resolution for image resizing |
| `conf` |  | `0.05` | confidence score threshold |
| `iou` |  | `0.5` | IoU (Intersection over Union) score threshold |
| `model_path` |  | `'yolo11n.pt'` | the path to the YOLO model. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_detection_yolo_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_detection_yolo_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)