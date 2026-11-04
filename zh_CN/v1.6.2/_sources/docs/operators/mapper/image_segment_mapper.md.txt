# image_segment_mapper

Perform segment-anything on images and return the bounding boxes.

This operator uses a FastSAM model to detect and segment objects in images, returning their bounding boxes. It processes each image in the sample, and stores the bounding boxes in the 'bbox_tag' field under the 'meta' key. If no images are present in the sample, an empty array is stored instead. The operator allows setting the image resolution, confidence threshold, and IoU (Intersection over Union) score threshold for the segmentation process. Bounding boxes are represented as N x M x 4 arrays, where N is the number of images, M is the number of detected boxes, and 4 represents the coordinates.

对图像执行 segment-anything 并返回边界框。

此算子使用 FastSAM 模型检测并分割图像中的对象，返回它们的边界框。它处理样本中的每张图像，并将边界框存储在 'meta' 键下的 'bbox_tag' 字段中。如果样本中没有图像，则存储一个空数组。该算子允许设置图像分辨率、置信度阈值和 IoU（交并比）得分阈值。边界框表示为 N x M x 4 数组，其中 N 是图像数量，M 是检测到的框数，4 代表坐标。

Type 算子类型: **mapper**

Tags 标签: gpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `imgsz` |  | `1024` | resolution for image resizing |
| `conf` |  | `0.05` | confidence score threshold |
| `iou` |  | `0.5` | IoU (Intersection over Union) score threshold |
| `model_path` |  | `'FastSAM-x.pt'` | the path to the FastSAM model. Model name should be one of ['FastSAM-x.pt', 'FastSAM-s.pt']. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_segment_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_segment_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)