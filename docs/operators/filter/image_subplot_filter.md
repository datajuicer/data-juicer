# image_subplot_filter

Filter to remove samples containing images with subplots using Hough Line Transform detection.

This operator detects subplots in images using Hough Line Transform to identify straight lines that form grid-like structures. It computes a confidence score based on the number of detected horizontal and vertical lines, their regularity, grid structure, and length consistency. Samples are filtered out if their subplot confidence exceeds the specified threshold. The operator supports two strategies: 'any' (filter if any image meets the condition) and 'all' (filter only if all images meet the condition). If no images are present in the sample, the sample is not filtered out.

**Note**: This subplot detection assumes that subplot edges are perfectly straight lines forming grid-like structures. The hyperparameters are sensitive to different image types and subplot layouts. It is recommended to run hyperparameter optimization for your specific dataset (see [Automated Hyperparameter Optimization](../../tools/hpo/README.md)).

使用Hough直线变换检测来过滤掉包含子图的图像样本。

该算子使用Hough直线变换检测图像中的子图，识别形成网格状结构的直线。它基于检测到的水平和垂直线条数量、规律性、网格结构和长度一致性计算置信度分数。如果样本的子图置信度超过指定阈值，则过滤掉该样本。该算子支持两种策略：'any'（如果有任何图像满足条件则过滤）和'all'（只有当所有图像都满足条件时才过滤）。如果样本中没有图像，则不会过滤掉该样本。

**注意**: 此子图检测假设子图的边缘是完美的直线，形成网格状结构。超参数对不同图像类型和子图布局很敏感。建议针对您的特定数据集运行超参数优化（参见[自动化超参优化](../../tools/hpo/README_ZH.md)）。

Type 算子类型: **filter**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_confidence` | <class 'float'> | `0.5` | Minimum confidence threshold for subplot detection. Samples with confidence above this threshold will be filtered out. |
| `min_horizontal_lines` | <class 'int'> | `3` | Minimum number of horizontal lines required for subplot detection. |
| `min_vertical_lines` | <class 'int'> | `3` | Minimum number of vertical lines required for subplot detection. |
| `canny_threshold1` | <class 'int'> | `70` | First threshold for Canny edge detector hysteresis procedure. |
| `canny_threshold2` | <class 'int'> | `190` | Second threshold for Canny edge detector hysteresis procedure. |
| `hough_threshold` | <class 'int'> | `110` | Accumulator threshold parameter for Hough transform. |
| `min_line_length` | <class 'int'> | `110` | Minimum line length in pixels for Hough transform. |
| `max_line_gap` | <class 'int'> | `18` | Maximum allowed gap between line segments to treat them as a single line. |
| `angle_tolerance` | <class 'float'> | `4.0` | Angle tolerance in degrees for classifying lines as horizontal or vertical. |
| `any_or_all` | <class 'str'> | `'any'` | Filter this sample with 'any' or 'all' strategy of all images. 'any': filter this sample if any images meet the condition. 'all': filter this sample only if all images meet the condition. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_subplot_detection
```python
ImageSubplotFilter(min_confidence=0.5)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_subplot.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_subplot.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_nosubplot1.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_nosubplot1.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_nosubplot1.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_nosubplot1.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator detects subplots in images using Hough Line Transform. The first sample contains a subplot image with clear grid lines, so it is filtered out. The second sample contains a regular image without subplots, so it is kept in the target list.

该算子使用Hough直线变换检测图像中的子图。第一个样本包含具有清晰网格线的子图图像，因此被过滤掉。第二个样本包含没有子图的常规图像，因此保留在目标列表中。

### test_any_strategy
```python
ImageSubplotFilter(min_confidence=0.5, any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_nosubplot1.jpg|image_subplot.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_nosubplot1.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/image_subplot.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_nosubplot1.jpg|image_subplot.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_nosubplot1.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/image_subplot.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
With the 'any' strategy, the operator filters samples if any image contains subplots. In this case, the sample contains both a subplot image and a regular image, so it is filtered out due to the presence of at least one subplot image.

使用'any'策略时，如果任何图像包含子图，则算子会过滤样本。在这种情况下，样本同时包含子图图像和常规图像，因此由于至少存在一个子图图像而被过滤掉。

### test_all_strategy
```python
ImageSubplotFilter(min_confidence=0.5, any_or_all='all')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_nosubplot1.jpg|image_subplot.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_nosubplot1.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/image_subplot.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_nosubplot1.jpg|image_subplot.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_nosubplot1.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/image_subplot.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
With the 'all' strategy, the operator only filters samples if all images contain subplots. In this case, the sample contains one subplot image and one regular image, so it is kept since not all images meet the subplot condition.

使用'all'策略时，只有当所有图像都包含子图时，算子才会过滤样本。在这种情况下，样本包含一个子图图像和一个常规图像，因此由于并非所有图像都满足子图条件而被保留。

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_subplot_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_subplot_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)