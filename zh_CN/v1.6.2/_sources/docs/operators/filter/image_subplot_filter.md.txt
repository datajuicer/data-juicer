# image_subplot_filter

Filter to detect and remove samples with images containing subplots.

This filter uses Hough Line Transform to detect straight lines in images, which is particularly effective for detecting grid-like subplot layouts with perfectly straight edges.

The algorithm works by: 1. Converting images to grayscale and applying edge detection 2. Using Hough Line Transform to detect straight lines 3. Classifying lines as horizontal or vertical based on angle 4. Counting lines that meet length and angle requirements 5. Calculating confidence based on line counts and distribution

用于检测并移除包含子图（subplots）图像的样本的过滤器。

该过滤器使用霍夫直线变换（Hough Line Transform）检测图像中的直线，对于检测具有完全笔直边缘的网格状子图布局尤其有效。

算法工作流程如下：
1. 将图像转换为灰度图并应用边缘检测
2. 使用霍夫直线变换检测直线
3. 根据角度将直线分类为水平或垂直
4. 统计满足长度和角度要求的直线数量
5. 基于直线数量及其分布计算置信度

Type 算子类型: **filter**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_horizontal_lines` | <class 'int'> | `3` | Minimum number of horizontal lines to consider an image as containing subplots. |
| `min_vertical_lines` | <class 'int'> | `3` | Minimum number of vertical lines to consider an image as containing subplots. |
| `min_confidence` | <class 'float'> | `0.5` | Minimum confidence score for filtering. Images with subplot confidence above this threshold will be considered as containing subplots. |
| `any_or_all` | <class 'str'> | `'any'` | Strategy for multi-image samples. 'any' filters the sample if any image contains subplots. 'all' filters the sample only if all images contain subplots. |
| `canny_threshold1` | <class 'int'> | `70` | First threshold for Canny edge detector. |
| `canny_threshold2` | <class 'int'> | `190` | Second threshold for Canny edge detector. |
| `hough_threshold` | <class 'int'> | `110` | Accumulator threshold for Hough transform. |
| `min_line_length` | <class 'int'> | `110` | Minimum line length to be detected. |
| `max_line_gap` | <class 'int'> | `18` | Maximum gap between line segments to be treated as a single line. |
| `angle_tolerance` | <class 'float'> | `4.0` | Tolerance in degrees for classifying lines as horizontal/vertical. |
| `args` |  | `''` | Extra args. |
| `kwargs` |  | `''` | Extra args. |

## 📊 Effect demonstration 效果演示
### test_any_strategy
```python
ImageSubplotFilter(any_or_all='any', min_confidence=0.5)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_nosubplot.jpg|image_subplot.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_nosubplot.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/image_subplot.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Nothing Left</strong></div><div class="media-section" style="margin-bottom:8px;"></div></div>

#### ✨ explanation 解释
With the 'any' strategy, the operator filters samples if any image contains subplots. In this case, the sample contains both a subplot image and a regular image, so it is filtered out due to the presence of at least one subplot image.

使用'any'策略时，如果任何图像包含子图，则算子会过滤样本。在这种情况下，样本同时包含子图图像和常规图像，因此由于至少存在一个子图图像而被过滤掉。

### test_all_strategy
```python
ImageSubplotFilter(any_or_all='all', min_confidence=0.5)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_nosubplot.jpg|image_subplot.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_nosubplot.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/image_subplot.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">image_nosubplot.jpg|image_subplot.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/image_nosubplot.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/image_subplot.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
With the 'all' strategy, the operator only filters samples if all images contain subplots. In this case, the sample contains one subplot image and one regular image, so it is kept since not all images meet the subplot condition.

使用'all'策略时，只有当所有图像都包含子图时，算子才会过滤样本。在这种情况下，样本包含一个子图图像和一个常规图像，因此由于并非所有图像都满足子图条件而被保留。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_subplot_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_subplot_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)