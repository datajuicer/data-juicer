# image_face_ratio_filter

Filter to keep samples with face area ratios within a specific range.

This operator filters samples based on the ratio of the largest face area to the total image area. It uses an OpenCV classifier for face detection. The key metric, 'face_ratios', is computed for each image in the sample. Samples are kept if the face area ratios fall within the specified min and max ratio range. The filtering strategy can be set to 'any' (keep if any image meets the condition) or 'all' (keep only if all images meet the condition). If no images are present in the sample, the sample is retained.

用于保留面部区域比率在特定范围内的样本的过滤器。

该算子根据最大面部区域与总图像面积的比率来过滤样本。它使用OpenCV分类器进行面部检测。关键指标'face_ratios'是为样本中的每个图像计算的。如果面部区域比率落在指定的最小和最大比率范围内，则保留样本。过滤策略可以设置为'any'（如果有任何图像满足条件则保留）或'all'（只有当所有图像都满足条件时才保留）。如果样本中没有图像，则保留该样本。

Type 算子类型: **filter**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `cv_classifier` | <class 'str'> | `''` | OpenCV classifier path for face detection. By default, we will use 'haarcascade_frontalface_alt.xml'. |
| `min_ratio` | <class 'float'> | `0.0` | Min ratio for the largest face area in an image. |
| `max_ratio` | <class 'float'> | `0.4` | Max ratio for the largest face area in an image. |
| `any_or_all` | <class 'str'> | `'any'` | Keep this sample with 'any' or 'all' strategy of all images. 'any': keep this sample if any images meet the condition. 'all': keep this sample only if all images meet the condition. |
| `args` |  | `''` | Extra positional arguments. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_filter_small
```python
ImageFaceRatioFilter(min_ratio=0.4, max_ratio=1.0)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">lena.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/lena.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator filters out images where the face area ratio is not within 0.4 to 1.0, keeping only those that meet the criteria. The sample with 'lena-face.jpg' is kept because its face area ratio falls within the specified range, while others are removed.
算子过滤掉脸部面积比例不在0.4到1.0范围内的图片，只保留满足条件的图片。包含'lena-face.jpg'的样本被保留，因为其脸部面积比例在指定范围内，而其他样本被移除。

### test_all
```python
ImageFaceRatioFilter(min_ratio=0.0, max_ratio=0.4, any_or_all='all')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|lena.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/lena.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">lena.jpg|lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/lena.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|lena.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/lena.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
This test checks if all images in a sample have a face area ratio within the 0.0 to 0.4 range. Only the sample containing both 'cat.jpg' and 'lena.jpg' is kept, as both of these images satisfy the condition, whereas other samples contain at least one image that does not meet the criteria.
此测试检查样本中的所有图片的脸部面积比例是否都在0.0到0.4范围内。只有同时包含'cat.jpg'和'lena.jpg'的样本被保留，因为这两张图片都满足条件，而其他样本至少包含一张不满足条件的图片。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_face_ratio_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_face_ratio_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)