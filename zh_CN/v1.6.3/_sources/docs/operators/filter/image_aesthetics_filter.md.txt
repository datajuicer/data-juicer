# image_aesthetics_filter

Filter to keep samples with aesthetics scores within a specific range.

This operator uses a Hugging Face model to predict the aesthetics score of images. It keeps samples where the predicted scores fall within the specified min and max score range. The operator supports two strategies: 'any' (keep if any image meets the condition) and 'all' (keep only if all images meet the condition). Aesthetics scores are cached in the 'image_aesthetics_scores' field. If no images are present, the sample is kept. Scores are normalized by dividing by 10 if the model name includes 'shunk031/aesthetics-predictor'.

过滤出美学评分在特定范围内的样本。

该算子使用 Hugging Face 模型来预测图像的美学评分。它保留那些预测评分落在指定最小和最大评分范围内的样本。该算子支持两种策略：'any'（只要有任何一张图片符合条件就保留）和 'all'（只有当所有图片都符合条件时才保留）。美学评分会被缓存在 'image_aesthetics_scores' 字段中。如果没有图片，样本也会被保留。如果模型名称包含 'shunk031/aesthetics-predictor'，则分数会被除以 10 进行归一化。

Type 算子类型: **filter**

Tags 标签: gpu, hf, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_scorer_model` | <class 'str'> | `''` | Huggingface model name for the aesthetics predictor. By default, we will use 'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE', refer to pypi.org/project/simple-aesthetics-predictor |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `min_score` | <class 'float'> | `0.5` | Min score for the predicted aesthetics in an image. |
| `max_score` | <class 'float'> | `1.0` | Max score for the predicted aesthetics in an image. |
| `any_or_all` | <class 'str'> | `'any'` | Keep this sample with 'any' or 'all' strategy of all images. 'any': keep this sample if any images meet the condition. 'all': keep this sample only if all images meet the condition. |
| `args` |  | `''` | Extra positional arguments. |
| `kwargs` |  | `''` | Extra keyword arguments. |

## 📊 Effect demonstration 效果演示
### test_filter_small
```python
ImageAestheticsFilter(hf_scorer_model='shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE', min_score=0.55, max_score=1.0)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">blip.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">blip.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator filters out images with aesthetics scores outside the 0.55-1.0 range, keeping only those that meet the condition. In this case, only the image with a score within the specified range is kept.
算子过滤掉美学评分不在0.55-1.0范围内的图片，只保留符合条件的图片。在这种情况下，只有评分在指定范围内的图片被保留。

### test_all
```python
ImageAestheticsFilter(hf_scorer_model='shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE', any_or_all='all')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|blip.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">blip.jpg|lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg|lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">blip.jpg|lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator requires all images in a sample to have aesthetics scores within the 0.4-0.55 range to be kept. Only the sample where both images meet this requirement is retained.
算子要求样本中的所有图片的美学评分都必须在0.4-0.55范围内才能被保留。只有当两张图片都满足此条件时，该样本才会被保留。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_aesthetics_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_aesthetics_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)