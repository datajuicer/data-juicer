# image_text_matching_filter

Filter to keep samples with image-text matching scores within a specific range.

This operator uses a Hugging Face BLIP model to compute the matching score between images and text. It keeps samples where the matching score falls within the specified `min_score` and `max_score` range. The key metric, `image_text_matching_score`, is computed for each image-text pair. If multiple images are associated with a single text, the scores can be reduced using 'avg', 'max', or 'min' modes. The operator supports horizontal and vertical flipping of images. Samples are kept based on either 'any' or 'all' strategy: 'any' keeps the sample if any image meets the condition, while 'all' keeps the sample only if all images meet the condition.

用来保留图像-文本匹配分数在特定范围内的样本的过滤器。

该算子使用 Hugging Face BLIP 模型计算图像和文本之间的匹配分数。它保留匹配分数落在指定 `min_score` 和 `max_score` 范围内的样本。关键指标 `image_text_matching_score` 为每个图像-文本对计算。如果多个图像与单个文本相关联，可以使用 'avg'、'max' 或 'min' 模式来归约分数。该算子支持图像的水平和垂直翻转。根据 'any' 或 'all' 策略保留样本：'any' 表示只要有任何一个图像满足条件就保留样本，而 'all' 则表示只有当所有图像都满足条件时才保留样本。

Type 算子类型: **filter**

Tags 标签: gpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_blip` | <class 'str'> | `'Salesforce/blip-itm-base-coco'` | blip model name on huggingface to compute the matching score between image and text. |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `min_score` | <class 'float'> | `0.003` | The min matching score to keep samples. |
| `max_score` | <class 'float'> | `1.0` | The max matching score to keep samples. |
| `horizontal_flip` | <class 'bool'> | `False` | Flip image horizontally (left to right). |
| `vertical_flip` | <class 'bool'> | `False` | Flip image vertically (top to bottom). |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all images. 'any': keep this sample if any images meet the condition. 'all': keep this sample only if all images meet the condition. |
| `reduce_mode` | <class 'str'> | `'avg'` | reduce mode when one text corresponds to multiple images in a chunk. 'avg': Take the average of multiple values 'max': Take the max of multiple values 'min': Take the min of multiple values |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_keep_any
```python
ImageTextMatchingFilter(hf_blip='Salesforce/blip-itm-base-coco', reduce_mode='avg', any_or_all='any', min_score=0.003, max_score=1.0)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__image&gt;a woman sitting on the beach with a dog &lt;|__dj__eoc|&gt; &lt;__dj__image&gt;a man sitting on the grass with a cat &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">blip.jpg|blip.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__image&gt;a woman sitting on the beach with a dog &lt;|__dj__eoc|&gt; &lt;__dj__image&gt;a man sitting on the grass with a cat &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">blip.jpg|blip.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator keeps samples if any of the image-text pairs have a matching score within the specified range. In this case, at least one of the image-text pairs meets the criteria, so the sample is kept.
算子在任何图像-文本对的匹配分数落在指定范围内时保留样本。此例中，至少有一组图像-文本对满足条件，因此该样本被保留。

### test_keep_all
```python
ImageTextMatchingFilter(hf_blip='Salesforce/blip-itm-base-coco', reduce_mode='avg', any_or_all='all', min_score=0.003, max_score=1.0)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | 2 images</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">&lt;__dj__image&gt;a woman sitting on the beach with a dog &lt;|__dj__eoc|&gt; &lt;__dj__image&gt;a man sitting on the grass with a cat &lt;|__dj__eoc|&gt;</pre><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">blip.jpg|blip.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/blip.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[]</pre></div>

#### ✨ explanation 解释
The operator only keeps samples if all of the image-text pairs have a matching score within the specified range. In this case, not all image-text pairs meet the criteria, so the sample is removed.
算子仅在所有图像-文本对的匹配分数都落在指定范围内时才保留样本。此例中，并非所有的图像-文本对都满足条件，因此该样本被移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_text_matching_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_text_matching_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)