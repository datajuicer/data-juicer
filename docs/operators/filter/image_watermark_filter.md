# image_watermark_filter

Filter to keep samples whose images have no watermark with high probability.

This operator uses a Hugging Face watermark detection model to filter samples based on the presence of watermarks in their images. It keeps samples where the predicted watermark probability is below a specified threshold. The operator supports two strategies: 'any' (keep if any image meets the condition) and 'all' (keep only if all images meet the condition). The key metric 'image_watermark_prob' is computed for each image, representing the probability that the image contains a watermark. If no images are present in the sample, the metric is set to an empty array.

筛选出高概率没有水印的图片样本。

该算子使用 Hugging Face 水印检测模型基于图像中的水印存在情况过滤样本。它保留预测水印概率低于指定阈值的样本。该算子支持两种策略：'any'（如果有任何图像满足条件则保留）和 'all'（只有当所有图像都满足条件时才保留）。关键指标 'image_watermark_prob' 为每个图像计算，表示图像包含水印的概率。如果样本中没有图像，则该指标设置为空数组。

Type 算子类型: **filter**

Tags 标签: gpu, hf, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_watermark_model` | <class 'str'> | `'amrul-hzz/watermark_detector'` | watermark detection model name on huggingface. |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `prob_threshold` | <class 'float'> | `0.8` | the predicted watermark probability threshold for samples. range from 0 to 1. Samples with watermark probability less than this threshold will be kept. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all images. 'any': keep this sample if any images meet the condition. 'all': keep this sample only if all images meet the condition. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_watermark_filter
```python
ImageWatermarkFilter(hf_watermark_model='amrul-hzz/watermark_detector', prob_threshold=0.8)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
The operator filters out samples with a high probability of containing watermarks, keeping only those below the 0.8 threshold. In this case, it keeps img1 and img3 because their watermark probabilities are below the threshold, while img2 is removed for having a higher probability.
算子过滤掉水印概率高于0.8的样本，只保留低于此阈值的样本。在这种情况下，它保留了img1和img3，因为它们的水印概率低于阈值，而img2由于概率较高被移除。

### test_any
```python
ImageWatermarkFilter(hf_watermark_model='amrul-hzz/watermark_detector', prob_threshold=0.4, any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img2.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img2.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 images</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">img1.png|img3.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/img1.png" width="160" style="margin:4px;"/><img src="../../../tests/ops/data/img3.jpg" width="160" style="margin:4px;"/></div></div></div>

#### ✨ explanation 解释
With the 'any' strategy, the operator retains samples if at least one image in each sample has a watermark probability below the 0.4 threshold. Here, both samples have at least one image (img1) meeting the condition, but only the second sample (with img1 and img3) is kept, as the first sample contains an image (img2) that exceeds the threshold.
使用'any'策略时，如果每个样本中至少有一张图片的水印概率低于0.4阈值，则算子会保留该样本。这里，两个样本都至少包含一张满足条件的图片（img1），但只有第二个样本（包含img1和img3）被保留下来，因为第一个样本包含了一张超过阈值的图片（img2）。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/image_watermark_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_image_watermark_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)