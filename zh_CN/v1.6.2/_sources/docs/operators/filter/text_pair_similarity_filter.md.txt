# text_pair_similarity_filter

Filter to keep text pairs with similarities within a specific range.

This operator computes the similarity between two texts in a pair using a Hugging Face CLIP model. It keeps samples where the similarity score falls within the specified min and max thresholds. The key metric, 'text_pair_similarity', is computed as the cosine similarity between the text embeddings. The operator supports two strategies for keeping samples: 'any' (keep if any pair meets the condition) and 'all' (keep only if all pairs meet the condition). If the second text key is not provided, the operator will raise an error. The similarity scores are cached under the 'text_pair_similarity' field in the sample's stats.

用于保留相似度在特定范围内的文本对的过滤器。

该算子使用Hugging Face CLIP模型计算一对文本之间的相似度。它保留相似度得分在指定最小值和最大值之间的样本。关键指标'text_pair_similarity'计算为文本嵌入之间的余弦相似度。该算子支持两种保留样本的策略：'any'（只要有任何一对满足条件就保留）和'all'（只有当所有对都满足条件时才保留）。如果不提供第二个文本键，该算子将引发错误。相似度分数会被缓存在样本统计信息的'text_pair_similarity'字段中。

Type 算子类型: **filter**

Tags 标签: gpu, hf, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_clip` |  | `'openai/clip-vit-base-patch32'` | clip model name on huggingface to compute the similarity between image and text. |
| `trust_remote_code` |  | `False` | whether to trust the remote code of HF models. |
| `min_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `0.1` | The min similarity to keep samples. |
| `max_score` | <class 'jsonargparse.typing.ClosedUnitInterval'> | `1.0` | The max similarity to keep samples. |
| `text_key_second` |  | `None` | used to store the other sentence in the text pair. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all images. 'any': keep this sample if any images meet the condition. 'all': keep this sample only if all images meet the condition. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_no_eoc_special_token
```python
TextPairSimilarityFilter(hf_clip='openai/clip-vit-base-patch32', any_or_all='any', min_score=0.85, max_score=0.99, text_key_second='target_text')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a lovely cat</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>target_text</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>a lovely cat</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a cute cat</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>target_text</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>a lovely cat</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a black dog</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>target_text</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>a lovely cat</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a cute cat</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>target_text</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>a lovely cat</td></tr></table></div></div>

#### ✨ explanation 解释
The operator filters the input data to keep only those text pairs with a similarity score between 0.85 and 0.99. It uses a Hugging Face CLIP model to calculate the cosine similarity between the embeddings of each text pair. The 'any' strategy is selected, meaning a sample is kept if any of its text pairs meet the condition. In this case, the second sample ('a cute cat' and 'a lovely cat') has a similarity within the specified range, so it is kept, while the other samples are removed because their similarities fall outside the defined thresholds.
算子过滤输入数据，只保留相似度得分在0.85到0.99之间的文本对。它使用Hugging Face的CLIP模型计算每对文本嵌入之间的余弦相似度。选择了'any'策略，意味着如果样本中的任何一对文本满足条件，则保留该样本。在这种情况下，第二个样本（'一只可爱的小猫'和'一只漂亮的小猫'）的相似度在指定范围内，因此被保留，而其他样本由于其相似度不在定义的阈值范围内而被移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/text_pair_similarity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_text_pair_similarity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)