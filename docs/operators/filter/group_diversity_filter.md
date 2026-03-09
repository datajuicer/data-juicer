# group_diversity_filter

Filter to keep samples based on their semantic diversity within a group.

This operator computes the semantic diversity of each sample relative to the group by embedding all input texts into vectors using either a HuggingFace model or an API-based embedding model. It calculates the cosine similarity between each sample's embedding and the mean embedding of the entire group. A lower cosine similarity indicates that the sample is more semantically distinct from the group average, i.e., higher diversity. The raw cosine similarity is stored as `llm_embd_diversity` in the sample's stats. During filtering, the similarities are normalized within the group using `norm_ratio` to produce a diversity score, and only samples whose score falls within [`min_score`, `max_score`] are kept. Note that `max_score` should not exceed `norm_ratio`, as `norm_ratio` defines the upper bound of the diversity score.


用于根据样本在组内语义多样性进行过滤的算子。

该算子通过HuggingFace模型或API嵌入模型将所有输入文本转换为向量，计算每个样本的嵌入向量与组内平均嵌入向量之间的余弦相似度。余弦相似度越低，说明该样本与组平均语义差异越大，即多样性越高。原始余弦相似度存储在样本统计信息的`llm_embd_diversity`字段中。过滤时，通过`norm_ratio`对组内相似度进行归一化得到多样性得分，只保留得分在[`min_score`, `max_score`]范围内的样本。注意`max_score`不应超过`norm_ratio`，因为`norm_ratio`定义了多样性得分的上限。

Type 算子类型: **filter**

Tags 标签: gpu, hf, text

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `api_or_hf_model` | `str` | `'text-embedding-v3'` | API or HuggingFace embedding model name or local path. |
| `is_hf_model` | `bool` | `False` | Whether the model is a HuggingFace local model. If False, uses API mode. |
| `api_endpoint` | `str` | `'/embeddings'` | Embedding URL endpoint for the API. |
| `response_path` | `str` | `'data.0.embedding'` | Path to extract embedding from the API response. Defaults to 'data.0.embedding' for embedding model. |
| `model_params` | `dict` | `{}` | Extra parameters for initializing the model. |
| `ebd_dim` | `<class 'jsonargparse.typing.PositiveInt'>` | `512` | Embedding dimension, only effective in API mode (`is_hf_model=False`). |
| `min_score` | `<class 'jsonargparse.typing.NonNegativeFloat'>` | `0.0` | Minimum diversity score to keep samples. |
| `max_score` | `<class 'jsonargparse.typing.NonNegativeFloat'>` | `1.0` | Maximum diversity score to keep samples. Should not exceed `norm_ratio`. |
| `norm_ratio` | `<class 'jsonargparse.typing.NonNegativeFloat'>` | `0.5` | Normalization ratio controlling the upper bound of diversity score. The valid range of score is `[0, norm_ratio]`. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_group_diversity_filter
```python
GroupDiversityFilter(api_or_hf_model='iic/gte_Qwen2-1.5B-instruct', is_hf_model=True, min_score=0.3, max_score=0.5, norm_ratio=0.5)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The little cat is playing with a ball in the garden.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">小猫正在花园里开心地玩着球。</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">A kitten is chasing a colorful ball on the green grass.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">花园里有一只可爱的小猫在追着小球跑。</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 5:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Quantum entanglement is a fundamental concept in quantum mechanics.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 6:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">量子纠缠是量子力学中描述粒子间关联性的核心理论。</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Quantum entanglement is a fundamental concept in quantum mechanics.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">量子纠缠是量子力学中描述粒子间关联性的核心理论。</pre></div>

#### ✨ explanation 解释
The operator filters the input data to keep only those samples with a diversity score between 0.3 and 0.5. It uses a local HuggingFace embedding model (`gte_Qwen2-1.5B`) to compute the cosine similarity between each sample's embedding and the group mean embedding, stored as `llm_embd_diversity`. The 4 cat-and-ball samples are semantically similar to each other, resulting in high cosine similarities (0.864~0.908) close to the group mean, and thus low diversity scores (0.000~0.102) that fall below `min_score=0.3`. The 2 quantum mechanics samples are semantically distant from the group mean, with lower cosine similarities (0.693~0.765) and higher diversity scores (0.333~0.500), so they are kept.

算子过滤输入数据，只保留多样性得分在0.3到0.5之间的样本。它使用本地HuggingFace嵌入模型（`gte_Qwen2-1.5B`）计算每个样本的嵌入向量与组平均嵌入向量之间的余弦相似度，存储为`llm_embd_diversity`。4条猫玩球的样本语义相近，余弦相似度较高（0.864~0.908），接近组平均，多样性得分较低（0.000~0.102），低于`min_score=0.3`因此被过滤。2条量子力学样本与组平均语义差异较大，余弦相似度较低（0.693~0.765），多样性得分较高（0.333~0.500），因此被保留。

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/group_diversity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_group_diversity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)
