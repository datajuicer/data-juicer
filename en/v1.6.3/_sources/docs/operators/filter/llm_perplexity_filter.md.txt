# llm_perplexity_filter

Filter to keep samples with perplexity scores within a specified range, computed using a specified LLM.

This operator computes the perplexity score for each sample using a Hugging Face LLM. It then filters the samples based on whether their perplexity scores fall within the specified minimum and maximum score range. The perplexity score is calculated as the exponential of the loss value from the LLM. The operator uses a query and response template to format the input text for the LLM. If the perplexity score is not already cached in the sample's stats under the key 'llm_perplexity', it will be computed.

用于保留使用指定 LLM 计算出的困惑度分数在指定范围内的样本的过滤器。

该算子使用 Hugging Face LLM 为每个样本计算困惑度分数。然后根据困惑度分数是否落在指定的最小和最大分数范围内来过滤样本。困惑度分数计算为 LLM 损失值的指数。该算子使用查询和响应模板来格式化输入 LLM 的文本。如果困惑度分数尚未缓存在样本的统计信息中 'llm_perplexity' 键下，则会进行计算。

Type 算子类型: **filter**

Tags 标签: gpu, hf

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_model` | <class 'str'> | `'Qwen/Qwen2.5-0.5B'` | huggingface embedding model name. |
| `model_params` | typing.Optional[typing.Dict] | `None` | Parameters for initializing the API model. |
| `min_score` | <class 'float'> | `1.0` | Minimum perplexity score. |
| `max_score` | <class 'float'> | `100.0` | Maximum perplexity score. |
| `query_template` | typing.Optional[str] | `None` | Template for building the query string. |
| `response_template` | typing.Optional[str] | `None` | Template for building the response string. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_hf_model
```python
LLMPerplexityFilter(hf_model='Qwen/Qwen2.5-0.5B', min_score=1, max_score=50)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is Sunday and it&#x27;s a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is Sund Sund Sund Sund Sunda and it&#x27;s a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a v s e c s f e f g a qkc</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">，。、„”“«»１」「《》´∶：？！（）；–—．～’…━〈〉【】％►</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 5:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Do you need a cup of coffee?</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 6:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">emoji表情测试下😊，😸31231</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is Sunday and it&#x27;s a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Do you need a cup of coffee?</pre></div>

#### ✨ explanation 解释
The operator calculates the perplexity score for each text sample and retains only those with scores between 1 and 50. The first and fifth samples meet this criterion, while others are filtered out due to their perplexity scores being outside the specified range.
算子计算每个文本样本的困惑度分数，并仅保留得分在1到50之间的样本。第一个和第五个样本符合此标准，而其他样本由于其困惑度分数超出指定范围而被过滤掉。

### test_rft_data
```python
LLMPerplexityFilter(hf_model='Qwen/Qwen2.5-0.5B', min_score=1, max_score=5, query_template='Question: {text}', response_template='Answer: {answer}')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text | answer</div><div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">What is the capital of France?</pre></div><div><strong>A:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The capital of France is Paris.</pre></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text | answer</div><div class="qa" style="margin-bottom:6px;"><div><strong>Q:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">What is the capital of China?</pre></div><div><strong>A:</strong> <pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">The capital of China is Paris.</pre></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">What is the capital of France?</pre></div>

#### ✨ explanation 解释
The operator uses a question-answer template to compute the perplexity score of each sample, keeping only those with scores within the 1 to 5 range. The first sample meets the condition, but the second does not, as its answer is incorrect leading to a higher perplexity score that exceeds the set limit.
算子使用问答模板来计算每个样本的困惑度分数，仅保留得分在1到5范围内的样本。第一个样本符合条件，但第二个样本不符合，因为其答案错误导致困惑度分数较高，超出了设定的限制。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/llm_perplexity_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_llm_perplexity_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)