# naive_grouper

Group all samples in a dataset into a single batched sample.

This operator takes a dataset and combines all its samples into one batched sample. If the input dataset is empty, it returns an empty dataset. The resulting batched sample is a dictionary where each key corresponds to a list of values from all samples in the dataset.

将数据集中的所有样本合并成一个批次样本。

该算子接受一个数据集，并将其所有样本合并为一个批次样本。如果输入的数据集为空，则返回一个空的数据集。生成的批次样本是一个字典，其中每个键对应于数据集中所有样本的值列表。

Type 算子类型: **grouper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_naive_group
```python
NaiveGrouper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is Sunday and it&#x27;s a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur la plateforme MT4, plusieurs manières d&#x27;accéder à 
ces fonctionnalités sont conçues simultanément.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">欢迎来到阿里巴巴！</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&quot;Today is Sunday and it&#x27;s a happy day!&quot;, &quot;Sur la plateforme MT4, plusieurs manières d&#x27;accéder à \nces fonctionnalités sont conçues simultanément.&quot;, &#x27;欢迎来到阿里巴巴！&#x27;]</pre></div>

#### ✨ explanation 解释
The NaiveGrouper operator combines all the individual samples in a dataset into one single sample. In this case, it takes three separate text entries and groups them together into a list under a single 'text' key. The output is a single sample with a 'text' field that contains a list of all the input texts. This is useful for processing or analyzing the entire dataset as a whole, rather than handling each sample individually.
NaiveGrouper 算子将数据集中的所有单独样本合并成一个单一的样本。在这个例子中，它将三个独立的文本条目组合在一起，并将它们放入一个列表中，该列表位于一个单独的 'text' 键下。输出是一个包含 'text' 字段的单一样本，该字段包含了所有输入文本的列表。这在需要将整个数据集作为一个整体进行处理或分析时非常有用，而不是单独处理每个样本。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/grouper/naive_grouper.py)
- [unit test 单元测试](../../../tests/ops/grouper/test_naive_grouper.py)
- [Return operator list 返回算子列表](../../Operators.md)