# naive_reverse_grouper

Split batched samples into individual samples.

This operator processes a dataset by splitting each batched sample into individual samples. It also handles and optionally exports batch metadata.
- If a sample contains 'batch_meta', it is separated and can be exported to a specified path.
- The operator converts the remaining data from a dictionary of lists to a list of dictionaries, effectively unbatching the samples.
- If `batch_meta_export_path` is provided, the batch metadata is written to this file in JSON format, one entry per line.
- If no samples are present in the dataset, the original dataset is returned.

将批量样本拆分为单个样本。

该算子通过将每个批量样本拆分为单个样本来处理数据集。它还处理并可选地导出批量元数据。
- 如果样本包含 'batch_meta'，则将其分离并可以导出到指定路径。
- 该算子将剩余数据从字典列表转换为字典列表，从而取消批量样本。
- 如果提供了 `batch_meta_export_path`，则批量元数据将以 JSON 格式写入此文件，每行一个条目。
- 如果数据集中没有样本，则返回原始数据集。

Type 算子类型: **grouper**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `batch_meta_export_path` |  | `None` | the path to export the batch meta. Just drop the batch meta if it is None. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_one_batched_sample
```python
NaiveReverseGrouper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[&quot;Today is Sunday and it&#x27;s a happy day!&quot;, &quot;Sur la plateforme MT4, plusieurs manières d&#x27;accéder à \nces fonctionnalités sont conçues simultanément.&quot;, &#x27;欢迎来到阿里巴巴！&#x27;]</pre></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Today is Sunday and it&#x27;s a happy day!</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">Sur la plateforme MT4, plusieurs manières d&#x27;accéder à 
ces fonctionnalités sont conçues simultanément.</pre></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">欢迎来到阿里巴巴！</pre></div>

#### ✨ explanation 解释
This example demonstrates the operator's basic functionality of unbatching. It takes a single batch containing multiple text samples and splits it into individual samples, each with its own 'text' field. The output is a list where each element is a dictionary with one 'text' entry.
这个例子展示了算子的基本功能，即将批量数据拆分成单个样本。它接收一个包含多个文本样本的批次，并将其拆分成每个都具有自己'text'字段的单独样本。输出是一个列表，其中每个元素都是一个包含一个'text'条目的字典。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/grouper/naive_reverse_grouper.py)
- [unit test 单元测试](../../../tests/ops/grouper/test_naive_reverse_grouper.py)
- [Return operator list 返回算子列表](../../Operators.md)