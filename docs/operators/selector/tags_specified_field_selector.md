# tags_specified_field_selector

Selector to filter samples based on the tags of a specified field.

This operator selects samples where the value of the specified field matches one of the target tags. The field can be multi-level, with levels separated by dots (e.g., 'level1.level2'). The operator checks if the specified field exists in the dataset and if the field value is a string, number, None, or a list of these types. If the scalar value or any list element matches a target tag, the sample is kept. The selection is case-sensitive.

- The `field_key` parameter specifies the field to check.
- The `target_tags` parameter is a list of tags to match against the field value.
- If the dataset has fewer than two samples or if `field_key` is empty, the dataset is returned unchanged.

根据指定字段的标签筛选样本的选择器。

该算子选择指定字段的值与目标标签之一匹配的样本。字段可以是多级的，级别之间用点分隔（例如，'level1.level2'）。该算子检查指定的字段是否存在于数据集中，以及字段值是否为字符串、数字、None 或由这些类型组成的列表。如果标量值或列表中的任一元素与目标标签匹配，则保留该样本。选择区分大小写。

- `field_key` 参数指定要检查的字段。
- `target_tags` 参数是要与字段值匹配的标签列表。
- 如果数据集中的样本少于两个或 `field_key` 为空，则返回不变的数据集。

Type 算子类型: **selector**

Tags 标签: cpu

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `field_key` | <class 'str'> | `''` | Selector based on the specified value corresponding to the target key. The target key corresponding to multi-level field information need to be separated by '.'. |
| `target_tags` | typing.List[str] | `None` | Target tags to be select. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## Multi-label data 多标签数据

A list-valued field is selected when **any** element matches a target tag. Empty lists do not match; duplicate matching tags do not duplicate rows. Matching remains case-sensitive and preserves input row order. List elements must be strings, numbers, or `None`; nested lists and objects are rejected. A scalar `None` or a `None` list element matches only when `target_tags` contains `None`.

列表字段中任一标签命中即保留该行，空列表不命中，重复标签不会产生重复行。匹配区分大小写并保持输入顺序；不支持嵌套列表或对象。原有单标签及少于两行时直接返回的行为保持不变。

For example, keep documents tagged with math or code without exploding the dataset:

```yaml
process:
  - tags_specified_field_selector:
      field_key: meta.topics
      target_tags: [math, code]
```

`{"meta": {"topics": ["math", "reasoning"]}}` matches, while `{"meta": {"topics": ["history"]}}` does not.

## 📊 Effect demonstration 效果演示
### test_tag_select
```python
TagsSpecifiedFieldSelector(field_key='meta.sentiment', target_tags=['happy', 'sad'])
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>sentiment</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>happy</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">b</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>sentiment</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>happy</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">c</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>sentiment</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>sad</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 4:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">d</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>sentiment</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>angry</td></tr></table></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">a</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>sentiment</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>happy</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">b</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>sentiment</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>happy</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">c</pre><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>meta</th></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>sentiment</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>sad</td></tr></table></div></div>

#### ✨ explanation 解释
The TagsSpecifiedFieldSelector operator filters the dataset by keeping samples where the 'meta.sentiment' field value matches either 'happy' or 'sad'. The sample with 'angry' sentiment is removed because it does not match any of the target tags.
TagsSpecifiedFieldSelector算子通过保留'meta.sentiment'字段值为'happy'或'sad'的样本，来过滤数据集。带有'angry'情绪的样本被移除，因为它与目标标签不匹配。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/selector/tags_specified_field_selector.py)
- [unit test 单元测试](../../../tests/ops/selector/test_tags_specified_field_selector.py)
- [Return operator list 返回算子列表](../../Operators.md)