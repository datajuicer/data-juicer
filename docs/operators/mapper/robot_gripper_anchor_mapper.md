# robot_gripper_anchor_mapper

Extract an anchor on the gripper-state timeline.

从夹爪状态时间序列中提取阶段锚点。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
| --- | --- | --- | --- |
| `output_dir` | str | `output/episodes` | Episode artifact directory. |

## 📊 Usage 使用示例

```yaml
process:
  - robot_gripper_anchor_mapper:
      output_dir: ./output/episodes
```

## Input and Output 输入与输出

Each sample includes `source` and `episode_id`.

**Input:** DROID: `metadata_path`, `state_path`, `episode_index`.

**Output:** Adds `instruction`, `instruction_source_column` and `episode_dir`; writes `ir/phase_anchor.json`.

## 🔗 Related Links 相关链接

- [Source code 源代码](../../../data_juicer/ops/mapper/robot_gripper_anchor_mapper.py)
- [DROID recipe and input preparation](https://github.com/datajuicer/data-juicer-hub/tree/main/demo/embodied_vqa)
- [Return operator list 返回算子列表](../../Operators.md)
