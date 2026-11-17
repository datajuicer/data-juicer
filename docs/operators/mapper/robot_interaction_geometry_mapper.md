# robot_interaction_geometry_mapper

Compose tracked geometry, state anchors and synchronized views into episode IR.

将跟踪几何、状态锚点和同步视角组合为 episode IR。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
| --- | --- | --- | --- |
| `output_dir` | str | `output/episodes` | Episode artifact directory. |

## 📊 Usage 使用示例

```yaml
process:
  - robot_interaction_geometry_mapper:
      output_dir: ./output/episodes
```

## Input and Output 输入与输出

Each sample includes `source` and `episode_id`.

**Input:** DROID: `object_outputs`, `gripper_output`, `primary_view`, instruction, phase anchor and destination IR.

**Output:** Writes `ir/episode_ir.json`, combining synchronized views, contact geometry and optional tracked depth.

## 🔗 Related Links 相关链接

- [Source code 源代码](../../../data_juicer/ops/mapper/robot_interaction_geometry_mapper.py)
- [DROID recipe and input preparation](https://github.com/datajuicer/data-juicer-hub/tree/main/demo/embodied_vqa)
- [Return operator list 返回算子列表](../../Operators.md)
