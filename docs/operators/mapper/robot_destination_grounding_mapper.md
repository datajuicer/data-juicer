# robot_destination_grounding_mapper

Select a visible final view and ground the demonstrated target location.

选择可见的末帧视角，并定位演示中的目标物体。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
| --- | --- | --- | --- |
| `output_dir` | str | `output/episodes` | Episode artifact directory. |
| `model` | str | None | VLM model name; defaults to `VLM_MODEL`. |
| `endpoint` | str | None | OpenAI-compatible endpoint; defaults to `VLM_URL`. |

## 📊 Usage 使用示例

```yaml
process:
  - robot_destination_grounding_mapper:
      output_dir: ./output/episodes
```

## Input and Output 输入与输出

Each sample includes `source` and `episode_id`.

**Input:** DROID: `object_outputs` and decoded camera views from object tracking.

**Output:** Writes `ir/destination.json` with a VLM-selected view and final target point.

## 🔗 Related Links 相关链接

- [Source code 源代码](../../../data_juicer/ops/mapper/robot_destination_grounding_mapper.py)
- [DROID recipe and input preparation](https://github.com/datajuicer/data-juicer-hub/tree/main/demo/embodied_vqa)
- [Return operator list 返回算子列表](../../Operators.md)
