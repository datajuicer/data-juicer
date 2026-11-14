# robot_object_tracking_mapper

Localize the instructed object with VLM/SAM3 and track the full episode.

使用 VLM 和 SAM3 定位指令中的目标物体，并跟踪完整 episode。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
| --- | --- | --- | --- |
| `output_dir` | str | `output/episodes` | Episode artifact directory. |
| `model` | str | None | VLM model name; defaults to `VLM_MODEL`. |
| `endpoint` | str | None | OpenAI-compatible endpoint; defaults to `VLM_URL`. |
| `sam3_model` | str | `facebook/sam3` | SAM3 checkpoint or model ID; configurable with `SAM3_MODEL`. |

## 📊 Usage 使用示例

```yaml
process:
  - robot_object_tracking_mapper:
      output_dir: ./output/episodes
```

## Input and Output 输入与输出

Each sample includes `source` and `episode_id`.

**Input:** DROID: `raw_dir`, `metadata_path`, `primary_view`; run after the gripper anchor mapper.

**Output:** Adds `object_outputs`; decodes every camera frame, writes `ir/sampling.json`, and tracks the instructed object.

## 🔗 Related Links 相关链接

- [Source code 源代码](../../../data_juicer/ops/mapper/robot_object_tracking_mapper.py)
- [DROID recipe and input preparation](https://github.com/datajuicer/data-juicer-hub/tree/main/demo/embodied_vqa)
- [Return operator list 返回算子列表](../../Operators.md)
