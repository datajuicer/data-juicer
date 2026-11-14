# robot_gripper_tracking_mapper

Segment the robot gripper and propagate its mask across all frames.

分割机器人夹爪，并在完整视频中传播其掩码。

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
  - robot_gripper_tracking_mapper:
      output_dir: ./output/episodes
```

## Input and Output 输入与输出

Each sample includes `source` and `episode_id`.

**Input:** DROID: `metadata_path`; run after object tracking, which prepares the full primary-view video.

**Output:** Adds `gripper_output`, pointing to gripper masks and tracks over the complete episode.

## 🔗 Related Links 相关链接

- [Source code 源代码](../../../data_juicer/ops/mapper/robot_gripper_tracking_mapper.py)
- [DROID recipe and input preparation](https://github.com/datajuicer/data-juicer-hub/tree/main/demo/embodied_vqa)
- [Return operator list 返回算子列表](../../Operators.md)
