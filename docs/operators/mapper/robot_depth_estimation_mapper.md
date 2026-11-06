# robot_depth_estimation_mapper

Estimate every camera frame and persist aligned depth IR outside Arrow metadata.

对每个相机帧估计深度，并将对齐的深度数组保存为 episode IR。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置

| name 参数名 | type 类型 | default 默认值 | desc 说明 |
| --- | --- | --- | --- |
| `output_dir` | str | `output/episodes` | Episode artifact directory. |
| `model_path` | str | `depth-anything/DA3METRIC-LARGE` | Model ID or checkpoint; configurable with `DA3_MODEL`. |
| `inference_batch_size` | int | 8 | Frames per batch; zero processes an entire view jointly. |
| `process_res` | int | 504 | Inference resolution. |
| `process_res_method` | str | `upper_bound_resize` | Full-image resize method. |

## 📊 Usage 使用示例

```yaml
process:
  - robot_depth_estimation_mapper:
      output_dir: ./output/episodes
```

## Input and Output 输入与输出

Each sample includes `source` and `episode_id`.

**Input:** DROID: decoded camera views in `ir/sampling.json`; run after object tracking and before interaction geometry.

**Output:** Adds `depth_ir`; writes aligned depth arrays for every source frame. Without camera intrinsics, units are focal-normalized. Install the Depth Anything 3 SDK separately. Any-view checkpoints require `inference_batch_size: 0`.

## 🔗 Related Links 相关链接

- [Source code 源代码](../../../data_juicer/ops/mapper/robot_depth_estimation_mapper.py)
- [DROID recipe and input preparation](https://github.com/datajuicer/data-juicer-hub/tree/main/demo/embodied_vqa)
- [Return operator list 返回算子列表](../../Operators.md)
