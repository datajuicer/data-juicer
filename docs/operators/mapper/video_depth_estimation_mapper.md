# video_depth_estimation_mapper

Perform depth estimation on the video.

对视频执行深度估计。

Type 算子类型: **mapper**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `video_depth_model_path` | <class 'str'> | `'video_depth_anything_vitb.pth'` | The path to the Video-Depth-Anything model. If the model is a 'metric' model, the code will automatically switch to metric mode, and the user should input the path for storing point clouds. |
| `point_cloud_dir_for_metric` | <class 'str'> | `DATA_JUICER_ASSETS_CACHE` | The path for storing point clouds (for a 'metric' model). |
| `max_res` | <class 'int'> | `1280` | The maximum resolution threshold for videos; videos exceeding this threshold will be resized. |
| `torch_dtype` | <class 'str'> | `'fp16'` | The floating point type used for model inference. Can be one of ['fp32', 'fp16'] |
| `if_save_visualization` | <class 'bool'> | `False` | Whether to save visualization results. |
| `save_visualization_dir` | <class 'str'> | `DATA_JUICER_ASSETS_CACHE` | The path for saving visualization results. |
| `grayscale` | <class 'bool'> | `False` | If True, the colorful palette will not be applied. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
### test
```python
VideoDepthEstimationMapper(video_depth_model_path='video_depth_anything_vits.pth', point_cloud_dir_for_metric=DATA_JUICER_ASSETS_CACHE, max_res=1280, torch_dtype='fp16', if_save_visualization=True, save_visualization_dir=DATA_JUICER_ASSETS_CACHE, grayscale=False)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video4.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video4.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>depth_data</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[673, 360, 480]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>fps</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>30.0</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>depth_data</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[1190, 640, 362]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>fps</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>24.0</td></tr></table></div></div>


### test_metric
```python
VideoDepthEstimationMapper(video_depth_model_path='metric_video_depth_anything_vits.pth', point_cloud_dir_for_metric=DATA_JUICER_ASSETS_CACHE, max_res=1280, torch_dtype='fp16', if_save_visualization=True, save_visualization_dir=DATA_JUICER_ASSETS_CACHE, grayscale=False)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video4.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video4.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>depth_data</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[673, 360, 480]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>fps</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>30.0</td></tr></table></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> empty</div><div class='meta' style='margin:6px 0;'><table class='meta-table' style='border-collapse:collapse; width:100%; border:1px solid #e3e3e3;'><tr><th colspan='2' style='text-align:left; vertical-align:top; padding:6px 8px; font-weight:600; border-bottom:1px solid #e3e3e3;'>depth_data</th></tr><tr><td colspan='2' style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:22px; border-bottom:1px solid #e3e3e3;'>[1190, 640, 362]</td></tr><tr><td style='text-align:left; vertical-align:top; padding:4px 8px; padding-left:8px; font-weight:500; color:#444; border-bottom:1px solid #e3e3e3; white-space:nowrap;'>fps</td><td style='text-align:left; vertical-align:top; padding:4px 6px; padding-left:4px; border-bottom:1px solid #e3e3e3;'>24.0</td></tr></table></div></div>



## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_depth_estimation_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_depth_estimation_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)