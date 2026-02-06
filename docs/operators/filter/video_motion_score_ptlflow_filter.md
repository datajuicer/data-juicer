# video_motion_score_ptlflow_filter

Filter to keep samples with video motion scores from ptlflow within a specified range.

This operator utilizes the ptlflow library (https://github.com/hmorimitsu/ptlflow) to predict optical flow between video frames. It keeps samples where the video motion score is within the given min and max score range. The motion score is computed based on the optical flow between frames, which is estimated using the models supported in ptlflow. The operator can sample frames at a specified FPS and apply transformations to the frames before computing the flow.

- The models in ptlflow is used to estimate the optical flow.
- Frames are preprocessed using a series of transformations including normalization and color channel flipping.
- The motion score is calculated from the optical flow data.
- The operator can be configured to filter based on any or all frames in the video.
- The device for model inference (CPU or CUDA) is automatically detected and set.

For further details, refer to the official documentation: https://ptlflow.readthedocs.io/

用于保留 ptlflow 视频运动分数在指定范围内的样本的过滤器。

该算子利用 ptlflow 库（https://github.com/hmorimitsu/ptlflow）预测视频帧之间的光流。它保留视频运动分数在给定最小值和最大值范围内的样本。运动分数基于帧间光流计算，而光流由 ptlflow 支持的模型进行估计。该算子可以按指定 FPS 采样帧，并在计算光流前对帧应用变换。

- 使用 ptlflow 中的模型估计光流。
- 帧通过一系列变换进行预处理，包括归一化和颜色通道翻转。
- 运动分数从光流数据中计算得出。
- 可配置为基于视频中任意帧或所有帧进行过滤。
- 模型推理设备（CPU 或 CUDA）会自动检测并设置。

更多详情请参阅官方文档：https://ptlflow.readthedocs.io/

Type 算子类型: **filter**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_score` | <class 'float'> | `1.0` |  |
| `max_score` | <class 'float'> | `1.7976931348623157e+308` |  |
| `frame_field` | typing.Optional[str] | `None` |  |
| `model_name` | <class 'str'> | `'dpflow'` |  |
| `ckpt_path` | typing.Optional[str] | `'things'` |  |
| `get_model_args` | typing.Optional[dict] | `None` |  |
| `sampling_fps` | typing.Annotated[float, Gt(gt=0)] | `2` |  |
| `size` | typing.Union[typing.Annotated[int, Gt(gt=0)], typing.Tuple[typing.Annotated[int, Gt(gt=0)]], typing.Tuple[typing.Annotated[int, Gt(gt=0)], typing.Annotated[int, Gt(gt=0)]], NoneType] | `None` |  |
| `max_size` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` |  |
| `divisible` | typing.Annotated[int, Gt(gt=0)] | `8` |  |
| `relative` | <class 'bool'> | `False` |  |
| `any_or_all` | <class 'str'> | `'any'` |  |
| `if_output_optical_flow` | <class 'bool'> | `False` |  |
| `optical_flow_key` | <class 'str'> | `'video_optical_flow'` |  |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |

## 📊 Effect demonstration 效果演示
### test_default
```python
VideoMotionScorePtlflowFilter()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>


### test_different_model
```python
VideoMotionScorePtlflowFilter(model_name='raft')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>



## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_motion_score_ptlflow_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_motion_score_ptlflow_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)