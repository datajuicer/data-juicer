# video_motion_score_filter

Filter to keep samples with video motion scores from OpenCV within a specific range.

The operator uses Farneback's algorithm from OpenCV to compute dense optical flow. It calculates the average motion score for each video and retains samples based on the specified minimum and maximum score thresholds. The 'any' or 'all' strategy determines whether to keep a sample if any or all videos meet the criteria. The motion score is computed as the mean magnitude of the optical flow, which can be normalized relative to the frame's diagonal length. The stats are cached under the key 'video_motion_score'.

用于保留 OpenCV 视频运动分数在指定范围内的样本的过滤器。

该算子使用 OpenCV 中的 Farneback 算法计算稠密光流。它为每个视频计算平均运动分数，并根据指定的最小和最大分数阈值保留样本。'any' 或 'all' 策略决定只要任意视频或所有视频满足条件时是否保留样本。运动分数被计算为光流幅值的均值，该值可相对于帧对角线长度进行归一化。统计信息缓存在键 'video_motion_score' 下。

Type 算子类型: **filter**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_score` | <class 'float'> | `0.25` | The minimum motion score to keep samples. |
| `max_score` | <class 'float'> | `1.7976931348623157e+308` | The maximum motion score to keep samples. |
| `frame_field` | typing.Optional[str] | `None` | the field name of video frames to compute motion score. If frame_field is None, extract frames from the video field. |
| `sampling_fps` | typing.Annotated[float, Gt(gt=0)] | `2` | The sampling rate in frames_per_second for optical flow calculations. |
| `size` | typing.Union[typing.Annotated[int, Gt(gt=0)], typing.Tuple[typing.Annotated[int, Gt(gt=0)]], typing.Tuple[typing.Annotated[int, Gt(gt=0)], typing.Annotated[int, Gt(gt=0)]], NoneType] | `None` | Resize frames before computing optical flow. If size is a sequence like (h, w), frame size will be matched to this. If size is an int, smaller edge of frames will be matched to this number. i.e, if height > width, then frame will be rescaled to (size * height / width, size). Default `None` to keep the original size. |
| `max_size` | typing.Optional[typing.Annotated[int, Gt(gt=0)]] | `None` | The maximum allowed for the longer edge of resized frames. If the longer edge of frames is greater than max_size after being resized according to size, size will be overruled so that the longer edge is equal to max_size. As a result, the smaller edge may be shorter than size. This is only supported if size is an int. |
| `divisible` | typing.Annotated[int, Gt(gt=0)] | `1` | The number that the dimensions must be divisible by. |
| `relative` | <class 'bool'> | `False` | If `True`, the optical flow magnitude is normalized to a [0, 1] range, relative to the frame's diagonal length. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all videos. 'any': keep this sample if any videos meet the condition. 'all': keep this sample only if all videos meet the condition. |
| `if_output_optical_flow` | <class 'bool'> | `False` | Determines whether to output the computed optical flows into the metas. The optical flows for each video will be stored in the shape of (num_frame, H, W, 2) |
| `optical_flow_key` | <class 'str'> | `'video_optical_flow'` | The field name to store the optical flows. It's "video_optical_flow" in default. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_middle
```python
VideoMotionScoreFilter(min_score=1.5, max_score=3.0)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### ✨ explanation 解释
The operator filters the videos to keep only those with a motion score between 1.5 and 3.0. In this case, video1 is kept because its motion score falls within the specified range, while the other videos are removed for not meeting the criteria.
算子过滤视频，仅保留运动分数在1.5到3.0之间的视频。在这种情况下，video1被保留，因为它的运动分数落在指定范围内，而其他视频因不符合条件而被移除。

### test_any
```python
VideoMotionScoreFilter(min_score=1.5, max_score=3.0, any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### ✨ explanation 解释
The operator is set to 'any' mode, meaning it keeps a sample if any of the videos in the sample meet the motion score criteria (between 1.5 and 3.0). The first and third samples contain at least one video that meets the criteria, so they are kept. The second sample does not have any video meeting the criteria, hence it is removed.
算子设置为'any'模式，这意味着如果样本中的任何一个视频满足运动分数标准（在1.5和3.0之间），则保留该样本。第一个和第三个样本中至少有一个视频满足条件，因此它们被保留。第二个样本没有任何视频满足条件，因此被移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_motion_score_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_motion_score_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)