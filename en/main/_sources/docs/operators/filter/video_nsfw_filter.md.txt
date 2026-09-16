# video_nsfw_filter

Filter to keep samples whose videos have nsfw scores in a specified range.

This operator uses a Hugging Face model to detect NSFW content in video frames. It keeps samples where the NSFW score is below a specified threshold. The operator supports two frame sampling methods: "all_keyframes" and "uniform". For "uniform", it extracts a specified number of frames. The NSFW scores are reduced using one of three modes: "avg", "max", or "min". The key metric, 'video_nsfw_score', is computed for each video and stored in the sample's stats. The operator can use either an "any" or "all" strategy to decide if a sample should be kept based on the NSFW scores of its videos.

用于保留视频 NSFW 得分在指定范围内的样本的过滤器。

该算子使用 Hugging Face 模型来检测视频帧中的 NSFW 内容。它保留 NSFW 得分低于指定阈值的样本。该算子支持两种帧采样方法："all_keyframes" 和 "uniform"。对于 "uniform"，它提取指定数量的帧。NSFW 得分通过三种模式之一进行归约："avg"、"max" 或 "min"。关键指标 'video_nsfw_score' 对每个视频进行计算并存储在样本的 stats 中。该算子可以使用 "any" 或 "all" 策略来决定是否基于其视频的 NSFW 得分保留样本。

Type 算子类型: **filter**

Tags 标签: gpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_nsfw_model` | <class 'str'> | `'Falconsai/nsfw_image_detection'` | nsfw detection model name on huggingface. |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `min_score` | <class 'float'> | `0.0` | the nsfw score threshold for samples. range from 0 to 1. Samples with nsfw score greater than this threshold will be kept. |
| `max_score` | <class 'float'> | `0.5` | the nsfw score threshold for samples. range from 0 to 1. Samples with nsfw score less than this threshold will be kept. |
| `frame_field` | typing.Optional[str] | `None` | the field name of video frames to calculate the nsfw score. If frame_field is None, extract frames from the video field. |
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame images from the videos. Should be one of ["all_keyframes", "uniform"]. The former one extracts all key frames (the number of which depends on the duration of the video) and the latter one extract specified number of frames uniformly from the video. Default: "all_keyframes". |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from the video. Only works when frame_sampling_method is "uniform" or "frame_field" is given. If it's 1, only the middle frame will be extracted. If it's 2, only the first and the last frames will be extracted. If it's larger than 2, in addition to the first and the last frames, other frames will be extracted uniformly within the video duration. |
| `reduce_mode` | <class 'str'> | `'avg'` | reduce mode for multiple sampled video frames. 'avg': Take the average of multiple values 'max': Take the max of multiple values 'min': Take the min of multiple values |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all videos. 'any': keep this sample if any videos meet the condition. 'all': keep this sample only if all videos meet the condition. |
| `video_backend` | <class 'str'> | `'av'` | video backend, can be `ffmpeg`, `av`. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_all_keyframes
```python
VideoNSFWFilter(hf_nsfw_model='Falconsai/nsfw_image_detection', max_score=0.1, frame_sampling_method='all_keyframes')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### ✨ explanation 解释
This operator filters out videos with a high NSFW score, keeping only those below the threshold. It uses all keyframes for scoring and averages the scores. The video1 is removed because its average NSFW score is above 0.1.
该算子过滤掉NSFW分数高的视频，只保留低于阈值的视频。它使用所有关键帧进行评分，并计算平均分。video1被移除是因为其平均NSFW分数高于0.1。

### test_any
```python
VideoNSFWFilter(hf_nsfw_model='Falconsai/nsfw_image_detection', max_score=0.01, frame_sampling_method='all_keyframes', any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### ✨ explanation 解释
This operator checks if any of the videos in a sample have an NSFW score below the threshold. It keeps the sample if at least one video meets the criteria. The second sample is removed because both videos in it have NSFW scores above 0.01.
该算子检查样本中的任何一个视频是否具有低于阈值的NSFW分数。只要有一个视频符合条件，就保留该样本。第二个样本被移除是因为其中两个视频的NSFW分数都高于0.01。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_nsfw_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_nsfw_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)