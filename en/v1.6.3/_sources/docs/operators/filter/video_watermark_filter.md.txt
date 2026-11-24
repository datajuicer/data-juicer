# video_watermark_filter

Filter to keep samples whose videos have no watermark with high probability.

This operator uses a Hugging Face watermark detection model to predict the probability of watermarks in video frames. It keeps samples where the predicted watermark probability is below a specified threshold. The key metric, 'video_watermark_prob', is computed by extracting frames from the video using a specified sampling method and then averaging, maximizing, or minimizing the probabilities based on the reduce mode. If multiple videos are present, the operator can use either an 'any' or 'all' strategy to determine if the sample should be kept. The frame sampling method can be 'all_keyframes' or 'uniform', and the reduce mode can be 'avg', 'max', or 'min'.

筛选出高概率没有水印的视频样本。

该算子使用 Hugging Face 水印检测模型预测视频帧中水印的概率。它保留预测水印概率低于指定阈值的样本。关键指标 'video_watermark_prob' 通过使用指定的采样方法从视频中提取帧，然后根据归约模式对概率进行平均、最大化或最小化来计算。如果有多个视频存在，该算子可以使用 'any' 或 'all' 策略来确定是否保留样本。帧采样方法可以是 'all_keyframes' 或 'uniform'，归约模式可以是 'avg'、'max' 或 'min'。

Type 算子类型: **filter**

Tags 标签: gpu, hf, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_watermark_model` | <class 'str'> | `'amrul-hzz/watermark_detector'` | watermark detection model name on huggingface. |
| `trust_remote_code` | <class 'bool'> | `False` | whether to trust the remote code of HF models. |
| `prob_threshold` | <class 'float'> | `0.8` | the predicted watermark probability threshold for samples. range from 0 to 1. Samples with watermark probability less than this threshold will be kept. |
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame images from the videos. Should be one of ["all_keyframes", "uniform"]. The former one extracts all key frames (the number of which depends on the duration of the video) and the latter one extract specified number of frames uniformly from the video. Default: "all_keyframes". |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from the video. Only works when frame_sampling_method is "uniform". If it's 1, only the middle frame will be extracted. If it's 2, only the first and the last frames will be extracted. If it's larger than 2, in addition to the first and the last frames, other frames will be extracted uniformly within the video duration. |
| `reduce_mode` | <class 'str'> | `'avg'` | reduce mode for multiple sampled video frames. 'avg': Take the average of multiple values 'max': Take the max of multiple values 'min': Take the min of multiple values |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all videos. 'any': keep this sample if any videos meet the condition. 'all': keep this sample only if all videos meet the condition. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_all_keyframes
```python
VideoWatermarkFilter(hf_watermark_model='amrul-hzz/watermark_detector', prob_threshold=0.8, frame_sampling_method='all_keyframes')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### ✨ explanation 解释
This operator filters out samples where the probability of watermarks in video frames, as determined by a Hugging Face model, is above 0.8. It uses all keyframes for sampling. In this case, only the sample with video3 is kept because its watermark probability is below the threshold.
该算子过滤掉视频帧中水印概率高于0.8的样本，使用Hugging Face模型进行预测，并对所有关键帧进行采样。在这种情况下，只有video3的样本被保留，因为它的水印概率低于阈值。

### test_reduce_max
```python
VideoWatermarkFilter(hf_watermark_model='amrul-hzz/watermark_detector', prob_threshold=0.9, frame_sampling_method='all_keyframes', reduce_mode='max')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[]</pre></div>

#### ✨ explanation 解释
This operator filters out samples if the maximum watermark probability among all keyframes exceeds 0.9. Here, all samples are removed since at least one keyframe in each video has a watermark probability higher than 0.9, demonstrating how 'max' reduce mode works under stricter filtering conditions.
该算子如果所有关键帧中的最大水印概率超过0.9，则会过滤掉这些样本。这里，所有的样本都被移除，因为每个视频中至少有一个关键帧的水印概率高于0.9，这展示了在更严格的过滤条件下'max'归约模式的工作方式。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_watermark_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_watermark_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)