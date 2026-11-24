# video_ocr_area_ratio_filter

Keep data samples whose detected text area ratios for specified frames in the video are within a specified range.

This operator filters data based on the ratio of the detected text area to the total frame area. It uses EasyOCR to detect text in the specified languages and calculates the area ratio for each sampled frame. The operator then determines whether to keep a sample based on the `any` or `all` strategy, which checks if any or all of the videos meet the specified area ratio range. The key metric, `video_ocr_area_ratio`, is computed as the mean of the text area ratios across the sampled frames. The number of sampled frames and the specific frames to be sampled can be configured.

保留视频中指定帧的检测文本区域比率在指定范围内的数据样本。

该算子根据检测到的文本区域与总帧面积的比例来过滤数据。它使用EasyOCR来检测指定语言中的文本，并计算每个采样帧的面积比例。然后，算子根据`any`或`all`策略决定是否保留样本，该策略检查是否有任何或所有视频满足指定的面积比例范围。关键指标`video_ocr_area_ratio`是通过采样帧中文本面积比例的平均值计算得出的。可以配置采样帧的数量和要采样的特定帧。

Type 算子类型: **filter**

Tags 标签: gpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_area_ratio` | <class 'float'> | `0` | The min ocr area ratio to keep samples. It's 0 by default. |
| `max_area_ratio` | <class 'float'> | `1.0` | The max ocr area ratio to keep samples. It's 1.0 by default. |
| `frame_sample_num` | typing.Annotated[int, Gt(gt=0)] | `3` | The number of sampled frames to calculate the ocr area ratio. If it's 1, only middle frame will be selected. If it's 2, only the first and the last frames will be selected. If it's larger than 2, in addition to the first and the last frames, other frames will be sampled evenly within the video duration. |
| `languages_to_detect` | typing.Union[str, typing.List[str]] | `['ch_sim', 'en']` | texts in which languages should be detected. Default: ['ch_sim', 'en']. Full language list can be found here: https://www.jaided.ai/easyocr/. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all videos. 'any': keep this sample if any videos meet the condition. 'all': keep this sample only if all videos meet the condition. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_filter_videos_within_range
```python
VideoOcrAreaRatioFilter(min_area_ratio=0.07, max_area_ratio=0.1)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### ✨ explanation 解释
The operator filters videos based on the detected text area ratio, keeping only those with a ratio between 0.07 and 0.1. In this case, only video3 meets the criteria, hence it is the only one retained in the target list.
算子根据检测到的文字区域比率过滤视频，只保留比率在0.07到0.1之间的视频。此情况下，只有video3符合标准，因此目标列表中只保留了video3。

### test_any
```python
VideoOcrAreaRatioFilter(min_area_ratio=0.07, max_area_ratio=0.1, any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +1 more:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 1 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### ✨ explanation 解释
This test uses the 'any' strategy to filter samples that contain at least one video meeting the specified area ratio range (0.07 to 0.1). The first sample is removed because none of its videos meet the criteria, while the other two samples each have at least one video within the desired range, so they are kept.
此测试使用“任意”策略来过滤至少包含一个满足指定区域比率范围（0.07至0.1）的视频的样本。第一个样本被移除，因为其视频都不符合标准；而其他两个样本各自至少有一个视频处于期望范围内，因此它们被保留。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_ocr_area_ratio_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_ocr_area_ratio_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)