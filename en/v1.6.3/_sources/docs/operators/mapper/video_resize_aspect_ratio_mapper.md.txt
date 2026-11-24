# video_resize_aspect_ratio_mapper

Resizes videos to fit within a specified aspect ratio range. This operator adjusts the dimensions of videos to ensure their aspect ratios fall within a defined range. It can either increase or decrease the video dimensions based on the specified strategy. The aspect ratio is calculated as width divided by height. If a video's aspect ratio is outside the given range, it will be resized to match the closest boundary (either the minimum or maximum ratio). The `min_ratio` and `max_ratio` should be provided as strings in the format "9:21" or "9/21". The resizing process uses the `ffmpeg` library to handle the actual video scaling. Videos that do not need resizing are left unchanged. The operator supports saving the modified videos to a specified directory or the same directory as the input files.

调整视频尺寸以适应指定的宽高比范围。此算子调整视频的尺寸，以确保其宽高比在定义的范围内。根据指定的策略，它可以增加或减少视频的尺寸。宽高比计算为宽度除以高度。如果视频的宽高比超出给定范围，它将被调整到最接近的边界（最小或最大比率）。`min_ratio` 和 `max_ratio` 应以 "9:21" 或 "9/21" 格式的字符串提供。调整过程使用 `ffmpeg` 库来处理实际的视频缩放。不需要调整尺寸的视频保持不变。该算子支持将修改后的视频保存到指定目录或与输入文件相同的目录。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_ratio` | <class 'str'> | `'9/21'` | The minimum aspect ratio to enforce videos with an aspect ratio below `min_ratio` will be resized to match this minimum ratio. The ratio should be provided as a string in the format "9:21" or "9/21". |
| `max_ratio` | <class 'str'> | `'21/9'` | The maximum aspect ratio to enforce videos with an aspect ratio above `max_ratio` will be resized to match this maximum ratio. The ratio should be provided as a string in the format "21:9" or "21/9". |
| `strategy` | <class 'str'> | `'increase'` | The resizing strategy to apply when adjusting the video dimensions. It can be either 'decrease' to reduce the dimension or 'increase' to enlarge it. Accepted values are ['decrease', 'increase']. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_default_params
```python
VideoResizeAspectRatioMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[(640, 360)], [(480, 640)], [(362, 640)]]</pre></div>

#### ✨ explanation 解释
This example demonstrates the default behavior of the operator, where no specific aspect ratio range is set. As a result, all videos remain unchanged because there are no constraints to modify their dimensions. The output data shows the (width, height) of each video, which is the same as the input data. For clarity, we show the (width, height) of each video in the raw output; the actual raw output from the operator is the original videos without any changes.
这个示例展示了算子的默认行为，即没有设置特定的宽高比范围。因此，所有视频保持不变，因为没有任何约束来修改它们的尺寸。输出数据显示了每个视频的（宽度，高度），这与输入数据相同。为了清晰起见，我们在原始输出中显示了每个视频的（宽度，高度）；算子的实际原始输出是没有做任何修改的原始视频。

### test_min_ratio_increase
```python
VideoResizeAspectRatioMapper(min_ratio='3/4', strategy='increase')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[(640, 360)], [(480, 640)], [(480, 640)]]</pre></div>

#### ✨ explanation 解释
This example sets a minimum aspect ratio of 3/4 and uses the 'increase' strategy. If a video's aspect ratio is below 3/4, it will be resized to match this minimum ratio. In this case, only the third video (with an initial aspect ratio of 181:320) is resized to 480x640 to meet the minimum ratio requirement. The other two videos remain unchanged. The output data shows the (width, height) of each video after processing. For clarity, we show the (width, height) of each video in the raw output; the actual raw output from the operator is the resized videos.
这个示例设置了最小宽高比为3/4，并使用了“增加”策略。如果一个视频的宽高比低于3/4，它将被调整以匹配这个最小比例。在这种情况下，只有第三个视频（初始宽高比为181:320）被调整为480x640以满足最小比例要求。其他两个视频保持不变。输出数据显示了处理后的每个视频的（宽度，高度）。为了清晰起见，我们在原始输出中显示了每个视频的（宽度，高度）；算子的实际原始输出是调整后的视频。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_resize_aspect_ratio_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_resize_aspect_ratio_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)