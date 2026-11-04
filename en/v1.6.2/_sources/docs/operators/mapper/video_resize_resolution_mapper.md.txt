# video_resize_resolution_mapper

Resizes video resolution based on specified width and height constraints.

This operator resizes videos to fit within the provided minimum and maximum width and height limits. It can optionally maintain the original aspect ratio by adjusting the dimensions accordingly. The resized videos are saved in the specified directory or the same directory as the input if no save directory is provided. The key metric for resizing is the video's width and height, which are adjusted to meet the constraints while maintaining the aspect ratio if configured. The `force_divisible_by` parameter ensures that the output dimensions are divisible by a specified integer, which must be a positive even number when used with aspect ratio adjustments.

根据指定的宽度和高度约束调整视频分辨率。

此算子调整视频尺寸以适应提供的最小和最大宽度和高度限制。它可以选择性地通过相应调整尺寸来保持原始宽高比。调整后的视频保存在指定目录中，如果没有提供保存目录，则保存在与输入文件相同的目录中。调整的关键指标是视频的宽度和高度，它们会根据约束进行调整，并在配置时保持宽高比。`force_divisible_by` 参数确保输出尺寸可以被指定的整数整除，当与宽高比调整一起使用时，该整数必须是正偶数。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_width` | <class 'int'> | `1` | Videos with width less than 'min_width' will be mapped to videos with equal or bigger width. |
| `max_width` | <class 'int'> | `9223372036854775807` | Videos with width more than 'max_width' will be mapped to videos with equal of smaller width. |
| `min_height` | <class 'int'> | `1` | Videos with height less than 'min_height' will be mapped to videos with equal or bigger height. |
| `max_height` | <class 'int'> | `9223372036854775807` | Videos with height more than 'max_height' will be mapped to videos with equal or smaller height. |
| `force_original_aspect_ratio` | <class 'str'> | `'disable'` | Enable decreasing or             increasing output video width or height if necessary             to keep the original aspect ratio, including ['disable',             'decrease', 'increase']. |
| `force_divisible_by` | typing.Annotated[int, Gt(gt=0)] | `2` | Ensures that both the output dimensions,             width and height, are divisible by the given integer when used             together with force_original_aspect_ratio, must be a positive             even number. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_default_mapper
```python
VideoResizeResolutionMapper()
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video2.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video3.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[(640, 360)], [(480, 640)], [(362, 640)]]</pre></div>

#### ✨ explanation 解释
This example shows the operator's behavior when no specific width or height constraints are set. The original videos are not resized, and their dimensions remain the same. The output data here is a list of (width, height) tuples for each video, which is derived from the processed videos to help understand the result.
此示例展示了当未设置特定的宽度或高度约束时，算子的行为。原始视频不会被调整大小，其尺寸保持不变。这里的输出数据是每个视频的（宽度，高度）元组列表，这是从处理后的视频中提取出来的，以帮助理解结果。

### test_keep_aspect_ratio_decrease_mapper
```python
VideoResizeResolutionMapper(min_width=400, max_width=480, min_height=480, max_height=480, force_original_aspect_ratio='decrease')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 video</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4:</div><div class="video-grid"><video src="../../../tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> list</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[(480, 270)]]</pre></div>

#### ✨ explanation 解释
In this case, the operator resizes the videos while maintaining the original aspect ratio and ensuring that both the width and height fall within the specified limits. If the original dimensions exceed the maximum, the video is scaled down. The output data here is a list of (width, height) tuples for each video, which is derived from the processed videos to help understand the result.
在这种情况下，算子在保持原始纵横比的同时调整视频大小，并确保宽度和高度都在指定范围内。如果原始尺寸超过最大值，则视频会被缩小。这里的输出数据是每个视频的（宽度，高度）元组列表，这是从处理后的视频中提取出来的，以帮助理解结果。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_resize_resolution_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_resize_resolution_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)