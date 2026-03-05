# video_ffmpeg_wrapped_mapper

Wraps FFmpeg video filters for processing video files in a dataset.

This operator applies a specified FFmpeg video filter to each video file in the dataset. It supports passing keyword arguments to the filter and global arguments to the FFmpeg command line. The processed videos are saved in a specified directory or the same directory as the input files. If no filter name is provided, the videos remain unmodified. The operator updates the source file paths in the dataset to reflect any changes.

封装 FFmpeg 视频滤镜以处理数据集中的视频文件。

该算子对数据集中的每个视频文件应用指定的 FFmpeg 视频滤镜。它支持向滤镜传递关键字参数以及向 FFmpeg 命令行传递全局参数。处理后的视频保存在指定的目录中，或者与输入文件相同的目录中。如果没有提供滤镜名称，视频将保持不变。该算子会更新数据集中源文件路径以反映任何更改。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `filter_name` | typing.Optional[str] | `None` | ffmpeg video filter name. |
| `filter_kwargs` | typing.Optional[typing.Dict] | `None` | keyword-arguments passed to ffmpeg filter. |
| `global_args` | typing.Optional[typing.List[str]] | `None` | list-arguments passed to ffmpeg command-line. |
| `capture_stderr` | <class 'bool'> | `True` | whether to capture stderr. |
| `overwrite_output` | <class 'bool'> | `True` | whether to overwrite output file. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_resize
```python
VideoFFmpegWrappedMapper(filter_kwargs={'width': 400, 'height': 480}, capture_stderr=False)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 3 videos</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">video1.mp4 +2 more:</div><div class="video-grid"><video src="../../../{PROJECT_ROOT}/tests/ops/data/video1.mp4" controls width="320" style="margin:4px;"></video></div><details style='margin:6px 0;'><summary style='cursor:pointer; color:#0366d6;'>Show 2 more videos 展开更多视频</summary><div class="video-grid"><video src="../../../{PROJECT_ROOT}/tests/ops/data/video2.mp4" controls width="320" style="margin:4px;"></video><video src="../../../{PROJECT_ROOT}/tests/ops/data/video3.mp4" controls width="320" style="margin:4px;"></video></div></details></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[[400, 480], [400, 480], [400, 480]]]</pre></div>

#### ✨ explanation 解释
This example demonstrates how the VideoFFmpegWrappedMapper operator resizes videos to a uniform size of 400x480 pixels. The input consists of three videos with different resolutions, and the output shows that all videos have been resized to 400x480. For clarity, we show the (width, height) of each video in the raw output; the actual raw output from the operator is the resized video files.
这个例子展示了VideoFFmpegWrappedMapper算子如何将视频统一调整为400x480像素的大小。输入包括三个不同分辨率的视频，输出显示所有视频都被调整为400x480。为了清晰起见，我们在原始输出中展示了每个视频的（宽度，高度）；实际上，算子的原始输出是调整大小后的视频文件。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_ffmpeg_wrapped_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_ffmpeg_wrapped_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)