# audio_ffmpeg_wrapped_mapper

Wraps FFmpeg audio filters for processing audio files in a dataset.

This operator applies specified FFmpeg audio filters to the audio files in the dataset. It supports passing custom filter parameters and global arguments to the FFmpeg command line. The processed audio files are saved to a specified directory or the same directory as the input files if no save directory is provided. The `DJ_PRODUCED_DATA_DIR` environment variable can also be used to set the save directory. If no filter name is provided, the audio files remain unmodified. The operator updates the source file paths in the dataset after processing.

包装FFmpeg音频滤镜以处理数据集中的音频文件。

该算子应用指定的FFmpeg音频滤镜到数据集中的音频文件。它支持传递自定义滤镜参数和全局参数到FFmpeg命令行。处理后的音频文件将保存到指定目录，或者如果没有提供保存目录，则保存到与输入文件相同的目录中。还可以使用`DJ_PRODUCED_DATA_DIR`环境变量设置保存目录。如果没有提供滤镜名称，音频文件将保持不变。算子在处理后更新数据集中的源文件路径。

Type 算子类型: **mapper**

Tags 标签: cpu, audio

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `filter_name` | typing.Optional[str] | `None` | ffmpeg audio filter name. |
| `filter_kwargs` | typing.Optional[typing.Dict] | `None` | keyword-arguments passed to ffmpeg filter. |
| `global_args` | typing.Optional[typing.List[str]] | `None` | list-arguments passed to ffmpeg command-line. |
| `capture_stderr` | <class 'bool'> | `True` | whether to capture stderr. |
| `overwrite_output` | <class 'bool'> | `True` | whether to overwrite output file. |
| `save_dir` | <class 'str'> | `None` | The directory where generated audio files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_resize
```python
AudioFFmpegWrappedMapper(filter_kwargs={'end': 6}, capture_stderr=False)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 3 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio2.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../{PROJECT_ROOT}/tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../{PROJECT_ROOT}/tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../{PROJECT_ROOT}/tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> text</div><pre style="padding:6px; background:#f6f8fa; border-radius:4px; overflow-x:auto; white-space:pre; word-wrap:normal;">[[5.501678004535147, 6.0, 6.0]]</pre></div>

#### ✨ explanation 解释
This example demonstrates the use of the AudioFFmpegWrappedMapper operator to trim audio files. The 'atrim' filter is applied with an 'end' parameter set to 6, which means that all audio files will be trimmed to a maximum duration of 6 seconds. In this case, the first audio file, which is already shorter than 6 seconds, remains unchanged. The other two audio files are trimmed to 6 seconds. The output data shows the durations of the processed audio files, but it's important to note that these durations are calculated after the processing and do not represent the actual output files themselves, which would be the trimmed audio files.
该示例展示了如何使用AudioFFmpegWrappedMapper算子来裁剪音频文件。这里应用了'atrim'滤镜，并将'end'参数设置为6，这意味着所有音频文件都将被裁剪到最多6秒的长度。在这种情况下，第一个音频文件已经短于6秒，因此保持不变。另外两个音频文件则被裁剪到6秒。输出数据显示的是处理后的音频文件的时长，但需要注意的是，这些时长是在处理后计算出来的，并不代表算子的实际输出文件本身，实际的输出文件应该是被裁剪后的音频文件。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/audio_ffmpeg_wrapped_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_audio_ffmpeg_wrapped_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)