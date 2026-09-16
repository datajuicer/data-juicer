# video_extract_frames_mapper

Mapper to extract frames from video files according to specified methods.

Extracts frames from video files using either all keyframes or a uniform sampling method.

Supported output formats are: ["path", "bytes"]. If format is "path", the output is a list of lists, where each inner list contains the path of the frames of a single video. e.g.[ [video1_frame1_path, video1_frame2_path, ...], [video2_frame1_path, video2_frame2_path, ...], ... ] (In the order of the videos). If format is "bytes", the output is a list of lists, where each inner list contains the bytes of the frames of a single video. e.g. [ [video1_byte1, video1_byte2, ...], [video2_byte1, video2_byte2, ...], ... ] (In the order of the videos).

- **Frame Sampling Methods**:
- "all_keyframes": Extracts all keyframes from the video.
- "uniform": Extracts a specified number of frames uniformly from the video.
- If `duration` is set, the video is segmented into multiple segments based on the duration, and frames are extracted from each segment.
- The output directory for the frames can be specified if output format is "path", else left to None.
- The field name in the sample's metadata where the frame information is stored can be customized.

用于根据指定方法从视频文件中提取帧的映射器。

使用全部关键帧或均匀采样方法从视频文件中提取帧。

支持的输出格式为：["path", "bytes"]。若格式为 "path"，输出为列表的列表，其中每个内层列表包含单个视频各帧的路径，例如：[ [video1_frame1_path, video1_frame2_path, ...], [video2_frame1_path, video2_frame2_path, ...], ... ]（按视频顺序排列）。若格式为 "bytes"，输出为列表的列表，其中每个内层列表包含单个视频各帧的字节数据，例如：[ [video1_byte1, video1_byte2, ...], [video2_byte1, video2_byte2, ...], ... ]（按视频顺序排列）。

- **帧采样方法**：
- "all_keyframes"：提取视频中的所有关键帧。
- "uniform"：从视频中均匀提取指定数量的帧。
- 若设置了 `duration`，视频将被分割为多个基于时长的片段，并从每个片段中提取帧。
- 若输出格式为 "path"，可指定帧的输出目录；否则设为 None。
- 可自定义样本元数据中存储帧信息的字段名称。

Type 算子类型: **mapper**

Tags 标签: cpu, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame videos from the videos. Should be one of ["all_keyframes", "uniform"]. The former one extracts all key frames (the number of which depends on the duration of the video) and the latter one extract specified number of frames uniformly from the video. If "duration" > 0, frame_sampling_method acts on every segment. Default: "all_keyframes". |
| `output_format` | <class 'str'> | `'path'` | The output format of the frame videos. Supported formats are: ["path", "bytes"]. If format is "path", the output is a list of lists, where each inner list contains the path of the frames of a single video. e.g.[         [video1_frame1_path, video1_frame2_path, ...],         [video2_frame1_path, video2_frame2_path, ...],         ...     ] (In the order of the videos). If format is "bytes", the output is a list of lists, where each inner list contains the bytes of the frames of a single video. e.g. [         [video1_byte1, video1_byte2, ...],         [video2_byte1, video2_byte2, ...],         ...     ] (In the order of the videos). |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from the video. Only works when frame_sampling_method is "uniform". If it's 1, only the middle frame will be extracted. If it's 2, only the first and the last frames will be extracted. If it's larger than 2, in addition to the first and the last frames, other frames will be extracted uniformly within the video duration. If "duration" > 0, frame_num is the number of frames per segment. |
| `duration` | <class 'float'> | `0` | The duration of each segment in seconds. If 0, frames are extracted from the entire video. If duration > 0, the video is segmented into multiple segments based on duration, and frames are extracted from each segment. |
| `frame_dir` | <class 'str'> | `None` | Output directory to save extracted frames. If output_format is "path", must specify a directory. |
| `frame_key` | <class 'str'> | `None` | The name of field to save generated frames info. |
| `frame_field` | <class 'str'> | `'video_frames'` | The name of field to save generated frames info. |
| `legacy_split_by_text_token` | <class 'bool'> | `True` | Whether to split by special tokens (e.g. <__dj__video>) in the text field and read videos in order, or use the 'videos' or 'frames' field directly. |
| `video_backend` | <class 'str'> | `'av'` | video backend, can be `ffmpeg`, `av`. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_extract_frames_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_extract_frames_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)