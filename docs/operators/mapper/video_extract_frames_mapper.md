# video_extract_frames_mapper

Mapper to extract frames from video files according to specified methods.

Extracts frames from video files using either all keyframes or a uniform sampling method. The extracted frames are saved in a directory, and the mapping from video keys to frame directories is stored in the sample's metadata. The data format for the extracted frames is a dictionary mapping video keys to their respective frame directories:
- "video_key_1": "/${frame_dir}/video_key_1_filename/"
- "video_key_2": "/${frame_dir}/video_key_2_filename/"

- **Frame Sampling Methods**:
- "all_keyframes": Extracts all keyframes from the video.
- "uniform": Extracts a specified number of frames uniformly from the video.
- If `duration` is set, the video is segmented into multiple segments based on the duration, and frames are extracted from each segment.
- The output directory for the frames can be specified; otherwise, a default directory is used.
- The field name in the sample's metadata where the frame information is stored can be customized.

映射器根据指定方法从视频文件中提取帧。

使用所有关键帧或均匀采样方法从视频文件中提取帧。提取的帧保存在一个目录中，并将视频键到帧目录的映射存储在样本的元数据中。提取帧的数据格式是一个字典，将视频键映射到其相应的帧目录：
- "video_key_1": "/${frame_dir}/video_key_1_filename/"
- "video_key_2": "/${frame_dir}/video_key_2_filename/"

- **帧采样方法**：
- "all_keyframes"：从视频中提取所有关键帧。
- "uniform"：从视频中均匀提取指定数量的帧。
- 如果设置了 `duration`，则根据持续时间将视频分割成多个片段，并从每个片段中提取帧。
- 可以指定帧的输出目录；否则，使用默认目录。
- 可以自定义样本元数据中存储帧信息的字段名称。

Type 算子类型: **mapper**

Tags 标签: cpu, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `frame_sampling_method` | <class 'str'> | `'all_keyframes'` | sampling method of extracting frame videos from the videos. Should be one of ["all_keyframes", "uniform"]. The former one extracts all key frames (the number of which depends on the duration of the video) and the latter one extract specified number of frames uniformly from the video. If "duration" > 0, frame_sampling_method acts on every segment. Default: "all_keyframes". |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `3` | the number of frames to be extracted uniformly from the video. Only works when frame_sampling_method is "uniform". If it's 1, only the middle frame will be extracted. If it's 2, only the first and the last frames will be extracted. If it's larger than 2, in addition to the first and the last frames, other frames will be extracted uniformly within the video duration. If "duration" > 0, frame_num is the number of frames per segment. |
| `duration` | <class 'float'> | `0` | The duration of each segment in seconds. If 0, frames are extracted from the entire video. If duration > 0, the video is segmented into multiple segments based on duration, and frames are extracted from each segment. |
| `frame_dir` | <class 'str'> | `None` | Output directory to save extracted frames. If None, a default directory based on the video file path is used. |
| `frame_key` |  | `'video_frames'` | The name of field to save generated frames info. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_extract_frames_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_extract_frames_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)