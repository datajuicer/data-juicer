# video_face_ratio_filter

Keep data samples whose videos' face-to-frame ratios are within a specified range. This operator uses dlib's frontal face detector to scan video frames at a specified interval and computes the ratio of frames containing faces. Samples are filtered based on whether this ratio meets a configurable threshold.

保留视频中人脸出现比例在指定范围内的数据样本。此算子使用 dlib 的正脸检测器按指定间隔扫描视频帧，计算包含人脸的帧的比例，并根据该比例是否达到可配置的阈值来过滤样本。

Type 算子类型: **filter**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `threshold` | <class 'float'> | `0.8` | minimum face-to-frame ratio to keep the sample. |
| `detect_interval` | <class 'int'> | `1` | interval (in frames) at which face detection is performed. |
| `any_or_all` | <class 'str'> | `'all'` | keep this sample with 'any' or 'all' strategy of all videos. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/video_face_ratio_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_video_face_ratio_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)
