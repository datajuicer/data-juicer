# video_undistort_mapper

Undistort raw videos with corresponding camera intrinsics and distortion coefficients.

使用对应的相机内参和畸变系数对原始视频进行去畸变处理。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `output_video_dir` | <class 'str'> | `DATA_JUICER_ASSETS_CACHE` | Output directory to save undistorted videos. |
| `tag_field_name` | <class 'str'> | `'video_undistortion_tags'` | The field name to store the tags. It's "video_undistortion_tags" in default. |
| `batch_size_each_video` | <class 'int'> | `1000` | Number of frames to process and save per temporary TS file batch. |
| `crf` | <class 'int'> | `22` | Constant Rate Factor (CRF) for FFmpeg encoding quality. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_undistort_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_undistort_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)