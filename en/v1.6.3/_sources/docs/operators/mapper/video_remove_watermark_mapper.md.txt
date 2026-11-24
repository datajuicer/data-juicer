# video_remove_watermark_mapper

Remove watermarks from videos based on specified regions.

This operator removes watermarks from video frames by detecting and masking the watermark areas. It supports two detection methods: 'pixel_value' and 'pixel_diversity'. The regions of interest (ROIs) for watermark detection can be specified as either pixel coordinates or ratios of the frame dimensions. The operator extracts a set number of frames uniformly from the video to detect watermark pixels. A pixel is considered part of a watermark if it meets the detection criteria in a minimum number of frames. The cleaned video is saved in the specified directory or the same directory as the input file if no save directory is provided.

根据指定区域去除视频中的水印。

该算子通过检测和遮罩水印区域来去除视频帧中的水印。它支持两种检测方法：'pixel_value' 和 'pixel_diversity'。可以通过像素坐标或帧尺寸的比例来指定感兴趣区域 (ROIs) 以进行水印检测。该算子从视频中均匀提取一定数量的帧以检测水印像素。如果某个像素在最少数量的帧中满足检测标准，则认为它是水印的一部分。清理后的视频保存在指定的目录中，或者如果未提供保存目录，则保存在与输入文件相同的目录中。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `roi_strings` | typing.List[str] | `['0,0,0.1,0.1']` | a given list of regions the watermarks locate. The format of each can be "x1, y1, x2, y2", "(x1, y1, x2, y2)", or "[x1, y1, x2, y2]". |
| `roi_type` | <class 'str'> | `'ratio'` | the roi string type. When the type is 'pixel', (x1, y1), (x2, y2) are the locations of pixels in the top left corner and the bottom right corner respectively. If the roi_type is 'ratio', the coordinates are normalized by widths and heights. |
| `roi_key` | typing.Optional[str] | `None` | the key name of fields in samples to store roi_strings for each sample. It's used for set different rois for different samples. If it's none, use rois in parameter "roi_strings". It's None in default. |
| `frame_num` | typing.Annotated[int, Gt(gt=0)] | `10` | the number of frames to be extracted uniformly from the video to detect the pixels of watermark. |
| `min_frame_threshold` | typing.Annotated[int, Gt(gt=0)] | `7` | a coordination is considered as the location of a watermark pixel when it is that in no less min_frame_threshold frames. |
| `detection_method` | <class 'str'> | `'pixel_value'` | the method to detect the pixels of watermark. If it is 'pixel_value', we consider the distribution of pixel value in each frame. If it is 'pixel_diversity', we will consider the pixel diversity in different frames. The min_frame_threshold is useless and frame_num must be greater than 1 in 'pixel_diversity' mode. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_remove_watermark_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_remove_watermark_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)