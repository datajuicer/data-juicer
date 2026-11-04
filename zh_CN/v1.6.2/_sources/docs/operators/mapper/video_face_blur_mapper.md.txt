# video_face_blur_mapper

Mapper to blur faces detected in videos.

This operator uses an OpenCV classifier for face detection and applies a specified blur type to the detected faces. The default classifier is 'haarcascade_frontalface_alt.xml'. Supported blur types include 'mean', 'box', and 'gaussian'. The radius of the blur kernel can be adjusted. If a save directory is not provided, the processed videos will be saved in the same directory as the input files. The `DJ_PRODUCED_DATA_DIR` environment variable can also be used to specify the save directory.

用于模糊检测到的视频中人脸的映射器。

该算子使用 OpenCV 分类器进行人脸检测，并对检测到的人脸应用指定的模糊类型。默认分类器是 'haarcascade_frontalface_alt.xml'。支持的模糊类型包括 'mean'、'box' 和 'gaussian'。可以调整模糊内核的半径。如果未提供保存目录，则处理后的视频将保存在与输入文件相同的目录中。也可以使用 `DJ_PRODUCED_DATA_DIR` 环境变量来指定保存目录。

Type 算子类型: **mapper**

Tags 标签: cpu, video

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `cv_classifier` | <class 'str'> | `''` | OpenCV classifier path for face detection. By default, we will use 'haarcascade_frontalface_alt.xml'. |
| `blur_type` | <class 'str'> | `'gaussian'` | Type of blur kernel, including ['mean', 'box', 'gaussian']. |
| `radius` | <class 'float'> | `2` | Radius of blur kernel. |
| `save_dir` | <class 'str'> | `None` | The directory where generated video files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_face_blur_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_face_blur_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)