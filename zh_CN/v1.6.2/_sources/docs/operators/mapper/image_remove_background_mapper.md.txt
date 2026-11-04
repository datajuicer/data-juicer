# image_remove_background_mapper

Mapper to remove the background of images.

This operator processes each image in the sample, removing its background. It uses the `rembg` library to perform the background removal. If `alpha_matting` is enabled, it applies alpha matting with specified thresholds and erosion size. The resulting images are saved in PNG format. The `bgcolor` parameter can be set to specify a custom background color for the cutout image. The processed images are stored in the directory specified by `save_dir`, or in the same directory as the input files if `save_dir` is not provided. The `source_file` field in the sample is updated to reflect the new file paths.

将图像的背景移除。

此算子处理样本中的每张图像，移除其背景。它使用 `rembg` 库来执行背景移除。如果启用了 `alpha_matting`，则应用带有指定阈值和腐蚀大小的 alpha 修边。生成的图像以 PNG 格式保存。可以通过设置 `bgcolor` 参数来指定剪切图像的自定义背景色。处理后的图像存储在由 `save_dir` 指定的目录中，如果没有提供 `save_dir`，则存储在与输入文件相同的目录中。样本中的 `source_file` 字段会更新以反映新的文件路径。

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `alpha_matting` | <class 'bool'> | `False` | (bool, optional) Flag indicating whether to use alpha matting. Defaults to False. |
| `alpha_matting_foreground_threshold` | <class 'int'> | `240` | (int, optional) Foreground threshold for alpha matting. Defaults to 240. |
| `alpha_matting_background_threshold` | <class 'int'> | `10` | (int, optional) Background threshold for alpha matting. Defaults to 10. |
| `alpha_matting_erode_size` | <class 'int'> | `10` | (int, optional) Erosion size for alpha matting. Defaults to 10. |
| `bgcolor` | typing.Optional[typing.Tuple[int, int, int, int]] | `None` | (Optional[Tuple[int, int, int, int]], optional) Background color for the cutout image. Defaults to None. |
| `save_dir` | <class 'str'> | `None` | The directory where generated image files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files.     This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. *args (Optional[Any]): Additional positional arguments. **kwargs (Optional[Any]): Additional keyword arguments. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_remove_background_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_remove_background_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)