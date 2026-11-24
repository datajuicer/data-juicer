# image_blur_mapper

Blurs images in the dataset with a specified probability and blur type.

This operator blurs images using one of three types: mean, box, or Gaussian. The probability of an image being blurred is controlled by the `p` parameter. The blur effect is applied using a kernel with a specified radius. Blurred images are saved to a directory, which can be specified or defaults to the input directory. If the save directory is not provided, the `DJ_PRODUCED_DATA_DIR` environment variable can be used to set it. The operator ensures that the blur type is one of the supported options and that the radius is non-negative.

以指定的概率和模糊类型对数据集中的图像进行模糊处理。

此算子使用三种类型之一（均值、框或高斯）对图像进行模糊处理。图像被模糊的概率由`p`参数控制。模糊效果通过具有指定半径的核来应用。模糊后的图像保存到一个目录中，该目录可以指定，默认为输入目录。如果未提供保存目录，可以使用`DJ_PRODUCED_DATA_DIR`环境变量来设置。算子确保模糊类型是支持的选项之一，并且半径是非负的。

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `p` | <class 'float'> | `0.2` | Probability of the image being blurred. |
| `blur_type` | <class 'str'> | `'gaussian'` | Type of blur kernel, including ['mean', 'box', 'gaussian']. |
| `radius` | <class 'float'> | `2` | Radius of blur kernel. |
| `save_dir` | <class 'str'> | `None` | The directory where generated image files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_blur_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_blur_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)