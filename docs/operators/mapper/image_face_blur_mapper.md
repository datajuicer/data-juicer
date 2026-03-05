# image_face_blur_mapper

Mapper to blur faces detected in images.

This operator uses an OpenCV classifier to detect faces in images and applies a specified blur type to the detected face regions. The blur types supported are 'mean', 'box', and 'gaussian'. The radius of the blur kernel can be adjusted. If no save directory is provided, the modified images will be saved in the same directory as the input files.

用于模糊图像中检测到的人脸的映射器。

该算子使用OpenCV分类器检测图像中的人脸，并对检测到的人脸区域应用指定的模糊类型。支持的模糊类型有'mean'、'box'和'gaussian'。可以调整模糊核的半径。如果没有提供保存目录，修改后的图像将保存在与输入文件相同的目录中。

Type 算子类型: **mapper**

Tags 标签: cpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `cv_classifier` | <class 'str'> | `''` | OpenCV classifier path for face detection. By default, we will use 'haarcascade_frontalface_alt.xml'. |
| `blur_type` | <class 'str'> | `'gaussian'` | Type of blur kernel, including ['mean', 'box', 'gaussian']. |
| `radius` | typing.Annotated[float, Ge(ge=0)] | `2` | Radius of blur kernel. |
| `save_dir` | <class 'str'> | `None` | The directory where generated image files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_box
```python
ImageFaceBlurMapper(blur_type='box')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">lena.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/lena.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据



### test_box_radius
```python
ImageFaceBlurMapper(blur_type='box', radius=10)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">lena.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/lena.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">lena-face.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/lena-face.jpg" width="160" style="margin:4px;"/></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 image</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">cat.jpg:</div><div class="image-grid"><img src="../../../tests/ops/data/cat.jpg" width="160" style="margin:4px;"/></div></div></div>

#### 📤 output data 输出数据




## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_face_blur_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_face_blur_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)