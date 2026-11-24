# image_sam_3d_body_mapper

SAM 3D Body (3DB) is a promptable model for single-image full-body 3D human mesh recovery (HMR).

SAM 3D Body（3DB）是一种可提示的模型，用于从单张图像中恢复完整人体的3D网格（HMR）。

Type 算子类型: **mapper**

Tags 标签: gpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `checkpoint_path` | <class 'str'> | `''` | Path to SAM 3D Body model checkpoint. |
| `detector_name` | <class 'str'> | `'vitdet'` | Human detection model for demo (Default `vitdet`, add your favorite detector if needed). |
| `segmentor_name` | <class 'str'> | `'sam2'` | Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed). |
| `fov_name` | <class 'str'> | `'moge2'` | FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed). |
| `mhr_path` | <class 'str'> | `''` | Path to MoHR/assets folder (or set SAM3D_mhr_path). |
| `detector_path` | <class 'str'> | `''` | Path to human detection model folder (or set SAM3D_DETECTOR_PATH). |
| `segmentor_path` | <class 'str'> | `''` | Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH). |
| `fov_path` | <class 'str'> | `''` | Path to fov estimation model folder (or set SAM3D_FOV_PATH). |
| `bbox_thresh` | <class 'float'> | `0.8` | Bounding box detection threshold. |
| `use_mask` | <class 'bool'> | `False` | Use mask-conditioned prediction (segmentation mask is automatically generated from bbox). |
| `visualization_dir` | <class 'str'> | `None` | Directory to save visualization results. If None, no visualization will be saved. |
| `tag_field_name` | <class 'str'> | `'sam_3d_body_data'` | Field name for storing the results. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_sam_3d_body_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_sam_3d_body_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)