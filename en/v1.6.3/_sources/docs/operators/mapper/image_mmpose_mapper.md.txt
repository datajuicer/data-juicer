# image_mmpose_mapper

Mapper to perform human keypoint detection inference using MMPose models. It requires three essential components for model initialization:
- deploy_cfg (str): Path to the deployment configuration file (defines inference settings)
- model_cfg (str): Path to the model configuration file (specifies model architecture)
- model_files (List[str]): Model weight files including pre-trained weights and parameters

The implementation follows the official MMPose deployment guidelines from MMDeploy. For detailed configuration requirements and usage examples, refer to: https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpose.md

用于使用 MMPose 模型执行人体关键点检测推理的 Mapper。模型初始化需要三个核心组件：  
- deploy_cfg (str)：部署配置文件路径（定义推理设置）  
- model_cfg (str)：模型配置文件路径（指定模型架构）  
- model_files (List[str])：模型权重文件，包括预训练权重和参数  

该实现遵循 MMDeploy 官方提供的 MMPose 部署指南。有关详细配置要求和使用示例，请参阅：https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/04-supported-codebases/mmpose.md

Type 算子类型: **mapper**

Tags 标签: gpu, image

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `deploy_cfg` | <class 'str'> | `None` | MMPose deployment config file. |
| `model_cfg` | <class 'str'> | `None` | MMPose model config file. |
| `model_files` | typing.Union[str, typing.Sequence[str], NoneType] | `None` | Path to the model weight files. |
| `pose_key` | <class 'str'> | `'pose_info'` | Key to store pose information. |
| `visualization_dir` | <class 'str'> | `None` | Directory to save visualization results. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/image_mmpose_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_image_mmpose_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)