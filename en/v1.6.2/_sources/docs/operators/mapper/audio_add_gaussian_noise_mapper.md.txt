# audio_add_gaussian_noise_mapper

Mapper to add Gaussian noise to audio samples.

This operator adds Gaussian noise to audio data with a specified probability. The amplitude of the noise is randomly chosen between `min_amplitude` and `max_amplitude`. If `save_dir` is provided, the modified audio files are saved in that directory; otherwise, they are saved in the same directory as the input files. The `p` parameter controls the probability of applying this transformation to each sample. If no audio is present in the sample, it is returned unchanged.

向音频样本添加高斯噪声的映射器。

该算子以指定的概率向音频数据添加高斯噪声。噪声的幅度在`min_amplitude`和`max_amplitude`之间随机选择。如果提供了`save_dir`，则修改后的音频文件将保存在该目录中；否则，它们将保存在与输入文件相同的目录中。`p`参数控制对每个样本应用此转换的概率。如果样本中没有音频，则原样返回。

Type 算子类型: **mapper**

Tags 标签: cpu, audio

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_amplitude` | <class 'float'> | `0.001` | float unit: linear amplitude. Default: 0.001. Minimum noise amplification factor. |
| `max_amplitude` | <class 'float'> | `0.015` | float unit: linear amplitude. Default: 0.015. Maximum noise amplification factor. |
| `p` | <class 'float'> | `0.5` | float range: [0.0, 1.0].  Default: 0.5. The probability of applying this transform. |
| `save_dir` | <class 'str'> | `None` | str. Default: None. The directory where generated audio files will be stored. If not specified, outputs will be saved in the same directory as their corresponding input files. This path can alternatively be defined by setting the `DJ_PRODUCED_DATA_DIR` environment variable. |
| `args` |  | `''` |  |
| `kwargs` |  | `''` |  |


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/audio_add_gaussian_noise_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_audio_add_gaussian_noise_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)