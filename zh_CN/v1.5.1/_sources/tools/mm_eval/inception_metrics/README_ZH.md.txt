# 视频生成测评工具

此文件夹包含一些测评脚本，用于测评模型生成的视频。

## 用法

使用 [calc_metrics_for_videos.py](calc_metrics_for_videos.py) 计算生成视频集的FVD或ISV

```shell
python tools/video_metrics/calc_metrics_for_videos.py        \
    --fake_data_path    <fake_data_path>        \
    --real_data_path    <real_data_path>        \
    [--fake_mm_dir      <fake_mm_dir>]          \
    [--real_mm_dir      <real_mm_dir>]          \
    --metric            <metric>                \
    [--detector_path    <detector_path>]        \
    --result_path       <result_path>           \
    --num_runs          <num_runs>              \
    --height            <height>                \
    --width             <width>                 \
    --replace_cache     <replace_cache>         \
    --verbose           <verbose>               \
    --seed              <seed>

# get help
python tools/video_metrics/calc_metrics_for_videos.py --help
```

- `fake_data_path`: 生成数据集的路径。目前只支持 `jsonl` 格式。每个sample的视频路径放在`videos`关键词下的列表里。
- `real_data_path`: 真实数据集的路径。目前只支持 `jsonl` 格式。每个sample的视频路径放在`videos`关键词下的列表里。计算FVD、FID、KID和PR时需要。
- `fake_mm_dir`: 存储生成视频的目录。如果不是none，在fake_data_path下的jonl文件中的路径为相对它的相对路径，否则为绝对路径。
- `real_mm_dir`: 存储真实视频的目录。如果不是none，在real_data_path下的jonl文件中的路径为相对它的相对路径，否则为绝对路径。
- `metric`: 测评的名称, 目前支持`fvd2048_16f`、`fvd2048_128f`、`fvd2048_128f_subsample8f`、`isv2048_ucf`、`prv2048_3n_16f`、`fid50k`、`kid50k`、`is50k`、`pr50k_3n`。
    - `fvd2048_16f`: 计算Frechet Video Distance (FVD)，在数据集中采样2048次，每次采样连续的16帧。
    - `fvd2048_128f`: 计算Frechet Video Distance (FVD)，在数据集中采样2048次，每次采样连续的128帧。
    - `fvd2048_128f_subsample8f`: 计算Frechet Video Distance (FVD)，在数据集中采样2048次，每次采样16帧，每帧间隔8帧。
    - `isv2048_ucf`: 计算Inception Score of Videos (ISV)，在数据集中采样2048次，每次采样连续的16帧，分成10份计算IS，返回均值和方差。
    - `prv2048_3n_16f`: 计算Precision/Recall of Videos (PRV)，在数据集中采样2048次，每次采样连续的16帧，采用第4近邻的特征距离来评估特征分布
    - `fid50k`: 计算视频帧的Frechet Inception Distance (FID)，从生成数据集中最多采样50000帧。
    - `kid50k`: 计算视频帧的Kernel Inception Distance (KID)，从生成数据集中最多采样50000帧，并将得到的特征拆分成100份来计算KID，返回其均值。
    - `is50k`: 计算视频帧的Inception Score(IS)，从生成数据集中最多采样50000帧，并将得到的特征拆分成10份来计算IS，返回其均值和方差。
    - `pr50k_3n`: 计算视频的Precision/Recall (PR)，从生成数据集中最多采样50000帧，采用第4近邻的特征距离来评估特征分布。
- `detector_path`: metric对应的视频分类模型的路径，如果为None则自动从网上下载。
- `result_path`: 结果存储路径，`jsonl` 格式。
- `num_runs`: 测评次数，大于1时最终结果会取平均。
- `height`: 每一帧测评时resize到这个高度
- `width`: 每一帧测评时resize到这个宽度
- `replace_cache`: 是否覆盖cache重新计算
- `verbose`: 是否打log
- `seed`: 随机种子

## 指标介绍

### FVD
Frechet Video Distance (FVD)<sup>[1](#reference)</sup>衡量了从真实数据集和生成数据集中提取的视频特征的分布距离。视频特征是由视频分类器提取的。视频分类器是一个在Kinetics-400数据集上训练的I3D模型，包含400种人类动作类别。

### KVD
Kernel Video Distance (KVD)是视频版本的Frechet Inception Distance (FID)<sup>[3](#reference)</sup>，它从视频中提取特征，使用了一个在包含400种人类动作类别的Kinetics-400数据集上训练的I3D模型。

### ISV
Inception Score of Videos (ISV)<sup>[2](#reference)</sup>基于生成视频的质量和多样性进行评估，其中更偏向于多样性。ISV利用了在UCF101动作识别数据集上训练的C3D视频分类模型，通过计算每个视频分类预测的负熵之和来评估质量。多样性则是通过预测概率分布的平均值的熵来衡量的。

### PRV
Precision/Recall of Videos (PRV)<sup>[5](#reference)</sup>通过在特征空间中划定到k个最近领特征的距离内的区域，来估计视频特征的分布。然后，通过确定样本是否落在真实数据集和生成数据集的分布内，来评估视频生成的精确度和召回率。这些特征是从Kinetics-400数据集上训练的包含400个人类动作类别的I3D模型中提取的。
Precision/Recall of Videos (PRV)是视频版本的Precision/Recall (PR)<sup>[5](#reference)</sup>，它从视频中提取特征，使用了一个在包含400种人类动作类别的Kinetics-400数据集上训练的I3D模型。

### FID
Frechet Inception Distance (FID)<sup>[3](#reference)</sup>与FVD的方法类似，这里提取的是每一帧的特征，模型采用在ImageNet上训练的图像分类模型。

### KID
Kernel Inception Distance (KID)<sup>[4](#reference)</sup>类似于FID，通过多次抽样计算帧特征的平方最大均值差异来量化两组视频之间的差异。不同于FID，KID是无偏估计，采用三阶核函数，更一致地符合人类感知。所应用的图像分类模型与FID中所应用的模型相同。

### IS
Inception Score (IS)<sup>[2](#reference)</sup>与ISV的方法类似，这里提利用了模型对每一帧的分类预测，模型采用在ImageNet上训练的图像分类模型。

### PR
The Precision/Recall (PR)<sup>[5](#reference)</sup>通过在视频帧的特征空间中划定到k个最近领特征的距离内的区域，来估计视频特征的分布。然后，通过确定样本帧是否落在真实数据集和生成数据集的分布内，来评估视频生成帧的精确度和召回率。这些特征是从ILSVRC-2012上训练的VGG图像分类模型<sup>[6](#reference)</sup>中提取的。


<h2 id="reference">参考文献：</h2>

- [1] Unterthiner, Thomas, et al. "Towards accurate generative models of video: A new metric & challenges." arXiv preprint arXiv:1812.01717 (2018).

- [2] Salimans, Tim, et al. "Improved techniques for training gans." Advances in neural information processing systems 29 (2016).

- [3] Heusel, Martin, et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium." Advances in neural information processing systems 30 (2017).

- [4] Bińkowski, Mikołaj, et al. "Demystifying mmd gans." arXiv preprint arXiv:1801.01401 (2018).

- [5] Kynkäänniemi, Tuomas, et al. "Improved precision and recall metric for assessing generative models." Advances in neural information processing systems 32 (2019).

- [6] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
