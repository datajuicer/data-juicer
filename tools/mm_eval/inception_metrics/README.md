# Metrics for video generation

This folder contains some postprocess scripts for evaluation of generated videos.

## Usage

Use [calc_metrics_for_videos.py](calc_metrics_for_videos.py) to compute FVD/ISV for generated videos.

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

- `fake_data_path`: The path to generated dataset. Only support for `jsonl` format. The video paths are put in the list under `videos` keys.
- `real_data_path`: The path to ground truth dataset. Only support for `jsonl` format. The video paths are put in the list under `videos` keys. Required when computing FVD, FID, KID, and PR.
- `fake_mm_dir`: The root directory to store the fake videos. If it is not none, the paths in jsonl file at fake_data_path are relative paths on it, else are absolute path.
- `real_mm_dir`: The root directory to store the real videos. If it is not none, the paths in jsonl file at real_data_path are relative paths on it, else are absolute path.
- `metric`: The name of metric applied, currently support `fvd2048_16f`, `fvd2048_128f`, `fvd2048_128f_subsample8f`, `kvd2048_16f`, `isv2048_ucf`, `prv2048_3n_16f`, `fid50k`, `kid50k`, `is50k`, `pr50k_3n`.
    - `fvd2048_16f`: Compute Frechet Video Distance (FVD), sample 2048 times in dataset, 16 adjacent frames each time.
    - `fvd2048_128f`: Compute Frechet Video Distance (FVD), sample 2048 times in dataset, 128 adjacent frames each time.
    - `fvd2048_128f_subsample8f`: Compute Frechet Video Distance (FVD), sample 2048 times in dataset, 16 frames each time, sample 1 frame every adjacent 8 frames.
    - `kvd2048_16f`: Compute Kernel Video Distance (KVD), sample 2048 times in dataset, 16 adjacent frames each time, split features to 100 subset to compute KVDs and return the mean.
    - `isv2048_ucf`: Compute Inception Score of Videos (ISV), sample 2048 times in dataset, 16 frames each time, split features to 10 subset to compute ISs and return the mean and std.
    - `prv2048_3n_16f`: Compute Precision/Recall of Videos (PRV), sample 2048 times in dataset, 16 adjacent frames each time, with the 4th nearest features to estimate the distributions.
    - `fid50k`: Compute Frechet Inception Distance (FID) of frames, sample 50000 frames from fake dataset at most.
    - `kid50k`: Compute Kernel Inception Distance (KID) of frames, sample 50000 frames from fake dataset at most, split features to 100 subset to compute KIDs and return the mean.
    - `is50k`: Compute Inception Score(IS) of frames, sample 50000 frames from fake dataset at most, split features to 10 subset to compute ISs and return the mean and std.
    - `pr50k_3n`: Compute Precision/Recall (PR) of frames, sample 50000 frames from fake dataset at most, with the 4th nearest features to estimate the distributions
- `detector_path`: Path to the corresponding detection model. Download the model from web if it is None.
- `result_path`: Path to JSON filename for saving results.
- `num_runs`: How many runs of the metric to average over.
- `height`: Sampled frames will be resized to this height.
- `width`: Sampled frames will be resized to this width.
- `replace_cache`: Whether to replace the dataset stats cache.
- `verbose`: Whether to log progress.
- `seed`: the random seed

## Introduction of Metrics

### FVD
The Frechet Video Distance (FVD)<sup>[1](#reference)</sup> measures the distance of distribution of video features from real dataset and fake dataset, extracted by a video classifier. The video classifier is an I3D model, trained on Kinetics-400, containing 400 human action classes.

### KVD
The Kernel Video Distance (KVD) is the video version of Frechet Inception Distance (FID)<sup>[3](#reference)</sup>, which extract features from videos by an I3D model, trained on Kinetics-400, containing 400 human action classes.

### ISV
The Inception Score of Videos (ISV)<sup>[2](#reference)</sup> evaluates the generated videos based on their quality and diversity, with a preference for diversity. Utilizing a C3D video classification model trained on the UCF101 action recognition dataset, ISV assesses quality through the classification certainty of each video—specifically, by computing the sum of the negative entropy of individual predictions. Meanwhile, diversity is gauged by the entropy of the prediction averages.

### PRV
The Precision/Recall of Videos (PRV) is the video version of Precision/Recall (PR)<sup>[5](#reference)</sup>, which extract features from videos by an I3D model, trained on Kinetics-400, containing 400 human action classes.

### FID
The Frechet Inception Distance (FID)<sup>[3](#reference)</sup> shares similarities with FVD in its approach, evaluating videos by analyzing the features of individual frames derived from an image classification model trained on ImageNet.

### KID
The Kernel Inception Distance (KID)<sup>[4](#reference)</sup> is similar to FID and quantifies the dissimilarity between two sets of videos by computing the squared maximum mean discrepancy of the features of frames through multiple samplings. Unlike FID, KID utilizes an unbiased estimator with a third-degree kernel, aligning more consistently with human perception. The applied image classification model is same as the model FID applied.

### IS
The Inception Score (IS)<sup>[2](#reference)</sup> shares similarities with ISV in its approach, evaluating videos by analyzing the predictions of individual frames derived from an image classification model trained on ImageNet.

### PR
The Precision/Recall (PR)<sup>[5](#reference)</sup> estimates the distribution of features of frames in the feature space by demarcating a region within the distance to the k-nearest neighbor features. It then assesses the precision and recall of frame generation by determining whether samples fall within the distributions of the real and fake datasets. The features are extracted from the VGG image classification model<sup>[6](#reference)</sup> trained on ILSVRC-2012.

<h2 id="reference">Reference:</h2>

- [1] Unterthiner, Thomas, et al. "Towards accurate generative models of video: A new metric & challenges." arXiv preprint arXiv:1812.01717 (2018).

- [2] Salimans, Tim, et al. "Improved techniques for training gans." Advances in neural information processing systems 29 (2016).

- [3] Heusel, Martin, et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium." Advances in neural information processing systems 30 (2017).

- [4] Bińkowski, Mikołaj, et al. "Demystifying mmd gans." arXiv preprint arXiv:1801.01401 (2018).

- [5] Kynkäänniemi, Tuomas, et al. "Improved precision and recall metric for assessing generative models." Advances in neural information processing systems 32 (2019).

- [6] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
