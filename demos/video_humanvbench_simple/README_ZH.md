# HumanVBench 算子演示

这是论文 **HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks (CVPR'26)** 的算子贡献页面。

## 论文

- **HumanVBench** (CVPR 2026)
- arXiv: <https://arxiv.org/abs/2412.17574>
- 项目主页: <https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_HumanVBench_Probing_Human-Centric_Video_Understanding_in_MLLMs_with_Automatically_Synthesized_CVPR_2026_paper.html>

## 相关文件

- **示例配置:** `demos/video_humanvbench_simple/analyzer.yaml`
- **算子定义:** `data_juicer/config/config_all.yaml`
- **第三方模型:** `thirdparty/humanvbench_models/`

## 算子列表

| 算子 | 类型 | 描述 |
|---|---|---|
| `video_face_ratio_filter` | filter | 按人脸帧比例过滤视频。 |
| `video_human_tracks_extraction_mapper` | mapper | 提取人脸和人体边界框轨迹。 |
| `video_human_tracks_face_demographic_mapper` | mapper | 通过 DeepFace 检测人脸属性。 |
| `video_tagging_from_audio_mapper` | mapper | 从音频生成视频标签（已有算子）。 |
| `video_audio_detect_age_gender_mapper` | mapper | 从语音音频检测年龄/性别。 |
| `video_audio_ASR_mapper` | mapper | 自动语音识别。 |
| `video_audio_speech_emotion_mapper` | mapper | 语音情感识别。 |
| `video_captioning_from_human_tracks_mapper` | mapper | 通过 VideoLLaMA3 生成人物描述。 |
| `video_captioning_face_attribute_emotion_mapper` | mapper | 通过 VideoLLaMA3 生成面部属性/情感描述。 |
| `video_active_speaker_detect_mapper` | mapper | 通过 Light-ASD 检测活跃说话者。 |

## 安装

> **注意：** 这些算子需要 `thirdparty/humanvbench_models/` 下的第三方补丁/模型，目前需**源码安装**。

```shell
git clone https://github.com/datajuicer/data-juicer.git
cd data-juicer
pip install -e .
```

## 快速开始

由于 HumanVBench 算子涉及外部仓库的修改，这些经过调整的仓库存储在 `thirdparty/humanvbench_models`。

### 方式一：自动模式（推荐）

直接运行即可——相关算子会自动处理 `git clone` 和 `.diff` 补丁合并：

```shell
dj-process --config demos/video_humanvbench_simple/analyzer.yaml
```

### 方式二：手动模式

按照 `thirdparty/humanvbench_models/README.md` 的指引手动完成 `git clone` 和 `.diff` 补丁合并，然后运行：

```shell
dj-process --config demos/video_humanvbench_simple/analyzer.yaml
```

## 管线依赖

算子构成一个处理管线，推荐执行顺序如下：

1. `video_face_ratio_filter` — 过滤以人为中心的视频
2. `video_human_tracks_extraction_mapper` — 提取人脸/人体轨迹
3. `video_human_tracks_face_demographic_mapper` — 人脸属性统计
4. `video_tagging_from_audio_mapper` — 音频标签（Speech/Music/EMPTY）
5. `video_audio_detect_age_gender_mapper` — 说话者年龄/性别
6. `video_captioning_from_human_tracks_mapper` — 人物描述
7. `video_captioning_face_attribute_emotion_mapper` — 面部属性/情感
8. `video_active_speaker_detect_mapper` — 活跃说话者检测
9. `video_audio_ASR_mapper` — 语音转文字
10. `video_audio_speech_emotion_mapper` — 语音情感
