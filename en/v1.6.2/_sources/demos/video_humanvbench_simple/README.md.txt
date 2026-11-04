# HumanVBench Operators Demo

This is the operator contribution page for the paper: **HumanVBench: Probing Human-Centric Video Understanding in MLLMs with Automatically Synthesized Benchmarks (CVPR'26)**.

## Paper

- **HumanVBench** (CVPR 2026)
- arXiv: <https://arxiv.org/abs/2412.17574>
- Project: <https://openaccess.thecvf.com/content/CVPR2026/html/Zhou_HumanVBench_Probing_Human-Centric_Video_Understanding_in_MLLMs_with_Automatically_Synthesized_CVPR_2026_paper.html>

## Related Files

- **Example Recipe:** `demos/video_humanvbench_simple/analyzer.yaml`
- **Operator Definitions:** `data_juicer/config/config_all.yaml`
- **Thirdparty Models:** `thirdparty/humanvbench_models/`

## Operators

| Operator | Type | Description |
|---|---|---|
| `video_face_ratio_filter` | filter | Filter videos by face-to-frame ratio. |
| `video_human_tracks_extraction_mapper` | mapper | Extract face and human bounding box tracks. |
| `video_human_tracks_face_demographic_mapper` | mapper | Detect face demographics via DeepFace. |
| `video_tagging_from_audio_mapper` | mapper | Generate video tags from audio (existing). |
| `video_audio_detect_age_gender_mapper` | mapper | Detect age/gender from speech audio. |
| `video_audio_ASR_mapper` | mapper | Automatic speech recognition. |
| `video_audio_speech_emotion_mapper` | mapper | Speech emotion recognition. |
| `video_captioning_from_human_tracks_mapper` | mapper | Per-person captioning via VideoLLaMA3. |
| `video_captioning_face_attribute_emotion_mapper` | mapper | Face attribute/emotion captioning via VideoLLaMA3. |
| `video_active_speaker_detect_mapper` | mapper | Active speaker detection via Light-ASD. |

## Installation

> **Note:** These OPs need third-party patches/models under `thirdparty/humanvbench_models/`, so a **source install** is required for now.

```shell
git clone https://github.com/datajuicer/data-juicer.git
cd data-juicer
pip install -e .
```

## Quick Start

Since HumanVBench operators involve modifications to external repositories, these adjusted repositories are stored in `thirdparty/humanvbench_models`.

### Option 1: Automatic Mode (Recommended)

Run directly — the operators handle `git clone` and `.diff` patch merging automatically:

```shell
dj-process --config demos/video_humanvbench_simple/analyzer.yaml
```

### Option 2: Manual Mode

Follow the instructions in `thirdparty/humanvbench_models/README.md` to manually complete the `git clone` and `.diff` patch merging, then run:

```shell
dj-process --config demos/video_humanvbench_simple/analyzer.yaml
```

## Pipeline Dependency

The operators form a processing pipeline. The recommended execution order is:

1. `video_face_ratio_filter` — filter human-centric videos
2. `video_human_tracks_extraction_mapper` — extract face/human tracks
3. `video_human_tracks_face_demographic_mapper` — face demographics
4. `video_tagging_from_audio_mapper` — audio tags (Speech/Music/EMPTY)
5. `video_audio_detect_age_gender_mapper` — speaker age/gender
6. `video_captioning_from_human_tracks_mapper` — per-person captions
7. `video_captioning_face_attribute_emotion_mapper` — face attributes/emotions
8. `video_active_speaker_detect_mapper` — active speaker detection
9. `video_audio_ASR_mapper` — speech transcription
10. `video_audio_speech_emotion_mapper` — speech emotion
