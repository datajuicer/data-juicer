# video_captioning_from_summarizer_mapper

Mapper to generate video captions by summarizing several kinds of generated texts (captions from video/audio/frames, tags from audio/frames, ...)

通过总结多种生成的文本（来自视频/音频/帧的字幕、来自音频/帧的标签等）来生成视频字幕的映射器。

Type 算子类型: **mapper**

Tags 标签: cpu, hf, multimodal

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `hf_summarizer` | <class 'str'> | `None` | the summarizer model used to summarize texts |
| `trust_remote_code` | <class 'bool'> | `False` |  |
| `consider_video_caption_from_video` | <class 'bool'> | `True` | whether to consider the video |
| `consider_video_caption_from_audio` | <class 'bool'> | `True` | whether to consider the video |
| `consider_video_caption_from_frames` | <class 'bool'> | `True` | whether to consider the |
| `consider_video_tags_from_audio` | <class 'bool'> | `True` | whether to consider the video |
| `consider_video_tags_from_frames` | <class 'bool'> | `True` | whether to consider the video |
| `vid_cap_from_vid_args` | typing.Optional[typing.Dict] | `None` | the arg dict for video captioning from |
| `vid_cap_from_frm_args` | typing.Optional[typing.Dict] | `None` | the arg dict for video captioning from |
| `vid_tag_from_aud_args` | typing.Optional[typing.Dict] | `None` | the arg dict for video tagging from audio |
| `vid_tag_from_frm_args` | typing.Optional[typing.Dict] | `None` | the arg dict for video tagging from |
| `keep_tag_num` | typing.Annotated[int, Gt(gt=0)] | `5` | max number N of tags from sampled frames to keep. |
| `keep_original_sample` | <class 'bool'> | `True` | whether to keep the original sample. If |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
not available 暂无

## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/mapper/video_captioning_from_summarizer_mapper.py)
- [unit test 单元测试](../../../tests/ops/mapper/test_video_captioning_from_summarizer_mapper.py)
- [Return operator list 返回算子列表](../../Operators.md)