# audio_size_filter

Keep data samples based on the size of their audio files.

This operator filters data samples by checking if the size of their audio files falls within a specified range. The size can be in bytes, kilobytes, megabytes, or any other unit. The key metric used is 'audio_sizes', which is an array of file sizes in bytes. If no audio files are present, the 'audio_sizes' field will be an empty array. The operator supports two strategies for keeping samples: 'any' and 'all'. In 'any' mode, a sample is kept if at least one of its audio files meets the size criteria. In 'all' mode, all audio files must meet the size criteria for the sample to be kept.

根据音频文件大小保留数据样本。

该算子通过检查音频文件的大小是否在指定范围内来过滤数据样本。文件大小可以是字节、千字节、兆字节或其他任何单位。使用的关键指标是'audio_sizes'，这是一个以字节为单位的文件大小数组。如果没有音频文件，则'audio_sizes'字段将是一个空数组。该算子支持两种保留样本的策略：'any'和'all'。在'any'模式下，只要有至少一个音频文件满足大小条件，样本就会被保留。在'all'模式下，所有音频文件都必须满足大小条件，样本才会被保留。

Type 算子类型: **filter**

Tags 标签: cpu, audio

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_size` | <class 'str'> | `'0'` | The min audio size to keep samples.  set to be "0" by default for no size constraint |
| `max_size` | <class 'str'> | `'1TB'` | The max audio size to keep samples.  set to be "1Tb" by default, an approximate for un-limited case |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all audios. 'any': keep this sample if any audios meet the condition. 'all': keep this sample only if all audios meet the condition. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_min_max
```python
AudioSizeFilter(min_size='800kb', max_size='1MB')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio2.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### ✨ explanation 解释
The operator filters the dataset to keep only those samples where the audio file size is between 800KB and 1MB. The first sample meets this criteria, while the others do not, leading to a filtered list containing just the first sample.
算子过滤数据集，仅保留音频文件大小在800KB到1MB之间的样本。第一个样本满足这个条件，而其他样本不满足，因此过滤后的列表只包含第一个样本。

### test_any
```python
AudioSizeFilter(min_size='800kb', max_size='1MB', any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio2.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio2.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio2.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### ✨ explanation 解释
The operator keeps samples if at least one of their audio files' sizes is within the specified range (800KB to 1MB) when using the 'any' mode. The first and third samples have at least one audio meeting the size requirement, so they are kept, while the second sample does not meet the condition for any of its audios, hence it's removed from the final list.
在使用'any'模式时，如果样本中至少有一个音频文件的大小在指定范围内（800KB到1MB），则算子会保留该样本。第一个和第三个样本至少有一个音频满足尺寸要求，所以它们被保留下来；而第二个样本的所有音频都不满足条件，因此从最终列表中移除。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/audio_size_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_audio_size_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)