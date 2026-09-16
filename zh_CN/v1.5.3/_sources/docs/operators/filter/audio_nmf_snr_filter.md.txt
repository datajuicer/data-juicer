# audio_nmf_snr_filter

Keep data samples whose audio Signal-to-Noise Ratios (SNRs) are within a specified range.

This operator computes the SNR of each audio in a sample using Non-negative Matrix Factorization (NMF). It then filters the samples based on whether their SNRs fall within the given minimum and maximum thresholds. The SNR is computed for each audio, and the filtering strategy can be set to either 'any' or 'all'. In 'any' mode, a sample is kept if at least one of its audios meets the SNR criteria. In 'all' mode, all audios must meet the criteria for the sample to be kept. The NMF computation uses a specified number of iterations. If no audio is present in the sample, the SNR is recorded as an empty array. The key metric is stored in the 'audio_nmf_snr' field.

保留音频信噪比（SNR）在指定范围内的数据样本。

该算子使用非负矩阵分解（NMF）计算每个样本中每个音频的SNR。然后根据音频的SNR是否在给定的最小值和最大值之间来过滤样本。每个音频的SNR都会被计算，并且过滤策略可以设置为'any'或'all'。在'any'模式下，只要至少有一个音频满足SNR条件，样本就会被保留。在'all'模式下，所有音频都必须满足条件，样本才会被保留。NMF计算使用指定的迭代次数。如果样本中没有音频，则SNR记录为空数组。关键指标存储在'audio_nmf_snr'字段中。

Type 算子类型: **filter**

Tags 标签: cpu, audio

## 🔧 Parameter Configuration 参数配置
| name 参数名 | type 类型 | default 默认值 | desc 说明 |
|--------|------|--------|------|
| `min_snr` | <class 'float'> | `0` | The min audio SNR to keep samples in dB. It's 0 by default. |
| `max_snr` | <class 'float'> | `9223372036854775807` | The max audio SNR to keep samples in dB. It's sys.maxsize by default. |
| `nmf_iter_num` | typing.Annotated[int, Gt(gt=0)] | `500` | The max number of iterations to run NMF. It's 500 in default. |
| `any_or_all` | <class 'str'> | `'any'` | keep this sample with 'any' or 'all' strategy of all audios. 'any': keep this sample if any audios meet the condition. 'all': keep this sample only if all audios meet the condition. |
| `args` |  | `''` | extra args |
| `kwargs` |  | `''` | extra args |

## 📊 Effect demonstration 效果演示
### test_filter_audios_within_range
```python
AudioNMFSNRFilter(min_snr=0, max_snr=5)
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio2.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 1 audio</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### ✨ explanation 解释
The operator filters the samples to keep only those with SNR values between 0 and 5. In this case, only the sample with audio3.ogg meets the criteria, so it is the only one kept in the target list.
算子过滤样本，只保留SNR值在0到5之间的样本。在这种情况下，只有包含audio3.ogg的样本满足条件，因此目标列表中仅保留了这一个样本。

### test_any
```python
AudioNMFSNRFilter(min_snr=0, max_snr=5, any_or_all='any')
```

#### 📥 input data 输入数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio2.wav:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio2.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 3:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### 📤 output data 输出数据
<div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 1:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio2.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio2.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div><div class="sample-card" style="border:1px solid #ddd; padding:12px; margin:8px 0; border-radius:6px; background:#fafafa; box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div class="sample-header" style="background:#f8f9fa; padding:4px 8px; margin-bottom:6px; border-radius:3px; font-size:0.9em; color:#666; border-left:3px solid #007acc;"><strong>Sample 2:</strong> 2 audios</div><div class="media-section" style="margin-bottom:8px;"><div class="media-label" style="font-size:0.85em; color:#666; margin-bottom:4px; font-weight:500;">audio1.wav|audio3.ogg:</div><div class="audio-list"><audio src="../../../tests/ops/data/audio1.wav" controls style="display:block; margin:4px 0;"></audio><audio src="../../../tests/ops/data/audio3.ogg" controls style="display:block; margin:4px 0;"></audio></div></div></div>

#### ✨ explanation 解释
The operator is set to 'any' mode, which means a sample is kept if at least one of its audios has an SNR value within the specified range (0-5). The first and third samples each have one audio meeting the condition, while the second sample has both audios within the range, so all three are kept in the target list.
算子设置为'any'模式，这意味着只要样本中的至少一个音频的SNR值位于指定范围内（0-5），则该样本就会被保留。第一个和第三个样本各自有一个音频满足条件，而第二个样本的两个音频都在范围内，因此这三个样本都被保留在目标列表中。


## 🔗 related links 相关链接
- [source code 源代码](../../../data_juicer/ops/filter/audio_nmf_snr_filter.py)
- [unit test 单元测试](../../../tests/ops/filter/test_audio_nmf_snr_filter.py)
- [Return operator list 返回算子列表](../../Operators.md)