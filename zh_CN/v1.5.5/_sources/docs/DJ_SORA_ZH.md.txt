# DJ-SORA
中文 | [English Page](DJ_SORA.md) 

---

数据是SORA等前沿大模型的关键，如何高效科学地获取和处理数据面临新的挑战！DJ-SORA旨在创建一系列大规模高质量开源多模态数据集，助力开源社区数据理解和模型训练。

DJ-SORA将基于Data-Juicer(包含上百个专用的视频、图像、音频、文本等多模态数据处理[算子](Operators.md)及工具)，形成一系列系统化可复用的多模态“数据菜谱”，用于分析、清洗及生成大规模高质量多模态数据。

本项目正在积极更新和维护中，我们热切地邀请您参与，共同打造一个更开放、更高质的多模态数据生态系统，激发大模型无限潜能！

![Overview](https://img.alicdn.com/imgextra/i4/O1CN01XphcBN2ACXcS6S1JH_!!6000000008167-2-tps-2289-1620.png)


## 动机 
- SORA仅简略提及使用了DALLE-3来生成高质量caption，且模型输入数据有变化的时长、分辨率和宽高比。
- 高质量大规模细粒度数据有助于稠密化数据点，帮助模型学好“文本 -> spacetime token”的条件映射，解决text-2-video模型的一系列现有挑战：
   - 画面流畅性和一致性，部分生成的视频有丢帧及静止状态
   - 文本理解能力和细粒度，生成出的结果和prompt匹配度较低
   - 视频内容较短，大多只有~10s，且场景画面不会有大的改变
   - 生成内容存在变形扭曲和物理规则违背情况，特别是在实体做出动作时

## 路线图
### 概览
- [DJ-SORA](#dj-sora)
  - [动机](#动机)
  - [路线图](#路线图)
    - [概览](#概览)
    - [支持视频数据的高性能加载和处理](#支持视频数据的高性能加载和处理)
    - [基础算子（视频时空维度）](#基础算子视频时空维度)
    - [进阶算子（细粒度模态间匹配及生成）](#进阶算子细粒度模态间匹配及生成)
    - [进阶算子（视频内容）](#进阶算子视频内容)
    - [DJ-SORA数据菜谱及数据集](#dj-sora数据菜谱及数据集)
    - [DJ-SORA数据验证及模型训练](#dj-sora数据验证及模型训练)

### 支持视频数据的高性能加载和处理
- [✅] 并行化数据加载存储：
  - [✅] lazy load with pyAV and ffmpeg
  - [✅] 多模态数据路径签名  
- [✅] 并行化算子处理：  
  - [✅] 支持单机多核  
  - [✅] GPU调用  
  - [✅] Ray多机分布式  
  - [✅] 基于阿里云PAI-DLC和Slurm的多机分布式 
- [✅] 分布式调度优化（OP-aware、自动化负载均衡）-->  Aliyun PAI-DLC
- [WIP] 视频相关算子的低精度加速支持, git tags: dj_op, dj_efficiency
- [WIP] 现有视频相关算子的SOTA模型增强, git tags: dj_op, dj_sota_models

### 基础算子（视频时空维度）
- 面向数据质量
  - [✅] video_resolution_filter （在分辨率维度进行过滤）
  - [✅] video_aspect_ratio_filter （在宽高比维度进行过滤）
  - [✅] video_duration_filter （在时间维度进行过滤）
  - [✅] video_motion_score_filter（在视频连续性维度过滤，计算光流，去除静态和极端动态）
  - [✅] video_ocr_area_ratio_filter （移除文本区域过大的样本）
- 面向数据多样性及数量
  - [✅] video_resize_resolution_mapper（在分辨率维度进行增强）
  - [✅] video_resize_aspect_ratio_mapper（在宽高比维度进行增强）
  - [✅] video_split_by_key_frame_mapper（基于关键帧进行切割）
  - [✅] video_split_by_duration_mapper（在时间维度进行切割）
  - [✅] video_split_by_scene_mapper (基于场景连续性进行切割)

### 进阶算子（细粒度模态间匹配及生成）
- 面向数据质量
  - [✅] video_frames_text_similarity_filter（在时空一致性维度过滤，计算关键/指定帧 和文本的匹配分）
- 面向数据多样性及数量
  - [✅] video_tagging_from_frames_mapper (轻量图生文模型，密集帧生成空间  概要信息)
  - [✅] video_captioning_from_frames_mapper（更重的图生文模型，少量帧生  成更详细空间信息）
  - [✅] video_tagging_from_audio_mapper (引入audio classification/category等meta信息)
  - [✅] video_captioning_from_audio_mapper（引入人声/对话等信息；  AudioCaption环境、场景等全局信息）
  - [✅] video_captioning_from_video_mapper（视频生文模型，连续帧生成时序信息）
  - [✅] video_captioning_from_summarizer_mapper（基于上述子能力的组合，使用纯文本大模型对不同种caption信息去噪、摘要）
  - [ ] [WIP] video_interleaved_mapper（在ICL、时间和跨模态维度增强），`interleaved_modes` include
    - text_image_interleaved（按时序交叉放置同一视频的的caption和frames）
    - text_audio_interleaved（按时序交叉放置同一视频的的ASR文本和frames）
    - text_image_audio_interleaved（交替拼接上述两种）

### 进阶算子（视频内容）
- [✅] video_deduplicator （比较MD5哈希值在文件样本级别去重）
- [✅] video_aesthetic_filter（拆帧后，进行美学度打分过滤）
- [✅]兼容ffmpeg已有的video commands
  - audio_ffmpeg_wrapped_mapper
  - video_ffmpeg_wrapped_mapper
- [WIP] 视频内容合规和隐私保护算子（图像、文字、音频）：
  - [✅] 马赛克
  - [✅] 版权水印
  - [✅] 人脸模糊
  - [✅] 黄暴恐
- [ ] [TODO] (Beyond Interpolation) 增强数据真实性和稠密性 
  - 碰撞、光影、重力、3D、场景切换（phase tranisition）、景深等
  - [ ] Filter类算子: caption是否描述真实性，该描述的相关性得分/正确性得分
  - [ ] Mapper类算子：增强video数据中对物理现象的文本描述
  - [ ] ...



### DJ-SORA数据菜谱及数据集
- 支持代表性数据的统一加载和转换（other-data <-> dj-data），方便DJ算子处理及扩展数据集
  - [✅] **Video-ChatGPT**: 100K video-instruction data:`{<question, answer, youtube_id>}`
  - [✅] **Youku-mPLUG-CN**: 36TB video-caption data：`{<caption, video_id>}`
  - [✅] **InternVid**: 234M data sample:`{<caption, youtube_id, start/end_time>}`
  - [✅] **MSR-VTT**: 10K video-caption data：`{<caption, video_id>}`
  - [✅] ModelScope数据集集成
  - [✅] VideoInstruct-100K, Panda70M, ......
- [ ] 大规模高质量DJ-SORA数据集
  - [✅] (Data sandbox) 基于DJ-video算子构建和优化多模态数据菜谱 (算子同期持续完善)
  - [✅] 数据源持续扩充：open-datasets, youku, web， ...
  - [ ] 基于DJ菜谱规模化分析、清洗、生成高质量多模态数据集
    -  [WIP] 多场景、高动态 
  - ...

### DJ-SORA数据验证及模型训练
  - [✅]  探索及完善多模态数据和模型的协同开发，形成benchmark和insights: [paper](https://arxiv.org/abs/2407.11784)
  - [] [WIP] 类SORA模型训练pipeline集成
    - [✅] [EasyAnimate](https://github.com/aigc-apps/EasyAnimate)
    - [✅] [T2V](https://t2v-turbo.github.io/)
    - [✅] [V-Bench](https://vchitect.github.io/VBench-project/)
    - ...
  - [✅] (Model-Data sandbox) 在相对小的模型和DJ-SORA数据集上，探索形成低开销、可迁移、有指导性的data-model co-design、配置及检查点
  - [ ] [WIP] 更大规模、更多场景使用DJ-SORA数据训练类SORA模型，提高模型性能
    - [✅] Data-Juicer-T2V, [V-Bench Top1 model](https://huggingface.co/datajuicer/Data-Juicer-T2V-v2)。详情请参考[这里](https://datajuicer.github.io/data-juicer-sandbox/zh_CN/main/index_ZH.html)。
    - ...
