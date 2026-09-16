# Awesome Data-Model Co-Development of MLLMs [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
Welcome to the "Awesome List" for data-model co-development of Multi-Modal Large Language Models (MLLMs), a continually updated resource tailored for the open-source community. This compilation features cutting-edge research, insightful articles focusing on improving MLLMs involving with the data-model co-development of MLLMs, and  tagged based on the proposed **taxonomy** from our data-model co-development [survey](https://arxiv.org/abs/2407.08583), as illustrated below.

![Overview of Our Taxonomy](https://img.alicdn.com/imgextra/i1/O1CN01aN3TVo1mgGZAuSHJ4_!!6000000004983-2-tps-3255-1327.png)
Due to the rapid development in the field, this repository and our paper are continuously being updated and synchronized with each other. **Please feel free to make pull requests or open issues to [contribute to](#contribution-to-this-survey) this list and add more related resources!** We will periodically update our [arXiv version](https://arxiv.org/abs/2407.08583) according to this repository.

## News
+ 🎉 [2025-06-04] Our [Data-Model Co-development Survey](hhttps://ieeexplore.ieee.org/document/11027559) has been accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (**TPAMI**)! Welcome to explore and contribute this awesome-list.
+ [2025-05-25] We added 20 academic papers related to this survey.
+ ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-10-23] We built a [dynamic table](https://datajuicer.github.io/data-juicer/en/main/_static/awesome-list.html) based on the [paper list](#paper-list) that supports filtering and searching.
+ ![new](https://img.alicdn.com/imgextra/i4/O1CN01kUiDtl1HVxN6G56vN_!!6000000000764-2-tps-43-19.png) [2024-10-22] We restructured our [paper list](#paper-list) to provide more streamlined information.


## Candidate Co-Development Tags
These tags correspond to the taxonomy in our paper, and each work may be assigned with more than one tags.
### Data4Model: Scaling
#### For Scaling Up of MLLMs: Larger Datasets
| Section Title | Tag |
|-------|-------|
|Data Acquisition|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|Data Augmentation|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|
|Data Diversity|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|

#### For Scaling Effectiveness of MLLMs: Better Subsets
| Section Title | Tag |
|-------|-------|
|Data Condensation|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Data Mixture|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|
|Data Packing|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|
|Cross-Modal Alignment|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|

### Data4Model: Usability
#### For Instruction Responsiveness of MLLMs
| Section Title | Tag |
|-------|-------|
|Prompt Design|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|
|ICL Data|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|
|Human-Behavior Alignment Data|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|

#### For Reasoning Ability of MLLMs
| Section Title | Tag |
|-------|-------|
|Data for Single-Hop Reasoning|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa)|
|Data for Multi-Hop Reasoning|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|

#### For Ethics of MLLMs
| Section Title | Tag |
|-------|-------|
|Data Toxicity|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|
|Data Privacy and Intellectual Property|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Privacy&IP-d3f0aa)|

#### For Evaluation of MLLMs
| Section Title | Tag |
|-------|-------|
|Benchmarks for Multi-Modal Understanding | ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|Benchmarks for Multi-Modal Generation: | ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|
|Benchmarks for Multi-Modal Retrieval: | ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Retrieval-d3f0aa)|
|Benchmarks for Multi-Modal Reasoning: | ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|

### Model4Data: Synthesis
| Section Title | Tag |
|-------|-------|
|Model as a Data Creator|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Model as a Data Mapper|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|Model as a Data Filter|![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb)|
|Model as a Data Evaluator|![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|

### Model4Data: Insights
| Section Title | Tag |
|-------|-------|
|Model as a Data Navigator|![](https://img.shields.io/badge/Model4Data--Insights--Navigator-f2c0c6)|
|Model as a Data Extractor|![](https://img.shields.io/badge/Model4Data--Insights--Extractor-f2c0c6)|
|Model as a Data Analyzer|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|
|Model as a Data Visualizer|![](https://img.shields.io/badge/Model4Data--Insights--Visualizer-f2c0c6)|


## Paper List
Below is a paper list summarized based on our survey. Additionally, we have provided a [dynamic table](https://datajuicer.github.io/data-juicer/_static/awesome-list.html) that supports filtering and searching, with the data source same as the list below.

| Title | Tags |
|-------|-------|
|No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|
|What Makes for Good Visual Instructions? Synthesizing Complex Visual Reasoning Instructions for Visual Instruction Tuning|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Med-MMHL: A Multi-Modal Dataset for Detecting Human- and LLM-Generated Misinformation in the Medical Domain|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|
|Probing Heterogeneous Pretraining Datasets with Small Curated Datasets|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)||
|ChartLlama: A Multimodal LLM for Chart Understanding and Generation|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)![](https://img.shields.io/badge/Model4Data--Insights--Visualizer-f2c0c6)|
|VideoChat: Chat-Centric Video Understanding|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|Aligned with LLM: a new multi-modal training paradigm for encoding fMRI activity in visual cortex|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|3DMIT: 3D Multi-modal Instruction Tuning for Scene Understanding|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|Audio Retrieval with WavText5K and CLAP Training|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Retrieval-d3f0aa)|
|The Devil is in the Details: A Deep Dive into the Rabbit Hole of Data Filtering|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Demystifying CLIP Data|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|3.2.2|
|Learning Transferable Visual Models From Natural Language Supervision|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|DataComp: In search of the next generation of multimodal datasets|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb)|
|Beyond neural scaling laws: beating power law scaling via data pruning|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Flamingo: a visual language model for few-shot learning|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|
|Quality not quantity: On the interaction between dataset design and robustness of clip|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|
|VBench: Comprehensive Benchmark Suite for Video Generative Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|
|EvalCraftr: Benchmarking and Evaluating Large Video Generation Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|
|Training Compute-Optimal Large Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|NExT-GPT: Any-to-Any Multimodal LLM|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|ChartThinker: A Contextual Chain-of-Thought Approach to Optimized Chart Summarization|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|ChartReformer: Natural Language-Driven Chart Image Editing|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)![](https://img.shields.io/badge/Model4Data--Insights--Visualizer-f2c0c6)|
|GroundingGPT: Language Enhanced Multi-modal Grounding Model|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|
|Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|
|Kosmos-2: Grounding Multimodal Large Language Models to the World|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|
|Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters|![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|
|3DBench: A Scalable 3D Benchmark and Instruction-Tuning Dataset|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|Structured Packing in LLM Training Improves Long Context Utilization|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|
|Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|
|MoDE: CLIP Data Experts via Clustering|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|
|Efficient Multimodal Learning from Data-centric Perspective|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Improved Baselines for Data-efficient Perceptual Augmentation of LLMs|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|
|MVBench: A Comprehensive Multi-modal Video Understanding Benchmark|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|Perception Test: A Diagnostic Benchmark for Multimodal Video Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|FunQA: Towards Surprising Video ComprehensionFunQA: Towards Surprising Video Comprehension|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|
|OneChart: Purify the Chart Structural Extraction via One Auxiliary Token|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|
|StructChart: Perception, Structuring, Reasoning for Visual Chart Understanding|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa)|
|MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|
|WorldGPT: Empowering LLM as Multimodal World Model|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|
|List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|
|TextSquare: Scaling up Text-Centric Visual Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|
|ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning?|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Insights--Navigator-f2c0c6)|
|Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|
|Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|
|Fewer Truncations Improve Language Modeling|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)|
|MedThink: Explaining Medical Visual Question Answering via Multimodal Decision-Making Rationale|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|AesExpert: Towards Multi-modality Foundation Model for Image Aesthetics Perception|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|UNIAA: A Unified Multi-modal Image Aesthetic Data AugmentationAssessment Baseline and Benchmark|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Improving Composed Image Retrieval via Contrastive Learning with Scaling Positives and Negatives|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|
|TextHawk: Exploring Efficient Fine-Grained Perception of Multimodal Large Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|The Wolf Within: Covert Injection of Malice into MLLM Societies via an MLLM Operative|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|
|BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|
|MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)| Sec. 4.3.1,  Sec. 4.4.2|
|Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|
|M3DBench: Let’s Instruct Large Models with Multi-modal 3D Prompts|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|MoqaGPT: Zero-Shot Multi-modal Open-domain Question Answering with Large Language Model|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|
|mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|
|mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|
|mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|
|Open-TransMind: A New Baseline and Benchmark for 1st Foundation Model Challenge of Intelligent Transportation|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Retrieval-d3f0aa)|
|On the Adversarial Robustness of Multi-Modal Foundation Models|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|
|What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|ShareGPT4V: Improving Large Multi-Modal Models with Better Captions|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|PaLM-E: An Embodied Multimodal Language Model|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|
|Multimodal Data Curation via Object Detection and Filter Ensembles|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Sieve: Multimodal Dataset Pruning Using Image Captioning Models|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Towards a statistical theory of data selection under weak supervision|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|𝐷2 Pruning: Message Passing for Balancing Diversity & Difficulty in Data Pruning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|UIClip: A Data-driven Model for Assessing User Interface Design|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|CapsFusion: Rethinking Image-Text Data at Scale|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|
|Improving CLIP Training with Language Rewrites|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb) ![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|
|OpenLEAF: Open-Domain Interleaved Image-Text Generation and Evaluation|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|
|A Decade's Battle on Dataset Bias: Are We There Yet?|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|
|Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|Data Filtering Networks|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|T-MARS: Improving Visual Representations by Circumventing Text Feature Learning|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Align and Attend: Multimodal Summarization with Dual Contrastive Losses|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|
|Text-centric Alignment for Multi-Modality Learning|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|Noisy Correspondence Learning with Meta Similarity Correction|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|Grounding-Prompter: Prompting LLM with Multimodal Information for Temporal Sentence Grounding in Long Videos|![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|
|Language-Image Models with 3D Understanding|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|
|Scaling Laws for Generative Mixed-Modal Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|BLINK: Multimodal Large Language Models Can See but Not Perceive|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|Visual Hallucinations of Multi-modal Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)|
|DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|
|EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)|
|Visual Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|
|Time-LLM: Time Series Forecasting by Reprogramming Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|
|On the De-duplication of LAION-2B|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|
|LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)|
|LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|
|Data Augmentation for Text-based Person Retrieval Using Large Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|Aligning Actions and Walking to LLM-Generated Textual Descriptions|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)|
|SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|
|AlignGPT: Multi-modal Large Language Models with Adaptive Alignment Capability|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|3.2.4|
|AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Probing Multimodal LLMs as World Models for Driving|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|
|Unified Hallucination Detection for Multimodal Large Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)![](https://img.shields.io/badge/Model4Data--Insights--Extractor-f2c0c6)![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|Semdedup: Data-efficient learning at web-scale through semantic deduplication|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|
|Automated Multi-level Preference for MLLMs|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|
|Silkie: Preference distillation for large visual language models|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|
|Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|
|M3it: A large-scale dataset towards multi-modal multilingual instruction tuning|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|
|Aligning Large Multimodal Models with Factually Augmented RLHF|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|
|DRESS: Instructing Large Vision-Language Models to Align and Interact with Humans via Natural Language Feedback|![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)|
|RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|
|MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa) ![](https://img.shields.io/badge/Data4Model--Usability--Eval--Retrieval-d3f0aa)|
|M3CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought|![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)|
|ImgTrojan: Jailbreaking Vision-Language Models with ONE Image|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa) ![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|VL-Trojan: Multimodal Instruction Backdoor Attacks against Autoregressive Visual Language Models|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|
|Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)|
|Improving Multimodal Datasets with Image Captioning|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)|3.2.1, 3.2.4, 8.2.2, 8.3.3|
|Bridging Research and Readers: A Multi-Modal Automated Academic Papers Interpretation System|![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)|6.3|
|LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition|![](https://img.shields.io/badge/Model4Data--Insights--Extractor-f2c0c6)|6.2|
|PDFChatAnnotator: A Human-LLM Collaborative Multi-Modal Data Annotation Tool for PDF-Format Catalogs|![](https://img.shields.io/badge/Model4Data--Insights--Extractor-f2c0c6)![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|CiT: Curation in Training for Effective Vision-Language Data|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)|
|InstructPix2Pix: Learning to Follow Image Editing Instructions|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Automated Data Visualization from Natural Language via Large Language Models: An Exploratory Study|![](https://img.shields.io/badge/Model4Data--Insights--Visualizer-f2c0c6)|
|ModelGo: A Practical Tool for Machine Learning License Analysis|![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Privacy&IP-d3f0aa)|
|Scaling Laws of Synthetic Images for Model Training ... for Now|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|
|Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)|
|Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V|![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)|
|Segment Anything|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|AIM: Let Any Multi-modal Large Language Models Embrace Efficient In-Context Learning|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|
|MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|
|All in an Aggregated Image for In-Image Learning|![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)|
|Panda-70m: Captioning 70m videos with multiple cross-modality teachers|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved With Text|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)|
|Imagebind: One embedding space to bind them all|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|UniBind: LLM-Augmented Unified and Balanced Representation Space to Bind Them All|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|FreeBind: Free Lunch in Unified Multimodal Space via Knowledge Fusion|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|Binding Touch to Everything: Learning Unified Multimodal Tactile Representations|![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)|
|Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|ZooProbe: A Data Engine for Evaluating, Exploring, and Evolving Large-scale Training Data for Multimodal LLMs|![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|
|MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb) ![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb)|
|MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|World to Code: Multi-modal Data Generation via Self-Instructed Compositional Captioning and Filtering|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|
|Model-in-the-Loop (MILO): Accelerating Multimodal AI Data Annotation with LLMs|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|FM2DS: Few-Shot Multimodal Multihop Data Synthesis with Knowledge Distillation for Question Answering|![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb)|
|MiniCPM-V: A GPT-4V Level MLLM on Your Phone|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|Descriptive Caption Enhancement with Visual Specialists for Multimodal Perception|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|REFINESUMM: Self-Refining MLLM for Generating a Multimodal Summarization Dataset|![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models|![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|
|A Comprehensive Study of Multimodal Large Language Models for Image Quality Assessment|![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)|
|Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models|![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d) ![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb)|
|Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation|![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)|

## Contribution to This Survey
Due to the rapid development in the field, this repository and our paper are continuously being updated and synchronized with each other. Please feel free to make pull requests or open issues to contribute to this list and add more related resources!
**You can add the titles of relevant papers to the table above, and (optionally) provide suggested tags along with the corresponding sections if possible.**
We will attempt to complete the remaining information and periodically update our survey based on the updated content of this document.

### References
If you find our work useful for your research or development, please kindly cite the following [paper](https://arxiv.org/abs/2407.08583).


```
@article{qin2024synergy,
  title={The Synergy between Data and Multi-Modal Large Language Models: A Survey from Co-Development Perspective},
  author={Qin, Zhen and Chen, Daoyuan and Zhang, Wenhao and Liuyi, Yao and Yilun, Huang and Ding, Bolin and Li, Yaliang and Deng, Shuiguang},
  journal={arXiv preprint arXiv:2407.08583},
  year={2024}
}
```


---

## "Section - Mentioned Papers" Retrieval List
We provide a collapsible list of back reference, allowing readers to see which (sub)section mention the papers from the table above. 
The collapsible list of back reference will be periodically updated based on the content of the table and our paper.

<details>
<summary>Sec. 3.1  For Scaling of MLLMs: Larger Datasets</summary>

- No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance
- Training Compute-Optimal Large Language Models

<details>
<summary>Sec. 3.1.1  Data Acquisition</summary>

![](https://img.shields.io/badge/Data4Model--Scaling--Up--Acquisition-f1db9d)

- No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance
- GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting
- Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation
- Audio Retrieval with WavText5K and CLAP Training
- DataComp: In search of the next generation of multimodal datasets
- Learning Transferable Visual Models From Natural Language Supervision
- NExT-GPT: Any-to-Any Multimodal LLM
- ChartThinker: A Contextual Chain-of-Thought Approach to Optimized Chart Summarization
- ChartReformer: Natural Language-Driven Chart Image Editing
- Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation
- Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models
- StructChart: Perception, Structuring, Reasoning for Visual Chart Understanding
- MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning
- List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs
- TextSquare: Scaling up Text-Centric Visual Instruction Tuning
- ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction
- TextHawk: Exploring Efficient Fine-Grained Perception of Multimodal Large Language Models
- BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs
- ShareGPT4V: Improving Large Multi-Modal Models with Better Captions
- UIClip: A Data-driven Model for Assessing User Interface Design
- EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought
- Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering
- Visual Instruction Tuning
- ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model
- Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding
- Probing Multimodal LLMs as World Models for Driving
- Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator
- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
- MiniCPM-V: A GPT-4V Level MLLM on Your Phone
- Descriptive Caption Enhancement with Visual Specialists for Multimodal Perception
- Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning
- LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models
</details>

<details>
<summary>Sec. 3.1.2  Data Augmentation</summary>

![](https://img.shields.io/badge/Data4Model--Scaling--Up--Augmentation-f1db9d)

- Improved Baselines for Data-efficient Perceptual Augmentation of LLMs
- mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration
- Improving Composed Image Retrieval via Contrastive Learning with Scaling Positives and Negatives
- mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding
- CapsFusion: Rethinking Image-Text Data at Scale
- Improving CLIP Training with Language Rewrites
- Data Augmentation for Text-based Person Retrieval Using Large Language Models
- Aligning Actions and Walking to LLM-Generated Textual Descriptions
- GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction
</details>

<details>
<summary>Sec. 3.1.3  Data Diversity</summary>

![](https://img.shields.io/badge/Data4Model--Scaling--Up--Diversity-f1db9d)

- No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance
- Audio Retrieval with WavText5K and CLAP Training
- DataComp: In search of the next generation of multimodal datasets
- Flamingo: a visual language model for few-shot learning
- Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation
- ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning
- Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models
- PaLM-E: An Embodied Multimodal Language Model
- SPHINX-X: Scaling Data and Parameters for a Family of Multi-modal Large Language Models
- Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs
- MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct
</details>

</details>

<details>
<summary>Sec. 3.2  For Scaling Effectiveness of MLLMs: Better Subsets</summary>

- No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance
- DataComp: In search of the next generation of multimodal datasets

<details>
<summary>Sec. 3.2.1  Data Condensation</summary>

![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Condensation-f1db9d)
- The Devil is in the Details: A Deep Dive into the Rabbit Hole of Data Filtering
- DataComp: In search of the next generation of multimodal datasets
- Beyond neural scaling laws: beating power law scaling via data pruning
- Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters
- Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training
- Efficient Multimodal Learning from Data-centric Perspective
- Multimodal Data Curation via Object Detection and Filter Ensembles
- Sieve: Multimodal Dataset Pruning Using Image Captioning Models
- Towards a statistical theory of data selection under weak supervision
- Data Filtering Networks
- T-MARS: Improving Visual Representations by Circumventing Text Feature Learning
- InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4
- Semdedup: Data-efficient learning at web-scale through semantic deduplication
- On the De-duplication of LAION-2B
- Improving Multimodal Datasets with Image Captioning
- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
</details>

<details>
<summary>Sec. 3.2.2  Data Mixture</summary>

![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Mixture-f1db9d)

- Learning Transferable Visual Models From Natural Language Supervision
- Flamingo: a visual language model for few-shot learning
- Quality not quantity: On the interaction between dataset design and robustness of clip
- List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs
- A Decade's Battle on Dataset Bias: Are We There Yet?
- Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding
- Demystifying CLIP Data
</details>

<details>
<summary>Sec. 3.2.3  Data Packing</summary>

![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--Packing-f1db9d)

- Structured Packing in LLM Training Improves Long Context Utilization
- Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models
- MoDE: CLIP Data Experts via Clustering
- Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution
- Fewer Truncations Improve Language Modeling
</details>

<details>
<summary>Sec. 3.2.4 Cross-Modal Alignment</summary>

![](https://img.shields.io/badge/Data4Model--Scaling--Effectiveness--CrossModalAlignment-f1db9d)

- No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance
- DataComp: In search of the next generation of multimodal datasets
- Multimodal Data Curation via Object Detection and Filter Ensembles
- Sieve: Multimodal Dataset Pruning Using Image Captioning Models
- ChartThinker: A Contextual Chain-of-Thought Approach to Optimized Chart Summarization
- Data Filtering Networks
- T-MARS: Improving Visual Representations by Circumventing Text Feature Learning
- Text-centric Alignment for Multi-Modality Learning
- Noisy Correspondence Learning with Meta Similarity Correction
- ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model
- Semdedup: Data-efficient learning at web-scale through semantic deduplication
- Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets
- AlignGPT: Multi-modal Large Language Models with Adaptive Alignment Capability
- Improving Multimodal Datasets with Image Captioning
- Imagebind: One embedding space to bind them all
- UniBind: LLM-Augmented Unified and Balanced Representation Space to Bind Them All
- FreeBind: Free Lunch in Unified Multimodal Space via Knowledge Fusion
- LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment
- Binding Touch to Everything: Learning Unified Multimodal Tactile Representations
</details>

</details>



<details>
<summary>Sec. 4.1  For Instruction Responsiveness of MLLMs</summary>

- ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model


<details>
<summary>Sec. 4.1.1  Prompt Design</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Following--Prompt-d3f0aa)

- Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic
- Kosmos-2: Grounding Multimodal Large Language Models to the World
- Draw-and-Understand: Leveraging Visual Prompts to Enable MLLMs to Comprehend What You Want
- Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation
- ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model
- Time-LLM: Time Series Forecasting by Reprogramming Large Language Models|
- Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V
- Scaling Laws of Synthetic Images for Model Training ... for Now
</details>

<details>
<summary>Sec. 4.1.2  ICL Data</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Following--ICL-d3f0aa)

- GroundingGPT: Language Enhanced Multi-modal Grounding Model
- List Items One by One: A New Data Source and Learning Paradigm for Multimodal LLMs
- Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models
- All in an Aggregated Image for In-Image Learning
- AIM: Let Any Multi-modal Large Language Models Embrace Efficient In-Context Learning
- MMICL: Empowering Vision-language Model with Multi-Modal In-Context Learning
</details>


<details>
<summary>Sec. 4.1.3  Human-Behavior Alignment Data</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Following--HumanBehavior-d3f0aa)

- Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation
- MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria
- ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model
- LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark
- Automated Multi-level Preference for MLLMs
- Silkie: Preference distillation for large visual language models
- Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning
- Aligning Large Multimodal Models with Factually Augmented RLHF
- DRESS: Instructing Large Vision-Language Models to Align and Interact with Humans via Natural Language Feedback
- RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback
</details>

</details>


<details>
<summary>Sec. 4.2  For Reasoning Ability of MLLMs</summary>


<details>
<summary>Sec. 4.2.1  Data for Single-Hop Reasoning</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--SingleHop-d3f0aa)
- FunQA: Towards Surprising Video ComprehensionFunQA: Towards Surprising Video Comprehension
- StructChart: Perception, Structuring, Reasoning for Visual Chart Understanding
- What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models
</details>

<details>
<summary>Sec. 4.2.2  Data for Multi-Hop Reasoning</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Reasoning--MultiHop-d3f0aa)

- MedThink: Explaining Medical Visual Question Answering via Multimodal Decision-Making Rationale
- Grounding-Prompter: Prompting LLM with Multimodal Information for Temporal Sentence Grounding in Long Videos
- Language-Image Models with 3D Understanding
- DDCoT: Duty-Distinct Chain-of-Thought Prompting for Multimodal Reasoning in Language Models
- EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought
- Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering
- Retrieval-augmented Multi-modal Chain-of-Thoughts Reasoning for Large Language Models
</details>

</details>



<details>
<summary>Sec. 4.3  For Ethics of MLLMs</summary>

<details>
<summary>Sec. 4.3.1  Data Toxicity</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Toxicity-d3f0aa)
- Med-MMHL: A Multi-Modal Dataset for Detecting Human- and LLM-Generated Misinformation in the Medical Domain
- Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation
- The Wolf Within: Covert Injection of Malice into MLLM Societies via an MLLM Operative
- MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models
- ImgTrojan: Jailbreaking Vision-Language Models with ONE Image
- VL-Trojan: Multimodal Instruction Backdoor Attacks against Autoregressive Visual Language Models
- Jailbreaking GPT-4V via Self-Adversarial Attacks with System Prompts
- On the Adversarial Robustness of Multi-Modal Foundation Models
</details>

<details>
<summary>Sec. 4.3.2  Data Privacy and Intellectual Property</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Ethic--Privacy&IP-d3f0aa)

- ModelGo: A Practical Tool for Machine Learning License Analysis
</details>

</details>




<details>
<summary>Sec. 4.4 For Evaluation of MLLMs</summary>

<details>
<summary>Sec. 4.4.1  Benchmarks for Multi-Modal Understanding</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Eval--Understanding-d3f0aa)

- DataComp: In search of the next generation of multimodal datasets
- 3DBench: A Scalable 3D Benchmark and Instruction-Tuning Dataset
- MVBench: A Comprehensive Multi-modal Video Understanding Benchmark
- SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension
- OneChart: Purify the Chart Structural Extraction via One Auxiliary Token
- MMC: Advancing Multimodal Chart Understanding with Large-scale Instruction Tuning
- ImplicitAVE: An Open-Source Dataset and Multimodal LLMs Benchmark for Implicit Attribute Value Extraction
- UNIAA: A Unified Multi-modal Image Aesthetic Data AugmentationAssessment Baseline and Benchmark
- M3DBench: Let’s Instruct Large Models with Multi-modal 3D Prompts
- Open-TransMind: A New Baseline and Benchmark for 1st Foundation Model Challenge of Intelligent Transportation
- BLINK: Multimodal Large Language Models Can See but Not Perceive
- LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark
</details>

<details>
<summary>Sec. 4.4.2 Benchmarks for Multi-Modal Generation</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Eval--Generation-d3f0aa)
- VBench: Comprehensive Benchmark Suite for Video Generative Models
- EvalCraftr: Benchmarking and Evaluating Large Video Generation Models
- Perception Test: A Diagnostic Benchmark for Multimodal Video Models
- WorldGPT: Empowering LLM as Multimodal World Model
- MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria
- MM-SafetyBench: A Benchmark for Safety Evaluation of Multimodal Large Language Models
- OpenLEAF: Open-Domain Interleaved Image-Text Generation and Evaluation
- Visual Hallucinations of Multi-modal Large Language Models
- Unified Hallucination Detection for Multimodal Large Language Models
- MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark
</details>


<details>
<summary>Sec. 4.4.3  Benchmarks for Multi-Modal Retrieval</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Eval--Retrieval-d3f0aa)
- Audio Retrieval with WavText5K and CLAP Training
- MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI
- Open-TransMind: A New Baseline and Benchmark for 1st Foundation Model Challenge of Intelligent Transportation
</details>

<details>
<summary>Sec. 4.4.4  Benchmarks for Multi-Modal Reasoning</summary>

![](https://img.shields.io/badge/Data4Model--Usability--Eval--Reasoning-d3f0aa)

- FunQA: Towards Surprising Video ComprehensionFunQA: Towards Surprising Video Comprehension
- ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning
- ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning
- Probing Multimodal LLMs as World Models for Driving
- M3CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought
</details>

</details>


<details>
<summary>Sec. 5.1  Model as a Data Creator</summary>

![](https://img.shields.io/badge/Model4Data--Synthesis--Creator-b4d4fb)

- What Makes for Good Visual Instructions? Synthesizing Complex Visual Reasoning Instructions for Visual Instruction Tuning
- ChartLlama: A Multimodal LLM for Chart Understanding and Generation
- VideoChat: Chat-Centric Video Understanding
- 3DMIT: 3D Multi-modal Instruction Tuning for Scene Understanding
- Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters
- Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation
- OneChart: Purify the Chart Structural Extraction via One Auxiliary Token
- ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning
- TextSquare: Scaling up Text-Centric Visual Instruction Tuning
- UNIAA: A Unified Multi-modal Image Aesthetic Data AugmentationAssessment Baseline and Benchmark
- Improving Composed Image Retrieval via Contrastive Learning with Scaling Positives and Negatives
- What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models
- EmbodiedGPT: Vision-Language Pre-Training via Embodied Chain of Thought
- AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling
- InstructPix2Pix: Learning to Follow Image Editing Instructions
- Genixer: Empowering Multimodal Large Language Model as a Powerful Data Generator
- World to Code: Multi-modal Data Generation via Self-Instructed Compositional Captioning and Filtering
- Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation
</details>

<details>
<summary>Sec. 5.2 Model as a Data Mapper</summary>

![](https://img.shields.io/badge/Model4Data--Synthesis--Mapper-b4d4fb) 

- VideoChat: Chat-Centric Video Understanding
- Aligned with LLM: a new multi-modal training paradigm for encoding fMRI activity in visual cortex
- GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting
- MedThink: Explaining Medical Visual Question Answering via Multimodal Decision-Making Rationale
- AesExpert: Towards Multi-modality Foundation Model for Image Aesthetics Perception
- BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs
- MoqaGPT: Zero-Shot Multi-modal Open-domain Question Answering with Large Language Model
- Improving CLIP Training with Language Rewrites
- Data Augmentation for Text-based Person Retrieval Using Large Language Models
- Aligning Actions and Walking to LLM-Generated Textual Descriptions
- Unified Hallucination Detection for Multimodal Large Language Models
- PDFChatAnnotator: A Human-LLM Collaborative Multi-Modal Data Annotation Tool for PDF-Format Catalogs
- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
- MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct
- Model-in-the-Loop (MILO): Accelerating Multimodal AI Data Annotation with LLMs
- MiniCPM-V: A GPT-4V Level MLLM on Your Phone
- Descriptive Caption Enhancement with Visual Specialists for Multimodal Perception
- REFINESUMM: Self-Refining MLLM for Generating a Multimodal Summarization Dataset
- Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning
- LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models
</details>

<details>
<summary>Sec. 5.3  Model as a Data Filter</summary>

![](https://img.shields.io/badge/Model4Data--Synthesis--Filter-b4d4fb)
- The Devil is in the Details: A Deep Dive into the Rabbit Hole of Data Filtering
- Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters
- DataComp: In search of the next generation of multimodal datasets
- TextSquare: Scaling up Text-Centric Visual Instruction Tuning
- What If the TV Was Off? Examining Counterfactual Reasoning Abilities of Multi-modal Language Models
- Towards a statistical theory of data selection under weak supervision
- Visual Hallucinations of Multi-modal Large Language Models
- MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models
- FM2DS: Few-Shot Multimodal Multihop Data Synthesis with Knowledge Distillation for Question Answering
</details>

<details>
<summary>Sec. 5.4  Model as a Data Evaluator</summary>

![](https://img.shields.io/badge/Model4Data--Synthesis--Evaluator-b4d4fb)
- Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation
- TextSquare: Scaling up Text-Centric Visual Instruction Tuning
- Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation
- MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria
- MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark
- ImgTrojan: Jailbreaking Vision-Language Models with ONE Image
- ZooProbe: A Data Engine for Evaluating, Exploring, and Evolving Large-scale Training Data for Multimodal LLMs
- FakeShield: Explainable Image Forgery Detection and Localization via Multi-modal Large Language Models
- A Comprehensive Study of Multimodal Large Language Models for Image Quality Assessment
</details>

<details>
<summary>Sec. 6.1  Model as a Data Navigator</summary>

![](https://img.shields.io/badge/Model4Data--Insights--Navigator-f2c0c6)
- How Does the Textual Information Affect the Retrieval of Multimodal In-Context Learning?
</details>

<details>
<summary>Sec. 6.2  Model as a Data Extractor</summary>

![](https://img.shields.io/badge/Model4Data--Insights--Extractor-f2c0c6)
- No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance
- LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition
- Unified Hallucination Detection for Multimodal Large Language Models
- LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition
</details>

<details>
<summary>Sec. 6.3  Model as a Data Analyzer</summary>

![](https://img.shields.io/badge/Model4Data--Insights--Analyzer-f2c0c6)
- ChartLlama: A Multimodal LLM for Chart Understanding and Generation
- OneChart: Purify the Chart Structural Extraction via One Auxiliary Token
- ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning
- StructChart: Perception, Structuring, Reasoning for Visual Chart Understanding
- ChartX & ChartVLM: A Versatile Benchmark and Foundation Model for Complicated Chart Reasoning
- mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding
- mPLUG-DocOwl 1.5: Unified Structure Learning for OCR-free Document Understanding
- mPLUG-PaperOwl: Scientific Diagram Analysis with the Multimodal Large Language Model
- Bridging Research and Readers: A Multi-Modal Automated Academic Papers Interpretation System
</details>

<details>
<summary>Sec. 6.4  Model as a Data Visualizer</summary>

![](https://img.shields.io/badge/Model4Data--Insights--Visualizer-f2c0c6)
- ChartLlama: A Multimodal LLM for Chart Understanding and Generation
- ChartReformer: Natural Language-Driven Chart Image Editing
- Automated Data Visualization from Natural Language via Large Language Models: An Exploratory Study
</details>

<details>
<summary>Sec. 8.1  Data-Model Co-Development Infrastructures</summary>

- DataComp: In search of the next generation of multimodal datasets
</details>

<details>
<summary>Sec. 8.2  Externally-Boosted MLLM Development</summary>

<details>
<summary>Sec. 8.2.1  MLLM-Based Data Discovery</summary> 

- No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance
- ModelGo: A Practical Tool for Machine Learning License Analysis
</details>

<details>
<summary>Sec. 8.2.2  Modality-Compatibility Detection with MLLMs</summary>

- Improving Multimodal Datasets with Image Captioning
</details>

<details>
<summary>Sec. 8.2.3  Automatic Knowledge Transfer for MLLMs</summary> 
- Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation
- MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria
</details>

</details>



<details>
<summary>Sec. 8.3  Self-Boosted MLLM Development</summary>

<details>
<summary>Sec. 8.3.1  Self Data Scaling with MLLMs</summary> 

- Sieve: Multimodal Dataset Pruning Using Image Captioning Models
- ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model
- Segment Anything
</details>

<details>
<summary>Sec. 8.3.2  Self Data Condensation with MLLMs</summary>
</details>

<details>
<summary>Sec. 8.3.3  RL from Self Feedback of MLLMs</summary>

- The Devil is in the Details: A Deep Dive into the Rabbit Hole of Data Filtering
- DataComp: In search of the next generation of multimodal datasets
- Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters
- Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training
- Multimodal Large Language Model is a Human-Aligned Annotator for Text-to-Image Generation
- TextSquare: Scaling up Text-Centric Visual Instruction Tuning
- Multimodal Data Curation via Object Detection and Filter Ensembles
- Sieve: Multimodal Dataset Pruning Using Image Captioning Models
- Data Filtering Networks
- T-MARS: Improving Visual Representations by Circumventing Text Feature Learning
- Semdedup: Data-efficient learning at web-scale through semantic deduplication
- MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark
- CiT: Curation in Training for Effective Vision-Language Data
- Improving Multimodal Datasets with Image Captioning
</details>

</details>

<details>
<summary>Tab. 2</summary>
- No "Zero-Shot" Without Exponential Data: Pretraining Concept Frequency Determines Multimodal Model Performance
- DataComp: In search of the next generation of multimodal datasets
- TextSquare: Scaling up Text-Centric Visual Instruction Tuning
- MLLM-Bench: Evaluating Multimodal LLMs with Per-sample Criteria
- Align and Attend: Multimodal Summarization with Dual Contrastive Losses
- MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?
- BLINK: Multimodal Large Language Models Can See but Not Perceive
- Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering
- ALLaVA: Harnessing GPT4V-synthesized Data for A Lite Vision-Language Model
- LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark
- Unified Hallucination Detection for Multimodal Large Language Models
- Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning
- M3it: A large-scale dataset towards multi-modal multilingual instruction tuning
- MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark
- MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating Large Vision-Language Models Towards Multitask AGI
- M3CoT: A Novel Benchmark for Multi-Domain Multi-step Multi-modal Chain-of-Thought
- Panda-70m: Captioning 70m videos with multiple cross-modality teachers
- Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved With Text
- ChartAssisstant: A Universal Chart Multimodal Language Model via Chart-to-Table Pre-training and Multitask Instruction Tuning
</details>
