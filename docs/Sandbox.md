# Sandbox

## User Guide


### What is DJ-Sandbox?

In Data-Juicer, the DJ-Sandbox is a middleware that links data and model feedback, enabling high performance and low-cost verification across a wide range of tasks. It aims to provide users with the best practices for continuously enhancing data-model recipes, featuring low overhead, portability, and guidance. In the sandbox, users can quickly experiment, iterate, and refine recipes based on small-scale datasets and models before scaling up to produce high-quality data to serve large-scale models.

In addition to the basic data optimization and recipe refinement features offered by Data-Juicer, users can seamlessly use configurable components such as data probing and analysis, model training and evaluation, and data and model feedback-based recipe refinement to form preferred pipelines for data-model research and development.

For more detailed information, please refer to our [paper](http://arxiv.org/abs/2407.11784) (ICML'25 spotlight).

### Applications and Use-Cases
We apply the sandbox to many cutting-edge models, such as Mini-Gemini and InternVL-2.0 (two LLaVA-inspired models for image-to-text generation), EasyAnimate and T2V-Turbo (two Diffusion Transformer-based models for text-to-video generation), and a CLIP model for image-text pre-training. Among these, we have secured a new leading position on the [VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) text-to-video leaderboard.
![top-1_in_vbench](https://img.alicdn.com/imgextra/i1/O1CN01I9wHW91UNnX9wtCWu_!!6000000002506-2-tps-1275-668.png)

The model is now publicly available on the ModelScope and HuggingFace platforms, and the training dataset has also been available.

| Open-source model or dataset | Link | Description |
| ------------ | --- | --- |
| Data-Juicer (T2V, 147k) |  [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V) | Corresponding to Data-Juicer (T2V-Turbo) model in VBench leaderboard |
| Data-Juicer (DJ, 228k) | [ModelScope](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V-v2) <br> [HuggingFace](https://huggingface.co/datajuicer/Data-Juicer-T2V-v2) | Corresponding to Data-Juicer (2024-09-23, T2V-Turbo) model in VBench leaderboard |
| data_juicer_t2v_optimal_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-optimal-data-pool)  <br> [HuggingFace](https://huggingface.co/datasets/datajuicer/data-juicer-t2v-optimal-data-pool) | The training dataset of Data-Juicer (T2V, 147k) |
| data_juicer_t2v_evolution_data_pool | [Aliyun](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_optimal_data_pool_s2.zip) <br> [ModelScope](https://modelscope.cn/datasets/Data-Juicer/data-juicer-t2v-evolution-data-pool) | The training dataset of Data-Juicer (2024-09-23, T2V-Turbo) |

Following is the case study for Data-Juicer (DJ, 228k) outputs.
  | Prompt | Generated Video |
  | --- | --- |
  | A beautiful coastal beach in spring, waves lapping on sand, zoom out | [![Case 0](https://img.alicdn.com/imgextra/i1/O1CN01KuJeOE1Ylqnk9zYkc_!!6000000003100-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case0.mp4) |
  | a boat accelerating to gain speed | [![Case 1](https://img.alicdn.com/imgextra/i2/O1CN01i1iMFE1TKlIUlqE8d_!!6000000002364-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case1.mp4) |
  | A boat sailing leisurely along the Seine River with the Eiffel Tower in background by Hokusai, in the style of Ukiyo | [![Case 2](https://img.alicdn.com/imgextra/i2/O1CN01u2cjJE1RBwRFeCFuo_!!6000000002074-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case2.mp4) |
  | a bottle on the left of a wine glass, front view | [![Case 3](https://img.alicdn.com/imgextra/i4/O1CN01vdMm6Q1xWc1CoJZW6_!!6000000006451-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case3.mp4) |
  | A corgi's head depicted as an explosion of a nebula | [![Case 4](https://img.alicdn.com/imgextra/i2/O1CN014oPB8Q1IrJg0AbUUg_!!6000000000946-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case4.mp4) |
  | A graceful ballerina doing a pirouette on a dimly lit stage, with soft spotlight highlighting her movements. | [![Case 5](https://img.alicdn.com/imgextra/i4/O1CN01yNlsVu1ymvkJgkvY8_!!6000000006622-2-tps-2048-320.png)](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/show_cases/case5.mp4) |

To reproduce the paper's experiments, please refer to the sandbox usage guide below, the experimental process in the following figure, the [initial dataset](http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/MM_data/our_refined_data/Data-Juicer-T2V/data_juicer_t2v_init_data_pool.zip), and the configuration file demos for the process: [1_single_op_pipeline.yaml](../configs/data_juicer_recipes/sandbox/easyanimate_text_to_video/1_single_op_pipeline.yaml), [2_multi_op_pipeline.yaml](../configs/data_juicer_recipes/sandbox/easyanimate_text_to_video/2_multi_op_pipeline.yaml), [3_duplicate_pipeline.yaml](../configs/data_juicer_recipes/sandbox/easyanimate_text_to_video/3_duplicate_pipeline.yaml).
![bench_bottom_up](https://img.alicdn.com/imgextra/i2/O1CN01xvu2fo1HU80biR6Q5_!!6000000000760-2-tps-7756-3693.png)


### Quick Start

#### Requirements

Before using sandbox, you might need to install sandbox-related dependencies by running the command below:
```shell
uv pip install -v -e .[sandbox]
```
And prepare third-party libraries used in sandbox (e.g., EasyAnimate, VBench, InternVL, etc.) according to their official instructions, or you can simply clone the third-party repositories from GitHub and leave the installation process to our `EnvManager` during sandbox running.

**NOTICE**: some sandbox-related dependencies require extra domain dependencies. 

1. To use [ModelScope](https://github.com/modelscope/modelscope), you need to install the related dependencies from its independent host:
```shell
uv pip install "modelscope[framework,nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
Please refer the the [ModelScope documentation](https://www.modelscope.cn/docs/Beginner-s-Guide/Environment-Setup) for more information.


2. To use [EasyAnimate](https://github.com/aigc-apps/EasyAnimate), you need to execute the following installation script:
```shell
cd thirdparty/models/
bash setup_easyanimate.sh
cd ../../
```

If some Module-Not-Found errors are raised by these third-party libraries when running the sandbox, users need to check their docs first.

#### Prepare Configuration Files for Sandbox

The sandbox will sequentially execute four types of jobs: Data/Model Probe (`probe_job_configs`), Iterative Recipe Refinement based on Probe Results(`refine_recipe_job_configs`), Dataset Processing and Model Training (`execution_job_configs`) and Data/Model Evaluation (`evaluation_job_configs`). Within each category of jobs, jobs are carried out in the order specified by the configured job list. Each task requires specifying: the hook for mounting this job (`hook`), the tag name for identifying the hook (`meta_name`), Data-Juicer data processing parameters (`dj_configs`), as well as other specific parameters for the job (`extra_configs`). Among these parameters, hook is required, while others may be left empty. dj_configs can refer to the full Data-Juicer data processing parameters available in [config_all.yaml](https://github.com/datajuicer/data-juicer/blob/main/configs/config_all.yaml). The `extra_configs` are task-specific parameters without restrictions. They can include parameters for model training, inference, evaluation, etc. For example, `path_k_sigma_recipe` can be used to specify the path for saving the data recipe refined using the k-sigma method. An example of a sandbox configuration file can be found at `configs/demo/sandbox/sandbox.yaml`:

```yaml
# Sandbox config example

# global parameters
project_name: 'demo-sandbox'
experiment_name: 'demo-sandbox-run0'              # for wandb tracer name
hpo_config: null                                  # path to a configuration file when using auto-HPO tool.

# configs for each job, the jobs will be executed according to the order in the list
probe_job_configs:
  - hook: 'ProbeViaAnalyzerHook'
    meta_name: 'analysis_ori_data'
    dj_configs: 'configs/demo/process.yaml'
    extra_configs:

refine_recipe_job_configs:
  - hook: 'RefineRecipeViaKSigmaHook'
    meta_name: 'analysis_ori_data'
    dj_configs: 'configs/demo/process.yaml'
    extra_configs:
      path_k_sigma_recipe: './outputs/demo-process/k_sigma_new_recipe.yaml'

execution_job_configs:
  - hook: 'ProcessDataHook'
    meta_name:
    dj_configs: './outputs/demo-process/k_sigma_new_recipe.yaml'
    extra_configs:
  - hook: 'TrainModelHook'
    meta_name:
    dj_configs:
    extra_configs: 'configs/demo/sandbox/gpt3_extra_train_config.json'

evaluation_job_configs:
  - hook: 'ProbeViaAnalyzerHook'
    meta_name: 'analysis_processed_data'
    dj_configs: 'configs/demo/process.yaml'
    extra_configs:
  - hook: 'EvaluateDataHook'
    meta_name: 'eval_data'
    dj_configs:
    extra_configs: 'configs/demo/sandbox/gpt3_data_quality_eval_config.yaml'
```
Based on this configuration file, sandbox:

1. Execute the Data-Juicer data analysis function to calculate specified metrics for each piece of data, for example, in `configs/demo/process.yaml`, the `language_id_score_filter` is designated to calculate language scores.

2. With the results from Data-Juicer data analysis, fine-tune the data recipe using the k-sigma method. Note that the `meta_name` here is set the same as the `meta_name` used during data analysis to query the results from W&B.

3. Execute Data-Juicer's data filtering function with the data recipe fine-tuned by the k-sigma method.

4. Train the model with the filtered data.

5. Analyze the data after filtering.

6. Score the data after filtering with a scorer.

When there are multiple pipelines needed in your config file, you can name each pipeline and organize them in a `pipelines` field:

```yaml
# Sandbox config example

# global parameters
project_name: 'demo-sandbox'
experiment_name: 'demo-sandbox-run0'              # for wandb tracer name
hpo_config: null                                  # path to a configuration file when using auto-HPO tool.

pipelines:
  pipeline_1:
    probe_job_configs:
      xxx
  pipeline_2:
    probe_job_configs:
      xxx
    refine_recipe_job_configs:
      xxx
  pipeline_3:
    probe_job_configs:
      xxx
    execution_job_configs:
      xxx
```

In this example, there are 3 pipelines organized in the `pipelines` field, named `pipeline_1`, `pipeline_2`, and `pipeline_3`. Each of them has their own different types of jobs. You can find a practical example of such config file for InternVL sandbox experiments in `configs/data_juicer_recipes/sandbox/internvl_coco_caption/sandbox_internvl_coco_caption.yaml`.

For the single-pipeline format, the only pipeline is named "anonymous" in default.

> [!Important]
> 
> The single pipeline format without `pipelines` field and the multi-pipeline format with `pipelines` field are both supported but can not be used at the same time.

#### Start Sandbox

The entry point for running the sandbox is `tools/sandbox_starter.py`. The usage is similar to the data processing and analysis tool, requiring specifying the sandbox configuration file:

```yaml
python tools/sandbox_starter.py --config configs/demo/sandbox/sandbox.yaml
```

Once the run is started, the sandbox will sequentially execute each of the predefined pipeline steps according to the configuration file. The default one trial of the pipeline mainly includes four major steps:

1. **Data/Model Probe**: This step provides probes into the input dataset/model, such as analysing the dataset or analysing the data produced by model inference, to guide the subsequent steps.
2. **Iterative Recipe Refinement based on Probe Results**: This step refines and optimizes the recipe hyperparameters based on the data/model probes. For example, the operator (OP) hyperparameters in the data recipe can be adjusted using the k-sigma method based on the data probes.
3. **Dataset Processing and Model Training**: This step processes and cleans the input dataset based on the refined recipe. If model training is configured in the sandbox, the processed dataset will also be used to train the configured model.
4. **Data/Model Evaluation**: This step evaluates the processed dataset and the model trained in the previous step (if applicable). The evaluation methods may include analysis of the processed dataset and specified benchmark evaluations based on the configuration.

Once this completes one trial of the sandbox pipeline run, the user can validate the effectiveness of the experiment in data production by comparing the probes and evaluation results before and after recipe refinement and dataset processing.

If the `hpo_config` is set in the configuration file and appropriate optimization objectives and OP hyperparameters to be refined are configured within it, the sandbox will perform multiple trials of pipeline runs in the form of Hyperparameter Optimization (HPO) and automatically conduct iterative refinement and optimization of the operator hyperparameters. The preparation of this configuration file can be referenced from the [HPO tool](https://github.com/datajuicer/data-juicer/tree/main/tools/hpo).

### Component Factory

In a single trial of the sandbox pipeline, four major steps involve various configurable components. Each of these components corresponds to a factory class used to initialize them:

- **Data Processing (DataExecutor)**: Executor for dataset processing, i.e., the Executor of Data-Juicer
- **Data Pool Manipulator (DataPoolManipulator)**: Manipulator for data pools, i.e., construction, combination
- **General Data Processing (GeneralDataExecutor)**: General executor for dataset processing, i.e., dataset format conversion
- **Data Analyzing（DataAnalyzer）**: Analyzer for dataset, i.e., the analyzer of Data-Juicer
- **Data Evaluation (DataEvaluator)**: Evaluator on the quality of the dataset
- **General Data Probe (GeneralProbe)**: General probe components for the dataset
- **Model-Data Evaluation (ModelInferEvaluator)**: Evaluator of dataset quality using the model's inference results
- **Model Training (ModelTrainExecutor)**: Executor for model training
- **Model Inference (ModelInferExecutor)**: Executor for model inference
- **Model Evaluation (ModelEvaluator)**: Evaluator on the performance of the model

Except for DataExecutor and DataAnalyzer, the rest of the components can be specified in the configuration file using the `type` parameter to choose a specific execution or evaluation type. For example, the data evaluation component supports a `type` of `"dj_text_quality_classifier"` to utilize Data-Juicer's text quality classifier tool for evaluating the dataset, while the model training component `type` can be set to `"modelscope"` to train a model from the ModelScope platform.

The currently supported component factories and the components supported within each factory are as follows:

- DataExecutorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `DJExecutor` | The data process module of Data-Juicer | - | - |

- DataPoolManipulatorFactory

| Component                | Function                                                  | Desc. of Method `run` | Reference Materials                               |
|--------------------------|-----------------------------------------------------------|-----------------------|---------------------------------------------------|
| `DataPoolConstruction`   | Construct data pool from specified analyzed data source   | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolCombination`    | Combine specified data pools                              | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolDuplication`    | Duplicate a data pool for specified times                 | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolDownsampling`   | Randomly downsample data pools to specified scale         | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolRanking`        | Rank data pools according to specified evaluation metrics | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolMerging`        | Merge data pools into one dataset or data pool            | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |
| `DataPoolCartesianJoin`  | Join two sets of data pools with Cartesian Join           | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |


- GeneralDataExecutorFactory

| Component                   | Function                                                     | Desc. of Method `run` | Reference Materials                                                                                     |
|-----------------------------|--------------------------------------------------------------|-----------------------|---------------------------------------------------------------------------------------------------------|
| `COCOCaptionToDJConversion` | Convert InternVL COCO Caption datasets to DJ format          | -                     | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html) |
| `COCOCaptionMetaGeneration` | Generate meta file for InternVL COCO Caption datasets        | -                     | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html) |

- DataAnalyzerFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `DJAnalyzer` | The data analysis module of Data-Juicer | - | - |

- DataEvaluatorFactory

| Component | Function                                                                                             | Desc. of Method `run` | Reference Materials |
| --- |------------------------------------------------------------------------------------------------------| --- | --- |
| `Gpt3QualityEvaluator` | Evaluate the quality of a dataset using the GPT-3 text quality classifier reproduced by Data-Juicer. | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`.<br />- `eval_obj`: A useless parameter.<br /> | [Data-Juicer Quality Classifier Toolkit](https://github.com/datajuicer/data-juicer/tree/main/tools/quality_classifier) |
| `VBenchEvaluator` | Evaluate the generated videos according to given prompts in multi dimensions                         | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter.<br />- Return: The average score of generated videos in multi dimensions.<br /> | [VBench paper](https://arxiv.org/abs/2311.17982) |
| `InceptionEvaluator` | Evaluate the generated videos by features extracted from video classification models.                | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter.<br />- Return: A dictionary of scores in the given metric. <br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |
| `AccuracyEvaluator` | Evaluate the accuracy to compare the labels in the predicted ones and ground truth                   | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter.<br />- Return: A dictionary of scores in the given metric. <br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |
| `MSEEvaluator` | Evaluate the MSE score between the predicted values and ground truth.                | <br />- `eval_type`: The type of the object to be evaluated by the evaluator, currently only supports `"data"`<br />- `eval_obj`: A useless parameter.<br />- Return: A dictionary of scores in the given metric. <br /> | [Inception Metrics](https://github.com/NVlabs/long-video-gan/tree/main/metrics) |

- GeneralProbeFactory

| Component         | Function                                                    | Desc. of Method `run` | Reference Materials                               |
|-------------------|-------------------------------------------------------------|-----------------------|---------------------------------------------------|
| `DataPoolRanking` | Rank data pools according to specified evaluation metrics   | -                     | [Sandbox Paper](https://arxiv.org/abs/2407.11784) |

- ModelInferEvaluatorFactory

| Component | Function | Desc. of Method `run` | Reference Materials |
| --- | --- | --- | --- |
| `ModelscopeInferExecutor` | Perform inference on a model from the ModelScope platform using a specified sampled dataset, and return the inference results. | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"modelscope"` in the component configuration file to activate this component.<br />- `run_obj`: Sampled dataset to be fed into model inference.<br /> | [ModelScope Docs of Model Inference](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%8E%A8%E7%90%86Pipeline) |

- ModelTrainExecutorFactory

| Component                          | Function                                                                                                                           | Desc. of Method `run`                                                                                                                                                                                   | Reference Materials                                                                                                |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| `ModelscopeTrainExecutor`          | Perform a training task on a model from the ModelScope platform using specified datasets, and monitor the change in training loss. | <br />- `run_type`: Type of model training. We need to set `type`  arg as `"modelscope"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br />  | [ModelScope Docs of Model Training](https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83Train) |
| `EasyAnimateTrainExecutor`         | Perform a LoRA training task on EasyAnimate text-to-video model, and monitor the change in training loss.                          | <br />- `run_type`: Type of model training. We need to set `type`  arg as `"easyanimate"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate)                                                            |
| `TrinityRFTTrainExecutor`          | Perform a reinforcement fine-tuning task based on Trinity-RFT framework, and monitor the change in training states.                | - `run_obj`: Could be the path to Trinity configs.                                                                                                                                                      | [Trinity-RFT](https://github.com/modelscope/Trinity-RFT)                                                           |
| `InternVLCOCOCaptionTrainExecutor` | Perform a LoRA fine-tuning task on InternVL2 for COCO Caption task, and monitor the change in training loss and learning rate.     | -                                                                                                                                                                                                       | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html)            |


- ModelInferExecutorFactory

| Component | Function                                                                                                          | Desc. of Method `run` | Reference Materials                                     |
| --- |-------------------------------------------------------------------------------------------------------------------| --- |---------------------------------------------------------|
| `EasyAnimateInferExecutor` | Perform inference on EasyAnimate text-to-video model with the prompts from VBench, and save the generated videos. | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"easyanimate"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) |
| `HFTransformersInferExecutor` | Perform inference with HuggingFace Transformers.                                                                  | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"huggingface"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | -                                                       |
| `VLLMInferExecutor` | Perform inference with vLLM.                                                                                      | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"vllm"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | -                                                       |
| `APIModelInferExecutor` | Perform inference with OpenAI API.                                                                                | <br />- `run_type`: Type of model inference. We need to set `type`  arg as `"api"` in the component configuration file to activate this component.<br />- `run_obj`: A useless parameter.<br /> | -                                                       |

- ModelEvaluatorFactory

| Component                      | Function                                                                         | Desc. of Method `run` | Reference Materials                                                                                                                     |
|--------------------------------|----------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `InternVLCOCOCaptionEvaluator` | Evaluate Bleu-1/2/3/4, METEOR, ROUGE_L, and CIDEr for InternVL COCO Caption task | -                     | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#evaluating-the-fine-tuned-model) |

Please refer to `data_juicer/core/sandbox/factories.py` for detailed definitions.

### Context Sharing

Sometimes, information needs to be shared between different hooks. For example, the `TrainModelHook` needs to share the model checkpoint path with the `EvaluateModelHook` to perform evaluation on the model after training.

To achieve this, we integrate a global, both cross-hook and cross-pipeline information container called `context_infos` to store the executed results of hooks in each pipeline. The `context_infos` are constructed automatically when the pipeline is executed.

#### Resume Mode

After each pipeline is finished, the `context_infos` are saved on disk. Based on this, Sandbox allows a pipeline-level resume mode to continue the sandbox execution from the last pipeline checkpoint. All we need to do is to set the `resume` parameter in the sandbox configuration file to `true`.

#### Input, Output, and Local Parameter of Hooks

To obtain the context infos, three new parameters are added to the hook:
- `input`: used to obtain results from the previous hook stored in the context infos, and update it in the configuration of the current hook.
  - Basic usage: `<key_to_updated>: <key_in_context_infos>`, where `<key_to_updated>` is the key in the configuration of the current hook to be updated, and `<key_in_context_infos>` is the key of the previous result stored in the context infos.
  - Each pair of input parameters would replace the values in the `<key_to_updated>` with the values in the context infos with key `<key_in_context_infos>`. Nested keys using dot (`.`) notation are supported for both of them.
  - If only the result from the last hook is needed, a simple `-1` can be used as the hook key in the `<key_in_context_infos>`.
- `output`: used to store the results of the current hook in the context infos.
  - Basic usage: `[<res_1_name>, <res_2_name>, ...]`, where `<res_i_name>` represents the `i`th output result of the current hook. If the `output` parameter is not specified, simple "res_i" is used automatically for the `i`th output result.
  - After the hook is executed, the results of the current hook are stored as dict in the context infos with the keys specified in the `output` parameter.
- `local`: used to update the configuration of the current hook locally with specified values.
  - Basic usage: `<key_to_updated>: <value>`, where `<key_to_updated>` is the key in the configuration of the current hook to be updated, and `<value>` is the target value.
  - Each pair of local parameters would replace the values in the `<key_to_updated>` with the values specified in the `<value>`. Nested keys using dot (`.`) notation are supported.

An example of a hook using these parameters is shown below:

```yaml
xxx_hook:
  meta_name: job_xxx
  input:
    dj_configs.dataset_path: pipeline1.name3.res4_key
    extra_configs.meta_paths: -1.res5_key
  output: ['res6_key', 'res7_key']
  local:
    extra_configs.arg2: 42
  dj_configs: '/path/to/dj_configs.yaml'
  extra_configs:
    arg1: "arg1_val"
    arg2: 404
    meta_paths: "<placeholder>"
```

In this hook, it uses all three parameters:
- In `input`, the hook replaces the `dataset_path` in the `dj_configs` in YAML format with the value of the `res4_key` stored in the context infos of the previous hook with `meta_name` "name3" in the pipeline named "pipeline1". Beside, it replace the `meta_paths` in the `extra_configs` with the value of the `res5_key` stored in the context infos of the previous hook specified by "-1".
- In `output`, the hook outputs two results named `res6_key` and `res7_key`, which will be stored in the context infos as following:
```python
{
  'meta_name': 'job_xxx',
  'res6_key': <output_1>,
  'res7_key': <output_2>,
}
```
- In `local`, the hook replaces the original value of `arg2` in the `extra_configs`, which is 404 before, with the target value 42.

## Developer Guide

As mentioned in the previous section, developers can develop customized configurable components and add them to the corresponding factory classes, then route to appropriate instantiation methods using the `type` parameter. Once the components are implemented, developers can encapsulate them as hooks and register the hooks into the job list. After the job list is orchestrated in the pipeline, when the sandbox pipeline is executed, each job in the job list will be executed in sequence at each step. Each of these parts - components, component factory, hooks, job lists, and the registration and execution orchestration of the pipeline - can be customized by the developer. The relationship among these parts is illustrated in the diagram below.
![sandbox-pipeline](https://img.alicdn.com/imgextra/i3/O1CN01ERmGre1uz3luKOn4n_!!6000000006107-2-tps-4655-1918.png)

### The Internal Implementation of Components
Currently, components are mainly divided into three major categories:

- **Executor**: Since the data executor is already handled by the Data-Juicer's Executor, the executor here specifically refers to the model executor, including model training, inference, evaluation, etc. The code is located in `data_juicer/core/sandbox/model_executors.py`.
- **Evaluator**: Used for evaluating the quality and performance of datasets or models. The code is located in `data_juicer/core/sandbox/evaluators.py`.
- **DataPoolManipulator**: Used for manipulating the data pool, such as construction, combination, sampling, etc. The code is located in `data_juicer/core/sandbox/data_pool_manipulators.py`.

#### Executor
The core function of the model executor is to train, infer, or evaluate the model specified in the configuration file with the specified dataset. The model executor needs to inherit from `BaseModelExecutor` and implement several core methods:

- The specific behavior of the model executor (training, inference, evaluation, etc.) needs to be defined in the `_run` method.
- The model executor does not return any value. Key metrics that need to be monitored during execution are usually parsed from the logs produced by the model executor (such as loss, evaluation results, etc.). The parsing and monitoring process needs to be defined by the `_watch_run` method.
- Model executor requires input from a dataset, so the `data_connector` method needs to be implemented to convert the dataset from the sandbox's format to the format required by the model framework or library.

It is important to note that, to monitor the change of training metrics (e.g., loss) promptly during the model training process, logs generated during training need to be monitored. Therefore, both the `_run` method for executing model training and the `watch_run` method for monitoring logs need to be asynchronous methods, indicated by the `async` keyword. In the `run` method, we redirect the standard output stream (stdout) and standard error stream (stderr) to a designated log file before the training starts. Then, we create two asynchronous tasks to execute the aforementioned two methods, each performing the following tasks:

- `_run` method: After loading the dataset, it starts model training based on the model training configuration. Upon completion of training, it outputs a predefined task completion identifier to the standard output stream, which has been redirected to the designated log file.
- `watch_run` method: It monitors the designated log file, reads it line by line, and calls the `_watch_run` method. The called method is customized based on the model training framework and used to parse the latest log content line, extract key metrics, and monitor them until the predefined task completion identifier is read.

#### Evaluator

The core function of the evaluator is to evaluate the quality and performance of the target using some specific methods and return the evaluation result, usually a numerical value. The evaluator needs to inherit from the base class `BaseEvaluator` and implement the `run` method. The `run` method typically takes two required parameters:

- `eval_type`: The type of evaluation, used for internal evaluation type routine within a certain evaluator.
- `eval_obj`: The object to be evaluated.

Users can also extend the usage of these two parameters based on their implementation.

#### DataPoolManipulator

The core function of the data pool manipulator is to manipulate the data pool, such as construction, combination, sampling, etc. The data pool manipulator needs to inherit from the base class `BaseDataPoolManipulator` and implement the `run` method. The `run` method. The necessary parameters usually come from the input data pool configs in the `__init__` method, covering input data pools, export paths, and specific parameters for each type of manipulators.

Users can refer to the doc string of the `run` method of each manipulator for more details in `data_juicer/core/sandbox/data_pool_manipulators.py`.

### Pipeline Hook

As mentioned at the start of this section, in the pipeline, we need to implement several hooks to connect components with the pipeline execution steps through the job list. Activated hooks will be registered in the pipeline's job list and then executed one by one during the pipeline execution at each step. The job lists for the four corresponding steps are as follows:

1. **Data/Model Probe**: Probe job list -- probe_jobs
2. **Iterative Recipe Refinement based on Probe Results**: Refinement job list -- refine_recipe_jobs
3. **Data Processing and Model Training**: Execution job list - execution_jobs
4. **Data/Model Evaluation**: Evaluation job list - evaluation_jobs

In general, we only need to implement one type of hook function for a type of component factory. In addition to hooks that depend on components, some hooks depend on the existing functionality and tools of Data-Juicer or other third-party libraries. The correspondence among these hooks, dependent components, tools, and job lists is as follows:

| Hook | Function                                                                                                      | Dependent Component Factory                          | Dependent Tool or Library                                                                                                                                               | Registered Job List |
| --- |---------------------------------------------------------------------------------------------------------------|------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| `ProbeViaAnalyzerHook` | Analyze and probe the quality and diversity distribution of the dataset                                       | DataAnalyzerFactory                                  | Data-Juicer Analyzer                                                                                                                                                    | - probe_jobs<br />- evaluation_jobs |
| `ProbeViaModelInferHook` | Analyze and understand the impact of the dataset on the model, explore and probe "difficult" and "dirty" data | DataExecutorFactory <br />ModelInferEvaluatorFactory | Data-Juicer Executor                                                                                                                                                    | - probe_jobs<br />- evaluation_jobs |
| `GeneralProbeHook` | General hook for probing the dataset, including ranking the datasets and so on                                | GeneralProbeFactory                                  | -                                                                                                                                                                       | - probe_jobs |
| `RefineRecipeViaKSigmaHook` | Refine data recipe hyperparameters using the k-sigma method based on the probe results of the dataset         | -                                                    | k-sigma recipe refinement tool of Data-Juicer Hyperparameter Optimization (HPO) toolkit                                                                                 | - refine_recipe_jobs |
| `RefineRecipeViaModelFeedbackHook` | Refine data recipe hyperparameters using model probe and feedback results                                     | TODO                                                 | -                                                                                                                                                                       | - refine_recipe_jobs |
| `ProcessDataHook` | Process and clean the dataset based on the current data recipe                                                | DataExecutorFactory                                  | Data-Juicer Executor                                                                                                                                                    | - execution_jobs |
| `DataPoolManipulationHook` | Data pool manipulation,  including construction, combination, sampling, etc.                                  | DataPoolManipulatorFactory                           | -                                                                                                                                                                       | - execution_jobs |
| `GeneralDataExecutorHook` | General data processing for dataset, including format conversion, etc.                                        | GeneralDataExecutorFactory                           | -                                                                                                                                                                       | - execution_jobs |
| `TrainModelHook` | Train a model based on the current dataset                                                                    | ModelTrainExecutorFactory                            | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate) <br/> [InternVL](https://internvl.readthedocs.io/en/latest/index.html)                                                                                                 | - execution_jobs |
| `InferModelHook` | The model generates output based on the given input                                                           | ModelInferExecutorFactory                            | [EasyAnimate](https://github.com/aigc-apps/EasyAnimate)                                                                                                                    | - execution_jobs |
| `EvaluateDataHook` | Evaluate the dataset in terms of data quality and other dimensions                                            | DataEvaluatorFactory                                 | [inception metrics](../tools/mm_eval/inception_metrics/README.md) for images and videos, such as FID and FVD <br /> [VBench](../tools/mm_eval/vbench_metrics/README.md) | - evaluation_jobs |
| `EvaluateModelHook` | Evaluate the trained model                                                                                    | ModelEvaluatorFactory                                | [InternVL COCO Caption](https://internvl.readthedocs.io/en/latest/tutorials/coco_caption_finetune.html#evaluating-the-fine-tuned-model)                                                                                                                                                                       | - evaluation_jobs |

It is worth noting that a hook can be registered in multiple job lists, as this hook can play different roles in different steps of the pipeline. For example, we can analyze and probe both the pre-processed and post-processed datasets to compare the changes in quality, diversity, and other dimensions before and after data processing.

### Customized Sandbox Pipeline
Users can directly modify the job configuration list in the parameter configuration file to achieve task modification and orchestration.

### Watcher
In the above sections, the concept of "monitoring" is repeatedly mentioned. The pipeline will monitor several metrics produced in each step, and these monitoring processes are implemented by `SandboxWatcher`.

`SandboxWatcher` is based on wandb library and mainly includes four methods:

- `setup_sweep`: This method is called in the multi-trial HPO mode, which is supported by the sweep module in wandb library. Therefore, the additional `hpo_config` configuration for sweep initialization is required to be passed into the sandbox configuration file.
- `watch_cfgs`: This method monitors and updates the sandbox experiments and configuration files of various components.
- `watch`: This method monitors a specific metric or experiment result and records it in the wandb log.
- `query`: This method queries a specific metric or experiment result from the wandb log.

### Details of Context Infos

The `context_infos` consists of two levels:

- pipeline level: it's the 1st level of `context_infos`, which is a dict with keys are the pipeline names and values are a list of context information of each job in this pipeline.
- job level: it's the 2nd level of `context_infos`, which is a list of dicts, each dict represents the context information of a specific job, with `meta_name` to identify the job and other key-value pairs with keys are the name of the outputs of this job and values are the output values.

Here is an example of `context_infos`:

```python
{
    'pipeline_0': [
        {
            'meta_name': 'name1',
            'res1_key': 'res1_value',
            'res2_key': <res2_value>,
        },
        {
            'meta_name': 'name2',
            'res3_key': 'res3_value',
        },
    ],
    'pipeline_1': [
        {
            'meta_name': 'name3',
            'res4_key': <res4_value>,
        },
    ],
    'pipeline_2': [
        {
            'meta_name': 'name4',
            'res5_key': ['res5_1_value', 'res5_2_value'],
        },
    ],
}
```

### Environment Manager

Sandbox supports different kinds of third-party libraries for training, evaluation and so on. If we put all of them into
one environment, version conflicts on some important and complex dependencies will occur. Therefore, we provide an 
easy-to-use environment manager to manage different environments for different third-party libraries, allow users to run
commands in isolated environments independently.

The basic class of environment is `Env` in `data_juicer/core/sandbox/env_manager.py` implemented as below:
```python
class Env(ABC):
  
    @abstractmethod
    def create(self):
        """
        Create an environment.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def check_availability(self):
        """
        Check the availability of the environment manager.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def exists(self):
        """
        Check if an environment exists.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def install_py_deps(self):
        """
        Install Python dependencies.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')

    @abstractmethod
    def run_cmd(self):
        """
        Run a command in this environment.
        """
        raise NotImplementedError(
            'This method must be implemented in subclass.')
```

It consists of five main abstract methods:
- `create`: create an environment if it's not existing.
- `check_availability`: check the availability of the environment manager (e.g., `conda`, `venv`).
- `exists`: check if an environment exists.
- `install_py_deps`: install Python dependencies. Usually support three ways: a "requirements.txt" file path, a dependency list, or a directory path to a library code base.
- `run_cmd`: run a command in this environment.

Now we provide two concrete implementations of `Env`:
- `CondaEnv`: use `conda` or `mamba` to manage environments.
- `VirtualEnv`: use `venv`, `virtualenv`, or `uv venv` to manage environments.

When initializing the environment manager, we can specify the environment manager to use by setting the `env_manager` parameter in the configuration file and the name of the environment by setting the `env_name` parameter. An example of the basic usage is as follows:
```python
from data_juicer.core.sandbox.env_manager import ENV_ROUTER

env_manager = 'conda'
env_name = 'new_conda_env'

# create an environment
env =  ENV_ROUTER[env_manager](
  env_name=env_name,
  env_manager=env_manager)
# check the availability
if not env.check_availability():
    # this env manager is not available
    exit()
# create a new env. If it's already existing, use the existing one
env.create()

# install extra dependencies
# use a "requirements.txt" file
env.install_py_deps("/path/to/requirements.txt")
# use a dependency list
env.install_py_deps(["torch", "torchvision"])
# use a directory path to a library code base, e.g., InternVL
env.install_py_deps("/path/to/a/third-party/library")

# run a command in this environment
cmd = "python train.py"
env.run_cmd(cmd)
```

A complete example of using the environment manager in a hook is available in the `InternVLCOCOCaptionEvaluator` class in `data_juicer/core/sandbox/specific_hooks/intervl_coco_captioning/model_hooks.py`.

## Q&A

1. `RuntimeError` when training InternVL:

```text
RuntimeError: 
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
```
- Reason: it might be the reason of incompatibility of CUDA, PyTorch, and bitsandbytes. Run `python -m bitsandbytes` for more details.
- Solution:
  - Remove the version limitation of bitsandbytes in `requirements/internvl_chat.txt` in the home of InternVL to avoid install the error version again when starting the env. Then reinstall it with `pip uninstall bitsandbytes && pip install bitsandbytes`.
  - If the above solution does not work, reinstall the PyTorch that is compatible with the CUDA version of your GPU, and repeat the above step, until the command `python -m bitsandbytes` outputs SUCCESS.
  - Then, the `flash-attn` needs to be reinstalled as well.

2. `AssertionError` when training InternVL:

```text
AssertionError: It is illegal to call Engine.step() inside no_sync context manager
```
- Solution: downgrade the version of `deepspeed` to `0.15.4`, and remove the version limitation of `deepspeed` in both `requirements/internvl_chat.txt` and `pyproject.toml` in the home of InternVL.

3. `java not found` when evaluating InternVL:
- Solution: install java.