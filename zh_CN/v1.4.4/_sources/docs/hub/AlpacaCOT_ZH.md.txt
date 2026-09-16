# Alpaca-CoT 完善配置文件

Data-Juicer-Hub 中的[文件夹](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/alpaca_cot)包含的配置文件能够让用户轻松快速地完善 [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT)。

在开始前，请将 [data-juicer-hub](https://github.com/datajuicer/data-juicer-hub) 仓库克隆到本地，其已经包含了大量的数据处理菜谱。
```shell
git clone https://github.com/datajuicer/data-juicer-hub.git
```

## 预处理

原始数据文件在 HuggingFace 中的 [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) 下载。

### 将 Alpaca-CoT 转换为 jsonl 文件
使用 [raw_alpaca_cot_merge_add_meta.py](../../tools/preprocess/raw_alpaca_cot_merge_add_meta.py) 选择数据集的 `instruction`, `input` 和 `output` 3个字段，并使用空格将它们合并到 `text`，同时在数据集中增加额外的[元信息]( #meta_info) ：

```shell
python tools/preprocess/raw_alpaca_cot_merge_add_meta.py    \
    --src_dir             <Alpaca-CoT_src_dir>              \
    --target_dir          <target_dir>                      \
    --num_proc            <num_proc>
```

### 按照语言将数据集拆分子数据集
使用 [dataset_split_by_language.py](../../tools/preprocess/dataset_split_by_language.py) 将数据集拆分为中文和英文：

```shell
python tools/preprocess/dataset_split_by_language.py    \
    --src_dir             <src_dir>                     \
    --target_dir          <target_dir>                  \
    --suffixes            jsonl                         \
    --num_proc            <num_proc>
```

## 处理
在预处理完成之后，修改 [alpaca-cot-en-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/alpaca_cot/alpaca-cot-en-refine.yaml) 和 [alpaca-cot-zh-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/alpaca_cot/alpaca-cot-zh-refine.yaml) 中的数据集路径，然后执行以下命令来复现完善过的 Alpaca-CoT 的处理流程。

```shell
# refine English dataset
python tools/process_data.py --config <path-to-data-juicer-hub>/refined_recipes/alpaca_cot/alpaca-cot-en-refine.yaml

# refine Chinese dataset
python tools/process_data.py --config <path-to-data-juicer-hub>/refined_recipes/alpaca_cot/alpaca-cot-zh-refine.yaml
```

### 元信息 <a name="meta_info"/>

在完善后的 Alpaca-CoT 的数据集中每个样本都包含元信息，标签说明如下：

#### Alpaca-CoT 元信息
* Language 标签:
    - EN: 英文数据集
    - CN: 中文数据集
    - ML: 多语言数据集
* Task 标签:
    - MT: 多任务数据集
    - TS: 特定任务数据集
* 产生方法:
    - HG: 人工产出数据集
    - SI: 机器产出数据集
    - MIX: 人工和机器混合数据集
    - COL: 从其他数据集合成的数据集

#### Data-Juicer 元信息
* `Dataset`: Alpaca-CoT 中的数据集名称
* `origin_path`: Alpaca-CoT 中的原始文件路径

* `IFT`：标记为指导(Instruct)微调数据集

* `CFT`：标记为聊天(Chat)微调数据集

  * `CFT-SR`：标记为聊天类的单轮对话数据集

  * `CFT-MR`：标记为聊天类的多轮对话数据集

  * `CFT-P`：标记为偏好数据集


#### 完善的 Alpaca-CoT 数据集元信息
|                      	| 任务  	| 产生方法 	| 语言  	| 数据集               	| IFT 	| CFT-SR 	| CFT-MR 	| CFT-P 	|
|----------------------	|-------	|----------	|-------	|----------------------	|-----	|--------	|--------	|--------	|
| Chain-of-Thought     	| MT    	| HG       	| EN/CN 	| Chain-of-Thought     	| ✅   	|        	|        	|        	|
| GPT4all              	| MT    	| COL      	| EN    	| GPT4all              	| ✅   	| ✅      	|        	|        	|
| GPTeacher            	| MT    	| SI       	| EN    	| GPTeacher            	|     	| ✅      	|        	|        	|
| Guanaco              	| MT    	| SI       	| ML    	| Guanaco              	|     	| ✅      	|        	|        	|
| HC3                  	| TS    	| MIX      	| EN/CN 	| HC3                  	|     	| ✅      	|        	| ✅      	|
| alpaca               	| MT    	| SI       	| EN    	| alpaca               	|     	| ✅      	|        	|        	|
| Natural-Instructions 	| MT    	| COL      	| ML    	| Natural-Instructions 	| ✅   	|        	|        	|        	|
| belle_cn             	| TS/MT 	| SI       	| CN    	| belle_cn             	|     	| ✅      	|        	|        	|
| instinwild           	| MT    	| SI       	| EN/CN 	| instinwild           	|     	| ✅      	|        	|        	|
| prosocial-dialog     	| TS    	| MIX      	| EN    	| prosocial-dialog     	|     	| ✅      	|        	|        	|
| finance              	| TS    	| COL      	| EN    	| finance              	|     	| ✅      	|        	|        	|
| xP3                  	| MT    	| COL      	| ML    	| xP3                  	| ✅   	|        	|        	|        	|
| firefly              	| MT    	| COL      	| CN    	| firefly              	| ✅   	|        	|        	|        	|
| instruct             	| MT    	| COL      	| EN    	| instruct             	|     	| ✅      	|        	|        	|
| CodeAlpaca           	| TS    	| SI       	| EN    	| CodeAlpaca           	| ✅   	|        	|        	|        	|
| alpacaGPT4           	| MT    	| SI       	| EN/CN 	| alpacaGPT4           	|     	| ✅      	|        	| ✅      	|
| webGPT               	| TS    	| MIX      	| EN    	| webGPT               	| ✅   	|        	|        	| ✅      	|
| dolly                	| TS    	| HG       	| EN    	| dolly                	|     	| ✅      	|        	|        	|
| baize                	| MT    	| COL      	| EN    	| baize                	|     	| ✅      	|        	|        	|
| hh-rlhf              	| TS    	| MIX      	| EN    	| hh-rlhf              	|     	| ✅      	| ✅      	| ✅      	|
| OIG                  	| MT    	| COL      	| EN    	| OIG                  	|     	| ✅      	|        	|        	|
| GAOKAO               	| MT    	| COL      	| CN    	| GAOKAO               	| ✅   	|        	|        	|        	|
| camel                	| MT    	| SI       	| EN    	| camel                	| ✅   	|        	|        	|        	|
| FLAN-Muffin          	| MT    	| COL      	| EN    	| FLAN-Muffin          	| ✅   	|        	|        	|        	|
| COIG                 	| MT    	| COL      	| CN    	| COIG                 	|     	| ✅      	|        	|        	|
| gpt4tools            	| MT    	| SI       	| EN    	| gpt4tools            	| ✅   	|        	|        	|        	|
| ShareGPT             	| MT    	| MIX      	| EN    	| ShareGPT             	|     	| ✅      	| ✅      	|        	|
| Auto-CoT             	| MT    	| COL      	| EN    	| Auto-CoT             	| ✅   	|        	|        	|        	|
| MOSS                 	| TS    	| SI       	| EN/CN 	| MOSS                 	|     	| ✅      	|        	|        	|
| ultrachat            	| TS    	| SI       	| EN    	| ultrachat            	|     	| ✅      	|        	|        	|
| Chinese-medical      	| TS    	| COL      	| CN    	| Chinese-medical      	|     	| ✅      	|        	|        	|
| CSL                  	| MT    	| COL      	| CN    	| CSL                  	| ✅   	|        	|        	|        	|
| pCLUE                	| MT    	| COL      	| CN    	| pCLUE                	| ✅   	|        	|        	|        	|
| news_commentary      	| TS    	| COL      	| CN    	| news_commentary      	| ✅   	|        	|        	|        	|
| StackExchange        	| MT    	| COL      	| EN    	| StackExchange        	|     	| ✅      	|        	| ✅      	|
| ConvAI2              	| TS    	| HG       	| EN    	| ConvAI2              	|     	| ✅      	|        	|        	|
| FastChat             	| MT    	| SI       	| EN    	| FastChat             	|     	| ✅      	|        	|        	|
| Tabular-LLM-Data     	| MT    	| COL      	| EN/CN 	| Tabular-LLM-Data     	| ✅   	|        	|        	|        	|
