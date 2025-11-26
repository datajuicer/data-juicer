# Refine Alpaca-CoT Config Files

The [folder](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/alpaca_cot) in Data-Juicer-Hub contains some configuration files to allow users to easily and quickly refine [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT).

Before starting, please clone the [data-juicer-hub](https://github.com/datajuicer/data-juicer-hub) to local, which contains lots of data processing recipes.
```shell
git clone https://github.com/datajuicer/data-juicer-hub.git
```

## Preprocess

The raw data files can be downloaded from [Alpaca-CoT](https://huggingface.co/datasets/QingyiSi/Alpaca-CoT) on HuggingFace.

### Convert raw Alpaca-CoT data to jsonl
Use [raw_alpaca_cot_merge_add_meta.py](../../tools/preprocess/raw_alpaca_cot_merge_add_meta.py) to select `instruction`, `input` and `output` columns and merge them to `text` field with a space, and add extra [ META ]( #meta_info) info to dataset:

```shell
python tools/preprocess/raw_alpaca_cot_merge_add_meta.py    \
    --src_dir             <Alpaca-CoT_src_dir>              \
    --target_dir          <target_dir>                      \
    --num_proc            <num_proc>
```

### Split datasets to sub-datasets by language
Use [dataset_split_by_language.py](../../tools/preprocess/dataset_split_by_language.py) to split the dataset to EN and ZH sub-datasets:

```shell
python tools/preprocess/dataset_split_by_language.py    \
    --src_dir             <src_dir>                     \
    --target_dir          <target_dir>                  \
    --suffixes            jsonl                         \
    --num_proc            <num_proc>
```

## Process
After preprocess, modify the dataset path in [alpaca-cot-en-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/alpaca_cot/alpaca-cot-en-refine.yaml) and [alpaca-cot-zh-refine.yaml](https://github.com/datajuicer/data-juicer-hub/tree/main/refined_recipes/alpaca_cot/alpaca-cot-zh-refine.yaml), and then execute the following command to reproduce the processing flow of refined Alpaca-CoT.
```shell
# refine English dataset
python tools/process_data.py --config <path-to-data-juicer-hub>/refined_recipes/alpaca_cot/alpaca-cot-en-refine.yaml

# refine Chinese dataset
python tools/process_data.py --config <path-to-data-juicer-hub>/refined_recipes/alpaca_cot/alpaca-cot-zh-refine.yaml
```

### Meta Info <a name="meta_info"/>

Each sample in refined data of Alpaca-CoT contains meta info listed as below:

#### Alpaca-CoT original meta info
* Language Tags:
    - EN: Instruction datasets in English
    - CN: Instruction datasets in Chinese
    - ML: [Multi-lingual] Instruction datasets in multiple languages
* Task Tags
    - MT: [Multi-task] Datasets containing multiple tasks
    - TS: [Task-specific] Datasets tailored for specific tasks
* Generation-method:
    - HG: [Human Generated Dataset] Datasets created by humans
    - SI: [Self-Instruct] Datasets generated using self-instruct methods
    - MIX: [Mixed Dataset] Dataset contains both human and machine generated data
    - COL: [Collection of Dataset] Dataset made from a collection of other datasets

#### Data-Juicer Meta info
* `Dataset`: dataset name in Alpaca-CoT
* `origin_path`: original file path in Alpaca-CoT

* `IFT`: tagged as Instruct Fine-Tuning datasets

* `CFT`: tagged as Chat Fine-Tuning datasets

  * `CFT-SR`: tagged as Single-round Dialog datasets

  * `CFT-MR`: tagged as Multi-round Dialog datasets

  * `CFT-P`: tagged as Preference datasets



#### Refined Alpaca-CoT dataset Meta info
|                      	| Task  	| Gen 	| Lang  	| Dataset              	| IFT 	| CFT-SR 	| CFT-MR 	| CFT-P 	|
|----------------------	|-------	|-----	|-------	|----------------------	|-----	|---------	|---------	|----------------	|
| Chain-of-Thought     	| MT    	| HG  	| EN/CN 	| Chain-of-Thought     	| ✅   	|         	|         	|                	|
| GPT4all              	| MT    	| COL 	| EN    	| GPT4all              	| ✅   	| ✅       	|         	|                	|
| GPTeacher            	| MT    	| SI  	| EN    	| GPTeacher            	|     	| ✅       	|         	|                	|
| Guanaco              	| MT    	| SI  	| ML    	| Guanaco              	|     	| ✅       	|         	|                	|
| HC3                  	| TS    	| MIX 	| EN/CN 	| HC3                  	|     	| ✅       	|         	| ✅              	|
| alpaca               	| MT    	| SI  	| EN    	| alpaca               	|     	| ✅       	|         	|                	|
| Natural-Instructions 	| MT    	| COL 	| ML    	| Natural-Instructions 	| ✅   	|         	|         	|                	|
| belle_cn             	| TS/MT 	| SI  	| CN    	| belle_cn             	|     	| ✅       	|         	|                	|
| instinwild           	| MT    	| SI  	| EN/CN 	| instinwild           	|     	| ✅       	|         	|                	|
| prosocial-dialog     	| TS    	| MIX 	| EN    	| prosocial-dialog     	|     	| ✅       	|         	|                	|
| finance              	| TS    	| COL 	| EN    	| finance              	|     	| ✅       	|         	|                	|
| xP3                  	| MT    	| COL 	| ML    	| xP3                  	| ✅   	|         	|         	|                	|
| firefly              	| MT    	| COL 	| CN    	| firefly              	| ✅   	|         	|         	|                	|
| instruct             	| MT    	| COL 	| EN    	| instruct             	|     	| ✅       	|         	|                	|
| CodeAlpaca           	| TS    	| SI  	| EN    	| CodeAlpaca           	| ✅   	|         	|         	|                	|
| alpacaGPT4           	| MT    	| SI  	| EN/CN 	| alpacaGPT4           	|     	| ✅       	|         	| ✅              	|
| webGPT               	| TS    	| MIX 	| EN    	| webGPT               	| ✅   	|         	|         	| ✅              	|
| dolly                	| TS    	| HG  	| EN    	| dolly                	|     	| ✅       	|         	|                	|
| baize                	| MT    	| COL 	| EN    	| baize                	|     	| ✅       	|         	|                	|
| hh-rlhf              	| TS    	| MIX 	| EN    	| hh-rlhf              	|     	| ✅       	| ✅       	| ✅              	|
| OIG                  	| MT    	| COL 	| EN    	| OIG                  	|     	| ✅       	|         	|                	|
| GAOKAO               	| MT    	| COL 	| CN    	| GAOKAO               	| ✅   	|         	|         	|                	|
| camel                	| MT    	| SI  	| EN    	| camel                	| ✅   	|         	|         	|                	|
| FLAN-Muffin          	| MT    	| COL 	| EN    	| FLAN-Muffin          	| ✅   	|         	|         	|                	|
| COIG                 	| MT    	| COL 	| CN    	| COIG                 	|     	| ✅       	|         	|                	|
| gpt4tools            	| MT    	| SI  	| EN    	| gpt4tools            	| ✅   	|         	|         	|                	|
| ShareGPT             	| MT    	| MIX 	| EN    	| ShareGPT             	|     	| ✅       	| ✅       	|                	|
| Auto-CoT             	| MT    	| COL 	| EN    	| Auto-CoT             	| ✅   	|         	|         	|                	|
| MOSS                 	| TS    	| SI  	| EN/CN 	| MOSS                 	|     	| ✅       	|         	|                	|
| ultrachat            	| TS    	| SI  	| EN    	| ultrachat            	|     	| ✅       	|         	|                	|
| Chinese-medical      	| TS    	| COL 	| CN    	| Chinese-medical      	|     	| ✅       	|         	|                	|
| CSL                  	| MT    	| COL 	| CN    	| CSL                  	| ✅   	|         	|         	|                	|
| pCLUE                	| MT    	| COL 	| CN    	| pCLUE                	| ✅   	|         	|         	|                	|
| news_commentary      	| TS    	| COL 	| CN    	| news_commentary      	| ✅   	|         	|         	|                	|
| StackExchange        	| MT    	| COL 	| EN    	| StackExchange        	|     	| ✅       	|         	| ✅              	|
| ConvAI2              	| TS    	| HG  	| EN    	| ConvAI2              	|     	| ✅       	|         	|                	|
| FastChat             	| MT    	| SI  	| EN    	| FastChat             	|     	| ✅       	|         	|                	|
| Tabular-LLM-Data     	| MT    	| COL 	| EN/CN 	| Tabular-LLM-Data     	| ✅   	|         	|         	|                	|
