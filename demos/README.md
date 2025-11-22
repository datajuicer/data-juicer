# Demos

This folder contains some demos, which allow users to easily experience the basic functions and tools of Data-Juicer.

## Usage

Use `app.py` in the subdirectory of `demos` to run the demos.

```shell
cd <subdir_of_demos>
streamlit run app.py
```

## Available Demos

- Data (`data`)
  - This folder contains some sample datasets.

- Overview scan (`overview_scan`)
  - This demo introduces the basic concepts and functions of Data-Juicer, such as features, configuration, operators, and so on.

- Data process loop (`data_process_loop`)
  - This demo analyzes and processes a dataset, providing a comparison of statistical information before and after the processing.

- Data visualization diversity (`data_visualization_diversity`)
  - This demo analyzes the verb-noun structure of the CFT dataset and plots its diversity in sunburst format.

- Data visualization op effect (`data_visualization_op_effect`)
  - This demo analyzes the statistics of dataset, and displays the effect of each Filter op by setting different thresholds.

- Data visualization statistics (`data_visualization_statistics`)
  - This demo analyzes the dataset and obtain up to 13 statistics.

- Process CFT Chinese data (`process_cft_zh_data`)
  - This demos analyzes and processes part of Chinese dataset in Alpaca-CoT to show how to process IFT or CFT data for LLM fine-tuning.

- Process SCI data (`process_sci_data`)
  - This demos analyzes and processes part of arXiv dataset to show how to process scientific literature data for LLM pre-training.

- Process code data (`process_code_data`)
  - This demos analyzes and processes part of Stack-Exchange dataset to show how to process code data for LLM pre-training.

- Text quality classifier (`tool_quality_classifier`)
  - This demo provides 3 text quality classifier to score the dataset.

- Dataset splitting by language (`tool_dataset_splitting_by_language`)
  - This demo splits a dataset to different sub-datasets by language.

- Data mixture (`data_mixture`)
  - This demo selects and mixes samples from multiple datasets and exports them into a new dataset.

- Partition and checkpoint (`partition_and_checkpoint`)
  - This demo showcases distributed processing with partitioning, checkpointing, and event logging. It demonstrates the new job management features including resource-aware partitioning, comprehensive event logging, and the processing snapshot utility for monitoring job progress.
