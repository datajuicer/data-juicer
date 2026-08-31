文档
====

欢迎来到 Data-Juicer 文档。

- **初次使用？** 请按顺序阅读「快速上手」：安装、跟随快速开始跑通第一个任务，再通过 DJ-Cookbook 探索更多资源。
- **构建自己的菜谱？** 「使用指南」涵盖算子库、数据集配置、导出、缓存与追踪。
- **大规模处理？** 「分布式处理」介绍 Ray 模式、分区与检查点、作业管理。
- **扩展 Data-Juicer？** 「扩展与开发」涵盖算子插件、API 服务与开发者指南。

.. toctree::
   :maxdepth: 2
   :caption: 快速上手

   docs/tutorial/Installation_ZH
   docs/tutorial/QuickStart_ZH
   docs/tutorial/DJ-Cookbook_ZH

.. toctree::
   :maxdepth: 2
   :caption: 使用指南

   docs/ProcessData_ZH
   docs/AnalyzeData_ZH
   docs/Playground_ZH
   docs/GlobalConfig_ZH
   docs/Operators
   docs/DatasetCfg_ZH
   docs/Export_ZH
   docs/Cache_ZH
   docs/Tracing_ZH

.. toctree::
   :maxdepth: 2
   :caption: 分布式处理

   docs/Distributed_ZH
   docs/PartitionAndCheckpoint_ZH
   docs/JobManagement_ZH

.. toctree::
   :maxdepth: 2
   :caption: 扩展与开发

   docs/OperatorPlugins_ZH
   docs/DJ_service_ZH
   docs/DJ_SORA_ZH
   docs/Juicer_ZH
   docs/DeveloperGuide_ZH

.. toctree::
   :maxdepth: 2
   :caption: 资源

   docs/awesome_llm_data
   docs/BadDataExhibition_ZH
   docs/news_zh

.. toctree::
   :maxdepth: 2
   :caption: 算子
   :hidden:
   :glob:

   docs/operators/aggregator/index
   docs/operators/deduplicator/index
   docs/operators/filter/index
   docs/operators/mapper/index
   docs/operators/formatter/index
   docs/operators/grouper/index
   docs/operators/selector/index
   docs/operators/pipeline/index

.. toctree::
   :maxdepth: 2
   :caption: demos
   :glob:

   demos/*
   demos/**/*

.. toctree::
   :maxdepth: 2
   :caption: 工具
   :glob:

   tools/*
   tools/**/*

.. toctree::
   :maxdepth: 2
   :caption: 第三方
   :glob:

   thirdparty/*
   thirdparty/**/*
