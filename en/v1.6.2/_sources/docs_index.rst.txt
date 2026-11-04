Documentation
=============

Welcome to the Data-Juicer documentation.

- **New to Data-Juicer?** Follow *Getting Started* in order: install it, walk through the Quick Start, then explore more resources in the DJ-Cookbook.
- **Building your own recipes?** The *Guides* cover processing, analysis, dataset configuration, export, caching, and tracing.
- **Looking up configuration or operators?** Use *Lookup Materials* for global options and the operator overview; use the top-level API tab for Python interfaces.
- **Processing at scale?** See *Distributed Processing* for Ray mode, partitioning/checkpointing, and job management.
- **Extending Data-Juicer?** *Extension & Development* covers operator plugins, the API service, and the developer guide.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   docs/tutorial/Installation
   docs/tutorial/QuickStart
   docs/tutorial/DJ-Cookbook

.. toctree::
   :maxdepth: 2
   :caption: Guides

   docs/ProcessData
   docs/AnalyzeData
   docs/Playground
   docs/DatasetCfg
   docs/Export
   docs/Cache
   docs/Tracing

.. toctree::
   :maxdepth: 2
   :caption: Lookup Materials

   docs/GlobalConfig
   docs/Operators

.. toctree::
   :maxdepth: 2
   :caption: Distributed Processing

   docs/Distributed
   docs/PartitionAndCheckpoint
   docs/JobManagement

.. toctree::
   :maxdepth: 2
   :caption: Extension & Development

   docs/OperatorPlugins
   docs/DJ_service
   docs/DJ_SORA
   docs/Juicer
   docs/DeveloperGuide

.. toctree::
   :maxdepth: 2
   :caption: Resources

   docs/awesome_llm_data
   docs/BadDataExhibition
   docs/news

.. toctree::
   :maxdepth: 1
   :caption: operators
   :hidden:

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
   :caption: tools
   :glob:

   tools/*
   tools/**/*

.. toctree::
   :maxdepth: 2
   :caption: thirdparty
   :glob:

   thirdparty/*
   thirdparty/**/*
