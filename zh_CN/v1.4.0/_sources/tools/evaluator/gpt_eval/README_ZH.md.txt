# GPT EVAL：使用 OpenAI API 评测大模型

## 快速上手

1. 准备待评测的模型以及对照模型。
    - 待评测模型：当前支持 Huggingface 以及 Megatron-LM 格式，后续会陆续支持加载其他常见模型检查点格式
    - 对照模型：Huggingface, Megatron-LM 模型或 OpenAI 提供的模型
    > 评测 Megatron-LM 模型需要使用 [`thirdparty`](../../../thirdparty) 文件夹提供的定制化 Megatron-LM

2. 使用 [`answer_generator.py`](answer_generator.py) 在评测数据集上分别生成待评测模型及对照模型的回答。
    1. 准备数据集。工具包内已经提供了 Vicuna 评测数据集 ([`config/question.jsonl`](config/question.jsonl))，同时支持用户使用自定义数据集生成回答，自定义数据集要求为单个 jsonl 文件，其中每个 json 对象包含以下3个域：
        - question_id: int 类型，用于标识该问题
        - text: string 类型，问题的具体内容
        - category: string 类型，该问题的类型

    2. 编写配置文件。 运行脚本需要的 yaml 文件格式如下：

        ```yaml
        answer_generation:
          model_name: <str>
          question_file: <str>  # 评测数据文件路径
          answer_file: <str>    # 模型生成回答文件路径
          batch_size: <int>     # 生成回答时的 batch size
          max_tokens: <int>     # 生成回答的最大 token 数量
          temperature: <float>
          # 以下配置根据模型来源选择一种即可
          # huggingface 配置
          huggingface:
            model_path: <str> # 文件路径或 huggingface model path
            tokenizer_path: <str> # 文件路径或 huggingface model path
          # megatron-lm 配置
          megatron:
            megatron_home: <str>    # Megatron-LM 代码根目录
            process_num: <int>      # 运行 megatron-lm 所需的进程数
            checkpoint_path: <str>  # megatron checkpoint 文件夹路径
            tokenizer_type: <str>   # 目前仅支持 'gpt2' 和 'sentencepiece'
            vocab_path: <str>       # gpt2 tokenizer 的 vocab 文件路径
            merge_path: <str>       # gpt2 tokenizer 的 merge 文件路径
            tokenizer_path: <str>   # sentencepiece tokenizer 的 model 文件路径
            iteration: <int>        # 待加载 checkpoint 的 iteration
          # openai 配置
          openai:
            openai_organization: <str>
            openai_api_key: <str>
            model: <str> # 评测模型的类型，例如 gpt-3.5-turbo
            max_retry: <int> # api 访问失败时最大重试次数
        ```

    3. 运行脚本。

        ```shell
        python answer_generator.py --config <path to config.yaml>
        ```

3. 通过 [`gpt_evaluator.py`](gpt_evaluator.py) 调用 OpenAI API 获得评价结果。
    1. 准备评测依赖项。运行脚本前需准备如下文件：
        - question_file: 即上一步中的评测数据文件
        - answer_file: 即上一步得到的待评测模型的回答文件
        - baseline_file: 即上一步得到的对照模型的回答文件
        - prompt_file: prompt 模板文件，工具包内已提供一份样本 ([`config/prompt.jsonl`](config/prompt.jsonl))
        - reviewer_file: reviewer 模板文件(包括评测时使用的模型类型和其他参数)，工具包内已提供一份样本 ([`config/reviewer.json`](config/reviewer.json))
    2. 编写配置文件。运行脚本所需的 yaml 文件格式如下：

        ```yaml
        gpt_evaluation:
          openai_organization: <str>
          openai_api_key: <str>
          question_file: <str>
          answer_file: <str>
          baseline_file: <str>
          prompt_file: <str>
          reviewer_file: <str>
          result_file: <str>    # 评价结果输出文件路径
        ```

    3. 运行脚本。

        ```shell
        python gpt_evaluator.py --config <path to config.yaml>
        ```
