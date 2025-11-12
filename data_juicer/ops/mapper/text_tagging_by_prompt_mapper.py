from typing import Dict

from loguru import logger

from data_juicer.ops.base_op import OPERATORS, UNFORKABLE, Mapper
from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.model_utils import get_model, prepare_model

OP_NAME = 'text_tagging_by_prompt_mapper'

with AvailabilityChecking(['torch', 'transformers', 'vllm'], OP_NAME):
    import torch
    import transformers  # noqa: F401
    import vllm  # noqa: F401

    # avoid hanging when calling model in multiprocessing
    torch.set_num_threads(1)


DEFAULT_CLASSIFICATION_PROMPT = """
请对下面的example文本回复的任务类别进行检测,并进行分类。
备选的分类包括：{tag_list}。
只回复对应的分类,不回复其他内容。
example文本:
{text}
""" # noqa

DEFAULT_CLASSIFICATION_LIST = [
    '数学', '代码', '翻译', '角色扮演', '开放领域问答', '特定领域问答', '提取', '生成', '头脑风暴', '分类',
    '总结', '改写', '其他'
]  # noqa

DEFAULT_IDENTITY_BINARY_PROMPT = """
检测下面的example文本的回复中是否包含人工智能模型的自我认知(例如表现出自己是一个AI人工助手)。
备选的分类包括：{tag_list}。
只回复对应的分类,不回复其他内容。

example文本:
{text}
""" # noqa

DEFAULT_BINARY_LIST = ['是', '否']


# TODO: Extend LLM-based OPs into API-based implementation.
@UNFORKABLE.register_module(OP_NAME)
@OPERATORS.register_module(OP_NAME)
class TextTaggingByPromptMapper(Mapper):
    """
    Mapper to generate text tags using prompt with LLM.
    Recommended model list: [
        'Qwen/Qwen2-7B-Instruct',
        'meta-llama/Meta-Llama-3.1-8B-Instruct',
    ]
    Other opensourced models with good instruction following ability
    also works.
    """

    _accelerator = 'cuda'

    def __init__(self,
                 hf_model: str = 'Qwen/Qwen2-7B-Instruct',
                 trust_remote_code: bool = False,
                 prompt: str = DEFAULT_CLASSIFICATION_PROMPT,
                 tag_list: str = DEFAULT_CLASSIFICATION_LIST,
                 enable_vllm: bool = True,
                 tensor_parallel_size: int = None,
                 max_model_len: int = None,
                 max_num_seqs: int = 256,
                 sampling_params: Dict = {},
                 *args,
                 **kwargs):
        """
        Initialization method.
        :param hf_model: Hugginface model id.
        :param trust_remote_code: passed to transformers
        :param prompt: the prompt used to generate text tags.
        :param tag_list: the list of tagging output options.
        :param enable_vllm: Whether to use vllm for inference acceleration.
        :param tensor_parallel_size: It is only valid when enable_vllm is True.
            The number of GPUs to use for distributed execution with tensor
            parallelism.
        :param max_model_len: It is only valid when enable_vllm is True.
            Model context length. If unspecified, will be automatically
            derived from the model config.
        :param max_num_seqs: It is only valid when enable_vllm is True.
            Maximum number of sequences to be processed in a single iteration.
        :param sampling_params: Sampling parameters for text generation.
            e.g {'temperature': 0.9, 'top_p': 0.95}
        :param args: extra args
        :param kwargs: extra args

        The default data format parsed by this interface is as follows:
        Model Input:
            请对下面的example文本回复的任务类别进行检测，并进行分类。备选的分类包括：["数学"，"代码"，"翻译"，"角色扮演"，"开放领域问答"，"特定领域问答", "提取", "生成", "头脑风暴", "分类"，"总结"，"改写"， "其他"]。只回复对应的分类，不回复其他内容。
            example文本:
            {
                "instruction": "找出方程 x2 - 3x = 0 的根。",
                "input": "",
                "output": "该方程可以写成 x(x-3)=0。\n\n根据乘法原理，x = 0或x - 3 = 0。\n\n因此，x1 = 0和x2 = 3是方程 x2 - 3x = 0 的两个根。"
            }
        Model Output:
            数学
        """ # noqa

        super().__init__(*args, **kwargs)
        self.num_proc = 1

        self.prompt = prompt
        self.tag_list = tag_list
        self.enable_vllm = enable_vllm

        if enable_vllm:
            import torch
            from vllm import SamplingParams

            assert torch.cuda.device_count() >= 1, 'must be executed in CUDA'
            if not tensor_parallel_size:
                tensor_parallel_size = torch.cuda.device_count()
                logger.info(f'Set tensor_parallel_size to \
                    {tensor_parallel_size} for vllm.')
            self.model_key = prepare_model(
                model_type='vllm',
                pretrained_model_name_or_path=hf_model,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs)
            self.sampling_params = SamplingParams(**sampling_params)
        else:
            self.model_key = prepare_model(
                model_type='huggingface',
                pretrained_model_name_or_path=hf_model,
                trust_remote_code=trust_remote_code)
            self.sampling_params = sampling_params

    def process(self, sample, rank=None):
        model, processor = get_model(self.model_key, rank, self.use_cuda())

        if self.enable_vllm:
            response = model.generate([
                self.prompt.format(text=sample[self.text_key],
                                   tag_list=self.tag_list)
            ], self.sampling_params)
            output = response[0].outputs[0].text
        else:
            inputs = processor([
                self.prompt.format(text=sample[self.text_key],
                                   tag_list=self.tag_list)
            ],
                               return_tensors='pt').to(model.device)
            response = model.generate(**inputs, **self.sampling_params)
            output = processor.decode(response.cpu()[0],
                                      skip_special_tokens=True)

        text_tags = []
        text_tags.append(output)
        sample[Fields.text_tags] = text_tags

        return sample
