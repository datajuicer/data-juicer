from itertools import chain

from data_juicer.config.config import get_default_cfg
from data_juicer.core.data.dataset_builder import DatasetBuilder
from data_juicer.utils.lazy_loader import LazyLoader

torch = LazyLoader("torch")
transformers = LazyLoader("transformers")


class TextTokenDistCollector(object):
    """Tokenize and collect distribution of tokens for given
    dataset with a specified tokenizer.
    """

    def __init__(self, tokenizer):
        """
        Initialization method.

        :param tokenizer: tokenizer name on huggingface
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        self.vocab_size = len(self.tokenizer)

    def collect(self, data_path, text_key, num_proc=1) -> "torch.distributions.Categorical":
        """
        Tokenize and collect tokens distribution of input dataset
        :param data_path: path to input dataset.
        :param text_key: field keys that will be considered into token counts.
        :param num_proc: number of processes to count tokens.
        :return: token distribution.
        """
        cfg = get_default_cfg()
        cfg.dataset_path = data_path
        builder = DatasetBuilder(cfg)
        dataset = builder.load_dataset(num_proc=num_proc)
        assert text_key in dataset.features, f"[{text_key} not find in dataset"

        def prepare_tokenizer(
            tokenizer,
            text_key,
        ):
            """
            Prepare a tokenizer function for dataset.
            :param tokenizer: a tokenizer to tokenize sample.
            :param text_key: field keys that will be
                considered into token counts.
            """

            def _tokenize_fn(
                example,
            ):
                example = tokenizer(example[text_key], add_special_tokens=False)
                return example

            return _tokenize_fn

        tokenize_proc = prepare_tokenizer(self.tokenizer, text_key)
        dataset = dataset.map(tokenize_proc, num_proc=num_proc, desc=f'tokenize {data_path.split("/")[-1]}')

        token_count = torch.zeros(self.vocab_size, dtype=torch.int64)
        token_ids = torch.tensor(list(chain.from_iterable(dataset["input_ids"])))
        indices, counts = token_ids.unique(return_counts=True)
        token_count.scatter_(0, indices, counts.to(token_count.dtype))
        dist = torch.distributions.Categorical(token_count)
        return dist
