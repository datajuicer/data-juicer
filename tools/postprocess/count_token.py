from multiprocessing import Pool

import fire
import jsonlines as jl
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

TOKENIZER = None


def count_token_single(sample, text_keys):
    global TOKENIZER
    num = 0
    for key in text_keys:
        num += len(TOKENIZER.tokenize(sample[key]))
    return num


def prepare_tokenizer(tokenizer_method):
    global TOKENIZER
    logger.info("Loading tokenizer from HuggingFace...")
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_method, trust_remote_code=True)


def main(data_path, text_keys="text", tokenizer_method="EleutherAI/pythia-6.9b-deduped", num_proc=1):
    """
    Count the number of tokens for given dataset and tokenizer.

    :param data_path: path to the input dataset. Only support 'jsonl' now.
    :param text_keys: field keys that will be considered into token counts.
    :param tokenizer_method: name of the Hugging Face tokenizer.
    :param num_proc: number of processes to count tokens.
    """
    prepare_tokenizer(tokenizer_method)

    if isinstance(text_keys, str):
        text_keys = [text_keys]

    with jl.open(data_path) as reader:
        token_count = 0
        result_list = []
        with Pool(num_proc) as p:
            for sample in tqdm(reader):
                result_list.append(
                    p.apply_async(
                        count_token_single,
                        args=(
                            sample,
                            text_keys,
                        ),
                    )
                )
            for res in tqdm(result_list):
                token_count += res.get()

        logger.info(f"Total num of tokens: {token_count}")


if __name__ == "__main__":
    fire.Fire(main)
