import argparse
import json
import os
import subprocess
import time
from abc import ABC, abstractmethod

import jsonlines
import openai
import requests
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def format_question(question):
    return f"Question: {question}\n\nAnswer:"


class AbstractGenerator(ABC):
    @abstractmethod
    def generate(self, texts, max_tokens, temperature):
        raise NotImplementedError("GENERATE is not implemented")

    def close(self):
        # do nothing
        return


class HuggingfaceGenerator(AbstractGenerator):
    def __init__(self, config):
        self.model = AutoModelForCausalLM.from_pretrained(config["model_path"]).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, texts, max_tokens, temperature):
        texts = [format_question(text) for text in texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
        return [
            self.tokenizer.decode(output[inputs.input_ids.shape[1] :], skip_special_tokens=True) for output in outputs
        ]


class OpenAIGenerator(AbstractGenerator):
    def __init__(self, config):
        openai.organization = config["openai_organization"]
        openai.api_key = config["openai_api_key"]
        self.model = config["model"]
        if "max_retry" in config:
            self.max_retry = config["max_retry"]
        else:
            self.max_retry = 5
        if "retry_wait" in config:
            self.retry_wait = config["retry_wait"]
        else:
            self.retry_wait = 5

    def generate(self, texts, max_tokens, temperature):
        outputs = []
        for text in texts:
            output = ""
            for _ in range(self.max_retry):
                try:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": text,
                            },
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    output = response["choices"][0]["message"]["content"]
                    break
                except Exception as e:
                    print(e)
                    time.sleep(self.retry_wait)
            if len(output) == 0:
                print(f"Failed to answer [{text}]")
            outputs.append(output)
        return outputs


class MegatronGenerator(AbstractGenerator):
    def __init__(self, config):
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "4"
        self.cur_dir = os.path.abspath(os.getcwd())
        self.process_num = config["process_num"]
        self.checkpoint_path = config["checkpoint_path"]
        self.load_iteration = config["iteration"]
        # for different tokenizer
        if config["tokenizer_type"] == "sentencepiece":
            self.tokenizer_type = "sentencepiece"
            self.vocab_path = None
            self.merge_path = None
            self.tokenizer_path = config["tokenizer_path"]
        elif config["tokenizer_type"] == "gpt2":
            self.tokenizer_type = "gpt2"
            self.vocab_path = config["vocab_path"]
            self.merge_path = config["merge_path"]
            self.tokenizer_path = None
        else:
            raise NotImplementedError("Unsupported tokenizer type")
        self.megatron_home = self.cur_dir
        if "megatron_home" in config:
            self.megatron_home = config["megatron_home"]
        print(f"Megatron-LM home: {self.megatron_home}")
        self.server_port = config["port"] if "port" in config else 5000
        self.handle = self._run_megatron_server()
        self.url = f"http://localhost:{self.server_port}/api"
        self.header = {
            "Content-Type": "application/json; charset=UTF-8",
        }
        print("Start Megatron text generation server")
        time.sleep(30)

    def _set_megatron_tokenizer(self, args):
        if self.tokenizer_type == "gpt2":
            args.append("GPT2BPETokenizer")
            args.append("--vocab-file")
            args.append(self.vocab_path)
            args.append("--merge-file")
            args.append(self.merge_path)
        elif self.tokenizer_type == "sentencepiece":
            args.append("SentencepieceTokenizer")
            args.append("--tokenizer-model")
            args.append(self.tokenizer_path)

    def _run_megatron_server(self):
        args = [
            "torchrun",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            "5950",
            "--nproc_per_node",
            "1",
            "--nnodes",
            str(self.process_num),
            "--node_rank",
            "0",
            "tools/run_text_generation_server.py",
            "--port",
            str(self.server_port),
            "--use-checkpoint-args",
            "--load",
            self.checkpoint_path,
            "--load-iteration",
            str(self.load_iteration),
            "--tokenizer-type",
        ]
        self._set_megatron_tokenizer(args)
        os.chdir(self.megatron_home)
        process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir(self.cur_dir)
        return process

    def _request(self, prompts, max_tokens, temperature):
        for _ in range(5):
            try:
                response = requests.put(
                    self.url,
                    headers=self.header,
                    data=json.dumps(
                        {
                            "prompts": prompts,
                            "tokens_to_generate": max_tokens,
                            "temperature": temperature,
                            "echo_prompts": False,
                        }
                    ),
                ).json()
            except Exception as e:
                response = {"message": e}
            if "text" not in response:
                print(f"Error in megatron response: {response}, retry in 10s")
                time.sleep(10)
            else:
                break
        return response["text"]

    def generate(self, texts, max_tokens, temperature):
        texts = [format_question(text) for text in texts]
        return self._request(texts, max_tokens, temperature)

    def close(self):
        self.handle.terminate()


class TextGenerator:
    def __init__(self, args):
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)["answer_generation"]
            self.questions = [q for q in jsonlines.open(config["question_file"], "r")]
            if not os.path.exists(os.path.dirname(config["answer_file"])):
                os.makedirs(os.path.dirname(config["answer_file"]))
            self.answer_writer = jsonlines.open(config["answer_file"], "w", flush=True)
            self.batch_size = config["batch_size"]
            self.max_tokens = config["max_tokens"]
            self.temperature = config["temperature"]
            self.model_name = config["model_name"]
            if "huggingface" in config:
                self.generator = HuggingfaceGenerator(config["huggingface"])
            elif "openai" in config:
                self.generator = OpenAIGenerator(config["openai"])
            elif "megatron" in config:
                self.generator = MegatronGenerator(config["megatron"])
            else:
                raise NotImplementedError("Generator not found")

    def generate(self, questions):
        texts = [question["text"] for question in questions]
        answer_texts = self.generator.generate(texts, self.max_tokens, self.temperature)
        for question, answer_text in zip(questions, answer_texts):
            self.answer_writer.write(
                {
                    "question_id": question["question_id"],
                    "model_id": self.model_name,
                    "text": answer_text,
                }
            )

    def run(self):
        questions = []
        for question in tqdm(self.questions):
            questions.append(question)
            if len(questions) % self.batch_size == 0:
                self.generate(questions)
                questions.clear()
        if len(questions) > 0:
            self.generate(questions)
        self.generator.close()
        self.answer_writer.close()


if __name__ == "__main__":
    args = parse_args()
    TextGenerator(args).run()
