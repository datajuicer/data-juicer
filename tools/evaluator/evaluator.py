import argparse
import os
import shutil
import subprocess
import time

import yaml
from gpt_eval.gpt_evaluator import GPTEvaluator
from recorder.wandb_writer import HelmWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-type", choices=["megatron", "huggingface"], default="megatron")
    parser.add_argument("--eval-type", choices=["helm", "gpt"], default="helm")
    parser.add_argument("--iteration-interval", type=int, default=1000)
    parser.add_argument("--begin-iteration", type=int, default=None)
    parser.add_argument("--end-iteration", type=int, default=None)
    parser.add_argument("--check-iterval", type=int, default=30)
    return parser.parse_args()


def check_args(args):
    if args.begin_iteration is None:
        print(
            f"--begin-iteration is not provided, use the value of " f"--iteration-interval ({args.iteration_interval})."
        )
        args.begin_iteration = args.iteration_interval
    if args.end_iteration is None:
        print("--end-iteration is not provided, evaluator will monitor the " "training process continuously.")
        args.end_iteration = float("inf")


class Evaluator:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)["auto_eval"]
        self.eval_type = args.eval_type
        self.iteration_interval = args.iteration_interval
        self.begin_iteration = args.begin_iteration
        self.end_iteration = args.end_iteration
        self.check_iterval = args.check_iterval
        self.load_config()

    def load_config(self):
        self.project_name = self.config["project_name"]
        self.model_name = self.config["model_name"]
        self.full_name = f"{self.project_name}-{self.model_name}"
        # load cache dir
        self.cur_dir = os.path.abspath(os.getcwd())
        self.cache_dir = self.config["cache_dir"] if "cache_dir" in self.config else os.path.join(self.cur_dir, "cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        # load megatron config
        if "megatron" in self.config:
            os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
            os.environ["OMP_NUM_THREADS"] = "4"
            self.megatron_process_num = self.config["megatron"]["process_num"]
            self.megatron_checkpoint_path = self.config["megatron"]["checkpoint_path"]
            # for different tokenizer
            if self.config["megatron"]["tokenizer_type"] == "sentencepiece":
                self.tokenizer_type = "sentencepiece"
                self.vocab_path = None
                self.merge_path = None
                self.tokenizer_path = self.config["megatron"]["tokenizer_path"]
            elif self.config["megatron"]["tokenizer_type"] == "gpt2":
                self.tokenizer_type = "gpt2"
                self.vocab_path = self.config["megatron"]["vocab_path"]
                self.merge_path = self.config["megatron"]["merge_path"]
                self.tokenizer_path = None
            else:
                raise NotImplementedError(
                    f"tokenizer type: " f"{self.config['megatron']['tokenizer_type']} is not " f"supported"
                )
            self.megatron_log_path = os.path.join(self.cache_dir, "megatron.log")
            if "log_path" in self.config["megatron"]:
                self.megatron_log_path = self.config["megatron"]["log_path"]
            self.megatron_server_port = 5000
            if "port" in self.config["megatron"]:
                self.megatron_server_port = self.config["megatron"]["port"]
            self.megatron_home = self.cur_dir
            if "megatron_home" in self.config["megatron"]:
                self.megatron_home = self.config["megatron"]["megatron_home"]
            self.max_tokens = 512
            if "max_tokens" in self.config["megatron"]:
                self.max_tokens = self.config["megatron"]["max_tokens"]
            self.megatron_token_per_iteration = 0
            if "token_per_iteration" in self.config["megatron"]:
                self.megatron_token_per_iteration = self.config["megatron"]["token_per_iteration"]
        # load helm config
        if "helm" in self.config:
            self.helm_spec_template_path = self.config["helm"]["helm_spec_template_path"]
            self.helm_output_path = self.config["helm"]["helm_output_path"]
            self.helm_spec_path = os.path.join(self.cache_dir, "helm_spec.conf")
            self.helm_cache_path = os.path.join(self.cache_dir, "helm_cache")
            self.helm_suite_name = self.full_name
            self.helm_conda_env = (
                self.config["helm"]["helm_env_name"] if "helm_env_name" in self.config["helm"] else "crfm-helm"
            )
            self.helm_eval_instances = (
                self.config["helm"]["eval_instances"] if "eval_instances" in self.config["helm"] else 100
            )
            self.helm_benchmarks = self.config["helm"]["benchmarks"] if "benchmarks" in self.config["helm"] else None
            self.helm_mymodel_config = os.path.join(self.cache_dir, "helm_config.yaml")
            with open(self.helm_mymodel_config, "w", encoding="utf-8") as f:
                mymodel_config = {
                    "port": self.megatron_server_port,
                    "tokenizer": {
                        "type": self.tokenizer_type,
                        "vocab_path": self.vocab_path,
                        "merge_path": self.merge_path,
                        "tokenizer_path": self.tokenizer_path,
                    },
                }
                yaml.dump(mymodel_config, f)
        if self.eval_type == "gpt":
            self.gpt_question_file = self.config["gpt_evaluation"]["question_file"]
            self.gpt_answer_file = self.config["gpt_evaluation"]["answer_file"]
        if "wandb" in self.config:
            self.wandb_base_url = self.config["wandb"]["base_url"] if "base_url" in self.config["wandb"] else None
            self.wandb_project = (
                self.config["wandb"]["project"] if "project" in self.config["wandb"] else self.project_name
            )

    def _set_megatron_tokenizer(self, args):
        if self.tokenizer_type == "gpt2":
            args.append("GPT2BPETokenizer")
            args.append("--vocab-file")
            args.append(self.vocab_path)
            args.append("--merge-file")
            args.append(self.merge_path)
        elif self.tokenizer_type == "sentencepiece":
            args.append("SentencePieceTokenizer")
            args.append("--tokenizer-model")
            args.append(self.tokenizer_path)

    def run_megatron_server(self, iteration):
        while not self.megatron_checkpoint_exists(iteration):
            print(f"Wait for megatron checkpoint {iteration}")
            time.sleep(self.check_iterval * 60)
        # setup megatron server
        print(f"Start megatron text generation server for checkpoint " f"iter_{iteration}")
        args = [
            "torchrun",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            "5950",
            "--nproc_per_node",
            str(self.megatron_process_num),
            "--nnodes",
            "1",
            "--node_rank",
            "0",
            os.path.join(self.megatron_home, "tools/run_text_generation_server.py"),
            "--port",
            str(self.megatron_server_port),
            "--use-checkpoint-args",
            "--load",
            self.megatron_checkpoint_path,
            "--load-iteration",
            str(iteration),
            "--tokenizer-type",
        ]
        self._set_megatron_tokenizer(args)
        logfile = open(self.megatron_log_path, "w")
        os.chdir(self.megatron_home)
        process = subprocess.Popen(args, stdout=logfile, stderr=logfile)
        os.chdir(self.cur_dir)
        return {"process": process, "logfile": logfile}

    def stop_megatron_server(self, process, logfile):
        process.terminate()
        logfile.close()
        print("Stop megatron text generation server")

    def run_megatron_inference(self, iteration):
        while not self.megatron_checkpoint_exists(iteration):
            time.sleep(self.check_iterval * 60)
            print(f"Wait for megatron checkpoint {iteration}")
        print(f"Start megatron inference for checkpoint iter_{iteration}")
        args = [
            "torchrun",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            "5950",
            "--nproc_per_node",
            "1",
            "--nnodes",
            str(self.megatron_process_num),
            "--node_rank",
            "0",
            "tools/inference.py",
            "--use-checkpoint-args",
            "--formatter",
            "gpt_eval",
            "--tokens-to-generate",
            str(self.max_tokens),
            "--input",
            self.gpt_question_file,
            "--output",
            self.gpt_answer_file,
            "--load",
            self.megatron_checkpoint_path,
            "--load-iteration",
            str(iteration),
            "--model-name",
            f"{self.full_name}/{iteration}",
            "--tokenizer-type",
        ]
        self._set_megatron_tokenizer(args)
        logfile = open(self.megatron_log_path, "w")
        os.chdir(self.megatron_home)
        subprocess.run(args)
        os.chdir(self.cur_dir)
        logfile.close()
        return {}

    def megatron_checkpoint_exists(self, iteration):
        with open(os.path.join(self.megatron_checkpoint_path, "latest_checkpointed_iteration.txt"), "r") as f:
            latest_checkpoint_iter = int(f.readline())
        if iteration > latest_checkpoint_iter:
            return False
        checkpoint_path = os.path.join(self.megatron_checkpoint_path, "iter_{:07d}".format(iteration))
        return os.path.exists(checkpoint_path)

    def replace_pattern(self, input_file, output_file, pattern, s):
        with open(input_file, "r", encoding="utf-8") as input, open(output_file, "w", encoding="utf-8") as output:
            lines = input.readlines()
            for i in range(len(lines)):
                lines[i] = lines[i].replace(pattern, s)
            output.writelines(lines)

    def run_helm_eval(self, iteration):
        print(f"Start helm evaluation for checkpoint iter_{iteration}")
        if os.path.exists(self.helm_cache_path):
            shutil.rmtree(self.helm_cache_path)
        self.replace_pattern(
            self.helm_spec_template_path, self.helm_spec_path, "<model>", f"mymodel/{self.full_name}/{iteration}"
        )
        helm_run_args = [
            "conda",
            "run",
            "-n",
            self.helm_conda_env,
            "--no-capture-output",
            "helm-run",
            "-n",
            "4",
            "-m",
            str(self.helm_eval_instances),
            "--conf-paths",
            self.helm_spec_path,
            "--my-config-path",
            self.helm_mymodel_config,
            "--local-path",
            self.helm_cache_path,
            "--suite",
            self.helm_suite_name,
            "-o",
            self.helm_output_path,
        ]
        subprocess.check_call(helm_run_args)
        print(f"run helm summarize for checkpoint iter_{iteration}")
        helm_summarize_args = [
            "conda",
            "run",
            "-n",
            self.helm_conda_env,
            "--no-capture-output",
            "helm-summarize",
            "--suite",
            self.helm_suite_name,
            "-o",
            self.helm_output_path,
        ]
        subprocess.check_call(helm_summarize_args)
        print(f"Finish helm evaluation for checkpoint iter_{iteration}")

    def run_gpt_eval(self, iteration):
        GPTEvaluator(self.config["gpt_evaluation"]).run()

    def write_wandb(self):
        if self.eval_type == "helm":
            helm_config = {
                "model_name": self.full_name,
                "source": "helm",
                "helm_output_dir": self.helm_output_path,
                "helm_suite_name": self.helm_suite_name,
                "token_per_iteration": self.megatron_token_per_iteration,
            }
            if self.helm_benchmarks is not None:
                helm_config["benchmarks"] = self.helm_benchmarks
            HelmWriter(project_name=self.wandb_project, base_url=self.wandb_base_url, helm_config=helm_config)

    def evaluate(self, start_gen_func, start_eval_func, stop_gen_func, stop_eval_func):
        cur_iter = self.begin_iteration
        while cur_iter <= self.end_iteration:
            states = start_gen_func(cur_iter)
            start_eval_func(cur_iter)
            stop_eval_func()
            stop_gen_func(**states)
            cur_iter += self.iteration_interval

    def dummy_stop(self, args=None):
        return

    def run(self):
        if self.eval_type == "helm":
            start_gen_func = self.run_megatron_server
            start_eval_func = self.run_helm_eval
            stop_gen_func = self.stop_megatron_server
            stop_eval_func = self.dummy_stop
        elif self.eval_type == "gpt":
            start_gen_func = self.run_megatron_inference
            start_eval_func = self.run_gpt_eval
            stop_gen_func = self.dummy_stop
            stop_eval_func = self.dummy_stop
        self.evaluate(start_gen_func, start_eval_func, stop_gen_func, stop_eval_func)


if __name__ == "__main__":
    args = parse_args()
    check_args(args)
    Evaluator(args).run()
