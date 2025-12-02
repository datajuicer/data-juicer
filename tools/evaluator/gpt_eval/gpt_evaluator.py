# Some code here has been modified from:
# https://github.com/lm-sys/FastChat
# --------------------------------------------------------

import argparse
import logging
import os
import time
from multiprocessing import Pool

import jsonlines
import openai
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--worker-num", type=int, default=4, help="Number of workers for OpenAI API")
    parser.add_argument("--max-retry", type=int, default=5, help="Retry times for OpenAI API")
    parser.add_argument("--debug", action="store_true", help="Run without calling OpenAI API")
    return parser.parse_args()


class GPTEvaluator:
    def __init__(self, config):
        openai.organization = config["openai_organization"]
        openai.api_key = config["openai_api_key"]
        self.questions = [q for q in jsonlines.open(config["question_file"], "r")]
        self.answers = [a for a in jsonlines.open(config["answer_file"], "r")]
        self.baseline = [b for b in jsonlines.open(config["baseline_file"], "r")]
        self.prompt_templates = {p["category"]: p for p in jsonlines.open(config["prompt_file"], "r")}
        self.reviewers = {z["category"]: z for z in jsonlines.open(config["reviewer_file"], "r")}
        if not os.path.exists(os.path.dirname(config["result_file"])):
            os.makedirs(os.path.dirname(config["result_file"]))
        self.result_writer = jsonlines.open(config["result_file"], "w", flush=True)
        self.worker_num = config["worker_num"] if "worker_num" in config else 4
        self.max_retry = config["max_retry"] if "max_retry" in config else 5
        self.debug = config["debug"] if "debug" in config else False

    def generate_prompt(self, question, answer, baseline, prompts):
        if question["category"] in self.reviewers.keys():
            reviewer = self.reviewers[question["category"]]
            prompt_json = prompts[question["category"]]
        else:
            reviewer = self.reviewers["general"]
            prompt_json = prompts["general"]
        sys_prompt = prompt_json["system_prompt"]
        prompt_template = prompt_json["prompt_template"]
        defaults = prompt_json["defaults"]
        prompt1 = prompt_template.format(
            question=question["text"], answer_1=answer["text"], answer_2=baseline["text"], **defaults
        )
        prompt2 = prompt_template.format(
            question=question["text"], answer_1=baseline["text"], answer_2=answer["text"], **defaults
        )
        return sys_prompt, prompt1, prompt2, reviewer

    def parse_score(self, review):
        review = review.strip("\n")
        score_pair = review.split("\n")[-1]
        score_pair.strip()
        sp = score_pair.split(",")
        try:
            if len(sp) == 2:
                return [float(sp[0]), float(sp[1])]
            else:
                logger.error("Invalid score pair.")
                return [0, 0]
        except Exception:
            logger.error("Invalid answer")
            return [0, 0]

    def run(self):
        results = []
        requests = []
        question_num = len(self.questions)
        for i in range(question_num):
            sys_prompt, prompt1, prompt2, reviewer = self.generate_prompt(
                self.questions[i], self.answers[i], self.baseline[i], self.prompt_templates
            )
            results.append(
                {
                    "question_id": self.questions[i]["question_id"],
                    "metadata": reviewer["metadata"],
                    "model1": self.answers[i]["model_id"],
                    "model2": self.baseline[i]["model_id"],
                }
            )
            pool = Pool(processes=self.worker_num)
            requests.append(
                {
                    "sys_prompt": sys_prompt,
                    "user_prompt": prompt1,
                    "temperature": reviewer["metadata"]["temperature"],
                    "max_tokens": reviewer["metadata"]["max_tokens"],
                    "model": reviewer["metadata"]["model"],
                    "debug": self.debug,
                    "retry": self.max_retry,
                }
            )
            requests.append(
                {
                    "sys_prompt": sys_prompt,
                    "user_prompt": prompt2,
                    "temperature": reviewer["metadata"]["temperature"],
                    "max_tokens": reviewer["metadata"]["max_tokens"],
                    "model": reviewer["metadata"]["model"],
                    "debug": self.debug,
                    "retry": self.max_retry,
                }
            )
        reviews = pool.map(eval, requests)
        target_score = 0.0
        baseline_score = 0.0
        cnt = 0
        for i, review in enumerate(tqdm(reviews)):
            scores = self.parse_score(review)
            idx = i // 2
            if i % 2 == 0:
                results[idx]["review1"] = review
                results[idx]["score1"] = scores
                target_score += scores[0]
                baseline_score += scores[1]
            else:
                results[idx]["review2"] = review
                results[idx]["score2"] = scores
                target_score += scores[1]
                baseline_score += scores[0]
                self.result_writer.write(results[idx])
                cnt += 1
        target_avg_score = target_score / cnt / 2
        baseline_avg_score = baseline_score / cnt / 2
        print("-------------------------")
        print(f"> {results[0]['model1']}: {target_avg_score}")
        print(f"> {results[0]['model2']}: {baseline_avg_score}")
        print("-------------------------")
        self.result_writer.write(
            {f"{results[0]['model1']}": target_avg_score, f"{results[0]['model2']}": baseline_avg_score}
        )
        self.result_writer.close()


def eval(request):
    if request["debug"]:
        logger.info(f"Fake response {request['user_prompt']}")
        return "Fake response\n10,9\n"
    for _ in range(request["retry"]):
        try:
            response = openai.ChatCompletion.create(
                model=request["model"],
                messages=[
                    {"role": "system", "content": request["sys_prompt"]},
                    {
                        "role": "user",
                        "content": request["user_prompt"],
                    },
                ],
                temperature=request["temperature"],
                max_tokens=request["max_tokens"],
            )
            content = response["choices"][0]["message"]["content"]
            logger.info(content)
            return content
        except Exception as e:
            logger.error(e)
            time.sleep(5)
    logger.error(f"Failed after {request['retry']} retries.")
    return "error"


if __name__ == "__main__":
    args = parse_args()
    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))["gpt_evaluation"]
    config["worker_num"] = args.worker_num
    config["max_retry"] = args.max_retry
    config["debug"] = args.debug
    evaluator = GPTEvaluator(config)
    evaluator.run()
