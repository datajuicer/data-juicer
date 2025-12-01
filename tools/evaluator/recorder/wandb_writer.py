import argparse
import json
import os

import wandb
import yaml


def get_args():
    parser = argparse.ArgumentParser(description="write evaluation result into wandb", allow_abbrev=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


class Writer:
    def __init__(self, project_name, base_url=None, print_only=False, summary_only=False) -> None:
        self.project = project_name
        self.base_url = base_url
        self.print_only = print_only
        self.summary_only = summary_only


DEFAULT_HELM_BENCHMARKS = [
    {"name": "mmlu", "metrics": ["EM"]},
    {"name": "raft", "metrics": ["EM"]},
    {"name": "imdb", "metrics": ["EM"]},
    {"name": "truthful_qa", "metrics": ["EM"]},
    {"name": "summarization_cnndm", "metrics": ["ROUGE-2"]},
    {"name": "summarization_xsum", "metrics": ["ROUGE-2"]},
    {"name": "boolq", "metrics": ["EM"]},
    {"name": "msmarco_trec", "metrics": ["NDCG@10"]},
    {"name": "msmarco_regular", "metrics": ["RR@10"]},
    {"name": "narrative_qa", "metrics": ["F1"]},
    {"name": "natural_qa_closedbook", "metrics": ["F1"]},
    {"name": "natural_qa_openbook_longans", "metrics": ["F1"]},
    {"name": "quac", "metrics": ["F1"]},
    {"name": "civil_comments", "metrics": ["EM"]},
    {"name": "hellaswag", "metrics": ["EM"]},
    {"name": "openbookqa", "metrics": ["EM"]},
]

DEFAULT_HELM_METRICS = [
    "mmlu.EM",
    "raft.EM",
    "imdb.EM",
    "truthful_qa.EM",
    "summarization_cnndm.ROUGE-2",
    "summarization_xsum.ROUGE-2",
    "boolq.EM",
    "msmarco_trec.NDCG@10",
    "msmarco_regular.RR@10",
    "narrative_qa.F1",
    "natural_qa_closedbook.F1",
    "natural_qa_openbook_longans.F1",
    "civil_comments.EM",
    "hellaswag.EM",
    "openbookqa.EM",
]


class HelmWriter(Writer):
    def __init__(
        self, project_name, helm_config=None, leaderboard=False, base_url=None, print_only=False, summary_only=False
    ) -> None:
        super().__init__(project_name, base_url, print_only, summary_only)
        self.conf = helm_config
        self.leaderboard = leaderboard
        if self.leaderboard:
            self.leaderboard_metrics = (
                self.conf["leaderboard_metrics"] if "leaderboard_metrics" in self.conf else DEFAULT_HELM_METRICS
            )
            self.excluded_models = self.conf["excluded_models"] if "excluded_models" in self.conf else []
            return
        self.parse_from_helm = False
        self.parse_from_file = False
        self.source = self.conf["source"] if "source" in self.conf else "helm"
        # parse from helm output dir
        if self.source == "helm":
            self.helm_root = self.conf["helm_output_dir"]
            self.suite_name = self.conf["helm_suite_name"]
            if "benchmarks" in self.conf:
                self.scenarios = self.conf["benchmarks"]
            else:
                self.scenarios = DEFAULT_HELM_BENCHMARKS
            self.parse_from_helm = True
        # parse from config file
        elif self.source == "file":
            self.eval_result = self.conf["eval_result"]
            self.parse_from_file = True
        self.default_iteration = 0
        self.model = None
        if "model_name" in self.conf:
            self.model = self.conf["model_name"]
        if "default_iteration" in self.conf:
            self.default_iteration = self.conf["default_iteration"]

    def make_leaderboard(self):
        api = wandb.Api(overrides={"base_url": self.base_url})
        runs = api.runs(path=f"{self.project}", filters={"tags": "summary"})
        result = {}
        token_num = {}
        token_per_iteration = {}
        for run in runs:
            if run.group == "leaderboard" or run.group in self.excluded_models:
                continue
            print(run.id)
            run_name = run.group
            history = run.scan_history(keys=["_step"] + self.leaderboard_metrics, page_size=2000, min_step=0)
            if "token_num" in run.config:
                token_num[run_name] = run.config["token_num"]
            if "token_per_iteration" in run.config:
                token_per_iteration[run_name] = run.config["token_per_iteration"]
            for step in history:
                for metric_name, score in step.items():
                    if metric_name in ["_step", "average"]:
                        continue
                    if metric_name not in result:
                        result[metric_name] = {}
                    if run_name not in result[metric_name]:
                        result[metric_name][run_name] = {}
                    result[metric_name][run_name][step["_step"]] = score
        sum_scores = {}
        for metric_scores in result.values():
            self.cal_score(metric_scores)
            for run_name, iters in metric_scores.items():
                for iter, score in iters.items():
                    if run_name not in sum_scores:
                        sum_scores[run_name] = {}
                    if iter not in sum_scores[run_name]:
                        sum_scores[run_name][iter] = score
                    else:
                        sum_scores[run_name][iter] += score
        if self.print_only:
            print(sum_scores)
        else:
            run = wandb.init(
                project=self.project,
                group="leaderboard",
                name="leaderboard",
                save_code=False,
                id=f"{self.project}-leaderboard",
                tags=["leaderboard"],
                reinit=True,
            )
            data = []
            for name, iters in sum_scores.items():
                for iter, score in iters.items():
                    if name in token_num:
                        data.append([name, token_num[name], score])
                    elif name in token_per_iteration:
                        data.append([name, iter * token_per_iteration[name], score])
                    else:
                        data.append([name, None, score])
            table = wandb.Table(data=data, columns=["model", "token_num", "score"])
            wandb.log({"benchmark_score": wandb.plot.bar(table, "model", "score")})
            run.finish()

    def cal_score(self, scores):
        max_score = 0.0
        min_score = 1.0
        for subject, iters in scores.items():
            max_score = max(max(iters.values()), max_score)
            min_score = min(min(iters.values()), min_score)
        for subject, iters in scores.items():
            for iter, score in iters.items():
                scores[subject][iter] = (score - min_score) / (max_score - min_score)

    def write(self):
        if self.leaderboard:
            self.make_leaderboard()
        elif self.parse_from_helm:
            self.parse_scenarios()
        elif self.parse_from_file:
            self.write_wandb("summary", {self.default_iteration: self.eval_result}, "summary")
        else:
            print("do nothing, please check your config file")

    def parse_scenarios(self):
        summary = {}
        for scenario in self.scenarios:
            try:
                result = self.parse_scenario(scenario["name"], scenario["metrics"], self.model)
                if not self.summary_only:
                    self.write_wandb(scenario["name"], result, "detail")
                self.make_summary(scenario["name"], result, summary)
            except Exception as e:
                print(f"Fail to parse {scenario['name']}: {e}")
        self.write_wandb("summary", summary, "summary")

    def make_summary(self, scenario_name, eval_result, summary):
        print(f"summarize for {scenario_name}")
        for iteration, scenarios in eval_result.items():
            if iteration not in summary:
                summary[iteration] = dict()
            if scenario_name not in summary[iteration]:
                summary[iteration][scenario_name] = dict()
            for _, metrics in scenarios[scenario_name].items():
                summary[iteration][scenario_name] = metrics
                break

    def make_average(self, summary):
        for iteration, scenarios in summary.items():
            score = 0.0
            count = 0
            for _, metrics in scenarios.items():
                for _, value in metrics.items():
                    score += value
                    count += 1
                    break
            summary[iteration]["average"] = score / count

    def parse_scenario(self, scenario_name, scenario_metrics, model=None):
        evaluate_result = {}
        with open(os.path.join(self.helm_root, "runs", self.suite_name, "groups", f"{scenario_name}.json")) as f:
            print(f"parsing {scenario_name}.json")
            subjects = json.load(f)
            for subject in subjects:
                print(f"  parsing {subject['title']}")
                record_column_idx = {}
                for i, column in enumerate(subject["header"]):
                    if column["value"] in scenario_metrics:
                        record_column_idx[column["value"]] = i
                for row in subject["rows"]:
                    iteration = self.default_iteration
                    try:
                        iteration = int(row[0]["value"].split("_")[-1])
                    except Exception:
                        pass
                    try:
                        iteration = int(row[0]["value"].split("/")[-1])
                    except Exception:
                        pass
                    if iteration not in evaluate_result:
                        evaluate_result[iteration] = dict()
                    if scenario_name not in evaluate_result[iteration]:
                        evaluate_result[iteration][scenario_name] = dict()
                    evaluate_result[iteration][scenario_name][subject["title"].split(",")[0]] = dict()
                    for metric, i in record_column_idx.items():
                        evaluate_result[iteration][scenario_name][subject["title"].split(",")[0]][metric] = row[i][
                            "value"
                        ]
        return evaluate_result

    def write_wandb(self, name, result, tag):
        if self.print_only:
            print(result)
            return
        config = {}
        if "token_num" in self.conf:
            config["token_num"] = self.conf["token_num"]
        if "token_per_iteration" in self.conf:
            config["token_per_iteration"] = self.conf["token_per_iteration"]
        run = wandb.init(
            project=self.project,
            group=self.model,
            name=name,
            save_code=False,
            id=f"{self.project}-{self.model}-{name}",
            tags=["evalate", tag],
            config=config,
            reinit=True,
        )
        print(f"write {name} to wandb")
        for iteration in sorted(result.keys()):
            print(f"  write iteration {iteration} to wandb")
            wandb.log(result[iteration], int(iteration))
        run.finish()


def main():
    args = get_args()
    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    eval_configs = config["evals"] if "evals" in config else []
    for eval in eval_configs:
        if eval["eval_type"] == "helm":
            HelmWriter(
                project_name=config["project"],
                base_url=config["base_url"],
                print_only=args.print_only,
                summary_only=args.summary_only,
                helm_config=eval,
            ).write()
        else:
            raise NotImplementedError(f"Unsupported type for eval type {eval['eval_type']}")
    if "leaderboard" in config and config["leaderboard"] is True:
        HelmWriter(
            project_name=config["project"],
            base_url=config["base_url"],
            leaderboard=True,
            helm_config=config,
            print_only=args.print_only,
        ).write()


if __name__ == "__main__":
    main()
