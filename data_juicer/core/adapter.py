import json
import os
from copy import deepcopy

from datasets import Dataset, concatenate_datasets
from datasets.config import DEFAULT_MAX_BATCH_SIZE
from jsonargparse import Namespace

from data_juicer.analysis.measure import RelatedTTestMeasure
from data_juicer.core.monitor import Monitor
from data_juicer.ops import UNFORKABLE
from data_juicer.utils.cache_utils import dataset_cache_control
from data_juicer.utils.constant import Fields
from data_juicer.utils.process_utils import setup_mp


class Adapter:
    MAX_BATCH_SIZE = 10000

    def __init__(self, cfg: Namespace):
        self.cfg = cfg

        # insight mining related
        self.enable_insight_mining = self.cfg.open_insight_mining if hasattr(self.cfg, "open_insight_mining") else False

        # resource probe related
        self.idle_resources = Monitor.monitor_current_resources()

    @staticmethod
    def execute_and_probe(dataset, operators, sample_interval=0.5):
        """
        Process the input dataset and probe related information for each OP in
        the specified operator list.

        For now, we support the following targets to probe:
        "resource": resource utilization for each OP.
        "speed": average processing speed for each OP.

        The probe result is a list and each item in the list is the probe
        result for each OP.
        """
        if operators is None or len(operators) == 0:
            return []

        # number of test samples
        sample_num = len(dataset)

        # resource utilization list
        resource_util_list = []
        # probe for each OP
        unforkable_operators = set(UNFORKABLE.modules.keys())
        for op in operators:
            # select suitable mp method for each OP
            mp_context = ["forkserver", "spawn"] if (op.use_cuda() or op._name in unforkable_operators) else None
            setup_mp(mp_context)
            # expand the test dataset according to the runtime number of
            # processes to ensure enough data for a batch and probe the true
            # resource utilization for each OP
            expanded_dataset = concatenate_datasets([dataset] * op.runtime_np())

            # set the test batch size and save the old one
            if op.is_batched_op():
                old_batch_size = op.batch_size
                op.batch_size = sample_num

            # run single op and monitor the resource utilization
            _, resource_util_per_op = Monitor.monitor_func(
                op.run, args=(expanded_dataset,), sample_interval=sample_interval
            )

            # calculate speed
            resource_util_per_op["speed"] = sample_num / resource_util_per_op["time"]
            resource_util_list.append(resource_util_per_op)

            # # restore the batch size
            if op.is_batched_op():
                op.batch_size = old_batch_size

        return resource_util_list

    @staticmethod
    def take_batch(dataset, config):
        """
        Split the dataset into batches based on configuration and load factor.

        :param dataset: The dataset to be split
        :param config: Configuration settings, including batch size
        :return: An iterator of batches
        """
        # get initial batch size
        batch_size = config.get("batch_size", DEFAULT_MAX_BATCH_SIZE)
        # should be in [1, 10000]
        batch_size = min(max(batch_size, 1), Adapter.MAX_BATCH_SIZE)

        # check if there are enough samples
        num_samples = len(dataset)
        if batch_size >= num_samples:
            return dataset
        else:
            return dataset.take(batch_size)

    def adapt_workloads(self, dataset, operators):
        """
        Manage the scheduling and load balancing for the dataset processing.

        :param dataset: The dataset that needs to be processed
        :param operators: Operators in the data recipe
        """
        # TODO: set batch size to 1 for all OPs for probing
        load_analysis_res, probed_batch_size = self.probe_small_batch(dataset, operators)

        # calculate batch size for each OP according to the analysis results
        bs_per_op = self.batch_size_strategy(load_analysis_res, base_bs=probed_batch_size)

        return bs_per_op

    @dataset_cache_control(on=True)
    def probe_small_batch(self, dataset, operators):
        """
        Perform small batch pre-execution to probe available resources,
        current load and estimated OP speed, returning load factors and speed
        ranks for each OP.

        Notice: the probe should be run with cache enabled to avoid removing
        the cache files of the input dataset.

        :param dataset: The dataset to pre-execute small batch on
        :param operators: The OP list to be pre-execution and probe
        :return: A list of probe results for each OP and the length of data
            batch to probe.
        """

        # take a small batch
        data_batch = self.take_batch(dataset, self.cfg)
        # process and monitor the resource utilization
        resource_util_list = self.execute_and_probe(data_batch, operators)
        # analyze resource utilization
        analysis_res = Monitor.analyze_resource_util_list(resource_util_list)

        return analysis_res, len(data_batch)

    def batch_size_strategy(self, load_analysis_res, base_bs=1, util_th=0.9):
        """
        Decide the batch size for each op according to their workload analysis
        result and expected utilization threshold. We need to guarantee that
        the resource utilization won't exceed the threshold. Now we only
        consider the buckets effect, which means the max batch size is decided
        by the max utilization of all types of resources except GPU util
        (decided by num_proc).
        """
        batch_size_per_op = []

        # compute left utils according to the util_th
        left_utils = {}
        for key in self.idle_resources:
            if "util." not in key or "GPU" in key:
                continue
            left_utils[key] = max(0, util_th - self.idle_resources[key])

        for item in load_analysis_res:
            max_util = 1e-5
            max_key = min(left_utils.items(), key=lambda it: it[1])[0]
            analysis_res = item["resource_analysis"]
            for key in analysis_res:
                if "util." not in key or "GPU" in key:
                    continue
                used_util = max(0, analysis_res[key]["max"] - self.idle_resources[key])
                if used_util > max_util:
                    max_util = used_util
                    max_key = key
            load_factor = left_utils[max_key] / max_util
            bs_this_op = min(max(int(base_bs * load_factor), 1), self.MAX_BATCH_SIZE)
            batch_size_per_op.append(bs_this_op)

        return batch_size_per_op

    @dataset_cache_control(on=True)
    def analyze_small_batch(self, dataset, current_state):
        """
        Perform small batch analysis to probe the current OP-wise stats/meta
        distributions. The analyzed results will be stored in the directory
        `{work_dir}/insight_mining`.

        Notice: the probe should be run with cache enabled to avoid removing
        the cache files of the input dataset.

        :param dataset: The dataset to analyze small batch on
        :param current_state: A string to indicate the current state of the
            input dataset. It usually consists of a number of the index of the
            OP processed just now and the OP name, e.g. "1_text_length_filter".
        """
        # prepare analyzer config
        new_cfg = deepcopy(self.cfg)
        # check ops to mine
        new_cfg.auto = True
        new_cfg.config = None
        if len(new_cfg.op_list_to_mine) > 0:
            new_cfg.process = [{op_name: {"num_proc": new_cfg.np}} for op_name in new_cfg.op_list_to_mine]
        # update work dir
        new_cfg.work_dir = os.path.join(new_cfg.work_dir, "insight_mining", current_state)
        new_cfg.export_path = os.path.join(new_cfg.work_dir, f"{current_state}.jsonl")
        # close insight mining and monitor for inner analysis
        new_cfg.open_insight_mining = False
        new_cfg.open_monitor = False

        # init the analyzer
        from data_juicer.core.analyzer import Analyzer

        analyzer = Analyzer(new_cfg)

        # remove existing stats and meta in dataset
        target_fields = {Fields.stats, Fields.meta}
        target_fields = target_fields.intersection(set(dataset.features))
        if len(target_fields) > 0:
            dataset = dataset.remove_columns(list(target_fields))
        analyzer.run(dataset, skip_return=True)

    def insight_mining(self, pval_th=0.05):
        """
        Mining the insights from the OP-wise analysis results. For now, we use
        T-Test to check the significance of stats/meta changes before and after
        each OP processing. If the p-value is less than a given threshold
        (usually 0.05), we think the stats/meta changes are significant. The
        insight mining results will be stored in the file
        `{work_dir}/insight_mining/insight_mining.json`.

        :param pval_th: the threshold of p-value.
        """
        work_dir = os.path.join(self.cfg.work_dir, "insight_mining")
        res_order = [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d))]
        res_order.sort()

        # collect analysis results
        analysis_results = {}
        for res_dir in res_order:
            res = Dataset.from_json(os.path.join(work_dir, res_dir, f"{res_dir}_stats.jsonl")).flatten()
            analysis_results[res_dir] = res

        # distribution change significance analysis
        ttest_measure = RelatedTTestMeasure()

        sig_res = {}
        # i = 0 is the original dataset
        for i in range(1, len(res_order)):
            prev_res = analysis_results[res_order[i - 1]]
            curr_res = analysis_results[res_order[i]]

            # only consider common stats and meta
            common_features = list(set(prev_res.features).intersection(set(curr_res.features)))
            curr_sig_res = {}
            for feat in common_features:
                ttest_res = ttest_measure(prev_res[feat], curr_res[feat])
                curr_sig_res[feat] = {
                    "t-statistic (standardized mean difference)": ttest_res.statistic,
                    "p-value": ttest_res.pvalue,
                    "significant": True if ttest_res.pvalue < pval_th else False,
                }
            sig_res[res_order[i]] = curr_sig_res

        with open(os.path.join(work_dir, "insight_mining.json"), "w") as out:
            json.dump(sig_res, out)
