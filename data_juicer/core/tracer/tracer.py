import os
from collections import defaultdict
from multiprocessing import Lock

import pandas as pd
from datasets import Dataset
from loguru import logger

from data_juicer.ops import OPERATORS
from data_juicer.utils.common_utils import deprecated
from data_juicer.utils.constant import Fields


class Tracer:
    """
    The tracer to trace the sample changes before and after an operator
    process.

    The comparison results will be stored in the work directory.
    Now supports sample-level tracing for better efficiency and accuracy.
    """

    def __init__(self, work_dir, op_list_to_trace=None, show_num=10, trace_keys=None, lock=None):
        """
        Initialization method.

        :param work_dir: the work directory to store the comparison
            results
        :param op_list_to_trace: the OP list to be traced.
        :param show_num: the maximum number of samples to show in the
            comparison result files.
        :param trace_keys: list of field names to include in trace output.
            If set, the specified fields' values will be included in each
            trace entry.
        """
        self.work_dir = os.path.join(work_dir, "trace")
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        # clear existing trace files in the work_dir
        for f in os.listdir(self.work_dir):
            os.remove(os.path.join(self.work_dir, f))
        self.op_list_to_trace = op_list_to_trace
        if not op_list_to_trace:
            logger.info("Trace for all ops.")
            self.op_list_to_trace = set(OPERATORS.modules.keys())
        else:
            self.op_list_to_trace = set(op_list_to_trace)
        self.show_num = show_num
        self.trace_keys = trace_keys or []

        # Sample-level tracing storage: op_name -> list of trace entries
        self._sample_traces = defaultdict(list)
        # Thread lock for thread-safe sample collection (only used in non-Ray mode)
        # In Ray mode, each worker will have its own tracer instance
        self._lock = lock or Lock()
        # Counter for each op to track how many samples have been collected
        self._collected_counts = defaultdict(int)

    def should_trace_op(self, op_name: str) -> bool:
        """
        Check if an operator should be traced.

        :param op_name: the operator name
        :return: True if the operator should be traced
        """
        return op_name in self.op_list_to_trace

    def is_collection_complete(self, op_name: str) -> bool:
        """
        Check if enough samples have been collected for an operator.

        :param op_name: the operator name
        :return: True if enough samples have been collected
        """
        with self._lock:
            return self._collected_counts[op_name] >= self.show_num

    def collect_mapper_sample(self, op_name: str, original_sample: dict, processed_sample: dict, text_key: str):
        """
        Collect a sample-level change for a Mapper operator.
        This method is thread-safe and will only collect up to show_num samples.

        :param op_name: the operator name
        :param original_sample: the original sample before processing
        :param processed_sample: the processed sample after processing
        :param text_key: the key name of the text field to compare
        :return: True if the sample was collected, False if collection is complete
        """
        if not self.should_trace_op(op_name):
            return False

        # Check if sample has changed
        original_text = original_sample.get(text_key, "")
        processed_text = processed_sample.get(text_key, "")

        if original_text == processed_text:
            return False

        with self._lock:
            # Double-check after acquiring lock
            if self._collected_counts[op_name] >= self.show_num:
                return False

            entry = {}
            # Add specified fields first (appears at start of output)
            for key in self.trace_keys:
                entry[key] = original_sample.get(key)
            # Add trace data
            entry["original_text"] = original_text
            entry["processed_text"] = processed_text

            logger.debug(f"Trace the entry in mapper [{op_name}]: {entry}")

            self._collected_counts[op_name] += 1
            with open(self.get_trace_file_path(op_name), "a") as f:
                entry_str = pd.DataFrame([entry]).to_json(orient="records", lines=True, force_ascii=False)
                f.write(entry_str)
                f.flush()

            return True

    def collect_filter_sample(self, op_name: str, sample: dict, should_keep: bool):
        """
        Collect a sample-level change for a Filter operator.
        This method is thread-safe and will only collect up to show_num samples.
        Only collects samples that are filtered out (should_keep=False).

        :param op_name: the operator name
        :param sample: the sample being filtered
        :param should_keep: True if the sample should be kept, False if filtered
        :return: True if the sample was collected, False if collection is complete
        """
        if not self.should_trace_op(op_name):
            return False

        # Only collect filtered samples (should_keep=False)
        if should_keep:
            return False

        with self._lock:
            # Double-check after acquiring lock
            if self._collected_counts[op_name] >= self.show_num:
                return False

            logger.debug(f"Trace the sample in filter [{op_name}]: {sample}")

            self._collected_counts[op_name] += 1
            with open(self.get_trace_file_path(op_name), "a") as f:
                entry_str = pd.DataFrame([sample]).to_json(orient="records", lines=True, force_ascii=False)
                f.write(entry_str)
                f.flush()

            return True

    def get_trace_file_path(self, op_name: str) -> str:
        """
        Get the file path for a trace file.

        :param op_name: the operator name
        :return: the file path
        """
        return os.path.join(self.work_dir, f"sample_trace-{op_name}.jsonl")

    @deprecated("This method will be deprecated in the future. Please apply the sample-level tracing method instead.")
    def trace_mapper(self, op_name: str, previous_ds: Dataset, processed_ds: Dataset, text_key: str):
        """
        Compare datasets before and after a Mapper.

        This will mainly show the different sample pairs due to the
        modification by the Mapper

        :param op_name: the op name of mapper
        :param previous_ds: dataset before the mapper process
        :param processed_ds: dataset processed by the mapper
        :param text_key: which text_key to trace
        :return:
        """
        if op_name not in self.op_list_to_trace:
            return

        assert len(previous_ds) == len(processed_ds)
        dif_dict = []
        num = 0

        # Find different samples orderly between previous and processed
        # datasets until the total number of found sample pairs is enough.
        for i in range(len(previous_ds)):
            previous_sample = previous_ds[i][text_key]
            processed_sample = processed_ds[i][text_key]
            if previous_sample != processed_sample:
                entry = {}
                # Add specified fields first (appears at start of output)
                for key in self.trace_keys:
                    entry[key] = previous_ds[i].get(key)
                # Add trace data (these take precedence over trace_keys)
                entry["original_text"] = previous_sample
                entry["processed_text"] = processed_sample
                dif_dict.append(entry)
                num += 1
                if num >= self.show_num:
                    break

        if len(dif_dict) == 0:
            logger.warning(
                f"Datasets before and after op [{op_name}] are all "
                f"the same. Thus no comparison results would be "
                f"generated."
            )
            return
        elif len(dif_dict) < self.show_num:
            logger.warning(
                f"There are {len(dif_dict)} different samples "
                f"before and after op [{op_name}] -- less than "
                f"expected {self.show_num} samples."
            )

        # export the tracer results.
        res_name = f"mapper-{op_name}.jsonl"
        dif_df = pd.DataFrame(dif_dict)
        dif_df.to_json(os.path.join(self.work_dir, res_name), orient="records", lines=True, force_ascii=False)

    @deprecated("This method will be deprecated in the future. Please apply the sample-level tracing method instead.")
    def trace_batch_mapper(self, op_name: str, previous_ds: Dataset, processed_ds: Dataset, text_key: str):
        """
        Compare datasets before and after a BatchMapper.

        This will mainly show the new samples augmented by the BatchMapper

        :param op_name: the op name of mapper
        :param previous_ds: dataset before the mapper process
        :param processed_ds: dataset processed by the mapper
        :param text_key: which text_key to trace
        :return:
        """
        if op_name not in self.op_list_to_trace:
            return

        assert previous_ds[0][text_key] == processed_ds[0][text_key]
        aug_dict = []

        # Get the first samples
        for i in range(len(processed_ds)):
            processed_sample = processed_ds[i]
            aug_dict.append(processed_sample)
            if i + 1 >= self.show_num:
                break

        if len(aug_dict) < self.show_num:
            logger.warning(f"There are only {len(aug_dict)} samples -- less " f"than expected {self.show_num} samples.")

        # export the tracer results.
        res_name = f"mapper-{op_name}.jsonl"
        dif_df = pd.DataFrame(aug_dict)
        dif_df.to_json(os.path.join(self.work_dir, res_name), orient="records", lines=True, force_ascii=False)

    @deprecated("This method will be deprecated in the future. Please apply the sample-level tracing method instead.")
    def trace_filter(self, op_name: str, previous_ds: Dataset, processed_ds: Dataset):
        """
        Compare datasets before and after a Filter.

        This will mainly show the filtered samples by the Filter

        :param op_name: the op name of filter
        :param previous_ds: dataset before the filter process
        :param processed_ds: dataset processed by the filter
        :return:
        """
        if op_name not in self.op_list_to_trace:
            return

        if len(previous_ds) == len(processed_ds):
            logger.warning(
                f"Datasets before and after op [{op_name}] are all "
                f"the same. Thus no comparison results would be "
                f"generated."
            )
            return

        # get the number of filtered samples.
        total_dif_num = len(previous_ds) - len(processed_ds)
        # index of the current sample in the previous dataset
        i = 0
        filter_dict = []
        # number of found filtered samples. It's the offset between two
        # datasets as well.
        num = 0
        previous_ds_no_stats = (
            previous_ds.remove_columns(Fields.stats) if Fields.stats in previous_ds.column_names else previous_ds
        )
        processed_ds_no_stats = (
            processed_ds.remove_columns(Fields.stats) if Fields.stats in processed_ds.column_names else processed_ds
        )
        while i < len(previous_ds):
            if i - num >= len(processed_ds) or previous_ds_no_stats[i] != processed_ds_no_stats[i - num]:
                # 1. If all samples in processed dataset are checked but there
                # still some samples left in the previous dataset, all of these
                # left samples are filtered.
                # 2. If the corresponding samples in previous and processed
                # datasets are different, samples in the previous dataset are
                # filtered.
                num += 1
                filter_dict.append(previous_ds[i])
            if num >= self.show_num or num >= total_dif_num:
                # If the total number of found filtered samples is enough or we
                # have found all filtered samples, just stop.
                break
            i += 1
        if len(filter_dict) < self.show_num:
            logger.warning(
                f"There are {len(filter_dict)} filtered samples "
                f"before and after op [{op_name}] -- less than "
                f"expected {self.show_num} samples."
            )

        # export the tracer results.
        res_name = f"filter-{op_name}.jsonl"
        filter_df = pd.DataFrame(filter_dict)
        filter_df.to_json(os.path.join(self.work_dir, res_name), orient="records", lines=True, force_ascii=False)

    def trace_deduplicator(self, op_name: str, dup_pairs: dict):
        """
        Compare datasets before and after a Deduplicator.

        This will mainly show the near-duplicate sample pairs extracted
        by the Deduplicator. Different from the other two trace methods,
        the trace process for deduplicator is embedded into the process
        method of deduplicator, but the other two trace methods are
        independent of the process method of mapper and filter operators

        :param op_name: the op name of deduplicator
        :param dup_pairs: duplicate sample pairs obtained from
            deduplicator
        :return:
        """
        if op_name not in self.op_list_to_trace:
            return

        if dup_pairs is None:
            logger.warning(
                f"Op [{op_name}] does not generate dup_pairs "
                f"correctly, thus no comparison results can be "
                f"obtained from this op."
            )
            return
        if len(dup_pairs) == 0:
            logger.warning(
                f"Datasets before and after op [{op_name}] are all "
                f"the same. Thus no comparison results would be "
                f"generated."
            )
            return
        elif len(dup_pairs) < self.show_num:
            logger.warning(
                f"There are {len(dup_pairs)} filtered samples "
                f"before and after op [{op_name}] -- less than "
                f"expected {self.show_num} samples."
            )

        # reorganize the duplicate pairs
        dup_dict = []
        for key in dup_pairs:
            dup_dict.append(
                {
                    "dup1": dup_pairs[key][0],
                    "dup2": dup_pairs[key][1],
                }
            )

        # export the tracer result.
        res_name = f"duplicate-{op_name}.jsonl"
        dup_df = pd.DataFrame(dup_dict)
        dup_df.to_json(os.path.join(self.work_dir, res_name), orient="records", lines=True, force_ascii=False)
