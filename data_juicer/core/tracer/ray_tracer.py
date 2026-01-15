import os
from collections import defaultdict

import pandas as pd
import ray

from data_juicer.ops import OPERATORS


@ray.remote
class RayTracer:
    """
    The tracer to trace the sample changes before and after an operator
    process.

    The comparison results will be stored in the work directory.
    Now supports sample-level tracing for better efficiency and accuracy.
    """

    def __init__(self, work_dir, op_list_to_trace=None, show_num=10, trace_keys=None):
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
            print("Trace for all ops.")
            self.op_list_to_trace = set(OPERATORS.modules.keys())
        else:
            self.op_list_to_trace = set(op_list_to_trace)
        self.show_num = show_num
        self.trace_keys = trace_keys or []

        # Sample-level tracing storage: op_name -> list of trace entries
        self._sample_traces = defaultdict(list)
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

        # Ray mode: no lock needed as each worker has its own instance
        if self._collected_counts[op_name] >= self.show_num:
            return False

        entry = {}
        # Add specified fields first (appears at start of output)
        for key in self.trace_keys:
            entry[key] = original_sample.get(key)
        # Add trace data
        entry["original_text"] = original_text
        entry["processed_text"] = processed_text

        self._sample_traces[op_name].append(entry)
        self._collected_counts[op_name] += 1

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

        # Ray mode: no lock needed as each worker has its own instance
        if self._collected_counts[op_name] >= self.show_num:
            return False

        self._sample_traces[op_name].append(sample)
        self._collected_counts[op_name] += 1

        return True

    def get_trace_file_path(self, op_name: str) -> str:
        """
        Get the file path for a trace file.

        :param op_name: the operator name
        :return: the file path
        """
        return os.path.join(self.work_dir, f"sample_trace-{op_name}.jsonl")

    def finalize_traces(self):
        """
        Export all collected sample-level traces to files.
        This should be called after all operators have finished processing.
        """
        print(f"Traced {len(self._sample_traces)} OPs.")
        for op_name, traces in self._sample_traces.items():
            if len(traces) == 0:
                print(
                    f"WARNING: Datasets before and after op [{op_name}] are all "
                    f"the same. Thus no comparison results would be "
                    f"generated."
                )
                continue

            if len(traces) < self.show_num:
                print(
                    f"WARNING: There are {len(traces)} traced samples for op [{op_name}] "
                    f"-- less than expected {self.show_num} samples."
                )

            # Determine file name based on operator type
            # We'll use a generic name for now, could be improved with operator type detection
            res_name = self.get_trace_file_path(op_name)
            dif_df = pd.DataFrame(traces)
            dif_df.to_json(res_name, orient="records", lines=True, force_ascii=False)
            print(f"Exported {len(traces)} traced samples for op [{op_name}] to {res_name}")
