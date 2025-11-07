import copy
from functools import wraps

import numpy as np
import pyarrow as pa

from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import size_to_bytes
from data_juicer.utils.model_utils import free_models
from data_juicer.utils.process_utils import calculate_np
from data_juicer.utils.ray_utils import is_ray_mode
from data_juicer.utils.registry import Registry
from data_juicer.utils.resource_utils import is_cuda_available

OPERATORS = Registry("Operators")
UNFORKABLE = Registry("Unforkable")
NON_STATS_FILTERS = Registry("Non-stats Filters")
TAGGING_OPS = Registry("Tagging Operators")
ATTRIBUTION_FILTERS = Registry("Attribution Filters")
DEFAULT_BATCH_SIZE = 1000


def convert_list_dict_to_dict_list(samples):
    # reconstruct samples from "list of dicts" to "dict of lists"
    keys = samples[0].keys()
    res_samples = {}
    for key in keys:
        res_samples[key] = [s[key] for s in samples]
    return res_samples


def convert_dict_list_to_list_dict(samples):
    # reconstruct samples from "dict of lists" to "list of dicts"
    reconstructed_samples = []
    keys = list(samples.keys())
    # take any key, since they should be of same length
    for i in range(len(samples[keys[0]])):
        reconstructed_samples.append({key: samples[key][i] for key in samples})
    return reconstructed_samples


def convert_arrow_to_python(method):
    @wraps(method)
    def wrapper(sample, *args, **kwargs):
        if isinstance(sample, pa.Table):
            sample = sample.to_pydict()
        return method(sample, *args, **kwargs)

    return wrapper


def catch_map_batches_exception(method, skip_op_error=False, op_name=None):
    """
    For batched-map sample-level fault tolerance.
    """

    if op_name is None:
        op_name = method.__name__

    @wraps(method)
    @convert_arrow_to_python
    def wrapper(samples, *args, **kwargs):
        try:
            return method(samples, *args, **kwargs)
        except Exception as e:
            if not skip_op_error:
                raise
            import traceback

            from loguru import logger

            logger.error(
                f"An error occurred in {op_name} when processing "
                f'samples "{samples}" -- {type(e)}: {e} -- '
                f"{traceback.format_exc()}"
            )
            ret = {key: [] for key in samples.keys()}
            ret[Fields.stats] = []
            ret[Fields.source_file] = []
            return ret

    return wrapper


def catch_map_single_exception(method, return_sample=True, skip_op_error=False, op_name=None):
    """
    For single-map sample-level fault tolerance.
    The input sample is expected batch_size = 1.
    """

    if op_name is None:
        op_name = method.__name__

    def is_batched(sample):
        val_iter = iter(sample.values())
        first_val = next(val_iter)
        if not isinstance(first_val, list):
            return False
        first_len = len(first_val)
        return all(isinstance(val, list) and len(val) == first_len for val in val_iter)

    @wraps(method)
    @convert_arrow_to_python
    def wrapper(sample, *args, **kwargs):
        if is_batched(sample):
            try:
                sample = convert_dict_list_to_list_dict(sample)[0]
                res = method(sample, *args, **kwargs)
                if return_sample:
                    return convert_list_dict_to_dict_list([res])
                else:
                    return [res]
            except Exception as e:
                if not skip_op_error:
                    raise
                import traceback

                from loguru import logger

                logger.error(
                    f"An error occurred in {op_name} when processing "
                    f'sample "{sample}" -- {type(e)}: {e} -- '
                    f"{traceback.format_exc()}"
                )
                ret = {key: [] for key in sample.keys()}
                ret[Fields.stats] = []
                ret[Fields.source_file] = []
                return ret
        else:
            # without fault tolerance
            return method(sample, *args, **kwargs)

    return wrapper


class OP:
    _accelerator = "cpu"
    _batched_op = False

    def __init__(self, *args, **kwargs):
        """
        Base class of operators.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        :param image_bytes_key: the key name of field that stores sample image bytes list
            to be processed
        :param query_key: the key name of field that stores sample queries
        :param response_key: the key name of field that stores responses
        :param history_key: the key name of field that stores history of
            queries and responses
        :param index_key: index the samples before process if not None
        :param batch_size: the batch size for processing
        :param work_dir: the working directory for this operator
        """
        # init data keys
        self.text_key = kwargs.get("text_key", "text")
        self.image_key = kwargs.get("image_key", "images")
        self.audio_key = kwargs.get("audio_key", "audios")
        self.video_key = kwargs.get("video_key", "videos")

        # extra mm bytes keys
        self.image_bytes_key = kwargs.get("image_bytes_key", "image_bytes")

        self.system_key = kwargs.get("system_key", "system")
        self.instruction_key = kwargs.get("instruction_key", "instruction")
        self.prompt_key = kwargs.get("prompt_key", "prompt")
        self.query_key = kwargs.get("query_key", "query")
        self.response_key = kwargs.get("response_key", "response")
        self.history_key = kwargs.get("history_key", "history")

        self.index_key = kwargs.get("index_key", None)

        self.batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
        self.work_dir = kwargs.get("work_dir", None)

        # for unittest, do not skip the error.
        # It would be set to be True in config init.
        self.skip_op_error = kwargs.get("skip_op_error", False)

        # whether the model can be accelerated using cuda
        _accelerator = kwargs.get("accelerator", None)
        if _accelerator is not None:
            self.accelerator = _accelerator
        else:
            self.accelerator = self._accelerator

        # parameters to determine the number of procs for this op
        self.num_proc = kwargs.get("num_proc", -1)  # -1 means automatic calculation of concurrency
        self.cpu_required = kwargs.get("cpu_required", None)
        self.gpu_required = kwargs.get("gpu_required", None)
        self.mem_required = kwargs.get("mem_required", None)
        if isinstance(self.mem_required, str):
            self.mem_required = size_to_bytes(self.mem_required) / 1024**3

        self.turbo = kwargs.get("turbo", False)

        # nested wrappers
        from data_juicer.core.data import wrap_func_with_nested_access

        for name in ["process", "compute_stats", "compute_hash"]:
            method = getattr(self, name, None)
            if method and callable(method):
                setattr(self, f"_{name}", method)
                method = wrap_func_with_nested_access(method)
                setattr(self, name, method)

    def use_auto_proc(self):
        if is_ray_mode() and not self.use_cuda():  # ray task
            return self.num_proc == -1
        else:
            return not self.num_proc or self.num_proc == -1

    def is_batched_op(self):
        return self._batched_op

    def process(self, *args, **kwargs):
        raise NotImplementedError

    def use_cuda(self):
        return self.accelerator == "cuda" and is_cuda_available()

    def runtime_np(self):
        # Local import to avoid logger being serialized in multiprocessing
        from loguru import logger

        op_proc = calculate_np(
            self._name, self.mem_required, self.cpu_required or 1, self.use_cuda(), self.gpu_required
        )
        if not self.use_auto_proc():
            op_proc = min(op_proc, self.num_proc)
        logger.debug(f"Op [{self._name}] running with number of procs:{op_proc}")
        return op_proc

    def remove_extra_parameters(self, param_dict, keys=None):
        """
        at the beginning of the init of the mapper op, call
        self.remove_extra_parameters(locals())
        to get the init parameter dict of the op for convenience

        """
        if keys is None:
            param_dict = {k: v for k, v in param_dict.items() if not k.startswith("_")}
            param_dict.pop("self", None)
        else:
            param_dict = {k: v for k, v in param_dict.items() if k not in keys}
        return param_dict

    def add_parameters(self, init_parameter_dict, **extra_param_dict):
        """
        add parameters for each sample, need to keep extra_param_dict
        and init_parameter_dict unchanged.
        """
        related_parameters = copy.deepcopy(init_parameter_dict)
        related_parameters.update(extra_param_dict)
        return related_parameters

    def run(self, dataset):
        from data_juicer.core.data import NestedDataset

        if not isinstance(dataset, NestedDataset):
            dataset = NestedDataset(dataset)
        # add meta field for OPs that produce tags
        from data_juicer.core.data import add_same_content_to_new_column

        if self._name in TAGGING_OPS.modules and Fields.meta not in dataset.features:
            dataset = dataset.map(
                add_same_content_to_new_column,
                fn_kwargs={"new_column_name": Fields.meta, "initial_value": {}},
                num_proc=self.runtime_np(),
                batch_size=self.batch_size,
                desc="Adding new column for meta",
            )
        # add stats field for Filters that produce stats
        if (
            isinstance(self, Filter)
            and self._name not in NON_STATS_FILTERS.modules
            and Fields.stats not in dataset.features
        ):
            dataset = dataset.map(
                add_same_content_to_new_column,
                fn_kwargs={"new_column_name": Fields.stats, "initial_value": {}},
                num_proc=self.runtime_np(),
                batch_size=self.batch_size,
                desc="Adding new column for stats",
            )
        if self.index_key is not None and self.index_key not in dataset.features:

            def add_index(sample, idx):
                sample[self.index_key] = idx
                return sample

            dataset = dataset.map(add_index, with_indices=True)

        return dataset

    def empty_history(self):
        if is_ray_mode():
            return []

        return np.empty((0, 0), dtype=str)


class Mapper(OP):
    def __init__(self, *args, **kwargs):
        """
        Base class that conducts data editing.

        :param text_key: the key name of field that stores sample texts
            to be processed.
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        :param image_bytes_key: the key name of field that stores sample image bytes list
            to be processed
        :param query_key: the key name of field that stores sample queries
        :param response_key: the key name of field that stores responses
        :param history_key: the key name of field that stores history of
            queries and responses
        """
        super(Mapper, self).__init__(*args, **kwargs)

        # runtime wrappers
        if self.is_batched_op():
            self.process = catch_map_batches_exception(
                self.process_batched, skip_op_error=self.skip_op_error, op_name=self._name
            )
        else:
            self.process = catch_map_single_exception(
                self.process_single, skip_op_error=self.skip_op_error, op_name=self._name
            )

    # set the process method is not allowed to be overridden
    @classmethod
    def __init_subclass__(cls, **kwargs):
        not_allowed_list = ["process"]
        for method_name in not_allowed_list:
            if method_name in cls.__dict__:
                raise TypeError(
                    f"Method {method_name} cannot be overridden by subclass "
                    f"{cls.__name__}. Please implement {method_name}_single "
                    f"or {method_name}_batched."
                )

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def process_batched(self, samples, *args, **kwargs):
        keys = samples.keys()
        first_key = next(iter(keys))
        num_samples = len(samples[first_key])

        new_keys = {}
        for i in range(num_samples):
            this_sample = {key: samples[key][i] for key in keys}
            res_sample = self.process_single(this_sample, *args, **kwargs)
            res_keys = res_sample.keys()
            for key in res_keys:
                if key not in keys:
                    if key not in new_keys:
                        new_keys.update({key: []})
                    new_keys[key].append(res_sample[key])
                else:
                    samples[key][i] = res_sample[key]

        for k, v in new_keys.items():
            samples[k] = v

        return samples

    def process_single(self, sample):
        """
        For sample level, sample --> sample

        :param sample: sample to process
        :return: processed sample
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None):
        dataset = super(Mapper, self).run(dataset)
        new_dataset = dataset.map(
            self.process,
            num_proc=self.runtime_np(),
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + "_process",
        )
        if tracer:
            tracer.trace_mapper(self._name, dataset, new_dataset, self.text_key)
        free_models()
        return new_dataset


class Filter(OP):
    def __init__(self, *args, **kwargs):
        """
        Base class that removes specific info.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        :param image_bytes_key: the key name of field that stores sample image bytes list
            to be processed
        :param query_key: the key name of field that stores sample queries
        :param response_key: the key name of field that stores responses
        :param history_key: the key name of field that stores history of
            queries and responses

        :param min_closed_interval: whether the min_val of the specified filter range is a closed interval. It's True
            by default.
        :param max_closed_interval: whether the max_val of the specified filter range is a closed interval. It's True
            by default.
        :param reversed_range: whether to reverse the target range [min_val, max_val] to (-∞, min_val) or (max_val, +∞).
            It's False by default.
        """
        super(Filter, self).__init__(*args, **kwargs)
        self.stats_export_path = kwargs.get("stats_export_path", None)

        # filter strategy related
        self.min_closed_interval = kwargs.get("min_closed_interval", True)
        self.max_closed_interval = kwargs.get("max_closed_interval", True)
        self.reversed_range = kwargs.get("reversed_range", False)
        if self.reversed_range:
            self.min_closed_interval = not self.min_closed_interval
            self.max_closed_interval = not self.max_closed_interval

        # runtime wrappers
        if self.is_batched_op():
            self.compute_stats = catch_map_batches_exception(
                self.compute_stats_batched, skip_op_error=self.skip_op_error, op_name=self._name
            )
            self.process = catch_map_batches_exception(
                self.process_batched, skip_op_error=self.skip_op_error, op_name=self._name
            )
        else:
            self.compute_stats = catch_map_single_exception(
                self.compute_stats_single, skip_op_error=self.skip_op_error, op_name=self._name
            )
            self.process = catch_map_single_exception(
                self.process_single, return_sample=False, skip_op_error=self.skip_op_error, op_name=self._name
            )

    # set the process method is not allowed to be overridden
    @classmethod
    def __init_subclass__(cls, **kwargs):
        not_allowed_list = ["compute_stats", "process"]
        for method_name in not_allowed_list:
            if method_name in cls.__dict__:
                raise TypeError(
                    f"Method {method_name} cannot be overridden by subclass "
                    f"{cls.__name__}. Please implement {method_name}_single "
                    f"or {method_name}_batched."
                )

    def __call__(self, *args, **kwargs):
        return self.compute_stats(*args, **kwargs)

    def get_keep_boolean(self, val, min_val=None, max_val=None):
        res_bool = True
        if min_val is not None:
            res_bool = res_bool and (val >= min_val if self.min_closed_interval else val > min_val)
        if max_val is not None:
            res_bool = res_bool and (val <= max_val if self.max_closed_interval else val < max_val)
        if self.reversed_range:
            res_bool = not res_bool
        return res_bool

    def compute_stats_batched(self, samples, *args, **kwargs):
        keys = samples.keys()
        num_samples = len(samples[Fields.stats])
        for i in range(num_samples):
            this_sample = {key: samples[key][i] for key in keys}
            res_sample = self.compute_stats_single(this_sample, *args, **kwargs)
            samples[Fields.stats][i] = res_sample[Fields.stats]
            if "context" in kwargs and kwargs["context"]:
                samples[Fields.context][i] = res_sample[Fields.context]

        return samples

    def process_batched(self, samples):
        return map(lambda stat: self.process_single({Fields.stats: stat}), samples[Fields.stats])

    def compute_stats_single(self, sample, context=False):
        """
        Compute stats for the sample which is used as a metric to decide
        whether to filter this sample.

        :param sample: input sample.
        :param context: whether to store context information of intermediate
            vars in the sample temporarily.
        :return: sample with computed stats
        """
        raise NotImplementedError

    def process_single(self, sample):
        """
        For sample level, sample --> Boolean.

        :param sample: sample to decide whether to filter
        :return: true for keeping and false for filtering
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None, reduce=True):
        dataset = super(Filter, self).run(dataset)
        new_dataset = dataset.map(
            self.compute_stats,
            num_proc=self.runtime_np(),
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + "_compute_stats",
        )
        if exporter and self.stats_export_path is not None:
            exporter.export_compute_stats(new_dataset, self.stats_export_path)
        if reduce:
            new_dataset = new_dataset.filter(
                self.process, num_proc=self.runtime_np(), batch_size=self.batch_size, desc=self._name + "_process"
            )
            if tracer:
                tracer.trace_filter(self._name, dataset, new_dataset)
        free_models()
        return new_dataset


class Deduplicator(OP):
    def __init__(self, *args, **kwargs):
        """
        Base class that conducts deduplication.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        :param image_bytes_key: the key name of field that stores sample image bytes list
            to be processed
        :param query_key: the key name of field that stores sample queries
        :param response_key: the key name of field that stores responses
        :param history_key: the key name of field that stores history of
            queries and responses
        """
        super(Deduplicator, self).__init__(*args, **kwargs)

        # runtime wrappers
        if self.is_batched_op():
            self.compute_hash = catch_map_batches_exception(
                self.compute_hash, skip_op_error=self.skip_op_error, op_name=self._name
            )
        else:
            self.compute_hash = catch_map_single_exception(
                self.compute_hash, skip_op_error=self.skip_op_error, op_name=self._name
            )

    def compute_hash(self, sample):
        """
        Compute hash values for the sample.

        :param sample: input sample
        :return: sample with computed hash value.
        """
        raise NotImplementedError

    def process(self, dataset, show_num=0):
        """
        For doc-level, dataset --> dataset.

        :param dataset: input dataset
        :param show_num: number of traced samples used when tracer is
            open.
        :return: deduplicated dataset and the sampled duplicate pairs.
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None, reduce=True):
        dataset = super(Deduplicator, self).run(dataset)
        new_dataset = dataset.map(
            self.compute_hash, num_proc=self.runtime_np(), with_rank=self.use_cuda(), desc=self._name + "_compute_hash"
        )
        if reduce:
            show_num = tracer.show_num if tracer else 0
            new_dataset, dup_pairs = self.process(new_dataset, show_num)
            if tracer:
                tracer.trace_deduplicator(self._name, dup_pairs)
        free_models()
        return new_dataset


class Selector(OP):
    def __init__(self, *args, **kwargs):
        """
        Base class that conducts selection in dataset-level.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        :param image_bytes_key: the key name of field that stores sample image bytes list
            to be processed
        :param query_key: the key name of field that stores sample queries
        :param response_key: the key name of field that stores responses
        :param history_key: the key name of field that stores history of
            queries and responses
        """
        super(Selector, self).__init__(*args, **kwargs)

    def process(self, dataset):
        """
        Dataset --> dataset.

        :param dataset: input dataset
        :return: selected dataset.
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None):
        dataset = super(Selector, self).run(dataset)
        new_dataset = self.process(dataset)
        if tracer:
            tracer.trace_filter(self._name, dataset, new_dataset)
        free_models()
        return new_dataset


class Grouper(OP):
    def __init__(self, *args, **kwargs):
        """
        Base class that group samples.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        :param image_bytes_key: the key name of field that stores sample image bytes list
            to be processed
        :param query_key: the key name of field that stores sample queries
        :param response_key: the key name of field that stores responses
        :param history_key: the key name of field that stores history of
            queries and responses
        """
        super(Grouper, self).__init__(*args, **kwargs)

    def process(self, dataset):
        """
        Dataset --> dataset.

        :param dataset: input dataset
        :return: dataset of batched samples.
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None):
        dataset = super(Grouper, self).run(dataset)
        batched_samples = self.process(dataset)
        from data_juicer.core.data import NestedDataset

        new_dataset = NestedDataset.from_list(batched_samples)
        if tracer:
            tracer.trace_filter(self._name, dataset, new_dataset)
        free_models()
        return new_dataset


class Aggregator(OP):
    def __init__(self, *args, **kwargs):
        """
        Base class that group samples.

        :param text_key: the key name of field that stores sample texts
            to be processed
        :param image_key: the key name of field that stores sample image list
            to be processed
        :param audio_key: the key name of field that stores sample audio list
            to be processed
        :param video_key: the key name of field that stores sample video list
            to be processed
        :param image_bytes_key: the key name of field that stores sample image bytes list
            to be processed
        :param query_key: the key name of field that stores sample queries
        :param response_key: the key name of field that stores responses
        :param history_key: the key name of field that stores history of
            queries and responses
        """
        super(Aggregator, self).__init__(*args, **kwargs)
        self.process = catch_map_single_exception(
            self.process_single, skip_op_error=self.skip_op_error, op_name=self._name
        )

    def process_single(self, sample):
        """
        For sample level, batched sample --> sample,
        the input must be the output of some Grouper OP.

        :param sample: batched sample to aggregate
        :return: aggregated sample
        """
        raise NotImplementedError

    def run(self, dataset, *, exporter=None, tracer=None):
        dataset = super(Aggregator, self).run(dataset)
        # add batched meta field for OPs that produce aggregations
        if Fields.batch_meta not in dataset.features:
            from data_juicer.core.data import add_same_content_to_new_column

            dataset = dataset.map(
                add_same_content_to_new_column,
                fn_kwargs={"new_column_name": Fields.batch_meta, "initial_value": {}},
                num_proc=self.runtime_np(),
                batch_size=self.batch_size,
                desc="Adding new column for aggregation",
            )
        new_dataset = dataset.map(
            self.process,
            num_proc=self.runtime_np(),
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + "_process",
        )
        if tracer:
            tracer.trace_mapper(self._name, dataset, new_dataset, self.text_key)
        free_models()
        return new_dataset
