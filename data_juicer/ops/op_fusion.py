from typing import List, Optional

import numpy as np
from loguru import logger

from data_juicer.ops.base_op import OP, OPERATORS, Filter, Mapper
from data_juicer.ops.load import load_ops
from data_juicer.utils.common_utils import check_op_method_param
from data_juicer.utils.constant import Fields, InterVars
from data_juicer.utils.registry import Registry

# Type of intermediate vars
# text
INTER_LINES = Registry(InterVars.lines)
INTER_WORDS = Registry(InterVars.words)

# images
LOADED_IMAGES = Registry(InterVars.loaded_images)

# audios
LOADED_AUDIOS = Registry(InterVars.loaded_audios)

# videos
LOADED_VIDEOS = Registry(InterVars.loaded_videos)
INTER_SAMPLED_FRAMES = Registry(InterVars.sampled_frames)

# all
ALL_INTER_VARS = [INTER_LINES, INTER_WORDS, LOADED_AUDIOS, LOADED_IMAGES, LOADED_VIDEOS, INTER_SAMPLED_FRAMES]

# supported fusion strategies
FUSION_STRATEGIES = {"greedy", "probe"}


def fuse_operators(ops, probe_res=None):
    """
    Fuse the input ops list and return the fused ops list.

    :param ops: the corresponding list of op objects.
    :param probe_res: the probed speed for each OP from Monitor.
    :return: a list of fused op objects.
    """
    if probe_res is None:
        probe_res = [None for _ in range(len(ops))]
    # detect filter groups and try to fuse them
    fused_ops = []
    filter_group = []
    for op, op_probe in zip(ops, probe_res):
        if isinstance(op, Filter):
            filter_group.append((op, op_probe))
        else:
            if filter_group:
                # got a filter group, try to fuse them
                fused_ops.extend(fuse_filter_group(filter_group))
                filter_group = []
            # and add the current non-filter op into fused_ops
            fused_ops.append(op)
    # the final filter group, try to fuse them
    if filter_group:
        fused_ops.extend(fuse_filter_group(filter_group))
    return fused_ops


def fuse_filter_group(original_filter_group):
    """
    Fuse single filter group and return the fused filter group.

    :param original_filter_group: the original filter group, including op
        definitions and objects.
    :return: the fused definitions and objects of the input filter group.
    """
    fused_group = []
    group_speed = []
    all_intermediate_vars = ALL_INTER_VARS
    all_fused_filters = {inter_vars: [] for inter_vars in all_intermediate_vars}
    # group these filters by their intermediate vars
    for op, probe_res in original_filter_group:
        op_name = op._name
        for inter_vars in all_intermediate_vars:
            if op_name in inter_vars.modules:
                all_fused_filters[inter_vars].append((op, probe_res))
                break
        else:
            # first apply other filters to decrease the number of samples, so
            # we add them into the fused_group list directly
            fused_group.append(op)
            group_speed.append(probe_res["speed"] if probe_res else 0)

    # try to fuse ops for each type of intermediate vars
    for inter_vars in all_intermediate_vars:
        inter_vars_filter = all_fused_filters[inter_vars]
        if len(inter_vars_filter) == 0:
            # no ops include this type of intermediate var
            pass
        elif len(inter_vars_filter) > 1:
            # more than 1 ops share the same intermediate var, try to fuse them
            ops, probe_res_list = zip(*inter_vars_filter)
            # new definition: new name and a definition list of fused op list
            fused_filter_name = "OpFusion:(%s)" % ",".join([op._name for op in ops])
            logger.info(f"Ops are fused into one op " f"{fused_filter_name}.")
            # use these ops to create a FusedFilter object, and add the fused
            # definition and op into the fused group
            fused_filter = FusedFilter(fused_filter_name, ops)
            fused_filter._op_cfg = {fused_filter_name: [op._op_cfg for op in ops]}
            fused_filter_speed = sum([1.0 / probe_res["speed"] for probe_res in probe_res_list if probe_res])
            if fused_filter_speed > 0:
                fused_filter_speed = 1.0 / fused_filter_speed
            fused_group.append(fused_filter)
            group_speed.append(fused_filter_speed)
        else:
            # only 1 op for this type of intermediate var, add it to the fused
            # group directly without fusion
            fused_group.append(inter_vars_filter[0][0])
            probe_res = inter_vars_filter[0][1]
            group_speed.append(probe_res["speed"] if probe_res else 0)

    # reorder according to the probed speed results in group_speed
    # 'greedy': all speed data in group_speed will be 0, which will keep the
    #   current order of fused group
    # 'probe': OPs in fused group will be reordered according to the speed data
    #   in group_speed in descending order
    fused_group = [op for op, _ in sorted(zip(fused_group, group_speed), key=lambda it: it[1], reverse=True)]

    return fused_group


class FusedFilter(Filter):
    """A fused operator for filters."""

    _batched_op = True

    def __init__(self, name: str, fused_filters: List):
        """
        Initialization method.

        :param fused_filters: a list of filters to be fused.
        """
        self._name = name
        super().__init__()
        self.fused_filters = fused_filters
        # set accelerator to 'cuda' if there exists any ops whose accelerator
        # is 'cuda'
        accelerator_methods = set([op.accelerator for op in self.fused_filters])
        if "cuda" in accelerator_methods:
            self.accelerator = "cuda"

        # update num_proc with the min num_proc of all fusible filters
        self.num_proc = min([op.runtime_np() for op in self.fused_filters])

    def compute_stats_batched(self, samples, rank=None):
        import av

        # context for the intermediate vars
        num_samples = len(samples[Fields.stats])
        samples[Fields.context] = [{} for _ in range(num_samples)]
        for op in self.fused_filters:
            # open the context for these fused ops
            if op.accelerator == "cuda":
                samples = op.compute_stats_batched(samples, rank=rank, context=True)
            else:
                samples = op.compute_stats_batched(samples, context=True)
        # clean up the contexts after processing
        # check if there are containers that need to be closed
        for ctx in samples[Fields.context]:
            for context_key in ctx:
                if isinstance(ctx[context_key], av.container.InputContainer):
                    ctx[context_key].streams.video[0].close()
                    ctx[context_key].close()
        _ = samples.pop(Fields.context)
        return samples

    def process_batched(self, samples):
        # Only return True when all filters return True
        res = None
        for op in self.fused_filters:
            this_res = np.array(list(op.process_batched(samples)))
            if res is not None:
                res = np.logical_and(res, this_res)
            else:
                res = this_res
        return res


@OPERATORS.register_module("general_fused_op")
class GeneralFusedOP(Mapper):
    """An explicitly fused operator designed to execute multiple sequential operations (OPs) on
    the same batch, enabling fine-grained control over data processing.

    This operator allows for the chaining of multiple data processing steps, such as mappers
    and filters, into a single pass. It processes each batch of samples sequentially through
    the defined operations, ensuring that all specified transformations are applied in
    order. The operator supports both mappers, which transform data, and filters, which
    remove or keep samples based on computed statistics. Context variables can be passed
    between operations if needed. The accelerator is set to 'cuda' if any of the fused
    operations use it. The number of processes is determined by the minimum value among all
    fused operations. After processing, any temporary context variables, such as those used
    for video containers, are cleaned up."""

    _batched_op = True

    def __init__(self, batch_size: int = 1, fused_op_list: Optional[List] = None, *args, **kwargs):
        """
        Initialization.

        :param batch_size: the batch size of the input samples.
        :param fused_op_list: a list of OPs to be fused.
        """
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        if fused_op_list is None:
            fused_op_list = []
        self.fused_ops = load_ops(fused_op_list)
        self._name = "GeneralFusedOP:(%s)" % ",".join([op._name for op in self.fused_ops])
        # set accelerator to 'cuda' if there exists any ops whose accelerator
        # is 'cuda'
        accelerator_methods = set([op.accelerator for op in self.fused_ops])
        if "cuda" in accelerator_methods:
            self.accelerator = "cuda"

        # update num_proc with the min num_proc of all fusible filters
        self.num_proc = min([op.runtime_np() for op in self.fused_ops]) if self.fused_ops else 1

    def process_batched(self, samples, rank=None):
        from copy import deepcopy

        import av

        tmp_samples = deepcopy(samples)

        # context for the intermediate vars
        sample_key = list(tmp_samples.keys())[0]
        num_samples = len(tmp_samples[sample_key])
        tmp_samples[Fields.context] = [{} for _ in range(num_samples)]

        for op in self.fused_ops:
            process_args = {"rank": rank} if op.accelerator == "cuda" else {}
            if isinstance(op, Mapper):
                if check_op_method_param(op.process, "context"):
                    # add context param only when the core process method of this OP contains this param
                    process_args["context"] = True
                samples = op.process_batched(tmp_samples, **process_args)
            elif isinstance(op, Filter):
                if check_op_method_param(op.compute_stats, "context"):
                    # add context param only when the core process method of this OP contains this param
                    process_args["context"] = True
                tmp_samples = op.compute_stats_batched(tmp_samples, **process_args)
                indicators = list(op.process_batched(tmp_samples))
                new_samples = {}
                for key in tmp_samples:
                    new_samples[key] = [val for val, indicator in zip(tmp_samples[key], indicators) if indicator]
                tmp_samples = new_samples
            else:
                raise NotImplementedError(
                    f"FusedOP does not support OP {op._name} of type "
                    f"{type(op)} and only supports Mapper and Filter now."
                )
        # clean up the contexts after processing
        # check if there are containers that need to be closed
        for ctx in tmp_samples[Fields.context]:
            for context_key in ctx:
                if isinstance(ctx[context_key], av.container.InputContainer):
                    ctx[context_key].streams.video[0].close()
                    ctx[context_key].close()
        _ = tmp_samples.pop(Fields.context)
        return tmp_samples

    def process_batched_for_validation(self, samples, rank=None):
        """Process samples and return boolean masks for validation purposes."""
        # Initialize mask as all True
        mask = [True] * len(samples.get("text", []))

        for op in self.fused_ops:
            process_args = {"rank": rank} if op.accelerator == "cuda" else {}
            if isinstance(op, Mapper):
                samples = op.process_batched(samples, **process_args)
            elif isinstance(op, Filter):
                samples = op.compute_stats_batched(samples, **process_args)
                indicators = list(op.process_batched(samples))
                # Apply AND logic with the current mask
                mask = [m and i for m, i in zip(mask, indicators)]
            else:
                raise NotImplementedError(
                    f"FusedOP does not support OP {op._name} of type "
                    f"{type(op)} and only supports Mapper and Filter now."
                )
        return mask

    def run(self, dataset, *, exporter=None, tracer=None):
        # prepare the dataset
        from data_juicer.core.data import NestedDataset

        if not isinstance(dataset, NestedDataset):
            dataset = NestedDataset(dataset)
        if not self.fused_ops:
            return dataset
        # initialize for different kinds of datasets
        for op in self.fused_ops:
            dataset = OP.run(op, dataset)

        new_dataset = dataset.map(
            self.process_batched,
            num_proc=self.num_proc,
            with_rank=self.use_cuda(),
            batch_size=self.batch_size,
            desc=self._name + "_process",
        )
        return new_dataset
