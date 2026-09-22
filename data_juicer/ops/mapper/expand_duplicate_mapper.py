import copy

from pydantic import PositiveInt

from ..base_op import OPERATORS, Mapper

OP_NAME = "expand_duplicate_mapper"


@OPERATORS.register_module(OP_NAME)
class ExpandDuplicateMapper(Mapper):
    """Deterministic row-EXPANDING (1:many) mapper for testing streaming recovery.

    For each input sample this emits ``expand_num`` output rows that are exact
    deep copies of the input, tagged with a stable copy index in ``marker_key``
    (0..expand_num-1). One input row therefore becomes ``expand_num`` output
    rows, so ``expand_num >= 2`` grows the dataset (out > in).

    It is intentionally CPU-only, model-free, and fully deterministic: given the
    same input the output multiset is identical every run. That makes it the
    ideal probe for the streaming tee-sink's 1:many recovery path -- exactly-once
    can be checked by counting and by the ``(row_id, copy_idx)`` identity of
    every output row (no copy lost, none duplicated, none resurrected on resume).
    The copied rows carry the parent's streaming row id unchanged, which is
    exactly the frontier behavior the recovery machinery must tolerate.
    """

    _batched_op = True

    def __init__(
        self,
        expand_num: PositiveInt = 2,
        marker_key: str = "__dup_copy_idx__",
        *args,
        **kwargs,
    ):
        """
        :param expand_num: number of output rows produced per input row.
        :param marker_key: field written on each output row with its 0-based
            copy index, so duplicates are distinguishable for verification.
        """
        super().__init__(*args, **kwargs)
        self.expand_num = expand_num
        self.marker_key = marker_key

    def process_batched(self, samples):
        keys = list(samples.keys())
        if not keys:
            return samples
        n = len(samples[keys[0]])
        out = {k: [] for k in keys}
        out[self.marker_key] = []
        for i in range(n):
            for copy_idx in range(self.expand_num):
                for k in keys:
                    out[k].append(copy.deepcopy(samples[k][i]))
                out[self.marker_key].append(copy_idx)
        return out
