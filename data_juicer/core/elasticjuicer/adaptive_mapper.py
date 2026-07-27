"""Lossless OOM-safe micro-batching for one mapper callable."""

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Callable, List, Optional

from .batch_controller import AdaptiveBatchController, MinimumBatchSizeOOM
from .oom import is_oom_error


def _batch_length(batch) -> int:
    if hasattr(batch, "num_rows"):
        return int(batch.num_rows)
    if isinstance(batch, Mapping):
        column_lengths = []
        for value in batch.values():
            if isinstance(value, (str, bytes)):
                continue
            try:
                column_lengths.append(len(value))
            except TypeError:
                continue
        if not column_lengths:
            raise ValueError("a mapping batch must contain at least one sized column")
        if len(set(column_lengths)) != 1:
            raise ValueError("all sized mapping columns must have the same row count")
        return column_lengths[0]
    return len(batch)


def _slice_batch(batch, start: int, end: int, total_rows: int):
    if hasattr(batch, "num_rows") and callable(getattr(batch, "slice", None)):
        return batch.slice(start, end - start)
    if isinstance(batch, Mapping):
        result = {}
        for key, value in batch.items():
            if isinstance(value, (str, bytes)):
                result[key] = value
                continue
            try:
                result[key] = value[start:end] if len(value) == total_rows else value
            except TypeError:
                result[key] = value
        return result
    return batch[start:end]


def _merge_outputs(outputs: List[Any]):
    if not outputs:
        return []
    if len(outputs) == 1:
        return outputs[0]

    first = outputs[0]
    if hasattr(first, "schema") and hasattr(first, "num_rows"):
        import pyarrow

        return pyarrow.concat_tables(outputs)
    if isinstance(first, Mapping):
        keys = list(first)
        expected_keys = set(keys)
        if any(set(output) != expected_keys for output in outputs):
            raise ValueError("mapper output schema changed across micro-batches")
        merged = {}
        for key in keys:
            values = [output[key] for output in outputs]
            first_value = values[0]
            if isinstance(first_value, (str, bytes)):
                if any(value != first_value for value in values[1:]):
                    raise ValueError(f"scalar output field {key!r} changed across micro-batches")
                merged[key] = first_value
                continue
            try:
                combined = []
                for value in values:
                    combined.extend(value)
            except TypeError:
                if any(value != first_value for value in values[1:]):
                    raise ValueError(f"scalar output field {key!r} changed across micro-batches")
                merged[key] = first_value
            else:
                merged[key] = tuple(combined) if isinstance(first_value, tuple) else combined
        return merged
    if isinstance(first, tuple):
        return tuple(item for output in outputs for item in output)

    merged = []
    for output in outputs:
        if isinstance(output, list):
            merged.extend(output)
        else:
            try:
                merged.extend(list(output))
            except TypeError:
                merged.append(output)
    return merged


class OOMSafeAdaptiveMapper:
    """Retry the same input slice at smaller sizes after classified OOMs."""

    def __init__(
        self,
        mapper: Callable,
        controller: AdaptiveBatchController,
        sampler=None,
        max_retries_per_slice: int = 16,
        oom_cleanup: Optional[Callable[[], None]] = None,
        validate_output_rows: bool = True,
        before_slice: Optional[Callable[[], None]] = None,
        snapshot_callback: Optional[Callable] = None,
    ):
        if max_retries_per_slice < 0:
            raise ValueError("max_retries_per_slice must be non-negative")
        self.mapper = mapper
        self.controller = controller
        self.sampler = sampler
        self.max_retries_per_slice = max_retries_per_slice
        self.oom_cleanup = oom_cleanup
        self.validate_output_rows = validate_output_rows
        self.before_slice = before_slice
        self.snapshot_callback = snapshot_callback
        self.oom_retries = 0
        self.successful_slices = 0

    def __call__(self, batch, *args, **kwargs):
        total_rows = _batch_length(batch)
        if total_rows == 0:
            return self.mapper(batch, *args, **kwargs)

        outputs = []
        offset = 0
        while offset < total_rows:
            retries = 0
            while True:
                if self.before_slice is not None:
                    self.before_slice()
                batch_size = self.controller.next_batch_size(total_rows - offset)
                microbatch = _slice_batch(batch, offset, offset + batch_size, total_rows)
                measurement = self.sampler.measure(batch_size) if self.sampler is not None else nullcontext()
                try:
                    with measurement:
                        output = self.mapper(microbatch, *args, **kwargs)
                except BaseException as error:
                    if not is_oom_error(error):
                        self._emit_snapshot(measurement)
                        raise
                    self.oom_retries += 1
                    retries += 1
                    try:
                        self.controller.observe_oom(batch_size)
                    except MinimumBatchSizeOOM:
                        self._emit_snapshot(measurement)
                        raise error
                    self._emit_snapshot(measurement)
                    if retries > self.max_retries_per_slice:
                        raise
                    if self.oom_cleanup is not None:
                        self.oom_cleanup()
                    continue

                if self.validate_output_rows:
                    output_rows = _batch_length(output)
                    if output_rows != batch_size:
                        self._emit_snapshot(measurement)
                        raise ValueError(
                            f"mapper returned {output_rows} rows for an {batch_size}-row input micro-batch"
                        )
                self.controller.observe_success(batch_size)
                self._emit_snapshot(measurement)
                outputs.append(output)
                offset += batch_size
                self.successful_slices += 1
                break

        return _merge_outputs(outputs)

    def _emit_snapshot(self, measurement) -> None:
        """Report a completed measurement after the controller transition."""

        if self.snapshot_callback is None:
            return
        snapshot = getattr(measurement, "snapshot", None)
        if snapshot is None:
            return
        try:
            self.snapshot_callback(snapshot)
        except Exception as error:
            from loguru import logger

            logger.warning(f"Failed to report ElasticJuicer actor metrics: {error}")
