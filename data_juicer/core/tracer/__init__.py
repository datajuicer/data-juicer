from data_juicer.utils.ray_utils import is_ray_mode

from .tracer import Tracer


def check_tracer_collect_complete(tracer_instance, op_name):
    if is_ray_mode():
        import ray

        return ray.get(tracer_instance.is_collection_complete.remote(op_name))
    else:
        return tracer_instance.is_collection_complete(op_name)


def should_trace_op(tracer_instance, op_name):
    if is_ray_mode():
        import ray

        return ray.get(tracer_instance.should_trace_op.remote(op_name))
    else:
        return tracer_instance.should_trace_op(op_name)


def collect_for_mapper(tracer_instance, op_name, original_sample_dict, processed_sample_dict, text_key):
    if is_ray_mode():
        import ray

        ray.get(
            tracer_instance.collect_mapper_sample.remote(op_name, original_sample_dict, processed_sample_dict, text_key)
        )
    else:
        tracer_instance.collect_mapper_sample(op_name, original_sample_dict, processed_sample_dict, text_key)


def collect_for_filter(tracer_instance, op_name, sample, should_keep):
    if is_ray_mode():
        import ray

        ray.get(tracer_instance.collect_filter_sample.remote(op_name, sample, should_keep))
    else:
        tracer_instance.collect_filter_sample(op_name, sample, should_keep)
