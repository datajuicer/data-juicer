from .base_op import OPERATORS


def load_ops(process_list, op_env_manager=None):
    """
    Load op list according to the process list from config file.

    :param process_list: A process list. Each item is an op name and its
        arguments.
    :param op_env_manager: The OPEnvManager to try to merge environment specs of different OPs that have common
        dependencies. Only available when min_common_dep_num_to_combine >= 0.
    :return: The op instance list.
    """
    ops = []
    new_process_list = []

    for process in process_list:
        op_name, args = list(process.items())[0]
        ops.append(OPERATORS.modules[op_name](**args))
        new_process_list.append(process)

    # store the OP configs into each OP
    for op_cfg, op in zip(new_process_list, ops):
        op._op_cfg = op_cfg

    # update op runtime environment if OPEnvManager is enabled
    if op_env_manager:
        # first round: record and merge possible common env specs
        for op in ops:
            op_name = op._name
            op_env_spec = op.get_env_spec()
            op_env_manager.record_op_env_spec(op_name, op_env_spec)
        # second round: update op runtime environment
        for op in ops:
            op_name = op._name
            op_env_spec = op_env_manager.get_op_env_spec(op_name)
            op._requirements = op_env_spec.pip_pkgs
            # if the runtime_env is not set for this OP, update the runtime_env as well
            if op.runtime_env is None:
                op.runtime_env = op_env_spec.to_dict()

    return ops
