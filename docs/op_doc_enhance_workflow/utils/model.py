from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
    update_sampling_params,
)

API_MODEL = "qwen3-max"


def chat(messages: list[dict]):
    sampling_params = update_sampling_params({}, API_MODEL, False)
    model_key = prepare_model(
        model_type="api",
        model=API_MODEL,
    )
    _model = get_model(model_key, None, False)
    return _model(messages, **sampling_params)
