from data_juicer.utils.model_utils import (
    get_model,
    prepare_model,
)

API_MODEL = "qwen3-max"


def chat(messages: list[dict]):
    model_key = prepare_model(
        model_type="api",
        model=API_MODEL,
    )
    _model = get_model(model_key, None, False)
    return _model(messages)
