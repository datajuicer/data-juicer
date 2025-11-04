import fnmatch
import inspect
import io
import os
from contextlib import redirect_stderr
from functools import partial
from pickle import UnpicklingError
from typing import Optional, Union

import httpx
import multiprocess as mp
import wget
from loguru import logger

from data_juicer.utils.common_utils import nested_access
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.nltk_utils import (
    ensure_nltk_resource,
    patch_nltk_pickle_security,
)
from data_juicer.utils.resource_utils import cuda_device_count

from .cache_utils import DATA_JUICER_EXTERNAL_MODELS_HOME as DJEMH
from .cache_utils import DATA_JUICER_MODELS_CACHE as DJMC

torch = LazyLoader("torch")
transformers = LazyLoader("transformers")
nn = LazyLoader("torch.nn")
fasttext = LazyLoader("fasttext", "fasttext-wheel")
sentencepiece = LazyLoader("sentencepiece")
kenlm = LazyLoader("kenlm")
nltk = LazyLoader("nltk")
aes_pred = LazyLoader("aesthetics_predictor", "simple-aesthetics-predictor")
vllm = LazyLoader("vllm")
diffusers = LazyLoader("diffusers")
ram = LazyLoader("ram", "git+https://github.com/xinyu1205/recognize-anything.git")
cv2 = LazyLoader("cv2", "opencv-python")
openai = LazyLoader("openai")
ultralytics = LazyLoader("ultralytics")
tiktoken = LazyLoader("tiktoken")
dashscope = LazyLoader("dashscope")

MODEL_ZOO = {}

# Default cached models links for downloading
MODEL_LINKS = "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/" "data_juicer/models/"

# Backup cached models links for downloading
BACKUP_MODEL_LINKS = {
    # language identification model from fasttext
    "lid.176.bin": "https://dl.fbaipublicfiles.com/fasttext/supervised-models/",
    # tokenizer and language model for English from sentencepiece and KenLM
    "*.sp.model": "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/",
    "*.arpa.bin": "https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/",
    # sentence split model from nltk punkt
    "punkt.*.pickle": "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/" "data_juicer/models/",
    # ram
    "ram_plus_swin_large_14m.pth": "http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/data_juicer/models/"
    "ram_plus_swin_large_14m.pth",
    # FastSAM
    "FastSAM-s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/" "FastSAM-s.pt",
    "FastSAM-x.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/" "FastSAM-x.pt",
    # spacy
    "*_core_web_md-3.*.0": "https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/" "data_juicer/models/",
    # YOLO
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
}


def get_backup_model_link(model_name):
    for pattern, url in BACKUP_MODEL_LINKS.items():
        if fnmatch.fnmatch(model_name, pattern):
            return url
    return None


def check_model(model_name, force=False):
    """
    Check whether a model exists in DATA_JUICER_MODELS_CACHE.
    If exists, return its full path.
    Else, download it from cached models links.

    :param model_name: a specified model name
    :param force: Whether to download model forcefully or not, Sometimes
        the model file maybe incomplete for some reason, so need to
        download again forcefully.
    """
    # check for local model
    if not force and os.path.exists(model_name):
        return model_name

    if not force and DJEMH:
        external_paths = DJEMH.split(os.pathsep)
        for path in external_paths:
            clean_path = path.strip()
            if not clean_path:
                continue
            model_path = os.path.join(clean_path, model_name)
            if os.path.exists(model_path):
                return model_path

    if not os.path.exists(DJMC):
        os.makedirs(DJMC)

    # check if the specified model exists. If it does not exist, download it
    cached_model_path = os.path.join(DJMC, model_name)
    if force:
        if os.path.exists(cached_model_path):
            os.remove(cached_model_path)
            logger.info(f"Model [{cached_model_path}] is invalid. Forcing download...")
        else:
            logger.info(f"Model [{cached_model_path}] is not found. Downloading...")

        model_link = os.path.join(MODEL_LINKS, model_name)
        try:
            wget.download(model_link, cached_model_path)
        except:  # noqa: E722
            backup_model_link = get_backup_model_link(model_name)
            if backup_model_link is not None:
                backup_model_link = os.path.join(backup_model_link, model_name)
            try:
                wget.download(backup_model_link, cached_model_path)
            except:  # noqa: E722
                import traceback

                traceback.print_exc()
                raise RuntimeError(
                    f"Downloading model [{model_name}] error. "
                    f"Please retry later or download it into {DJMC} "
                    f"manually from {model_link} or {backup_model_link} "
                )
    return cached_model_path


def check_model_home(model_name):
    if not DJEMH:
        return model_name

    cached_model_path = os.path.join(DJEMH, model_name)
    if os.path.exists(cached_model_path):
        return cached_model_path
    return model_name


def filter_arguments(func, args_dict):
    """
    Filters and returns only the valid arguments for a given function
    signature.

    :param func: The function or callable to inspect.
    :param args_dict: A dictionary of argument names and values to filter.
    :return: A dictionary containing only the arguments that match the
                function's signature, preserving any **kwargs if applicable.
    """
    params = inspect.signature(func).parameters
    filtered_args = {}
    for name, param in params.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return args_dict
        if name not in {"self", "cls"} and name in args_dict:
            filtered_args[name] = args_dict[name]
    return filtered_args


class ChatAPIModel:
    def __init__(self, model=None, endpoint=None, response_path=None, **kwargs):
        """
        Initializes an instance of the APIModel class.

        :param model: The name of the model to be used for making API
            calls. This should correspond to a valid model identifier
            recognized by the API server. If it's None, use the first available model from the server.
        :param endpoint: The URL endpoint for the API. If provided as a
            relative path, it will be appended to the base URL (defined by the
            `OPENAI_BASE_URL` environment variable or through an additional
            `base_url` parameter). Defaults to '/chat/completions' for
            OpenAI compatibility.
        :param response_path: A dot-separated string specifying the path to
            extract the desired content from the API response. The default
            value is 'choices.0.message.content', which corresponds to the
            typical structure of an OpenAI API response.
        :param kwargs: Additional keyword arguments for configuring the
            internal OpenAI client.
        """
        self.model = model
        self.endpoint = endpoint or "/chat/completions"
        self.response_path = response_path or "choices.0.message.content"

        client_args = filter_arguments(openai.OpenAI, kwargs)
        self._client = openai.OpenAI(**client_args)
        if self.model is None:
            logger.warning("No model specified. Using the first available model from the server.")
            models_list = self._client.models.list().data
            if len(models_list) == 0:
                raise ValueError("No models available on the server.")
            self.model = models_list[0].id

    def __call__(self, messages, **kwargs):
        """
        Sends messages to the configured API model and returns the parsed
        response content.

        :param messages: A list of message dictionaries to send to the API.
                         Each message should have a 'role' (e.g., 'user',
                         'assistant') and 'content' (the message text).
        :param kwargs: Additional parameters for the API call.
        :return: The parsed response content from the API call, or an empty
            string if an error occurs.
        """
        body = {
            "messages": messages,
            "model": self.model,
        }
        body.update(kwargs)
        stream = kwargs.get("stream", False)
        stream_cls = openai.Stream[openai.types.chat.ChatCompletionChunk]

        try:
            response = self._client.post(
                self.endpoint, body=body, cast_to=httpx.Response, stream=stream, stream_cls=stream_cls
            )
            result = response.json()
            return nested_access(result, self.response_path)
        except Exception as e:
            logger.exception(e)
            return ""


class EmbeddingAPIModel:
    def __init__(self, model=None, endpoint=None, response_path=None, **kwargs):
        """
        Initializes an instance specialized for embedding APIs.

        :param model: The model identifier for embedding API calls.
            If it's None, use the first available model from the server.
        :param endpoint: API endpoint URL. Defaults to '/embeddings'.
        :param response_path: Path to extract embeddings from response.
            Defaults to 'data.0.embedding'.
        :param kwargs: Configuration for the OpenAI client.
        """
        self.model = model
        self.endpoint = endpoint or "/embeddings"
        self.response_path = response_path or "data.0.embedding"

        client_args = filter_arguments(openai.OpenAI, kwargs)
        self._client = openai.OpenAI(**client_args)
        if self.model is None:
            logger.warning("No model specified. Using the first available model from the server.")
            if len(self._client.models.list().data) == 0:
                raise ValueError("No models available on the server.")
            self.model = self._client.models.list().data[0].id

    def __call__(self, input, **kwargs):
        """
        Processes input text and returns embeddings.

        :param input: Input text or list of texts to embed.
        :param kwargs: Additional API parameters.
        :return: Extracted embeddings or empty list on error.
        """
        body = {
            "model": self.model,
            "input": input,
        }
        body.update(kwargs)

        try:
            response = self._client.post(self.endpoint, body=body, cast_to=httpx.Response)
            result = response.json()
            return nested_access(result, self.response_path) or []
        except Exception as e:
            logger.exception(f"Embedding API error: {e}")
            return []


def prepare_api_model(
    model, *, endpoint=None, response_path=None, return_processor=False, processor_config=None, **model_params
):
    """Creates a callable API model for interacting with OpenAI-compatible API.
    The callable supports custom response parsing and works with proxy servers
    that may be incompatible.

    :param model: The name of the model to interact with.
    :param endpoint: The URL endpoint for the API. If provided as a relative
        path, it will be appended to the base URL (defined by the
        `OPENAI_BASE_URL` environment variable or through an additional
        `base_url` parameter). Supported endpoints include:
        - '/chat/completions' for chat models
        - '/embeddings' for embedding models
        Defaults to `/chat/completions` for OpenAI compatibility.
    :param response_path: The dot-separated  path to extract desired content
        from the API response. Defaults to 'choices.0.message.content'
        for chat models and 'data.0.embedding' for embedding models.
    :param return_processor: A boolean flag indicating whether to return a
        processor along with the model. The processor can be used for tasks
        like tokenization or encoding. Defaults to False.
    :param processor_config: A dictionary containing configuration parameters
        for initializing a Hugging Face processor. It is only relevant if
        `return_processor` is set to True.
    :param model_params: Additional parameters for configuring the API model.
    :return: A callable APIModel instance, and optionally a processor
        if `return_processor` is True.
    """
    endpoint = endpoint or "/chat/completions"

    ENDPOINT_CLASS_MAP = {
        "chat": ChatAPIModel,
        "embeddings": EmbeddingAPIModel,
    }

    API_Class = next((cls for keyword, cls in ENDPOINT_CLASS_MAP.items() if keyword in endpoint.lower()), None)

    if API_Class is None:
        raise ValueError(f"Unsupported endpoint: {endpoint}")

    client = API_Class(model=model, endpoint=endpoint, response_path=response_path, **model_params)

    if not return_processor:
        return client

    def get_processor():
        try:
            return tiktoken.encoding_for_model(check_model_home(model))
        except Exception:
            pass

        try:
            return dashscope.get_tokenizer(check_model_home(model))
        except Exception:
            pass

        try:
            processor = transformers.AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=check_model_home(model), **processor_config
            )
            return processor
        except Exception:
            pass

        raise ValueError(
            "Failed to initialize the processor. Please check the following:\n"  # noqa: E501
            "- For OpenAI models: Install 'tiktoken' via `pip install tiktoken`.\n"  # noqa: E501
            "- For DashScope models: Install both 'dashscope' and 'tiktoken' via `pip install dashscope tiktoken`.\n"  # noqa: E501
            "- For custom models: Use the 'processor_config' parameter to configure a Hugging Face processor."  # noqa: E501
        )

    if processor_config is not None and "pretrained_model_name_or_path" in processor_config:
        processor = transformers.AutoProcessor.from_pretrained(**processor_config)
    else:
        processor = get_processor()
    return (client, processor)


def prepare_diffusion_model(pretrained_model_name_or_path, diffusion_type, **model_params):
    """
    Prepare and load an Diffusion model from HuggingFace.

    :param pretrained_model_name_or_path: input Diffusion model name
        or local path to the model
    :param diffusion_type: the use of the diffusion model. It can be
        'image2image', 'text2image', 'inpainting'
    :return: a Diffusion model.
    """

    TORCH_DTYPE_MAPPING = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    LazyLoader.check_packages(["torch", "transformers"])

    device = model_params.pop("device", None)
    if not device:
        model_params["device_map"] = "balanced"
    if "torch_dtype" in model_params:
        model_params["torch_dtype"] = TORCH_DTYPE_MAPPING[model_params["torch_dtype"]]

    diffusion_type_to_pipeline = {
        "image2image": diffusers.AutoPipelineForImage2Image,
        "text2image": diffusers.AutoPipelineForText2Image,
        "inpainting": diffusers.AutoPipelineForInpainting,
    }

    if diffusion_type not in diffusion_type_to_pipeline.keys():
        raise ValueError(
            f"Not support {diffusion_type} diffusion_type for diffusion "
            "model. Can only be one of "
            '["image2image", "text2image", "inpainting"].'
        )

    pipeline = diffusion_type_to_pipeline[diffusion_type]
    model = pipeline.from_pretrained(check_model_home(pretrained_model_name_or_path), **model_params)
    if device:
        model = model.to(device)

    return model


def prepare_fastsam_model(model_path, **model_params):
    device = model_params.pop("device", "cpu")
    model = ultralytics.FastSAM(check_model(model_path)).to(device)
    return model


def prepare_fasttext_model(model_name="lid.176.bin", **model_params):
    """
    Prepare and load a fasttext model.

    :param model_name: input model name
    :return: model instance.
    """
    logger.info("Loading fasttext language identification model...")
    try:
        # Suppress FastText warnings by redirecting stderr
        with redirect_stderr(io.StringIO()):
            ft_model = fasttext.load_model(check_model(model_name))
        # Verify the model has the predict method (for language identification)
        if not hasattr(ft_model, "predict"):
            raise AttributeError("Loaded model does not support prediction")
    except Exception as e:
        logger.warning(f"Error loading model: {e}. Attempting to force download...")
        try:
            with redirect_stderr(io.StringIO()):
                ft_model = fasttext.load_model(check_model(model_name, force=True))
            if not hasattr(ft_model, "predict"):
                raise AttributeError("Loaded model does not support prediction")
        except Exception as e:
            logger.error(f"Failed to load model after download attempt: {e}")
            raise
    return ft_model


def prepare_huggingface_model(
    pretrained_model_name_or_path, *, return_model=True, return_pipe=False, pipe_task="text-generation", **model_params
):
    """
    Prepare and load a huggingface model.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :param return_pipe: return pipeline or not
    :param pipe_task: task for pipeline
    :return: a tuple (model, processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    # Check if we need accelerate for device_map
    if "device" in model_params:
        device = model_params.pop("device")
        if device.startswith("cuda"):
            try:
                model_params["device_map"] = device
            except ImportError:
                # If accelerate is not available, use device directly
                model_params["device"] = device
                logger.warning("accelerate not found, using device directly")

    pretrained_model_name_or_path = check_model_home(pretrained_model_name_or_path)
    processor = transformers.AutoProcessor.from_pretrained(pretrained_model_name_or_path, **model_params)

    if return_model:
        config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path, **model_params)
        if hasattr(config, "auto_map"):
            class_name = next((k for k in config.auto_map if k.startswith("AutoModel")), "AutoModel")
        else:
            # TODO: What happens if more than one
            class_name = config.architectures[0]

        model_class = getattr(transformers, class_name)
        model = model_class.from_pretrained(pretrained_model_name_or_path, **model_params)

        if return_pipe:
            if isinstance(processor, transformers.PreTrainedTokenizerBase):
                pipe_params = {"tokenizer": processor}
            elif isinstance(processor, transformers.SequenceFeatureExtractor):
                pipe_params = {"feature_extractor": processor}
            elif isinstance(processor, transformers.BaseImageProcessor):
                pipe_params = {"image_processor": processor}
            pipe = transformers.pipeline(task=pipe_task, model=model, config=config, **pipe_params)
            model = pipe

    return (model, processor) if return_model else processor


def prepare_kenlm_model(lang, name_pattern="{}.arpa.bin", **model_params):
    """
    Prepare and load a kenlm model.

    :param model_name: input model name in formatting syntax.
    :param lang: language to render model name
    :return: model instance.
    """
    model_params.pop("device", None)

    model_name = name_pattern.format(lang)

    logger.info("Loading kenlm language model...")
    try:
        kenlm_model = kenlm.Model(check_model(model_name), **model_params)
    except:  # noqa: E722
        kenlm_model = kenlm.Model(check_model(model_name, force=True), **model_params)
    return kenlm_model


def prepare_nltk_model(lang, name_pattern="punkt.{}.pickle", **model_params):
    """
    Prepare and load a nltk punkt model with enhanced resource handling.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """
    model_params.pop("device", None)

    # Ensure pickle security is patched
    patch_nltk_pickle_security()

    nltk_to_punkt = {"en": "english", "fr": "french", "pt": "portuguese", "es": "spanish"}
    assert lang in nltk_to_punkt.keys(), "lang must be one of the following: {}".format(list(nltk_to_punkt.keys()))

    logger.info("Loading nltk punkt split model...")

    try:
        # Resource path and fallback for the punkt model
        resource_path = f"tokenizers/punkt/{nltk_to_punkt[lang]}.pickle"

        # Ensure the resource is available
        if ensure_nltk_resource(resource_path, "punkt"):
            logger.info(f"Successfully verified resource {resource_path}")
        else:
            logger.warning(f"Could not verify resource {resource_path}, model may not " f"work correctly")

        # Load the model
        nltk_model = nltk.data.load(resource_path, **model_params)
    except Exception as e:
        # Fallback to downloading and retrying
        logger.warning(f"Error loading model: {e}. Attempting to download...")
        try:
            nltk.download("punkt", quiet=False)
            nltk_model = nltk.data.load(resource_path, **model_params)
        except Exception as download_error:
            logger.error(f"Failed to load model after download " f"attempt: {download_error}")
            raise

    return nltk_model


def prepare_nltk_pos_tagger(**model_params):
    """
    Prepare and load NLTK's part-of-speech tagger with enhanced resource
      handling.

    :return: The POS tagger model
    """
    model_params.pop("device", None)

    # Ensure pickle security is patched
    patch_nltk_pickle_security()

    logger.info("Loading NLTK POS tagger model...")

    try:
        # Resource path and fallback for the averaged_perceptron_tagger
        resource_path = "taggers/averaged_perceptron_tagger/english.pickle"

        # Ensure the resource is available
        if ensure_nltk_resource(resource_path, "averaged_perceptron_tagger"):
            logger.info(f"Successfully verified resource {resource_path}")
        else:
            logger.warning(f"Could not verify resource {resource_path}, model may not " f"work correctly")

        # Import the POS tagger
        import nltk.tag

        tagger = nltk.tag.pos_tag
    except Exception as e:
        # Fallback to downloading and retrying
        logger.warning(f"Error loading POS tagger: {e}. Attempting to download...")
        try:
            nltk.download("averaged_perceptron_tagger", quiet=False)
            import nltk.tag

            tagger = nltk.tag.pos_tag
        except Exception as download_error:
            logger.error(f"Failed to load POS tagger after download " f"attempt: {download_error}")
            raise

    return tagger


def prepare_opencv_classifier(model_path, **model_params):
    model = cv2.CascadeClassifier(model_path)
    return model


def prepare_recognizeAnything_model(
    pretrained_model_name_or_path="ram_plus_swin_large_14m.pth", input_size=384, **model_params
):
    """
    Prepare and load recognizeAnything model.

    :param model_name: input model name.
    :param input_size: the input size of the model.
    """
    logger.info("Loading recognizeAnything model...")

    try:
        model = ram.models.ram_plus(
            pretrained=check_model(pretrained_model_name_or_path), image_size=input_size, vit="swin_l"
        )
    except (RuntimeError, UnpicklingError) as e:  # noqa: E722
        logger.warning(e)
        model = ram.models.ram_plus(
            pretrained=check_model(pretrained_model_name_or_path, force=True), image_size=input_size, vit="swin_l"
        )
    device = model_params.pop("device", "cpu")
    model.to(device).eval()
    return model


def prepare_sdxl_prompt2prompt(pretrained_model_name_or_path, pipe_func, torch_dtype="fp32", device="cpu"):
    pretrained_model_name_or_path = check_model_home(pretrained_model_name_or_path)
    if torch_dtype == "fp32":
        model = pipe_func.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.float32, use_safetensors=True
        ).to(device)
    else:
        model = pipe_func.from_pretrained(
            pretrained_model_name_or_path, torch_dtype=torch.float16, use_safetensors=True
        ).to(device)
    return model


def prepare_sentencepiece_model(model_path, **model_params):
    """
    Prepare and load a sentencepiece model.

    :param model_path: input model path
    :return: model instance
    """
    logger.info("Loading sentencepiece model...")
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    try:
        sentencepiece_model.load(check_model(model_path))
    except:  # noqa: E722
        sentencepiece_model.load(check_model(model_path, force=True))
    return sentencepiece_model


def prepare_sentencepiece_for_lang(lang, name_pattern="{}.sp.model", **model_params):
    """
    Prepare and load a sentencepiece model for specific language.

    :param lang: language to render model name
    :param name_pattern: pattern to render the model name
    :return: model instance.
    """

    model_name = name_pattern.format(lang)
    return prepare_sentencepiece_model(model_name)


def prepare_simple_aesthetics_model(pretrained_model_name_or_path, *, return_model=True, **model_params):
    """
    Prepare and load a simple aesthetics model.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :return: a tuple (model, input processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    # Check if we need accelerate for device_map
    if "device" in model_params:
        device = model_params.pop("device")
        if device.startswith("cuda"):
            try:
                model_params["device_map"] = device
            except ImportError:
                # If accelerate is not available, use device directly
                model_params["device"] = device
                logger.warning("accelerate not found, using device directly")

    pretrained_model_name_or_path = check_model_home(pretrained_model_name_or_path)
    processor = transformers.CLIPProcessor.from_pretrained(pretrained_model_name_or_path, **model_params)
    if not return_model:
        return processor
    else:
        if "v1" in pretrained_model_name_or_path:
            model = aes_pred.AestheticsPredictorV1.from_pretrained(pretrained_model_name_or_path, **model_params)
        elif "v2" in pretrained_model_name_or_path and "linear" in pretrained_model_name_or_path:
            model = aes_pred.AestheticsPredictorV2Linear.from_pretrained(pretrained_model_name_or_path, **model_params)
        elif "v2" in pretrained_model_name_or_path and "relu" in pretrained_model_name_or_path:
            model = aes_pred.AestheticsPredictorV2ReLU.from_pretrained(pretrained_model_name_or_path, **model_params)
        else:
            raise ValueError("Not support {}".format(pretrained_model_name_or_path))
        return (model, processor)


def prepare_spacy_model(lang, name_pattern="{}_core_web_md-3.7.0", **model_params):
    """
    Prepare spacy model for specific language.

    :param lang: language of sapcy model. Should be one of ["zh",
        "en"]
    :return: corresponding spacy model
    """
    import spacy

    assert lang in ["zh", "en"], "Diversity only support zh and en"
    model_name = name_pattern.format(lang)
    logger.info(f"Loading spacy model [{model_name}]...")
    compressed_model = "{}.tar.gz".format(model_name)

    # decompress the compressed model if it's not decompressed
    def decompress_model(compressed_model_path):
        if not compressed_model_path.endswith(".tar.gz"):
            raise ValueError("Only .tar.gz files are supported")

        decompressed_model_path = compressed_model_path.replace(".tar.gz", "")
        if os.path.exists(decompressed_model_path) and os.path.isdir(decompressed_model_path):
            return decompressed_model_path

        ver_name = os.path.basename(decompressed_model_path)
        unver_name = ver_name.rsplit("-", maxsplit=1)[0]
        target_dir_in_archive = f"{ver_name}/{unver_name}/{ver_name}/"

        import tarfile

        with tarfile.open(compressed_model_path, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith(target_dir_in_archive):
                    # relative path without unnecessary directory levels
                    relative_path = os.path.relpath(member.name, start=target_dir_in_archive)
                    target_path = os.path.join(decompressed_model_path, relative_path)

                    if member.isfile():
                        # ensure the directory exists
                        target_directory = os.path.dirname(target_path)
                        os.makedirs(target_directory, exist_ok=True)
                        # for files, extract to the specific location
                        with tar.extractfile(member) as source:
                            with open(target_path, "wb") as target:
                                target.write(source.read())
        return decompressed_model_path

    try:
        diversity_model = spacy.load(decompress_model(check_model(compressed_model)))
    except:  # noqa: E722
        diversity_model = spacy.load(decompress_model(check_model(compressed_model, force=True)))
    return diversity_model


def prepare_video_blip_model(pretrained_model_name_or_path, *, return_model=True, **model_params):
    """
    Prepare and load a video-clip model with the corresponding processor.

    :param pretrained_model_name_or_path: model name or path
    :param return_model: return model or not
    :param trust_remote_code: passed to transformers
    :return: a tuple (model, input processor) if `return_model` is True;
        otherwise, only the processor is returned.
    """
    if "device" in model_params:
        model_params["device_map"] = model_params.pop("device")

    class VideoBlipVisionModel(transformers.Blip2VisionModel):
        """A simple, augmented version of Blip2VisionModel to handle
        videos."""

        def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            interpolate_pos_encoding: bool = False,
        ) -> Union[tuple, transformers.modeling_outputs.BaseModelOutputWithPooling]:
            """Flatten `pixel_values` along the batch and time dimension,
            pass it through the original vision model,
            then unflatten it back.

            :param pixel_values: a tensor of shape
            (batch, channel, time, height, width)

            :returns:
                last_hidden_state: a tensor of shape
                (batch, time * seq_len, hidden_size)
                pooler_output: a tensor of shape
                (batch, time, hidden_size)
                hidden_states:
                    a tuple of tensors of shape
                    (batch, time * seq_len, hidden_size),
                    one for the output of the embeddings +
                    one for each layer
                attentions:
                    a tuple of tensors of shape
                    (batch, time, num_heads, seq_len, seq_len),
                    one for each layer
            """
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            batch, _, time, _, _ = pixel_values.size()

            # flatten along the batch and time dimension to create a
            # tensor of shape
            # (batch * time, channel, height, width)
            flat_pixel_values = pixel_values.permute(0, 2, 1, 3, 4).flatten(end_dim=1)

            vision_outputs: transformers.modeling_outputs.BaseModelOutputWithPooling = super().forward(  # noqa: E501
                pixel_values=flat_pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                interpolate_pos_encoding=interpolate_pos_encoding,
            )

            # now restore the original dimensions
            # vision_outputs.last_hidden_state is of shape
            # (batch * time, seq_len, hidden_size)
            seq_len = vision_outputs.last_hidden_state.size(1)
            last_hidden_state = vision_outputs.last_hidden_state.view(batch, time * seq_len, -1)
            # vision_outputs.pooler_output is of shape
            # (batch * time, hidden_size)
            pooler_output = vision_outputs.pooler_output.view(batch, time, -1)
            # hidden_states is a tuple of tensors of shape
            # (batch * time, seq_len, hidden_size)
            hidden_states = (
                tuple(hidden.view(batch, time * seq_len, -1) for hidden in vision_outputs.hidden_states)
                if vision_outputs.hidden_states is not None
                else None
            )
            # attentions is a tuple of tensors of shape
            # (batch * time, num_heads, seq_len, seq_len)
            attentions = (
                tuple(hidden.view(batch, time, -1, seq_len, seq_len) for hidden in vision_outputs.attentions)
                if vision_outputs.attentions is not None
                else None
            )
            if return_dict:
                return transformers.modeling_outputs.BaseModelOutputWithPooling(  # noqa: E501
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooler_output,
                    hidden_states=hidden_states,
                    attentions=attentions,
                )
            return (last_hidden_state, pooler_output, hidden_states, attentions)

    class VideoBlipForConditionalGeneration(transformers.Blip2ForConditionalGeneration):
        def __init__(self, config: transformers.Blip2Config) -> None:
            # HACK: we call the grandparent super().__init__() to bypass
            # transformers.Blip2ForConditionalGeneration.__init__() so we can
            # replace self.vision_model
            super(transformers.Blip2ForConditionalGeneration, self).__init__(config)

            self.vision_model = VideoBlipVisionModel(config.vision_config)

            self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
            self.qformer = transformers.Blip2QFormerModel(config.qformer_config)

            self.language_projection = nn.Linear(config.qformer_config.hidden_size, config.text_config.hidden_size)
            if config.use_decoder_only_language_model:
                language_model = transformers.AutoModelForCausalLM.from_config(config.text_config)
            else:
                language_model = transformers.AutoModelForSeq2SeqLM.from_config(config.text_config)  # noqa: E501
            self.language_model = language_model

            # Initialize weights and apply final processing
            self.post_init()

    pretrained_model_name_or_path = check_model_home(pretrained_model_name_or_path)
    processor = transformers.AutoProcessor.from_pretrained(pretrained_model_name_or_path, **model_params)
    if return_model:
        model_class = VideoBlipForConditionalGeneration
        model = model_class.from_pretrained(pretrained_model_name_or_path, **model_params)
    return (model, processor) if return_model else processor


def prepare_video_depth_anything(model_path, **model_params):
    import subprocess

    from data_juicer.utils.cache_utils import DATA_JUICER_ASSETS_CACHE

    video_depth_anything_repo_path = os.path.join(DATA_JUICER_ASSETS_CACHE, "Video-Depth-Anything")
    if not os.path.exists(video_depth_anything_repo_path):
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/DepthAnything/Video-Depth-Anything.git",
                video_depth_anything_repo_path,
            ],
            check=True,
        )
    import sys

    sys.path.append(video_depth_anything_repo_path)

    from video_depth_anything.video_depth import VideoDepthAnything

    device = model_params.pop("device", "cpu")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }

    model_links = {
        "video_depth_anything_vits": "https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth",
        "video_depth_anything_vitb": "https://huggingface.co/depth-anything/Video-Depth-Anything-Base/resolve/main/video_depth_anything_vitb.pth",
        "video_depth_anything_vitl": "https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth",
        "metric_video_depth_anything_vits": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/resolve/main/metric_video_depth_anything_vits.pth",
        "metric_video_depth_anything_vitb": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/resolve/main/metric_video_depth_anything_vitb.pth",
        "metric_video_depth_anything_vitl": "https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth",
    }

    if "vits" in model_path:
        encoder_type = "vits"
    elif "vitb" in model_path:
        encoder_type = "vitb"
    else:
        encoder_type = "vitl"

    if "metric" in model_path:
        metric = True
    else:
        metric = False

    if "metric_video_depth_anything_vitl" in model_path:
        model_type = "metric_video_depth_anything_vitl"
    elif "metric_video_depth_anything_vitb" in model_path:
        model_type = "metric_video_depth_anything_vitb"
    elif "metric_video_depth_anything_vits" in model_path:
        model_type = "metric_video_depth_anything_vits"
    elif "video_depth_anything_vitb" in model_path:
        model_type = "video_depth_anything_vitb"
    elif "video_depth_anything_vitl" in model_path:
        model_type = "video_depth_anything_vitl"
    else:
        model_type = "video_depth_anything_vits"

    if not os.path.exists(model_path):
        if not os.path.exists(DJMC):
            os.makedirs(DJMC)

        model_path = os.path.join(DJMC, model_type + ".pth")
        wget.download(model_links[model_type], model_path)

    model = VideoDepthAnything(**model_configs[encoder_type], metric=metric)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()

    return model


def prepare_yolo_model(model_path, **model_params):
    device = model_params.pop("device", "cpu")
    model = ultralytics.YOLO(check_model(model_path)).to(device)
    return model


def prepare_vllm_model(pretrained_model_name_or_path, **model_params):
    """
    Prepare and load a HuggingFace model with the corresponding processor.

    :param pretrained_model_name_or_path: model name or path
    :param model_params: LLM initialization parameters.
    :return: a tuple of (model, tokenizer)
    """
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    if model_params.get("device", "").startswith("cuda:"):
        model_params["device"] = "cuda"

    model = vllm.LLM(model=check_model_home(pretrained_model_name_or_path), generation_config="auto", **model_params)
    tokenizer = model.get_tokenizer()

    return (model, tokenizer)


def prepare_embedding_model(model_path, **model_params):
    """
    Prepare and load an embedding model using transformers.

    :param model_path: Path to the embedding model.
    :param model_params: Optional model parameters.
    :return: Model with encode() returning embedding list.
    """
    logger.info("Loading embedding model using transformers...")
    if "device" in model_params:
        device = model_params.pop("device")
    else:
        device = "cpu"
        logger.warning("'device' not specified in 'model_params'. Using 'cpu'.")
    if "pooling" in model_params:
        # pooling strategy to extract embedding from the hidden states. https://arxiv.org/abs/2503.01807
        # None: default option, the hidden state of the last token.
        # "mean": uniform mean of hidden states.
        # "weighted_mean": weighted mean of hidden states. https://arxiv.org/abs/2202.08904
        pooling = model_params.pop("pooling")
    else:
        pooling = None

    model_path = check_model_home(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = transformers.AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        mask = None
        if pooling not in ["mean", "weighted_mean"]:
            # return the embedding of the last token
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        elif pooling == "mean":
            mask = attention_mask
        elif pooling == "weighted_mean":
            if left_padding:
                sequence_lengths = attention_mask.sum(dim=1)
                tmp = list(range(1, attention_mask.shape[1] + 1))
                mask = torch.tensor([tmp[seq_len:] + tmp[:seq_len] for seq_len in sequence_lengths.tolist()]).to(
                    attention_mask.device
                )
            else:
                mask = torch.arange(1, attention_mask.shape[1] + 1)
            mask = mask * attention_mask / attention_mask.shape[1]
        masked_hidden_states = last_hidden_states * mask.unsqueeze(-1)
        return torch.mean(masked_hidden_states, dim=1)

    def encode(text, prompt_name=None, max_len=4096):
        if prompt_name:
            text = f"{prompt_name}: {text}"

        input_dict = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_len).to(device)

        with torch.no_grad():
            outputs = model(**input_dict)

        embedding = last_token_pool(outputs.last_hidden_state, input_dict["attention_mask"])
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding[0].tolist()

    return type("EmbeddingModel", (), {"encode": encode})()


def update_sampling_params(sampling_params, pretrained_model_name_or_path, enable_vllm=False):
    if enable_vllm:
        update_keys = {"max_tokens"}
    else:
        update_keys = {"max_new_tokens"}
    generation_config_keys = {
        "max_tokens": ["max_tokens", "max_new_tokens"],
        "max_new_tokens": ["max_tokens", "max_new_tokens"],
    }
    generation_config_thresholds = {
        "max_tokens": (max, 512),
        "max_new_tokens": (max, 512),
    }

    # try to get the generation configs
    from transformers import GenerationConfig

    pretrained_model_name_or_path = check_model_home(pretrained_model_name_or_path)
    try:
        model_generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path).to_dict()
    except:  # noqa: E722
        logger.warning(f"No generation config found for the model " f"[{pretrained_model_name_or_path}]")
        model_generation_config = {}

    for key in update_keys:
        # if there is this param in the sampling_prams, compare it with the
        # thresholds and apply the specified updating function
        if key in sampling_params:
            logger.debug(f"Found param {key} in the input `sampling_params`.")
            continue
        # if not, try to find it in the generation_config of the model
        found = False
        for config_key in generation_config_keys[key]:
            if config_key in model_generation_config and model_generation_config[config_key]:
                sampling_params[key] = model_generation_config[config_key]
                found = True
                break
        if found:
            logger.debug(f"Found param {key} in the generation config as " f"{sampling_params[key]}.")
            continue
        # if not again, use the threshold directly
        _, th = generation_config_thresholds[key]
        sampling_params[key] = th
        logger.debug(f"Use the threshold {th} as the sampling param {key}.")
    return sampling_params


MODEL_FUNCTION_MAPPING = {
    "api": prepare_api_model,
    "diffusion": prepare_diffusion_model,
    "fasttext": prepare_fasttext_model,
    "fastsam": prepare_fastsam_model,
    "huggingface": prepare_huggingface_model,
    "kenlm": prepare_kenlm_model,
    "nltk": prepare_nltk_model,
    "nltk_pos_tagger": prepare_nltk_pos_tagger,
    "opencv_classifier": prepare_opencv_classifier,
    "recognizeAnything": prepare_recognizeAnything_model,
    "sdxl-prompt-to-prompt": prepare_sdxl_prompt2prompt,
    "sentencepiece": prepare_sentencepiece_for_lang,
    "simple_aesthetics": prepare_simple_aesthetics_model,
    "spacy": prepare_spacy_model,
    "video_blip": prepare_video_blip_model,
    "video_depth_anything": prepare_video_depth_anything,
    "vllm": prepare_vllm_model,
    "yolo": prepare_yolo_model,
    "embedding": prepare_embedding_model,
}

_MODELS_WITHOUT_FILE_LOCK = {"fasttext", "fastsam", "kenlm", "nltk", "recognizeAnything", "sentencepiece", "spacy"}


def prepare_model(model_type, **model_kwargs):
    assert model_type in MODEL_FUNCTION_MAPPING.keys(), "model_type must be one of the following: {}".format(
        list(MODEL_FUNCTION_MAPPING.keys())
    )
    model_func = MODEL_FUNCTION_MAPPING[model_type]
    model_key = partial(model_func, **model_kwargs)
    if model_type in _MODELS_WITHOUT_FILE_LOCK:
        # initialize once in the main process to safely download model files
        model_key()
    return model_key


def get_model(model_key=None, rank=None, use_cuda=False):
    if model_key is None:
        return None

    global MODEL_ZOO
    if model_key not in MODEL_ZOO:
        logger.debug(f"{model_key} not found in MODEL_ZOO ({mp.current_process().name})")
        if use_cuda and cuda_device_count() > 0:
            rank = rank if rank is not None else 0
            rank = rank % cuda_device_count()
            device = f"cuda:{rank}"
        else:
            device = "cpu"
        MODEL_ZOO[model_key] = model_key(device=device)
    return MODEL_ZOO[model_key]


def free_models(clear_model_zoo=True):
    global MODEL_ZOO
    for model_key in MODEL_ZOO:
        try:
            model = MODEL_ZOO[model_key]
            model.to("cpu")
            if clear_model_zoo:
                del model
        except Exception:
            pass
    if clear_model_zoo:
        MODEL_ZOO.clear()
    torch.cuda.empty_cache()
