import string
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset
from jsonargparse.typing import ClosedUnitInterval
from loguru import logger
from tqdm import tqdm

from data_juicer.ops.base_op import ATTRIBUTION_FILTERS, OPERATORS, Filter
from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

torch = LazyLoader("torch")
transformers = LazyLoader("transformers")

OP_NAME = "text_embd_similarity_filter"


@OPERATORS.register_module(OP_NAME)
@ATTRIBUTION_FILTERS.register_module(OP_NAME)
class TextEmbdSimilarityFilter(Filter):
    """Filter to keep texts whose average embedding similarity to a set of given validation
    texts falls within a specific range.

    This operator computes the cosine similarity between the text embeddings and a set of
    validation text embeddings. It keeps samples where the average similarity score is
    within the specified range. The key metric, 'text_embd_similarity', is computed as the
    mean cosine similarity. The operator supports both API-based and Hugging Face model-
    based embeddings. If no valid dataset is provided, the `prepare_valid_feature` method
    must be called manually before applying the filter."""

    # TODO: save embeddings in local cache to save computation for multiple validation samples
    # TODO: aggregation strategies
    # TODO: truncation option

    _accelerator = "cuda"

    def __init__(
        self,
        api_or_hf_model: str = "text-embedding-v4",
        is_hf_model: bool = False,
        api_endpoint: str = "embeddings",
        response_path: str = "data.0.embedding",
        model_params: Optional[Dict] = None,
        min_score: ClosedUnitInterval = 0.1,
        max_score: ClosedUnitInterval = 1.0,
        valid_dataset: Optional[List[Dict]] = None,
        ebd_dim: int = 4096,
        pooling: Optional[str] = None,
        input_template: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialization method.

        :param api_or_hf_model: API or huggingface embedding model name.
        :param is_hf_model: Indicates if the model is from HuggingFace.
        :param api_endpoint: Embedding URL endpoint for the API.
        :param response_path: Path to extract content from the API response.
            Defaults to 'data.0.embedding' for embedding model.
        :param model_params: Parameters for initializing the API model.
        :param min_score: The min average similarity to keep samples.
        :param max_score: The max average similarity to keep samples.
        :param valid_dataset: The dataset to use for validation.
            If None, 'self.prepare_valid_feature' should be manually called before applying the filter.
        :param ebd_dim: The embedding's dimension via API.
            API specific parameter, i.e., if is_hf_model=True, this parameter will not take effect.
        :param pooling: strategy to extract embedding from the hidden states. https://arxiv.org/abs/2503.01807
            None: default option, the hidden state of the last token.
            "mean": uniform mean of hidden states.
            "weighted_mean": weighted mean of hidden states. https://arxiv.org/abs/2202.08904
            HF_MODEL specific parameter, i.e., if is_hf_model=False, this parameter will not take effect.
        :param input_template: Template for building the model input.
        """

        if model_params is None:
            model_params = {}
        kwargs["memory"] = "1500MB" if kwargs.get("memory", 0) == 0 else kwargs["memory"]
        super().__init__(*args, **kwargs)
        self.api_or_hf_model = api_or_hf_model
        self.is_hf_model = is_hf_model
        self.api_endpoint = api_endpoint
        self.response_path = response_path
        self.min_score = min_score
        self.max_score = max_score
        self.ebd_dim = ebd_dim
        self.pooling = pooling
        self.input_template = input_template or "{" + self.text_key + "}"

        if is_hf_model:
            self.model_key = prepare_model(
                model_type="embedding", model_path=api_or_hf_model, pooling=pooling, **model_params
            )
        else:
            self.model_key = prepare_model(
                model_type="api",
                model=api_or_hf_model,
                endpoint=self.api_endpoint,
                response_path=self.response_path,
                **model_params,
            )

        self.valid_feature = {}
        if valid_dataset is None:
            logger.warning(
                f"valid_dataset is None when initializing {OP_NAME}. \
            prepare_valid_feature' method should be manually called before applying the filter."
            )
        else:
            self.prepare_valid_feature(Dataset.from_list(valid_dataset))

    @property
    def valid_feature_ready(self):
        return "embeddings" in self.valid_feature

    def prepare_valid_feature(self, dataset, n_shot=None, *args, **kwargs):
        n_shot = n_shot or len(dataset)
        dataset = dataset.select(range(0, n_shot))
        embeddings = self._get_embd(dataset)
        self.valid_feature.update({"embeddings": embeddings})
        # mean(cosine(query, reference)) is the dot product of the normalized
        # query with the mean of normalized references. Do not normalize that
        # mean again: its magnitude is part of the average similarity score.
        self.valid_feature.pop("mean_normalized_embedding", None)
        if embeddings.ndim == 2 and all(embeddings.shape):
            references = torch.from_numpy(embeddings)
            self.valid_feature["mean_normalized_embedding"] = torch.nn.functional.normalize(
                references, dim=1, eps=1e-8
            ).mean(dim=0)

    def _sample_to_text(self, sample):
        formatter = string.Formatter()
        keys = [field_name for _, field_name, _, _ in formatter.parse(self.input_template)]
        valid_sample = True
        for k in keys:
            if k not in sample:
                valid_sample = False
                break
        if valid_sample:
            return self.input_template.format(**sample)
        assert "messages" in sample
        msgs = sample["messages"]
        contents = [msg["content"] for msg in msgs]
        return self.input_template.format(**dict(zip(keys, contents[: len(keys)])))

    def _get_embd_single(self, sample, rank=None):
        model = get_model(self.model_key, rank, self.use_cuda())
        text = self._sample_to_text(sample)
        if self.is_hf_model:
            # Embeddings extract via local models
            with torch.no_grad():
                embedding = model.encode(text)
        else:
            # Embeddings extract via API
            embedding = None
            try:
                embedding = model(text, dimensions=self.ebd_dim, encoding_format="float")
            except Exception as e:
                logger.warning(f"Exception: {e}")
            if embedding is not None:
                return embedding
            logger.debug("Trying to request with a list of texts.")
            sub_seq_length = len(text) // 9
            text_list = [text[i * sub_seq_length : (i + 1) * sub_seq_length] for i in range(10)]
            try:
                embedding = model(text_list, dimensions=self.ebd_dim, encoding_format="float")
            except Exception as e:
                logger.warning(f"Exception: {e}")
            if embedding is None:
                logger.warning("Failed to extract embedding from text.")

        return embedding

    def _get_embd(self, dataset, rank=None):
        embeddings = []
        for sample in tqdm(dataset, desc="Embedding", unit="sample"):
            embedding = self._get_embd_single(sample, rank)
            embeddings.append(embedding)
        embeddings = np.array(embeddings, dtype=np.float64)

        return embeddings

    def compute_stats_single(self, sample, rank=None):
        # check if it's computed already
        if StatsKeys.text_embd_similarity in sample[Fields.stats]:
            return sample

        assert self.valid_feature_ready, "Validation feature not ready yet. Call prepare_valid_feature first."

        embedding = self._get_embd_single(sample, rank)
        try:
            query = torch.tensor(embedding).view(1, -1)
            mean_reference = self.valid_feature.get("mean_normalized_embedding")
            if mean_reference is not None and query.shape[1] == mean_reference.numel() and not query.is_complex():
                # cosine_similarity normalizes floating inputs in their own
                # dtype before multiplication promotes them to float64.
                if not query.is_floating_point():
                    query = query.to(dtype=mean_reference.dtype)
                similarity = (torch.nn.functional.normalize(query, dim=1, eps=1e-8) * mean_reference).sum().item()
            else:
                # Retain the original broadcasting/error behavior for unusual
                # shapes and callers supplying validation embeddings directly.
                similarity = (
                    torch.nn.functional.cosine_similarity(query, torch.from_numpy(self.valid_feature["embeddings"]))
                    .mean()
                    .item()
                )
        except Exception as e:
            logger.warning(f"Exception: {e}")
            similarity = None

        sample[Fields.stats][StatsKeys.text_embd_similarity] = similarity

        return sample

    def process_single(self, sample, rank=None):
        similarity = sample[Fields.stats][StatsKeys.text_embd_similarity]
        if similarity is None:
            return True
        return self.get_keep_boolean(similarity, self.min_score, self.max_score)
