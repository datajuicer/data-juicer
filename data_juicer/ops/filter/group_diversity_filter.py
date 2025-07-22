import sys
from typing import Dict, List

import numpy as np
from jsonargparse.typing import NonNegativeFloat, PositiveInt
from tqdm import tqdm

from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.lazy_loader import LazyLoader
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter

# Lazy load torch to improve startup time
torch = LazyLoader("torch")


@OPERATORS.register_module("group_diversity_filter")
class GroupDiversityFilter(Filter):
    """
    Filter samples based on their semantic diversity within a group.
    """

    _accelerator = "cuda"
    _batched_op = True

    def __init__(
        self,
        api_or_hf_model: str = "text-embedding-v3",
        is_hf_model: bool = False,
        api_endpoint: str = "/embeddings",
        response_path: str = "data.0.embedding",
        model_params: Dict = {},
        ebd_dim: PositiveInt = 512,
        min_score: NonNegativeFloat = 0.0,
        max_score: NonNegativeFloat = 1.0,
        norm_ratio: NonNegativeFloat = 0.5,
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
        :param ebd_dim: The embedding's dimension via API.
        :param min_score: Minimum score for filtering.
        :param max_score: Maximum score for filtering.
        :param norm_ratio: Ratio to normalize the score.
        :param args: extra args
        :param kwargs: extra args
        """
        kwargs.setdefault("mem_required", "20GB")
        super().__init__(*args, **kwargs)

        self.min_score = min_score
        self.max_score = max_score
        self.norm_ratio = norm_ratio
        self.is_hf_model = is_hf_model
        self.ebd_dim = ebd_dim

        if self.is_hf_model:
            self.model_key = prepare_model(model_type="embedding", model_path=api_or_hf_model, **model_params)
        else:
            self.model_key = prepare_model(
                model_type="api",
                model=api_or_hf_model,
                endpoint=api_endpoint,
                response_path=response_path,
                **model_params,
            )

    def _embed_texts(self, texts: List[str], rank: int) -> np.ndarray:
        # Embed a list of texts using the initialized model
        embeddings = []
        model = get_model(self.model_key, rank, self.use_cuda())

        for text in tqdm(texts, desc="Embedding texts", leave=False):
            try:
                if self.is_hf_model:
                    embedding = model.encode(text)
                else:
                    embedding = model(text, dimensions=self.ebd_dim, encoding_format="float")
                embeddings.append(np.array(embedding, dtype=np.float32))
            except Exception as e:
                dim = model.get_sentence_embedding_dimension() if self.is_hf_model else self.ebd_dim
                embeddings.append(np.zeros(dim, dtype=np.float32))
                print(f"Failed to embed text: '{text}'. Error: {e}. Using zero vector.", file=sys.stderr)

        return np.array(embeddings)

    def compute_stats_batched(self, samples: Dict, rank: int = 0) -> Dict:
        stats_list = samples[Fields.stats]
        if stats_list and StatsKeys.text_ebd_diversity_score in stats_list[0]:
            return samples

        texts_to_embed = samples[self.text_key]
        if not texts_to_embed:
            for stat in stats_list:
                stat[StatsKeys.text_ebd_diversity] = 0.0
                stat[StatsKeys.text_ebd_diversity_score] = 0.0
            return samples

        embeddings_array = self._embed_texts(texts_to_embed, rank=rank)

        avg_embedding = np.mean(embeddings_array, axis=0)

        cos_sims = (
            torch.nn.functional.cosine_similarity(
                torch.from_numpy(embeddings_array), torch.from_numpy(avg_embedding).unsqueeze(0), dim=1
            )
            .cpu()
            .numpy()
            .tolist()
        )

        min_sim, max_sim = min(cos_sims), max(cos_sims)
        range_sim = max_sim - min_sim

        normalized_scores = []
        if range_sim < 1e-8:
            normalized_scores = [0.0] * len(cos_sims)
        else:
            for sim in cos_sims:
                normalized_sim = self.norm_ratio * (max_sim - sim) / range_sim
                normalized_scores.append(normalized_sim)

        for i, stat in enumerate(stats_list):
            stat[StatsKeys.text_ebd_diversity] = cos_sims[i]
            stat[StatsKeys.text_ebd_diversity_score] = normalized_scores[i]

        return samples

    def process_batched(self, samples: Dict) -> List[bool]:
        stats_list = samples[Fields.stats]
        return [self.min_score <= stat[StatsKeys.text_ebd_diversity_score] <= self.max_score for stat in stats_list]
