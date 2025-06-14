from importlib import resources
import torch


class EmbeddingRange:
    """
    Class to store the value range of an embedding tensor.
    Used to restrict the search space to reasonable values.
    Able to generate random embeddings within the range.
    Part of the utility suite.
    """

    def __init__(self, min_values: torch.Tensor, max_values: torch.Tensor):
        self._min_values = min_values
        self._max_values = max_values
        self._minimum = min_values.min()
        self._maximum = max_values.max()

    @property
    def min_values(self) -> torch.Tensor:
        return self._min_values

    @property
    def max_values(self) -> torch.Tensor:
        return self._max_values

    @property
    def minimum(self) -> float:
        return self._minimum.item()

    @property
    def maximum(self) -> float:
        return self._maximum.item()

    def random_tensor_in_range(self) -> torch.Tensor:
        """
        Generate a random tensor with values in the range of the embedding tensor.
        """
        return torch.rand_like(self._min_values) * (self._max_values - self._min_values) + self._min_values


class SDXLTurboEmbeddingRange(EmbeddingRange):
    """
    Class to load and store the value range of SDXL Turbo embedding tensors.
    """

    def __init__(self):
        with resources.path('evolutionary_prompt_embedding.tensors', 'sdxl_turbo_min_tensor.pt') as min_tensor_path:
            min_values = torch.load(min_tensor_path, weights_only=True)
        with resources.path('evolutionary_prompt_embedding.tensors', 'sdxl_turbo_max_tensor.pt') as max_tensor_path:
            max_values = torch.load(max_tensor_path, weights_only=True)
        super().__init__(min_values, max_values)

class SDXLTurboPooledEmbeddingRange(EmbeddingRange):
    """
    Class to load and store the value range of SDXL Turbo pooled embedding tensors.
    """

    def __init__(self):
        with resources.path('evolutionary_prompt_embedding.tensors', 'sdxl_turbo_min_tensor_pooled.pt') as min_tensor_path:
            min_values = torch.load(min_tensor_path, weights_only=True)
        with resources.path('evolutionary_prompt_embedding.tensors', 'sdxl_turbo_max_tensor_pooled.pt') as max_tensor_path:
            max_values = torch.load(max_tensor_path, weights_only=True)
        super().__init__(min_values, max_values)

class SDTurboEmbeddingRange(EmbeddingRange):
    """
    Class to load and store the value range of SD Turbo embedding tensors.
    """

    def __init__(self):
        with resources.path('evolutionary_prompt_embedding.tensors', 'sd_turbo_min_tensor.pt') as min_tensor_path:
            min_values = torch.load(min_tensor_path, weights_only=True)
        with resources.path('evolutionary_prompt_embedding.tensors', 'sd_turbo_max_tensor.pt') as max_tensor_path:
            max_values = torch.load(max_tensor_path, weights_only=True)
        super().__init__(min_values, max_values)

class AudioLDMFullEmbeddingRange(EmbeddingRange):
    """
    Class to load and store the value range of AudioLDM Full embedding tensors.
    """
    def __init__(self):
        with resources.path('evolutionary_prompt_embedding.tensors', 'audioldmfull_min_tensor.pt') as min_tensor_path:
            min_values = torch.load(min_tensor_path, weights_only=True)
        with resources.path('evolutionary_prompt_embedding.tensors', 'audioldmfull_max_tensor.pt') as max_tensor_path:
            max_values = torch.load(max_tensor_path, weights_only=True)
        super().__init__(min_values, max_values)