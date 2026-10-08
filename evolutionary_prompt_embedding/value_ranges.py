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

def load_embedding_bounds(*, path=None, source="diffusiondb"):
    """Load verified FP32 coordinate limits from packaged data or an external input."""
    import hashlib
    import json
    from pathlib import Path
    from safetensors.torch import load_file

    if source not in {"diffusiondb", "parti"}:
        raise ValueError("Unknown bounds source")
    if path is not None:
        bounds_path = Path(path).expanduser().resolve(strict=True)
        manifest = json.loads(bounds_path.with_suffix(".json").read_text())
        content = bounds_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
            raise ValueError("Bounds input checksum differs")
        tensors = load_file(str(bounds_path), device="cpu")
    elif source == "diffusiondb":
        data = resources.files("evolutionary_prompt_embedding").joinpath("tensors")
        manifest = json.loads(data.joinpath("diffusiondb-full-bounds.json").read_text())
        content = data.joinpath("diffusiondb-full-bounds.safetensors").read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["sha256"]:
            raise ValueError("Packaged bounds checksum differs")
        from safetensors.torch import load

        tensors = load(content)
    else:
        token, pooled = SDXLTurboEmbeddingRange(), SDXLTurboPooledEmbeddingRange()
        tensors = {
            "token_min": token.min_values.float(),
            "token_max": token.max_values.float(),
            "pooled_min": pooled.min_values.float(),
            "pooled_max": pooled.max_values.float(),
        }
        digest = hashlib.sha256()
        for name in sorted(tensors):
            digest.update(name.encode())
            digest.update(tensors[name].contiguous().numpy().tobytes())
        manifest = {
            "dataset": "Parti prompts",
            "model_id": "stabilityai/sdxl-turbo",
            "sha256": digest.hexdigest(),
            "extrema_dtype": "float32",
            "initialization": "synthetic uniform coordinate draws",
        }
    shapes = {
        "token_min": (1, 77, 2048),
        "token_max": (1, 77, 2048),
        "pooled_min": (1, 1280),
        "pooled_max": (1, 1280),
    }
    if set(tensors) != set(shapes):
        raise ValueError("Expected four token/pooled bounds tensors")
    for name, shape in shapes.items():
        value = tensors[name]
        if (
            tuple(value.shape) != shape
            or value.dtype != torch.float32
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid bounds tensor {name}")
    for name in ("token", "pooled"):
        if not torch.all(tensors[name + "_min"] <= tensors[name + "_max"]):
            raise ValueError("Bounds minimum exceeds maximum")
    return tensors, manifest


class SDXLTurboDiffusionDBEmbeddingRange(EmbeddingRange):
    """Full-corpus DiffusionDB token-coordinate bounds for SDXL-Turbo."""

    def __init__(self):
        tensors, _ = load_embedding_bounds()
        super().__init__(tensors["token_min"], tensors["token_max"])


class SDXLTurboDiffusionDBPooledEmbeddingRange(EmbeddingRange):
    """Full-corpus DiffusionDB pooled-coordinate bounds for SDXL-Turbo."""

    def __init__(self):
        tensors, _ = load_embedding_bounds()
        super().__init__(tensors["pooled_min"], tensors["pooled_max"])
