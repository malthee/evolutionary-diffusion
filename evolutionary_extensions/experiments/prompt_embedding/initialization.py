"""Verified bounds and reproducible uniform or frozen embedding initialization."""

from pathlib import Path

import torch

from evolutionary_model_helpers.tensor_variation import repair_bounds
from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData


def load_bounds(path=None, source="diffusiondb"):
    from evolutionary_prompt_embedding.value_ranges import load_embedding_bounds

    tensors, manifest = load_embedding_bounds(path=path, source=source)
    for key in ("token", "pooled"):
        repair_bounds(
            tensors[f"{key}_min"],
            (tensors[f"{key}_min"], tensors[f"{key}_max"]),
            dtype=torch.float16,
        )
    return tensors, manifest


def initial_arguments(bounds, count, config=None):
    def sample(key):
        lo, hi = bounds[f"{key}_min"], bounds[f"{key}_max"]
        return repair_bounds(
            lo + torch.rand_like(lo) * (hi - lo), (lo, hi), dtype=torch.float16
        )

    if config is None or config.initial_embeddings_file is None:
        return [
            PooledPromptEmbedData(sample("token"), sample("pooled"))
            for _ in range(count)
        ]
    from safetensors.torch import load_file

    from evolutionary_prompt_embedding.archive import sha256_file

    path = Path(config.initial_embeddings_file).expanduser()
    if sha256_file(path) != config.initial_embeddings_sha256:
        raise ValueError("Initial embedding checksum differs")
    tensors = load_file(str(path))
    if set(tensors) != {"prompt_embeds", "pooled_prompt_embeds"}:
        raise ValueError("Initial embeddings require token and pooled tensors")
    for key, name in [("token", "prompt_embeds"), ("pooled", "pooled_prompt_embeds")]:
        values = tensors[name]
        shape = bounds[f"{key}_min"].shape
        if values.ndim != len(shape) + 1 or tuple(values.shape[1:]) != tuple(shape):
            raise ValueError(
                f"Initial {key} embeddings have invalid shape; expected [N,{tuple(shape)}]"
            )
        if not torch.isfinite(values).all():
            raise ValueError("Initial embeddings must be finite")
        for value in values:
            repaired = repair_bounds(
                value, (bounds[f"{key}_min"], bounds[f"{key}_max"]), dtype=torch.float16
            )
            if not torch.equal(value.to(torch.float16), repaired):
                raise ValueError("Frozen initial embedding is outside safe FP16 bounds")
    n = tensors["prompt_embeds"].shape[0]
    if tensors["pooled_prompt_embeds"].shape[0] != n:
        raise ValueError("Initial embedding row counts differ")
    if config.initialization == "uniform":
        if n != count:
            raise ValueError("Frozen global population size differs")
        return [
            PooledPromptEmbedData(
                tensors["prompt_embeds"][i].clone(),
                tensors["pooled_prompt_embeds"][i].clone(),
            )
            for i in range(count)
        ]
    if n != 1:
        raise ValueError("Local initialization requires exactly one anchor")
    scales = config.initial_perturbation_std or {"token": 0.25, "pooled": 0.05}

    def neighbor(key, name):
        anchor = tensors[name][0].float()
        return repair_bounds(
            anchor + torch.randn_like(anchor) * scales[key],
            (bounds[f"{key}_min"], bounds[f"{key}_max"]),
            dtype=torch.float16,
        )

    return [
        PooledPromptEmbedData(
            tensors["prompt_embeds"][0].clone(),
            tensors["pooled_prompt_embeds"][0].clone(),
        ),
        *[
            PooledPromptEmbedData(
                neighbor("token", "prompt_embeds"),
                neighbor("pooled", "pooled_prompt_embeds"),
            )
            for _ in range(count - 1)
        ],
    ]
