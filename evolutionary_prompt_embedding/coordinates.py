"""Bounded SDXL embedding coordinates for vector optimizers."""

import hashlib
import math

import numpy as np


class EmbeddingCoordinates:
    """Normalize variable coordinates while retaining exact fixed coordinates."""

    def __init__(self, bounds):
        self.bounds = bounds
        self.shapes = {
            key: tuple(bounds[f"{key}_min"].shape) for key in ("token", "pooled")
        }
        self.sizes = {key: math.prod(shape) for key, shape in self.shapes.items()}
        self.lower = np.concatenate(
            [bounds[f"{key}_min"].cpu().float().numpy().ravel() for key in self.shapes]
        ).astype(np.float64)
        self.upper = np.concatenate(
            [bounds[f"{key}_max"].cpu().float().numpy().ravel() for key in self.shapes]
        ).astype(np.float64)
        if (
            not np.isfinite(self.lower).all()
            or not np.isfinite(self.upper).all()
            or (self.lower > self.upper).any()
        ):
            raise ValueError("Require finite ordered coordinate bounds")
        self.active = self.upper > self.lower
        if int(self.active.sum()) < 2:
            raise ValueError("CMA-ES requires at least two variable coordinates")
        self.identity = hashlib.sha256(
            self.lower.tobytes() + self.upper.tobytes()
        ).hexdigest()

    def encode(self, argument):
        values = np.concatenate(
            [
                argument.prompt_embeds.cpu().float().numpy().ravel(),
                argument.pooled_prompt_embeds.cpu().float().numpy().ravel(),
            ]
        )
        if (
            values.shape != self.lower.shape
            or not np.isfinite(values).all()
            or (values < self.lower).any()
            or (values > self.upper).any()
        ):
            raise ValueError("Initial embeddings differ from the bounded geometry")
        return (values[self.active] - self.lower[self.active]) / (
            self.upper[self.active] - self.lower[self.active]
        )

    def decode(self, normalized):
        import torch

        from evolutionary_model_helpers.tensor_variation import repair_bounds
        from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData

        normalized = np.asarray(normalized, dtype=np.float64)
        if (
            normalized.shape != (int(self.active.sum()),)
            or not np.isfinite(normalized).all()
            or (normalized < 0).any()
            or (normalized > 1).any()
        ):
            raise ValueError("CMA proposal is outside normalized coordinate bounds")
        values = self.lower.copy()
        values[self.active] += normalized * (
            self.upper[self.active] - self.lower[self.active]
        )
        tensors, offset = {}, 0
        for key, shape in self.shapes.items():
            size = self.sizes[key]
            tensor = torch.from_numpy(
                values[offset : offset + size].astype(np.float32)
            ).reshape(shape)
            tensors[key] = repair_bounds(
                tensor,
                (self.bounds[f"{key}_min"], self.bounds[f"{key}_max"]),
                dtype=torch.float16,
            )
            offset += size
        return PooledPromptEmbedData(tensors["token"], tensors["pooled"])
