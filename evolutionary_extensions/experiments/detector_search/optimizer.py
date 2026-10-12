"""Compatibility imports for existing detector-search checkpoints."""

import inspect

from evolutionary.algorithms.sep_cma import CMA_VERSION, SepCMAOptimizer
from evolutionary_extensions.persistence.packaging import hashes
from evolutionary_prompt_embedding.coordinates import EmbeddingCoordinates

__all__ = ["CMA_VERSION", "EmbeddingCoordinates", "SearchOptimizer"]


class SearchOptimizer(SepCMAOptimizer):
    """Retain the serialized class path used by completed and resumable searches."""


def optimizer_source_files():
    """Bind search identity to implementations outside the experiment package."""
    return {
        "evolutionary/algorithms/sep_cma.py": hashes(inspect.getfile(SepCMAOptimizer))[
            "sha256"
        ],
        "evolutionary_prompt_embedding/coordinates.py": hashes(
            inspect.getfile(EmbeddingCoordinates)
        )["sha256"],
    }
