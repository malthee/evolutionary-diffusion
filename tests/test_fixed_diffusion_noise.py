"""Mock pipeline calls; no model loading, network calls, or actual diffusion."""

from types import SimpleNamespace

import pytest
import torch

from evolutionary_imaging.image_base import ImageCreator
from evolutionary_prompt_embedding.argument_types import (
    PooledPromptEmbedData,
    PromptEmbedData,
)
from evolutionary_prompt_embedding.image_creation import (
    SDPromptEmbeddingImageCreator,
    SDXLPromptEmbeddingImageCreator,
)


@pytest.mark.parametrize(
    "creator_type,pooled",
    [(SDPromptEmbeddingImageCreator, False), (SDXLPromptEmbeddingImageCreator, True)],
)
def test_fixed_noise_identical_between_candidates_and_consuming_retry(
    monkeypatch, creator_type, pooled
):
    draws = []
    failures = [True, False, False]

    class Pipeline:
        device = "cpu"

        def __call__(self, **kwargs):
            noise = torch.randn(6, generator=kwargs["generator"][0])
            draws.append(noise)
            if failures.pop(0):
                raise RuntimeError("mock retry after consuming noise")
            return SimpleNamespace(images=[noise])

    monkeypatch.setattr(
        ImageCreator, "_setup_diffusers_pipeline", lambda self: Pipeline()
    )
    creator = creator_type(3, 1, fixed_noise_seeds=[0])
    data = (
        PooledPromptEmbedData(torch.zeros(1, 3, 4), torch.zeros(1, 4))
        if pooled
        else PromptEmbedData(torch.zeros(1, 3, 4))
    )
    torch.manual_seed(17)
    before = torch.random.get_rng_state().clone()
    first = creator.create_solution(data)
    other = (
        PooledPromptEmbedData(torch.ones(1, 3, 4), torch.ones(1, 4))
        if pooled
        else PromptEmbedData(torch.ones(1, 3, 4))
    )
    second = creator.create_solution(other)
    assert len(draws) == 3 and all(torch.equal(draws[0], noise) for noise in draws)
    assert torch.equal(first.result.images[0], second.result.images[0])
    assert torch.equal(before, torch.random.get_rng_state())
    state = creator.__getstate__()
    assert "_pipeline" not in state and "_generators" not in state
    restored = creator_type.__new__(creator_type)
    restored.__setstate__(state)
    assert torch.equal(
        torch.randn(6, generator=restored._generation_generators()[0]), draws[0]
    )


def test_omitted_seeds_preserve_advancing_stream_and_fixed_batch_seeds(monkeypatch):
    monkeypatch.setattr(
        ImageCreator,
        "_setup_diffusers_pipeline",
        lambda self: SimpleNamespace(device="cpu"),
    )
    creator = SDPromptEmbeddingImageCreator(3, 1)
    first = torch.randn(5, generator=creator._generation_generators()[0])
    second = torch.randn(5, generator=creator._generation_generators()[0])
    assert not torch.equal(first, second)
    creator = SDPromptEmbeddingImageCreator(
        3, 2, deterministic=False, fixed_noise_seeds=[2, 7]
    )
    generators = creator._generation_generators()
    assert len(generators) == 2
    for generator, seed in zip(generators, [2, 7]):
        assert torch.equal(
            torch.randn(5, generator=generator),
            torch.randn(5, generator=torch.Generator().manual_seed(seed)),
        )


@pytest.mark.parametrize("seeds", [[], [1, 2], [1.5], [True], [-1], [2**64]])
def test_invalid_fixed_noise_seeds_rejected_before_pipeline_loading(seeds, monkeypatch):
    def fail(self):
        raise AssertionError("Should validate before model loading")

    monkeypatch.setattr(ImageCreator, "_setup_diffusers_pipeline", fail)
    with pytest.raises(ValueError):
        SDPromptEmbeddingImageCreator(3, 1, fixed_noise_seeds=seeds)
