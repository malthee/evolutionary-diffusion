import math
import random
from types import SimpleNamespace

import pytest
import torch

from evolutionary.operators import MultiCrossover, MultiMutator
from evolutionary_model_helpers import tensor_variation as tv
from evolutionary_prompt_embedding import variation as v
from evolutionary_prompt_embedding.argument_types import (
    PooledPromptEmbedData,
    PromptEmbedData,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "alpha,expected", [(0.0, [-0.5, -0.25, 0.5]), (0.5, [-1.0, -0.5, 1.0])]
)
def test_blx_known_uniform_quantiles(monkeypatch, dtype, alpha, expected):
    monkeypatch.setattr(
        torch, "rand_like", lambda x: torch.tensor([0.0, 0.25, 1.0], device=x.device)
    )
    a = torch.full((3,), -0.5, dtype=dtype)
    b = -a
    output = tv.blx_alpha_crossover(a, b, (-2, 2), alpha)
    assert torch.allclose(output.float(), torch.tensor(expected), atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "quantile,magnitude", [(1 / 7, 0.25), (0.75, 2 / math.sqrt(11)), (1.0, 1.0)]
)
def test_bounded_sbx_closed_form_quantiles(monkeypatch, dtype, quantile, magnitude):
    # With eta=1, parents +/-0.5 and bounds +/-1, alpha=7/4.
    # u=1/7 contracts to +/-1/4; u=3/4 expands to +/-2/sqrt(11);
    # the bounded distribution's u=1 endpoint is exactly the bounds.
    a, b = torch.tensor([-0.5], dtype=dtype), torch.tensor([0.5], dtype=dtype)
    for child_index, expected in [(0, -magnitude), (1, magnitude)]:
        draws = iter([quantile, 0.75, 0.0])  # Quantile, no swap, full participation.
        monkeypatch.setattr(
            torch, "rand_like", lambda x: torch.full_like(x, next(draws))
        )
        out = tv.bounded_sbx_crossover(
            a, b, (-1, 1), index=1, participation=1, child_index=child_index
        )
        assert out.item() == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "quantile,expected",
    [
        (0.0, 0.0),
        (0.25, math.sqrt(5 / 8) - 0.5),
        (0.75, 1.5 - math.sqrt(5 / 8)),
        (1.0, 1.0),
    ],
)
def test_polynomial_closed_form_quantiles(monkeypatch, dtype, quantile, expected):
    # Eta=1 at the midpoint of [0,1] gives square-root quantiles in both branches.
    draws = iter([quantile, 0.0])
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.full_like(x, next(draws)))
    out = tv.bounded_polynomial_mutate_tensor(
        torch.tensor([0.5], dtype=dtype), (0, 1), index=1, participation=1
    )
    assert out.item() == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    "quantile,sbx_magnitude,polynomial_value",
    [
        (0.25, 0.4788014120380568, 0.4675318004931773),
        (0.75, 0.5221361442999747, 0.5324681995068227),
    ],
)
def test_default_distribution_indices_match_reference_quantiles(
    monkeypatch, quantile, sbx_magnitude, polynomial_value
):
    # Independently calculated with 50-digit decimal arithmetic from Deb's
    # inverse CDFs at eta=15 (SBX) and eta=20 (polynomial mutation).
    draws = iter([quantile, 0.75, 0.0])
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.full_like(x, next(draws)))
    sbx = tv.bounded_sbx_crossover(
        torch.tensor([-0.5]),
        torch.tensor([0.5]),
        (-1, 1),
        participation=1,
        child_index=0,
    )
    assert sbx.item() == pytest.approx(-sbx_magnitude, abs=1e-6)
    draws = iter([quantile, 0.0])
    polynomial = tv.bounded_polynomial_mutate_tensor(
        torch.tensor([0.5]), (0, 1), participation=1
    )
    assert polynomial.item() == pytest.approx(polynomial_value, abs=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_repair_bounds_rounds_inward_in_final_dtype(dtype):
    low, high = torch.tensor([-0.3, -0.1]), torch.tensor([0.3, 0.1])
    source = torch.tensor([[-2.0, 2.0], [2.0, -2.0]], dtype=dtype)
    saved = source.clone()
    output = tv.repair_bounds(source, (low, high))
    assert output.dtype == dtype and output.shape == source.shape
    assert (output.float() >= low).all() and (output.float() <= high).all()
    assert torch.equal(source, saved)
    # Bounds outside the dtype range still admit all its finite values.
    wide = tv.repair_bounds(source, (-100_000, 100_000))
    assert torch.isfinite(wide).all()


def test_repair_rejects_interval_with_no_representable_half_value():
    with pytest.raises(ValueError, match="representable"):
        tv.repair_bounds(torch.tensor([0.3]), (0.3, 0.3), dtype=torch.float16)
    with pytest.raises(ValueError, match="representable"):
        tv.repair_bounds(
            torch.tensor([100_000.0]), (100_000, 200_000), dtype=torch.float16
        )


@pytest.mark.parametrize(
    "family", ["gaussian", "spherical", "slerp", "blx", "sbx", "polynomial"]
)
def test_every_repaired_family_respects_nonrepresentable_half_bounds(family):
    torch.manual_seed(0)
    a = torch.full((2, 5, 10), -0.25, dtype=torch.float16)
    b = -a
    bounds = (-0.3, 0.3)
    if family == "gaussian":
        output = tv.uniform_gaussian_mutate_tensor(a, 1, 5, bounds)
    elif family == "spherical":
        output = tv.spherical_rotation_mutate_tensor(a, 1, 1, bounds)
    elif family == "slerp":
        output = tv.slerp_crossover(a, b, 0.5, bounds)
    elif family == "blx":
        output = tv.blx_alpha_crossover(a, b, bounds)
    elif family == "sbx":
        output = tv.bounded_sbx_crossover(a, b, bounds, index=1, participation=1)
    else:
        output = tv.bounded_polynomial_mutate_tensor(
            a, bounds, index=1, participation=1
        )
    assert output.dtype == a.dtype and output.shape == a.shape
    # Compare in float32: a half-precision comparison would round the bounds too.
    assert (output.float() >= bounds[0]).all() and (output.float() <= bounds[1]).all()


@pytest.mark.parametrize(
    "weights", [[0, 0], [1], [-1, 1], [float("nan"), 1], [float("inf"), 1]]
)
def test_invalid_pool_weights(weights):
    with pytest.raises(ValueError):
        MultiMutator({"a": object(), "b": object()}, weights)


def test_seeded_pool_selection_and_zero_weight():
    def pool():
        return MultiCrossover(
            {
                "a": SimpleNamespace(crossover=lambda a, b: "a"),
                "b": SimpleNamespace(crossover=lambda a, b: "b"),
            }
        )

    random.seed(10)
    first = pool()
    sequence = [first.crossover(0, 0) for _ in range(30)]
    random.seed(10)
    second = pool()
    assert sequence == [second.crossover(0, 0) for _ in range(30)]
    assert set(sequence) == {"a", "b"}
    assert second.last_operator_name == sequence[-1]
    weighted = MultiMutator(
        {"never": object(), "always": SimpleNamespace(mutate=lambda a: a + 1)}, [0, 1]
    )
    assert weighted.mutate(2) == 3 and weighted.last_operator_name == "always"
    with pytest.raises(ValueError):
        MultiCrossover({})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("pooled", [False, True])
def test_every_operator_shape_dtype_bounds_seed_and_parent_integrity(dtype, pooled):
    torch.manual_seed(5)
    prompt1, prompt2 = torch.rand(2, 5, 8).to(dtype), -torch.rand(2, 5, 8).to(dtype)
    pool1, pool2 = torch.rand(2, 8).to(dtype), -torch.rand(2, 8).to(dtype)
    a = PooledPromptEmbedData(prompt1, pool1) if pooled else PromptEmbedData(prompt1)
    b = PooledPromptEmbedData(prompt2, pool2) if pooled else PromptEmbedData(prompt2)
    saved = [t.clone() for t in (prompt1, prompt2, pool1, pool2)]
    bounds = (-1.0, 1.0)
    crossovers = (
        [
            v.PooledUniformCrossover(0.5, 0.5),
            v.PooledRowUniformCrossover(),
            v.PooledOnePointCrossover(),
            v.PooledTwoPointCrossover(),
            v.PooledArithmeticCrossover(0.5, 0.5),
            v.PooledSlerpCrossover(0.5, 0.5, bounds, bounds),
            v.PooledBLXAlphaCrossover(bounds, bounds),
            v.PooledSBXCrossover(bounds, bounds),
        ]
        if pooled
        else [
            v.UniformCrossover(0.5),
            v.RowUniformCrossover(),
            v.OnePointCrossover(),
            v.TwoPointCrossover(),
            v.ArithmeticCrossover(0.5),
            v.SlerpCrossover(0.5, bounds),
            v.BLXAlphaCrossover(bounds),
            v.SBXCrossover(bounds),
        ]
    )
    gaussian_args = v.UniformGaussianMutatorArguments(0.2, 0.4, bounds)
    mutations = (
        [
            v.PooledUniformGaussianMutator(gaussian_args, gaussian_args),
            v.PooledSphericalRotationMutator(1.0, 0.1, 1.0, 0.1, bounds, bounds),
            v.PooledDonorReplacementMutator([b], 0.5, bounds, bounds),
            v.PooledPolynomialMutator(bounds, bounds),
        ]
        if pooled
        else [
            v.UniformGaussianMutator(gaussian_args),
            v.SphericalRotationMutator(1.0, 0.1, bounds),
            v.DonorReplacementMutator([b], 0.5, bounds),
            v.PolynomialMutator(bounds),
        ]
    )
    for operator in crossovers + mutations:

        def invoke(operator=operator):
            return (
                operator.crossover(a, b)
                if operator in crossovers
                else operator.mutate(a)
            )

        random.seed(3)
        torch.manual_seed(3)
        result = invoke()
        random.seed(3)
        torch.manual_seed(3)
        repeat = invoke()
        tensors = [(result.prompt_embeds, prompt1, repeat.prompt_embeds)]
        if pooled:
            tensors.append(
                (result.pooled_prompt_embeds, pool1, repeat.pooled_prompt_embeds)
            )
        for output, source, repeated in tensors:
            assert (
                output.shape == source.shape
                and output.dtype == source.dtype
                and output.device == source.device
            )
            assert (
                torch.isfinite(output).all()
                and (output >= -1).all()
                and (output <= 1).all()
            )
            assert torch.equal(output, repeated)
            assert output.data_ptr() != source.data_ptr()
        assert all(
            torch.equal(t, copy)
            for t, copy in zip((prompt1, prompt2, pool1, pool2), saved)
        )


def test_point_crossover_boundaries_and_short_axes():
    a, b = torch.zeros(1, 7, 5), torch.ones(1, 7, 5)
    for points in (1, 2):
        torch.manual_seed(2)
        out = tv.point_crossover(a, b, points, -2)
        assert torch.equal(out, out[..., :1].expand_as(out))  # Whole rows only.
        changes = (out[0, 1:, 0] != out[0, :-1, 0]).sum()
        assert changes == points
        assert out[0, 0, 0] == 0 and out[0, -1, 0] == (1 if points == 1 else 0)
        pooled = tv.point_crossover(torch.zeros(1, 8), torch.ones(1, 8), points, -1)
        assert ((pooled[0, 1:] != pooled[0, :-1]).sum()) == points
    short = a[:, :2]
    assert torch.equal(tv.point_crossover(short, b[:, :2], 2), short)


def test_row_uniform_keeps_rows_and_probability_endpoints():
    a, b = torch.zeros(2, 7, 8), torch.ones(2, 7, 8)
    torch.manual_seed(12)
    out = tv.row_uniform_crossover(a, b)
    assert torch.equal(out, out[..., :1].expand_as(out))
    assert torch.equal(tv.row_uniform_crossover(a, b, 0), a)
    assert torch.equal(tv.row_uniform_crossover(a, b, 1), b)
    assert torch.equal(tv.uniform_crossover_tensors(a, b, 0), a)
    assert torch.equal(tv.uniform_crossover_tensors(a, b, 1), b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_spherical_preserves_nonunit_row_norms_zero_rows_and_three_dimensions(dtype):
    torch.manual_seed(2)
    source = (torch.randn(2, 7, 10) * 30).to(dtype)
    source[0, 0] = 0
    result = tv.spherical_rotation_mutate_tensor(source, 1, 0.1)
    tolerance = 0.001 if dtype == torch.float16 else 1e-6
    assert torch.allclose(
        source.float().norm(dim=-1), result.float().norm(dim=-1), rtol=tolerance
    )
    assert torch.equal(result[0, 0], source[0, 0])
    assert torch.equal(tv.spherical_rotation_mutate_tensor(source, 0, 0.1), source)
    assert not torch.equal(source, result)
    repaired = tv.spherical_rotation_mutate_tensor(source, 1, 0.1, (-1, 1))
    assert repaired.abs().max() <= 1
    single = torch.tensor([4.0, 0.0], dtype=dtype)
    assert tv.spherical_rotation_mutate_tensor(single, 1).shape == single.shape
    assert torch.equal(
        tv.spherical_rotation_mutate_tensor(torch.tensor([4.0]), 1), torch.tensor([4.0])
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_slerp_zero_parallel_opposite_and_endpoints(dtype):
    a = torch.tensor([[2.0, 0.0], [2.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    b = torch.tensor([[4.0, 0.0], [-4.0, 0.0], [4.0, 0.0], [0.0, 1.0]], dtype=dtype)
    output = tv.slerp_crossover(a, b, 0.5)
    expected = torch.tensor(
        [[3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [2**-0.5, 2**-0.5]], dtype=dtype
    )
    assert torch.allclose(output, expected, atol=1e-3)
    assert torch.equal(tv.slerp_crossover(a, b, 0), a)
    assert torch.equal(tv.slerp_crossover(a, b, 1), b)
    assert torch.equal(
        tv.slerp_crossover(torch.tensor([2.0]), torch.tensor([-4.0]), 0.5),
        torch.tensor([-1.0]),
    )
    assert torch.isfinite(tv.slerp_crossover(a.unsqueeze(0), b.unsqueeze(0), 0.5)).all()


def test_sbx_and_polynomial_constant_coordinates_tensor_bounds_and_nonparticipation():
    low = torch.tensor([0.0, -1.0, 2.0])
    high = torch.tensor([0.0, 1.0, 2.0])
    a = torch.tensor([[0.0, -0.8, 2.0]], dtype=torch.float16)
    b = torch.tensor([[0.0, 0.8, 2.0]], dtype=torch.float16)
    for index in (0, 1):
        out = tv.bounded_sbx_crossover(
            a, b, (low, high), participation=1, child_index=index
        )
        assert torch.isfinite(out).all() and (out >= low).all() and (out <= high).all()
        assert out.dtype == a.dtype
    assert torch.equal(
        tv.bounded_sbx_crossover(a, b, (low, high), participation=0, child_index=0), a
    )
    assert torch.equal(
        tv.bounded_sbx_crossover(a, b, (low, high), participation=0, child_index=1), b
    )
    assert torch.equal(tv.bounded_sbx_crossover(a, a, (low, high)), a)
    mutated = tv.bounded_polynomial_mutate_tensor(a, (low, high), participation=1)
    assert torch.isfinite(mutated).all() and mutated.dtype == a.dtype
    assert mutated[0, 0] == 0 and mutated[0, 2] == 2 and -1 <= mutated[0, 1] <= 1
    assert torch.equal(
        tv.bounded_polynomial_mutate_tensor(a, (low, high), participation=0), a
    )


def test_donor_bank_is_frozen_and_uses_positions_and_same_donor_for_both_tensors():
    source = PooledPromptEmbedData(
        torch.arange(12.0).reshape(1, 3, 4), torch.arange(4.0).reshape(1, 4) + 20
    )
    mutator = v.PooledDonorReplacementMutator([source], participation=1)
    saved_prompt, saved_pool = (
        source.prompt_embeds.clone(),
        source.pooled_prompt_embeds.clone(),
    )
    source.prompt_embeds.fill_(99)
    source.pooled_prompt_embeds.fill_(99)
    target = PooledPromptEmbedData(torch.zeros(1, 3, 4), torch.zeros(1, 4))
    out = mutator.mutate(target)
    assert torch.equal(out.prompt_embeds, saved_prompt)
    assert torch.equal(out.pooled_prompt_embeds, saved_pool)
    assert target.prompt_embeds.count_nonzero() == 0
    with pytest.raises(ValueError):
        v.DonorReplacementMutator([])


def test_pooled_event_selects_one_family_for_both_tensors():
    a = PooledPromptEmbedData(torch.zeros(1, 3, 4), torch.zeros(1, 4))
    b = PooledPromptEmbedData(torch.ones(1, 3, 4), torch.ones(1, 4))
    pool = MultiCrossover(
        {
            "arithmetic": v.PooledArithmeticCrossover(0.25, 0.25),
            "rows": v.PooledRowUniformCrossover(1),
        },
        weights=[1, 0],
    )
    out = pool.crossover(a, b)
    assert pool.last_operator_name == "arithmetic"
    assert torch.all(out.prompt_embeds == 0.75) and torch.all(
        out.pooled_prompt_embeds == 0.75
    )


def test_bounded_sbx_one_complete_branch_for_pooled_event():
    a = PooledPromptEmbedData(torch.zeros(1, 3, 4), torch.zeros(1, 4))
    b = PooledPromptEmbedData(torch.ones(1, 3, 4), torch.ones(1, 4))
    op = v.PooledSBXCrossover((-1, 1), (-1, 1), participation=0)
    for seed in range(6):
        random.seed(seed)
        out = op.crossover(a, b)
        assert torch.all(out.prompt_embeds == out.pooled_prompt_embeds[0, 0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_partial_arithmetic_noncontiguous_inputs_and_slerp_endpoint_copies(dtype):
    a = torch.zeros(2, 4, 3, dtype=dtype).transpose(1, 2)
    b = torch.ones_like(a)
    torch.manual_seed(3)
    output = tv.arithmetic_crossover(a, b, 0.5, 0.5)
    assert output.dtype == dtype and output.shape == a.shape
    assert (output == 0.5).sum() == a.numel() // 2
    assert a.count_nonzero() == 0 and torch.all(b == 1)
    for ratio in (0, 1):
        output = tv.slerp_crossover(a, b, ratio)
        assert output.data_ptr() != (a if ratio == 0 else b).data_ptr()
