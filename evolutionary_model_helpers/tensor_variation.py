"""
Implements ways to perform variation on tensors, such as crossover and mutation.
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def uniform_gaussian_mutate_tensor(
    tensor: torch.Tensor,
    mutation_rate: float = 0.05,
    mutation_strength: float = 0.1,
    clamp_range: Tuple[float, float] = (-1, 1),
) -> torch.Tensor:
    """
    Perform a uniform gaussian mutation on a tensor, returning a mutated version of the tensor.

    :param tensor: (torch.Tensor): The tensor to mutate.
    :param mutation_rate: (float): Fraction of elements to mutate (between 0 and 1).
    :param mutation_strength: (float): The strength of the mutation, influencing how much each element can change.
    :param clamp_range: (tuple): A tuple of (min, max) to clamp the mutated values.

    Returns:
    - torch.Tensor: The new mutated tensor.
    """
    _probability(mutation_rate, "mutation_rate")
    if not math.isfinite(mutation_strength) or mutation_strength < 0:
        raise ValueError("mutation_strength must be finite and nonnegative")
    device = tensor.device
    # Clone the tensor to ensure the original is not modified
    cloned_tensor = tensor.float().clone()

    num_elements_to_mutate = int(torch.numel(cloned_tensor) * mutation_rate)
    indices_to_mutate = torch.randperm(torch.numel(cloned_tensor), device=device)[
        :num_elements_to_mutate
    ]

    mutations = torch.randn(num_elements_to_mutate, device=device) * mutation_strength
    flat_tensor = cloned_tensor.flatten()
    flat_tensor[indices_to_mutate] += mutations
    mutated_tensor = flat_tensor.view(cloned_tensor.shape)

    # Clamp the values of the mutated tensor
    return repair_bounds(mutated_tensor, clamp_range, dtype=tensor.dtype)


def spherical_rotation_mutate_tensor(
    tensor, mutation_rate=0.1, mutation_angle=0.1, clamp_range=None
):
    """Rotate selected [..., D] rows, preserving their original norm before optional repair."""
    _probability(mutation_rate, "mutation_rate")
    if not math.isfinite(mutation_angle) or mutation_angle < 0:
        raise ValueError("mutation_angle must be finite and nonnegative")
    rows = tensor.float().reshape(-1, tensor.shape[-1])
    out = rows.clone()
    norms = rows.norm(dim=-1, keepdim=True)
    selected = (torch.rand(rows.shape[0], device=tensor.device) < mutation_rate) & (
        norms[:, 0] > 0
    )
    if tensor.shape[-1] > 1 and selected.any():
        u = F.normalize(rows[selected], dim=-1)
        v = torch.randn_like(u)
        tangent = v - (v * u).sum(-1, keepdim=True) * u
        tangent = F.normalize(tangent, dim=-1)
        # A degenerate random tangent leaves the row unchanged.
        angle = (torch.rand((len(u), 1), device=tensor.device) * 2 - 1) * mutation_angle
        rotated = F.normalize(u * angle.cos() + tangent * angle.sin(), dim=-1)
        out[selected] = rotated * norms[selected]
    out = out.reshape(tensor.shape)
    return repair_bounds(out, clamp_range, dtype=tensor.dtype)


def uniform_crossover_tensors(
    tensor1: torch.Tensor, tensor2: torch.Tensor, swap_rate: float = 0.5
) -> torch.Tensor:
    """
    Perform a uniform crossover operation between two tensors.

    Args:
    :param tensor1: (torch.Tensor): The first parent tensor.
    :param tensor2: (torch.Tensor): The second parent tensor.
    :param swap_rate: (float): The rate at which elements from the second tensor are introduced into the first.

    Returns:
    - torch.Tensor: The resulting tensor after crossover.
    """
    assert tensor1.shape == tensor2.shape, (
        "Both tensors must have the same shape for crossover."
    )

    _probability(swap_rate, "swap_rate")
    crossover_mask = torch.rand(tensor1.shape, device=tensor1.device) < swap_rate
    offspring = torch.where(crossover_mask, tensor2.to(tensor1), tensor1)

    return offspring


def arithmetic_crossover(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    interpolation_weight: float = 0.5,
    proportion: float = 1.0,
) -> torch.Tensor:
    """
    Perform an interpolation-based crossover between two tensors.

    Args:
    :param tensor1: (torch.Tensor): The first parent tensor.
    :param tensor2: (torch.Tensor): The second parent tensor.
    :param interpolation_weight: (float): The weight for interpolation (between 0 and 1). A weight of 0.5 results in an
    equal blend of both tensors.
    :param proportion: (float): The proportion of elements to interpolate. If 1.0 then full arithmetic crossover
    is performed. When not selected for crossover, elements are taken from the first tensor.

    Returns:
    - torch.Tensor: The resulting tensor after interpolation.
    """
    assert tensor1.shape == tensor2.shape, (
        "Both tensors must have the same shape for crossover."
    )
    _probability(proportion, "proportion")
    _probability(interpolation_weight, "interpolation_weight")

    device = tensor1.device
    tensor2 = tensor2.to(device)

    # If proportion is 1, perform full crossover and return immediately
    if proportion == 1.0:
        return (
            tensor1.float() * interpolation_weight
            + tensor2.float() * (1 - interpolation_weight)
        ).to(tensor1.dtype)

    # For partial crossover
    offspring = tensor1.float().contiguous().clone()
    num_elements = tensor1.numel()
    num_crossover = int(
        num_elements * proportion
    )  # Number of elements to apply crossover

    # Randomly choose indices for crossover
    indices = torch.randperm(num_elements, device=device)[:num_crossover]

    # Apply crossover only to selected indices
    flat_offspring = offspring.view(-1)
    flat_tensor1 = tensor1.float().reshape(-1)
    flat_tensor2 = tensor2.float().reshape(-1)
    flat_offspring[indices] = flat_tensor1[
        indices
    ] * interpolation_weight + flat_tensor2[indices] * (1 - interpolation_weight)
    return offspring.to(tensor1.dtype)


def slerp_crossover(tensor1, tensor2, ratio=0.5, clamp_range=None):
    """Interpolate direction on the sphere and radius linearly, using float32 intermediates.

    Zero rows and parallel directions use linear interpolation. Opposite directions
    use a deterministic orthogonal great circle (linear in one dimension).
    """
    _probability(ratio, "ratio")
    _same_shape(tensor1, tensor2)
    a, b = tensor1.float(), tensor2.to(device=tensor1.device, dtype=torch.float32)
    if ratio == 0 or ratio == 1:
        return repair_bounds(
            a if ratio == 0 else b, clamp_range, dtype=tensor1.dtype
        ).clone()
    na, nb = a.norm(dim=-1, keepdim=True), b.norm(dim=-1, keepdim=True)
    u, v = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    dot = (u * v).sum(-1, keepdim=True).clamp(-1, 1)
    omega = dot.acos()
    tangent = v - dot * u
    direction = (
        u * (ratio * omega).cos() + F.normalize(tangent, dim=-1) * (ratio * omega).sin()
    )
    opposite = dot < -1 + 1e-6
    if a.shape[-1] > 1:
        # The axis least aligned with u gives a stable, reproducible orthogonal direction.
        axis = torch.zeros_like(u).scatter_(-1, u.abs().argmin(-1, keepdim=True), 1)
        orthogonal = F.normalize(axis - (axis * u).sum(-1, keepdim=True) * u, dim=-1)
        antipodal = u * math.cos(math.pi * ratio) + orthogonal * math.sin(
            math.pi * ratio
        )
        direction = torch.where(opposite, antipodal, direction)
    out = direction * (na * (1 - ratio) + nb * ratio)
    linear = a.lerp(b, ratio)
    fallback = (na == 0) | (nb == 0) | (dot > 1 - 1e-6)
    if a.shape[-1] == 1:
        fallback |= opposite
    out = torch.where(fallback, linear, out)
    return repair_bounds(out, clamp_range, dtype=tensor1.dtype)


def _probability(value, name):
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _same_shape(a, b):
    if a.shape != b.shape:
        raise ValueError("Parent tensors must have the same shape")


def _bounds(tensor, clamp_range):
    low = torch.as_tensor(clamp_range[0], device=tensor.device, dtype=torch.float32)
    high = torch.as_tensor(clamp_range[1], device=tensor.device, dtype=torch.float32)
    if (
        not torch.isfinite(low).all()
        or not torch.isfinite(high).all()
        or (low > high).any()
    ):
        raise ValueError("Bounds must be finite and ordered")
    return low, high


def repair_bounds(tensor, clamp_range, *, dtype=None):
    """Clamp to bounds representable inside the interval in the output dtype.

    Round bounds inward before casting so half precision cannot undo repair.
    An interval containing no representable output value is rejected.
    """
    dtype = tensor.dtype if dtype is None else dtype
    if clamp_range is None:
        return tensor.to(dtype)
    low, high = _bounds(tensor, clamp_range)
    output_low, output_high = low.to(dtype), high.to(dtype)
    output_low = torch.where(
        output_low.float() < low,
        torch.nextafter(output_low, torch.full_like(output_low, math.inf)),
        output_low,
    )
    output_high = torch.where(
        output_high.float() > high,
        torch.nextafter(output_high, torch.full_like(output_high, -math.inf)),
        output_high,
    )
    if (output_low > output_high).any():
        raise ValueError("Bounds contain no value representable in the output dtype")
    return torch.maximum(
        torch.minimum(tensor, output_high.float()), output_low.float()
    ).to(dtype)


def row_uniform_crossover(a, b, swap_rate=0.5):
    _same_shape(a, b)
    _probability(swap_rate, "swap_rate")
    mask = torch.rand(a.shape[:-1] + (1,), device=a.device) < swap_rate
    return torch.where(mask, b.to(a), a)


def point_crossover(a, b, points=1, axis=-2):
    """Cut only at internal row/coordinate boundaries; short axes return a clone."""
    _same_shape(a, b)
    if points not in (1, 2):
        raise ValueError("points must be 1 or 2")
    size = a.shape[axis]
    if size <= points:
        return a.clone()
    cuts = (torch.randperm(size - 1, device=a.device)[:points] + 1).sort().values
    positions = torch.arange(size, device=a.device)
    mask = positions >= cuts[0]
    if points == 2:
        mask &= positions < cuts[1]
    shape = [1] * a.ndim
    shape[axis] = size
    return torch.where(mask.reshape(shape), b.to(a), a)


def blx_alpha_crossover(a, b, clamp_range, alpha=0.5):
    _same_shape(a, b)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    x, y = a.float(), b.to(device=a.device, dtype=torch.float32)
    lower, upper = torch.minimum(x, y), torch.maximum(x, y)
    span = upper - lower
    out = lower - alpha * span + torch.rand_like(x) * ((1 + 2 * alpha) * span)
    return repair_bounds(out, clamp_range, dtype=a.dtype)


def bounded_sbx_crossover(
    a, b, clamp_range, index=15, participation=0.5, child_index=None
):
    """Deb's bounded SBX; return one randomly chosen complete child, not two solutions."""
    _same_shape(a, b)
    _probability(participation, "participation")
    if not math.isfinite(index) or index <= 0:
        raise ValueError("index must be finite and positive")
    x, y = (
        repair_bounds(a.float(), clamp_range),
        repair_bounds(b.to(a.device).float(), clamp_range),
    )
    low, high = _bounds(x, clamp_range)
    lower, upper = torch.minimum(x, y), torch.maximum(x, y)
    delta = upper - lower
    safe_delta = delta.clamp_min(1e-12)
    rand = torch.rand_like(x)
    power = 1 / (index + 1)

    def beta_q(beta):
        alpha = 2 - beta.pow(-(index + 1))
        return torch.where(
            rand <= 1 / alpha,
            (rand * alpha).pow(power),
            (1 / (2 - rand * alpha).clamp_min(1e-12)).pow(power),
        )

    q1 = beta_q(1 + 2 * (lower - low) / safe_delta)
    q2 = beta_q(1 + 2 * (high - upper) / safe_delta)
    c1 = repair_bounds(0.5 * (lower + upper - q1 * delta), clamp_range)
    c2 = repair_bounds(0.5 * (lower + upper + q2 * delta), clamp_range)
    swap = torch.rand_like(x) < 0.5
    first, second = torch.where(swap, c2, c1), torch.where(swap, c1, c2)
    mask = (torch.rand_like(x) < participation) & (delta > 1e-12) & (high > low)
    first, second = torch.where(mask, first, x), torch.where(mask, second, y)
    if child_index is None:
        child_index = int(torch.randint(2, (), device=a.device))
    if child_index not in (0, 1):
        raise ValueError("child_index must be 0 or 1")
    return repair_bounds(
        first if child_index == 0 else second, clamp_range, dtype=a.dtype
    )


def bounded_polynomial_mutate_tensor(tensor, clamp_range, index=20, participation=0.1):
    _probability(participation, "participation")
    if not math.isfinite(index) or index <= 0:
        raise ValueError("index must be finite and positive")
    x = repair_bounds(tensor.float(), clamp_range)
    low, high = _bounds(x, clamp_range)
    span = high - low
    safe_span = span.clamp_min(1e-12)
    d1, d2 = (x - low) / safe_span, (high - x) / safe_span
    rand = torch.rand_like(x)
    power = 1 / (index + 1)
    left = (2 * rand + (1 - 2 * rand) * (1 - d1).pow(index + 1)).clamp_min(0).pow(
        power
    ) - 1
    right = 1 - (2 * (1 - rand) + 2 * (rand - 0.5) * (1 - d2).pow(index + 1)).clamp_min(
        0
    ).pow(power)
    delta = torch.where(rand <= 0.5, left, right)
    mask = (torch.rand_like(x) < participation) & (span > 0)
    out = torch.where(mask, x + delta * span, x)
    return repair_bounds(out, clamp_range, dtype=tensor.dtype)
