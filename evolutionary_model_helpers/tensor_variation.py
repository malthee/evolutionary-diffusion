"""
Implements ways to perform variation on tensors, such as crossover and mutation.
"""

import torch
from typing import Tuple, Union


def uniform_gaussian_mutate_tensor(tensor: torch.Tensor, mutation_rate: float = 0.05, mutation_strength: float = 0.1,
                                   clamp_range: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    """
    Perform a uniform gaussian mutation on a tensor, returning a mutated version of the tensor.

    :param tensor: (torch.Tensor): The tensor to mutate.
    :param mutation_rate: (float): Fraction of elements to mutate (between 0 and 1).
    :param mutation_strength: (float): The strength of the mutation, influencing how much each element can change.
    :param clamp_range: (tuple): A tuple of (min, max) to clamp the mutated values.

    Returns:
    - torch.Tensor: The new mutated tensor.
    """
    device = tensor.device
    # Clone the tensor to ensure the original is not modified
    cloned_tensor = tensor.clone()

    num_elements_to_mutate = int(torch.numel(cloned_tensor) * mutation_rate)
    indices_to_mutate = torch.randperm(torch.numel(cloned_tensor), device=device)[:num_elements_to_mutate]

    mutations = torch.randn(num_elements_to_mutate, device=device) * mutation_strength
    flat_tensor = cloned_tensor.flatten()
    flat_tensor[indices_to_mutate] += mutations
    mutated_tensor = flat_tensor.view(cloned_tensor.shape)

    # Clamp the values of the mutated tensor
    clamp_min, clamp_max = clamp_range
    mutated_tensor = torch.clamp(mutated_tensor, min=clamp_min, max=clamp_max)

    return mutated_tensor

def spherical_rotation_mutate_tensor(
    tensor: torch.Tensor,
    mutation_rate: float = 0.05,
    mutation_angle: float = 0.1
) -> torch.Tensor:
    """
    Rotate each unit-normalized embedding by a small random angle.

    :param tensor: torch.Tensor of shape [D] or [B, D], L2-normalized vectors.
    :param mutation_rate: Fraction of embeddings (rows) to perturb.
    :param mutation_angle: Maximum angular deviation (radians).

    :returns: torch.Tensor same shape as input, with selected vectors rotated.
    """
    single = (tensor.dim() == 1)
    if single:
        tensor = tensor.unsqueeze(0)
    B, D = tensor.shape
    device = tensor.device
    dtype = tensor.dtype

    out = tensor.clone()
    mask = torch.rand(B, device=device, dtype=dtype) < mutation_rate
    if mask.any():
        idxs = mask.nonzero(as_tuple=False).view(-1)
        u = tensor[idxs]  # [M, D]

        # random direction in ambient space
        v = torch.randn((idxs.size(0), D), device=device, dtype=dtype)
        # project onto tangent space at u
        proj = (v * u).sum(dim=1, keepdim=True)
        v_tangent = v - proj * u
        v_tangent = torch.nn.functional.normalize(v_tangent, dim=-1)

        # sample angles and compute trig in same dtype
        theta = (torch.rand((idxs.size(0), 1), device=device, dtype=dtype) * 2 - 1) * mutation_angle
        cos_t = torch.cos(theta).to(device=device, dtype=dtype)
        sin_t = torch.sin(theta).to(device=device, dtype=dtype)

        # rotate along the sphere
        rotated = (u * cos_t) + (v_tangent * sin_t)
        out[idxs] = rotated

    return out.squeeze(0) if single else out

def uniform_crossover_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor,
                              swap_rate: float = 0.5) -> torch.Tensor:
    """
    Perform a uniform crossover operation between two tensors.

    Args:
    :param tensor1: (torch.Tensor): The first parent tensor.
    :param tensor2: (torch.Tensor): The second parent tensor.
    :param swap_rate: (float): The rate at which elements from the second tensor are introduced into the first.

    Returns:
    - torch.Tensor: The resulting tensor after crossover.
    """
    assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape for crossover."

    crossover_mask = torch.rand(tensor1.shape, device=tensor1.device) < swap_rate
    offspring = torch.where(crossover_mask, tensor1, tensor2)

    return offspring

def arithmetic_crossover(tensor1: torch.Tensor, tensor2: torch.Tensor,
                         interpolation_weight: float = 0.5, proportion: float = 1.0) -> torch.Tensor:
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
    assert tensor1.shape == tensor2.shape, "Both tensors must have the same shape for crossover."
    assert 0 < proportion <= 1, "Proportion must be > 0 and <= 1."

    device = tensor1.device
    tensor2 = tensor2.to(device)

    # If proportion is 1, perform full crossover and return immediately
    if proportion == 1.0:
        return tensor1 * interpolation_weight + tensor2 * (1 - interpolation_weight)

    # For partial crossover
    offspring = tensor1.clone()
    num_elements = tensor1.numel()
    num_crossover = int(num_elements * proportion)  # Number of elements to apply crossover

    # Randomly choose indices for crossover
    indices = torch.randperm(num_elements, device=device)[:num_crossover]

    # Apply crossover only to selected indices
    flat_offspring = offspring.view(-1)
    flat_tensor1 = tensor1.view(-1)
    flat_tensor2 = tensor2.view(-1)
    flat_offspring[indices] = flat_tensor1[indices] * interpolation_weight + flat_tensor2[indices] * (
                1 - interpolation_weight)
    return offspring

def slerp_crossover(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    ratio: float
) -> torch.Tensor:
    """
    Perform spherical linear interpolation (SLERP) between two tensors.

    :param tensor1: First tensor of shape [..., D].
    :param tensor2: Second tensor of shape [..., D].
    :param ratio: Interpolation parameter in [0,1].
    :returns: torch.Tensor result of SLERP, same shape as inputs.
    """
    # ensure same dtype/device
    device = tensor1.device
    dtype = tensor1.dtype

    u_norm = torch.nn.functional.normalize(tensor1, dim=-1)
    v_norm = torch.nn.functional.normalize(tensor2, dim=-1)
    dot = (u_norm * v_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-6)

    t = torch.tensor(ratio, device=device, dtype=dtype)
    part1 = torch.sin((1 - t) * omega) / sin_omega * tensor1
    part2 = torch.sin(t * omega) / sin_omega * tensor2

    return part1 + part2
