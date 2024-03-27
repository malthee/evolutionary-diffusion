"""
Implements ways to perform variation on tensors, such as crossover and mutation.
"""

import torch
from typing import Tuple, Union


def uniform_gaussian_mutate_tensor(tensor: torch.Tensor, mutation_rate: float = 0.05, mutation_strength: float = 0.1,
                                   clamp_range: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    """
    Perform a uniform gaussian mutation on a tensor, returning a mutated version of the tensor.

    Args:
    - tensor (torch.Tensor): The tensor to mutate.
    - mutation_rate (float): Fraction of elements to mutate (between 0 and 1).
    - mutation_strength (float): The strength of the mutation, influencing how much each element can change.
    - clamp_range (tuple): A tuple of (min, max) to clamp the mutated values.

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


def uniform_crossover_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor,
                              crossover_rate: float = 0.5) -> torch.Tensor:
    """
    Perform a uniform crossover operation between two tensors, assuming they are on the same device.

    Args:
    - tensor1 (torch.Tensor): The first parent tensor.
    - tensor2 (torch.Tensor): The second parent tensor.
    - crossover_rate (float): The rate at which elements from the second tensor are introduced into the first.

    Returns:
    - torch.Tensor: The resulting tensor after crossover.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Both tensors must have the same shape for crossover.")

    crossover_mask = torch.rand(tensor1.shape, device=tensor1.device) < crossover_rate
    offspring = torch.where(crossover_mask, tensor2, tensor1)

    return offspring


def arithmetic_crossover(tensor1: torch.Tensor, tensor2: torch.Tensor,
                         crossover_rate: float = 0.5) -> torch.Tensor:
    """
    Perform an interpolation-based crossover between two tensors.

    Args:
    - tensor1 (torch.Tensor): The first parent tensor.
    - tensor2 (torch.Tensor): The second parent tensor.
    - interpolation_weight (float): The weight for interpolation (between 0 and 1). A weight of 0.5 results in an
      equal blend of both tensors.

    Returns:
    - torch.Tensor: The resulting tensor after interpolation.
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Both tensors must have the same shape for interpolation.")

    device = tensor1.device
    tensor2 = tensor2.to(device)

    offspring = tensor1 * crossover_rate + tensor2 * (1 - crossover_rate)

    return offspring
