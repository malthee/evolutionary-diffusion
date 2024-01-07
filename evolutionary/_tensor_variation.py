"""
Implements ways to perform variation on tensors, such as crossover and mutation.
"""

import torch


def uniform_gaussian_mutate_tensor(tensor, mutation_rate=0.05, mutation_strength=0.1, clamp_range=(-1, 1)):
    """
    Perform a uniform gaussian mutation on the tensor while keeping it on the same device.

    Args:
    - tensor (torch.Tensor): The tensor to mutate.
    - mutation_rate (float): Fraction of elements to mutate (between 0 and 1).
    - mutation_strength (float): The strength of the mutation, influencing how much each element can change.
    - clamp_range (tuple): A tuple of (min, max) to clamp the mutated values.

    Returns:
    - torch.Tensor: The mutated tensor.
    """
    device = tensor.device  # Get the device of the input tensor
    num_elements_to_mutate = int(torch.numel(tensor) * mutation_rate)
    indices_to_mutate = torch.randperm(torch.numel(tensor), device=device)[:num_elements_to_mutate]

    # Generate mutations
    mutations = torch.randn(num_elements_to_mutate, device=device) * mutation_strength
    flat_tensor = tensor.flatten()
    flat_tensor[indices_to_mutate] += mutations
    mutated_tensor = flat_tensor.view(tensor.shape)

    # Clamp values to ensure they remain within a reasonable range
    mutated_tensor = torch.clamp(mutated_tensor, min=clamp_range[0], max=clamp_range[1])

    return mutated_tensor


# Some examples for possible crossover and mutation in prompt encoding space

def uniform_crossover_tensors(tensor1, tensor2, crossover_rate=0.5):
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

    # Create a mask for crossover
    crossover_mask = torch.rand(tensor1.shape, device=tensor1.device) < crossover_rate

    # Perform crossover
    offspring = torch.where(crossover_mask, tensor2, tensor1)

    return offspring


def arithmetic_crossover(tensor1, tensor2, interpolation_weight=0.5):
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

    # Ensure tensors are on the same device
    device = tensor1.device
    tensor2 = tensor2.to(device)

    # Perform interpolation
    offspring = tensor1 * (1 - interpolation_weight) + tensor2 * interpolation_weight

    return offspring
