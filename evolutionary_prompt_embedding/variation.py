from typing import Tuple, Optional
from evolutionary.evolution_base import Mutator, Crossover
from evolutionary_prompt_embedding.argument_types import PromptEmbedData, PooledPromptEmbedData
from evolutionary_model_helpers.tensor_variation import (uniform_crossover_tensors, uniform_gaussian_mutate_tensor,
                                                         arithmetic_crossover)


class ArithmeticCrossover(Crossover[PromptEmbedData]):
    def __init__(self, interpolation_weight: float, proportion: float = 1.0):
        """
        :param interpolation_weight: The weight for interpolation (between 0 and 1). A weight of 0.5 results in an
        equal blend of both tensors. First tensor is multiplied by the weight, second tensor by (1 - weight).
        :param proportion: The proportion of elements to interpolate. If 1.0 then full arithmetic crossover
        is performed.
        """
        self._interpolation_weight = interpolation_weight
        self._proportion = proportion

    def crossover(self, argument1: PromptEmbedData, argument2: PromptEmbedData) -> PromptEmbedData:
        new_embeds = arithmetic_crossover(argument1.prompt_embeds, argument2.prompt_embeds, self._interpolation_weight,
                                          self._proportion)
        return PromptEmbedData(new_embeds)


class UniformCrossover(Crossover[PromptEmbedData]):
    def __init__(self, swap_rate: float):
        """
        :param swap_rate: The rate at which elements from the second tensor are introduced into the first.
        """
        self._swap_rate = swap_rate

    def crossover(self, argument1: PromptEmbedData, argument2: PromptEmbedData) -> PromptEmbedData:
        new_embeds = uniform_crossover_tensors(argument1.prompt_embeds, argument2.prompt_embeds,
                                               self._swap_rate)
        return PromptEmbedData(new_embeds)


class PooledArithmeticCrossover(Crossover[PooledPromptEmbedData]):
    def __init__(self, interpolation_weight: float, interpolation_weight_pooled: float,
                 proportion: float = 1.0, proportion_pooled: float = 1.0):
        """
        See ArithmeticCrossover. Added crossover rate for pooled embeddings.
        """
        self._interpolation_weight = interpolation_weight
        self._interpolation_weight_pooled = interpolation_weight_pooled
        self._proportion = proportion
        self._proportion_pooled = proportion_pooled

    def crossover(self, argument1: PooledPromptEmbedData, argument2: PooledPromptEmbedData) -> PooledPromptEmbedData:
        new_embeds = arithmetic_crossover(argument1.prompt_embeds, argument2.prompt_embeds, self._interpolation_weight,
                                          self._proportion)
        new_pooled_embeds = arithmetic_crossover(argument1.pooled_prompt_embeds, argument2.pooled_prompt_embeds,
                                                 self._interpolation_weight_pooled, self._proportion_pooled)
        return PooledPromptEmbedData(new_embeds, new_pooled_embeds)


class PooledUniformCrossover(Crossover[PooledPromptEmbedData]):
    def __init__(self, swap_rate: float, swap_rate_pooled: float):
        """
        See UniformCrossover. Added interpolation weight for pooled embeddings.
        """
        self._swap_rate = swap_rate
        self._swap_rate_pooled = swap_rate_pooled

    def crossover(self, argument1: PooledPromptEmbedData, argument2: PooledPromptEmbedData) -> PooledPromptEmbedData:
        new_embeds = uniform_crossover_tensors(argument1.prompt_embeds, argument2.prompt_embeds,
                                               self._swap_rate)
        new_pooled_embeds = uniform_crossover_tensors(argument1.pooled_prompt_embeds, argument2.pooled_prompt_embeds,
                                                      self._swap_rate_pooled)
        return PooledPromptEmbedData(new_embeds, new_pooled_embeds)


class UniformGaussianMutatorArguments:
    def __init__(self, mutation_rate: float, mutation_strength: float, clamp_range: Tuple[float, float]):
        """
        Arguments for the uniform gaussian mutator.

        :param mutation_rate: Rate of elements (how many) of the tensor which will have their value changed.
        :param mutation_strength: The strength of the mutation. The value of the mutated element will be changed by a
        random value from a normal distribution multiplied with the mutation_strength.
        :param clamp_range: The range to clamp the values of the mutated tensor to.
        """
        self._mutation_rate = mutation_rate
        self._mutation_strength = mutation_strength
        self._clamp_range = clamp_range

    @property
    def mutation_rate(self) -> float:
        return self._mutation_rate

    @property
    def mutation_strength(self) -> float:
        return self._mutation_strength

    @property
    def clamp_range(self) -> Tuple[float, float]:
        return self._clamp_range


class UniformGaussianMutator(Mutator[PromptEmbedData]):
    def __init__(self, args: UniformGaussianMutatorArguments):
        """
        :param args: The arguments for the uniform gaussian mutation.
        """
        self._args = args

    def mutate(self, argument: PromptEmbedData) -> PromptEmbedData:
        new_embeds = uniform_gaussian_mutate_tensor(argument.prompt_embeds, self._args.mutation_rate,
                                                    self._args.mutation_strength, self._args.clamp_range)
        return PromptEmbedData(new_embeds)


class PooledUniformGaussianMutator(Mutator[PooledPromptEmbedData]):
    def __init__(self, embed_args: UniformGaussianMutatorArguments, pooled_args: UniformGaussianMutatorArguments):
        """
        See UniformGaussianMutator. Added arguments for pooled embeddings.
        """
        self._embed_args = embed_args
        self._pooled_args = pooled_args

    def mutate(self, argument: PooledPromptEmbedData) -> PooledPromptEmbedData:
        new_embeds = uniform_gaussian_mutate_tensor(argument.prompt_embeds, self._embed_args.mutation_rate,
                                                    self._embed_args.mutation_strength, self._embed_args.clamp_range)
        new_pooled_embeds = uniform_gaussian_mutate_tensor(argument.pooled_prompt_embeds,
                                                           self._pooled_args.mutation_rate,
                                                           self._pooled_args.mutation_strength,
                                                           self._pooled_args.clamp_range)
        return PooledPromptEmbedData(new_embeds, new_pooled_embeds)
