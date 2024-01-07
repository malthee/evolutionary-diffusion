from typing import Tuple
from evolutionary.evolution_base import Mutator, Crossover
from argument_types import PromptEmbedData, PooledPromptEmbedData
from evolutionary._tensor_variation import (uniform_crossover_tensors, uniform_gaussian_mutate_tensor,
                                            arithmetic_crossover)


class ArithmeticCrossover(Crossover[PromptEmbedData]):
    def __init__(self, crossover_rate: float):
        self._crossover_rate = crossover_rate

    def crossover(self, argument1: PromptEmbedData, argument2: PromptEmbedData) -> PromptEmbedData:
        new_embeds = arithmetic_crossover(argument1.prompt_embeds, argument2.prompt_embeds, self._crossover_rate)
        return PromptEmbedData(new_embeds)


class UniformCrossover(Crossover[PromptEmbedData]):
    def __init__(self, interpolation_weight: float):
        self._interpolation_weight = interpolation_weight

    def crossover(self, argument1: PromptEmbedData, argument2: PromptEmbedData) -> PromptEmbedData:
        new_embeds = uniform_crossover_tensors(argument1.prompt_embeds, argument2.prompt_embeds,
                                               self._interpolation_weight)
        return PromptEmbedData(new_embeds)


class PooledArithmeticCrossover(Crossover[PooledPromptEmbedData]):
    def __init__(self, crossover_rate: float, crossover_rate_pooled: float):
        self._crossover_rate = crossover_rate
        self._crossover_rate_pooled = crossover_rate_pooled

    def crossover(self, argument1: PooledPromptEmbedData, argument2: PooledPromptEmbedData) -> PooledPromptEmbedData:
        new_embeds = arithmetic_crossover(argument1.prompt_embeds, argument2.prompt_embeds, self._crossover_rate)
        new_pooled_embeds = arithmetic_crossover(argument1.pooled_prompt_embeds, argument2.pooled_prompt_embeds,
                                                 self._crossover_rate_pooled)
        return PooledPromptEmbedData(new_embeds, new_pooled_embeds)


class PooledUniformCrossover(Crossover[PooledPromptEmbedData]):
    def __init__(self, interpolation_weight: float, interpolation_weight_pooled: float):
        self._interpolation_weight = interpolation_weight
        self._interpolation_weight_pooled = interpolation_weight_pooled

    def crossover(self, argument1: PooledPromptEmbedData, argument2: PooledPromptEmbedData) -> PooledPromptEmbedData:
        new_embeds = uniform_crossover_tensors(argument1.prompt_embeds, argument2.prompt_embeds,
                                               self._interpolation_weight)
        new_pooled_embeds = uniform_crossover_tensors(argument1.pooled_prompt_embeds, argument2.pooled_prompt_embeds,
                                                      self._interpolation_weight_pooled)
        return PooledPromptEmbedData(new_embeds, new_pooled_embeds)


class UniformGaussianMutatorArguments:
    def __init__(self, mutation_rate: float, mutation_strength: float, clamp_range: Tuple[float, float]):
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
        self._args = args

    def mutate(self, argument: PromptEmbedData) -> PromptEmbedData:
        new_embeds = uniform_gaussian_mutate_tensor(argument.prompt_embeds, self._args.mutation_rate,
                                                    self._args.mutation_strength, self._args.clamp_range)
        return PromptEmbedData(new_embeds)


class PooledUniformGaussianMutator(Mutator[PooledPromptEmbedData]):
    def __init__(self, embed_args: UniformGaussianMutatorArguments, pooled_args: UniformGaussianMutatorArguments):
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
