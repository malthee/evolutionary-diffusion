from typing import Tuple
import random
import torch
from evolutionary.evolution_base import Mutator, Crossover
from evolutionary_prompt_embedding.argument_types import PromptEmbedData, PooledPromptEmbedData
from evolutionary_model_helpers.tensor_variation import (uniform_crossover_tensors, uniform_gaussian_mutate_tensor,
                                                         arithmetic_crossover, slerp_crossover,
                                                         spherical_rotation_mutate_tensor, row_uniform_crossover, point_crossover,
                                                         blx_alpha_crossover, bounded_sbx_crossover,
                                                         bounded_polynomial_mutate_tensor, repair_bounds, _probability)


class ArithmeticCrossover(Crossover[PromptEmbedData]):
    def __init__(self, interpolation_weight: float, proportion: float = 1.0):
        """
        :param interpolation_weight: The weight for interpolation (between 0 and 1). A weight of 0.5 results in an
        equal blend of both tensors. First tensor is multiplied by the weight, second tensor by (1 - weight).
        :param proportion: The proportion of elements to interpolate. If 1.0 then full arithmetic crossover
        is performed. When not selected for crossover, elements are taken from the first embedding.
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


class SlerpCrossover(Crossover[PromptEmbedData]):
    def __init__(self, ratio: float = 0.5, clamp_range=None):
        """
        Perform spherical linear interpolation (SLERP) between two prompt embeddings.

        :param ratio: (float): Interpolation parameter between 0 and 1.
        """
        self._ratio = ratio
        self._clamp_range = clamp_range

    def crossover(
        self, argument1: PromptEmbedData, argument2: PromptEmbedData
    ) -> PromptEmbedData:
        new_embeds = slerp_crossover(
            argument1.prompt_embeds,
            argument2.prompt_embeds,
            self._ratio,
            self._clamp_range,
        )
        return PromptEmbedData(new_embeds)


class PooledSlerpCrossover(Crossover[PooledPromptEmbedData]):
    def __init__(
        self,
        ratio: float = 0.5,
        ratio_pooled: float = 0.5,
        clamp_range=None,
        clamp_range_pooled=None,
    ):
        """
        Perform spherical linear interpolation (SLERP) between two pooled prompt embeddings.

        :param ratio: (float): Interpolation parameter between 0 and 1 for prompt embeddings.
        :param ratio_pooled: (float): Interpolation parameter between 0 and 1 for pooled prompt embeddings.
        """
        self._clamp_range = clamp_range
        self._clamp_range_pooled = clamp_range_pooled
        self._ratio = ratio
        self._ratio_pooled = ratio_pooled

    def crossover(
        self, argument1: PooledPromptEmbedData, argument2: PooledPromptEmbedData
    ) -> PooledPromptEmbedData:
        new_embeds = slerp_crossover(
            argument1.prompt_embeds,
            argument2.prompt_embeds,
            self._ratio,
            self._clamp_range,
        )
        new_pooled_embeds = slerp_crossover(
            argument1.pooled_prompt_embeds,
            argument2.pooled_prompt_embeds,
            self._ratio_pooled,
            self._clamp_range_pooled,
        )
        return PooledPromptEmbedData(new_embeds, new_pooled_embeds)


class SphericalRotationMutator(Mutator[PromptEmbedData]):
    def __init__(
        self, mutation_rate: float = 0.1, mutation_angle: float = 0.1, clamp_range=None
    ):
        """
        Rotate each embedding row by a small random angle, preserving its original norm.

        :param mutation_rate: (float): Fraction of embeddings (rows) to perturb.
        :param mutation_angle: (float): Maximum angular deviation (radians).
        """
        self._clamp_range = clamp_range
        self._mutation_rate = mutation_rate
        self._mutation_angle = mutation_angle

    def mutate(self, argument: PromptEmbedData) -> PromptEmbedData:
        new_embeds = spherical_rotation_mutate_tensor(
            argument.prompt_embeds,
            self._mutation_rate,
            self._mutation_angle,
            self._clamp_range,
        )
        return PromptEmbedData(new_embeds)


class PooledSphericalRotationMutator(Mutator[PooledPromptEmbedData]):
    def __init__(
        self,
        mutation_rate: float = 0.1,
        mutation_angle: float = 0.1,
        mutation_rate_pooled: float = 0.1,
        mutation_angle_pooled: float = 0.1,
        clamp_range=None,
        clamp_range_pooled=None,
    ):
        """
        Rotate prompt and pooled rows by small random angles, preserving original norms.

        :param mutation_rate: (float): Fraction of prompt embeddings (rows) to perturb.
        :param mutation_angle: (float): Maximum angular deviation (radians) for prompt embeddings.
        :param mutation_rate_pooled: (float): Fraction of pooled embeddings (rows) to perturb.
        :param mutation_angle_pooled: (float): Maximum angular deviation (radians) for pooled embeddings.
        """
        self._clamp_range = clamp_range
        self._clamp_range_pooled = clamp_range_pooled
        self._mutation_rate = mutation_rate
        self._mutation_angle = mutation_angle
        self._mutation_rate_pooled = mutation_rate_pooled
        self._mutation_angle_pooled = mutation_angle_pooled

    def mutate(self, argument: PooledPromptEmbedData) -> PooledPromptEmbedData:
        new_embeds = spherical_rotation_mutate_tensor(
            argument.prompt_embeds,
            self._mutation_rate,
            self._mutation_angle,
            self._clamp_range,
        )
        new_pooled_embeds = spherical_rotation_mutate_tensor(
            argument.pooled_prompt_embeds,
            self._mutation_rate_pooled,
            self._mutation_angle_pooled,
            self._clamp_range_pooled,
        )
        return PooledPromptEmbedData(new_embeds, new_pooled_embeds)


class RowUniformCrossover(Crossover[PromptEmbedData]):
    def __init__(self, swap_rate=0.5):
        self.swap_rate = swap_rate

    def crossover(
        self, argument1: PromptEmbedData, argument2: PromptEmbedData
    ) -> PromptEmbedData:
        return PromptEmbedData(
            row_uniform_crossover(
                argument1.prompt_embeds, argument2.prompt_embeds, self.swap_rate
            )
        )


class PooledRowUniformCrossover(Crossover[PooledPromptEmbedData]):
    def __init__(self, swap_rate=0.5):
        self.swap_rate = swap_rate

    def crossover(
        self, argument1: PooledPromptEmbedData, argument2: PooledPromptEmbedData
    ) -> PooledPromptEmbedData:
        return PooledPromptEmbedData(
            row_uniform_crossover(
                argument1.prompt_embeds, argument2.prompt_embeds, self.swap_rate
            ),
            uniform_crossover_tensors(
                argument1.pooled_prompt_embeds,
                argument2.pooled_prompt_embeds,
                self.swap_rate,
            ),
        )


class OnePointCrossover(Crossover[PromptEmbedData]):
    points = 1

    def crossover(
        self, argument1: PromptEmbedData, argument2: PromptEmbedData
    ) -> PromptEmbedData:
        return PromptEmbedData(
            point_crossover(
                argument1.prompt_embeds, argument2.prompt_embeds, self.points, -2
            )
        )


class TwoPointCrossover(OnePointCrossover):
    points = 2


class PooledOnePointCrossover(Crossover[PooledPromptEmbedData]):
    points = 1

    def crossover(
        self, argument1: PooledPromptEmbedData, argument2: PooledPromptEmbedData
    ) -> PooledPromptEmbedData:
        return PooledPromptEmbedData(
            point_crossover(
                argument1.prompt_embeds, argument2.prompt_embeds, self.points, -2
            ),
            point_crossover(
                argument1.pooled_prompt_embeds,
                argument2.pooled_prompt_embeds,
                self.points,
                -1,
            ),
        )


class PooledTwoPointCrossover(PooledOnePointCrossover):
    points = 2


class BLXAlphaCrossover(Crossover[PromptEmbedData]):
    def __init__(self, clamp_range, alpha=0.5):
        self.clamp_range, self.alpha = clamp_range, alpha

    def crossover(
        self, argument1: PromptEmbedData, argument2: PromptEmbedData
    ) -> PromptEmbedData:
        return PromptEmbedData(
            blx_alpha_crossover(
                argument1.prompt_embeds,
                argument2.prompt_embeds,
                self.clamp_range,
                self.alpha,
            )
        )


class PooledBLXAlphaCrossover(Crossover[PooledPromptEmbedData]):
    def __init__(self, clamp_range, clamp_range_pooled, alpha=0.5):
        self.clamp_range, self.alpha = clamp_range, alpha
        self.clamp_range_pooled = clamp_range_pooled

    def crossover(
        self, argument1: PooledPromptEmbedData, argument2: PooledPromptEmbedData
    ) -> PooledPromptEmbedData:
        return PooledPromptEmbedData(
            blx_alpha_crossover(
                argument1.prompt_embeds,
                argument2.prompt_embeds,
                self.clamp_range,
                self.alpha,
            ),
            blx_alpha_crossover(
                argument1.pooled_prompt_embeds,
                argument2.pooled_prompt_embeds,
                self.clamp_range_pooled,
                self.alpha,
            ),
        )


class SBXCrossover(Crossover[PromptEmbedData]):
    def __init__(self, clamp_range, index=15, participation=0.5):
        self.clamp_range, self.index, self.participation = (
            clamp_range,
            index,
            participation,
        )

    def crossover(
        self, argument1: PromptEmbedData, argument2: PromptEmbedData
    ) -> PromptEmbedData:
        return PromptEmbedData(
            bounded_sbx_crossover(
                argument1.prompt_embeds,
                argument2.prompt_embeds,
                self.clamp_range,
                self.index,
                self.participation,
            )
        )


class PooledSBXCrossover(Crossover[PooledPromptEmbedData]):
    def __init__(self, clamp_range, clamp_range_pooled, index=15, participation=0.5):
        self.clamp_range, self.index, self.participation = (
            clamp_range,
            index,
            participation,
        )
        self.clamp_range_pooled = clamp_range_pooled

    def crossover(
        self, argument1: PooledPromptEmbedData, argument2: PooledPromptEmbedData
    ) -> PooledPromptEmbedData:
        child_index = random.randrange(2)
        return PooledPromptEmbedData(
            bounded_sbx_crossover(
                argument1.prompt_embeds,
                argument2.prompt_embeds,
                self.clamp_range,
                self.index,
                self.participation,
                child_index,
            ),
            bounded_sbx_crossover(
                argument1.pooled_prompt_embeds,
                argument2.pooled_prompt_embeds,
                self.clamp_range_pooled,
                self.index,
                self.participation,
                child_index,
            ),
        )


class PolynomialMutator(Mutator[PromptEmbedData]):
    def __init__(self, clamp_range, index=20, participation=0.1):
        self.clamp_range, self.index, self.participation = (
            clamp_range,
            index,
            participation,
        )

    def mutate(self, argument: PromptEmbedData) -> PromptEmbedData:
        return PromptEmbedData(
            bounded_polynomial_mutate_tensor(
                argument.prompt_embeds, self.clamp_range, self.index, self.participation
            )
        )


class PooledPolynomialMutator(Mutator[PooledPromptEmbedData]):
    def __init__(self, clamp_range, clamp_range_pooled, index=20, participation=0.1):
        self.clamp_range, self.index, self.participation = (
            clamp_range,
            index,
            participation,
        )
        self.clamp_range_pooled = clamp_range_pooled

    def mutate(self, argument: PooledPromptEmbedData) -> PooledPromptEmbedData:
        return PooledPromptEmbedData(
            bounded_polynomial_mutate_tensor(
                argument.prompt_embeds, self.clamp_range, self.index, self.participation
            ),
            bounded_polynomial_mutate_tensor(
                argument.pooled_prompt_embeds,
                self.clamp_range_pooled,
                self.index,
                self.participation,
            ),
        )


class DonorReplacementMutator(Mutator[PromptEmbedData]):
    """Copy selected rows from the same positions of a frozen initial-population donor."""

    def __init__(self, initial_arguments, participation=0.1, clamp_range=None):
        _probability(participation, "participation")
        if not initial_arguments:
            raise ValueError("The donor bank must not be empty")
        self._donors = tuple(
            argument.prompt_embeds.detach().clone() for argument in initial_arguments
        )
        self.participation, self.clamp_range = participation, clamp_range

    def mutate(self, argument: PromptEmbedData) -> PromptEmbedData:
        donor = random.choice(self._donors).to(argument.prompt_embeds)
        return PromptEmbedData(
            repair_bounds(
                row_uniform_crossover(
                    argument.prompt_embeds, donor, self.participation
                ),
                self.clamp_range,
            )
        )


class PooledDonorReplacementMutator(Mutator[PooledPromptEmbedData]):
    def __init__(
        self,
        initial_arguments,
        participation=0.1,
        clamp_range=None,
        clamp_range_pooled=None,
    ):
        _probability(participation, "participation")
        if not initial_arguments:
            raise ValueError("The donor bank must not be empty")
        self._donors = tuple(
            argument.prompt_embeds.detach().clone() for argument in initial_arguments
        )
        self.participation, self.clamp_range = participation, clamp_range
        self._pooled_donors = tuple(
            argument.pooled_prompt_embeds.detach().clone()
            for argument in initial_arguments
        )
        self.clamp_range_pooled = clamp_range_pooled

    def mutate(self, argument: PooledPromptEmbedData) -> PooledPromptEmbedData:
        index = random.randrange(len(self._donors))
        donor = self._donors[index].to(argument.prompt_embeds)
        pooled = argument.pooled_prompt_embeds
        mask = torch.rand(pooled.shape, device=pooled.device) < self.participation
        return PooledPromptEmbedData(
            repair_bounds(
                row_uniform_crossover(
                    argument.prompt_embeds, donor, self.participation
                ),
                self.clamp_range,
            ),
            repair_bounds(
                torch.where(mask, self._pooled_donors[index].to(pooled), pooled),
                self.clamp_range_pooled,
            ),
        )
