"""Fixed random operator pools using the existing single-child interfaces."""

import math
import random

from evolutionary.evolution_base import A, Crossover, Mutator


class _OperatorPool:
    def __init__(self, operators, weights=None):
        # Mapping names are explicit stable labels for analysis and exported configurations.
        self.operators = dict(operators)
        if not self.operators or any(
            not isinstance(name, str) or not name for name in self.operators
        ):
            raise ValueError(
                "Provide a nonempty mapping of operator names to operators"
            )
        self.weights = (
            tuple(weights) if weights is not None else (1.0,) * len(self.operators)
        )
        if len(self.weights) != len(self.operators) or any(
            not math.isfinite(w) or w < 0 for w in self.weights
        ):
            raise ValueError(
                "Weights must be finite, nonnegative, and match the operator count"
            )
        if not math.isfinite(sum(self.weights)) or sum(self.weights) <= 0:
            raise ValueError("Weights must have a finite positive sum")
        self.last_operator_name = None

    def _choose(self):
        self.last_operator_name = random.choices(
            tuple(self.operators), weights=self.weights, k=1
        )[0]
        return self.operators[self.last_operator_name]


class MultiCrossover(_OperatorPool, Crossover[A]):
    def crossover(self, argument1: A, argument2: A) -> A:
        return self._choose().crossover(argument1, argument2)


class MultiMutator(_OperatorPool, Mutator[A]):
    def mutate(self, argument: A) -> A:
        return self._choose().mutate(argument)
