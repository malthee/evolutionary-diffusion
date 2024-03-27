from math import log
from typing import Sequence

from evolutionary.evolution_base import SingleObjectiveEvaluator, R_covariant, SingleObjectiveFitness, Evaluator, \
    MultiObjectiveFitness, Fitness


class MultiObjectiveEvaluator(Evaluator[R_covariant, MultiObjectiveFitness]):
    def __init__(self, evaluators: Sequence[Evaluator[R_covariant, Fitness]]):
        """
        Initializes the multi-objective evaluator with a list of single- and multi-objective evaluators.
        The result of multiple evaluators will extend the result of this evaluator.
        """
        self._evaluators = evaluators

    def evaluate(self, result: R_covariant) -> MultiObjectiveFitness:
        fitness = []
        for evaluator in self._evaluators:
            evaluation_result = evaluator.evaluate(result)
            if isinstance(evaluation_result, list):  # Extend with multi-objective results, append single-objective
                fitness.extend(evaluation_result)
            else:
                fitness.append(evaluation_result)
        return fitness


class InverseEvaluator(SingleObjectiveEvaluator):
    """
    Inverts the fitness value of a single-objective evaluator.
    Used to minimize instead of maximize, as by default evaluators are meant to maximize.
    """
    def __init__(self, evaluator: SingleObjectiveEvaluator[R_covariant]):
        self._evaluator = evaluator

    def evaluate(self, result: R_covariant) -> SingleObjectiveFitness:
        return -self._evaluator.evaluate(result)


class CappedEvaluator(SingleObjectiveEvaluator):
    """
    Caps the fitness value of a single-objective evaluator to a maximum value.
    Useful for limiting the fitness value to a certain threshold.
    """

    def __init__(self, evaluator: SingleObjectiveEvaluator[R_covariant], cap_value: float):
        """
        Initializes the capped evaluator with a base evaluator and a cap value.

        Args:
            evaluator: The base evaluator providing initial fitness values.
            cap_value: The maximum fitness value allowed. If the base evaluator returns a value greater than this,
                       the capped value will be used instead.
        """
        self._evaluator = evaluator
        self._cap_value = cap_value

    def evaluate(self, result: R_covariant) -> SingleObjectiveFitness:
        original_fitness = self._evaluator.evaluate(result)
        return min(original_fitness, self._cap_value)


class GoalDiminishingEvaluator(SingleObjectiveEvaluator):
    """
    Modifies fitness values from an underlying evaluator based on a goal value.
    For fitness values exceeding the goal, a logarithmic diminishing returns effect is applied.
    """

    def __init__(self, evaluator: SingleObjectiveEvaluator[R_covariant], goal_value: SingleObjectiveFitness,
                 diminish_scale: float = 1.0):
        """
        Initializes the evaluator with a goal value and a scale for diminishing returns.

        Args:
            evaluator: The base evaluator providing initial fitness values.
            goal_value: The fitness threshold beyond which diminishing returns are applied.
            diminish_scale: Adjusts the rate of diminishing returns for values above the goal. The lower, the stronger
                            the diminishing effect. Default is 1.0.
        """
        assert diminish_scale > 0, "Diminish scale must be greater than 0."

        self._evaluator = evaluator
        self._goal_value = goal_value
        self._diminish_scale = diminish_scale

    def evaluate(self, result: R_covariant) -> SingleObjectiveFitness:
        original_fitness = self._evaluator.evaluate(result)

        # Apply the diminishing returns logic only for values exceeding the goal
        if original_fitness <= self._goal_value:
            return original_fitness
        else:
            # Apply diminishing returns to the portion of the fitness above the goal.
            return self._goal_value + log(1 + (original_fitness - self._goal_value) * self._diminish_scale)