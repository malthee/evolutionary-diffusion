import random
from typing import List, Optional

from evolutionary.algorithms._multi_objective import _MultiObjectiveAlgorithm
from evolutionary.algorithms.algorithm_base import Algorithm
from evolutionary.evaluators import MultiObjectiveEvaluator
from evolutionary.evolution_base import (
    A,
    Crossover,
    MultiObjectiveFitness,
    Mutator,
    R,
    Selector,
    SolutionCandidate,
    SolutionCreator,
)


class NSGASolutionCandidate(SolutionCandidate[A, R, MultiObjectiveFitness]):
    def __init__(self, arguments: A, result: R):
        super().__init__(arguments, result)
        self.rank = None
        self.crowding_distance = None


class NSGATournamentSelector(Selector[MultiObjectiveFitness]):
    """Original crowded comparison: lower rank, then larger crowding distance."""

    def select(self, candidates: List[NSGASolutionCandidate]) -> NSGASolutionCandidate:
        first, second = random.choices(candidates, k=2)
        if any(c.rank is None or c.crowding_distance is None for c in (first, second)):
            raise RuntimeError(
                "NSGA-II tournament requires evaluated rank and crowding distance."
            )
        if first.rank != second.rank:
            return first if first.rank < second.rank else second
        if first.crowding_distance != second.crowding_distance:
            return (
                first if first.crowding_distance > second.crowding_distance else second
            )
        return random.choice((first, second))


class NSGA_II(_MultiObjectiveAlgorithm[A, R]):
    """Elitist parent/offspring survival with front-wise crowding (Deb et al., 2002)."""

    candidate_type = NSGASolutionCandidate

    def __init__(
        self,
        num_generations: int,
        population_size: int,
        solution_creator: SolutionCreator[A, R],
        selector: Selector[MultiObjectiveFitness],
        mutator: Mutator[A],
        crossover: Crossover[A],
        evaluator: MultiObjectiveEvaluator[R],
        initial_arguments: List[A],
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        elitism_count: Optional[int] = None,
        normalize_crowding_distance: bool = True,
        post_evaluation_callback: Optional[Algorithm.GenerationCallback] = None,
        post_non_dominated_sort_callback: Optional[Algorithm.GenerationCallback] = None,
        ident: Optional[int] = None,
    ):
        if elitism_count not in (None, 0):
            raise ValueError(
                "NSGA-II already uses elitist union survival; omit elitism_count."
            )
        super().__init__(
            num_generations,
            population_size,
            solution_creator,
            evaluator,
            initial_arguments,
            post_evaluation_callback,
            ident,
        )
        self._configure_variation(
            selector,
            mutator,
            crossover,
            mutation_rate,
            crossover_rate,
            post_non_dominated_sort_callback,
        )
        self._normalize_crowding_distance = normalize_crowding_distance

    def _fast_non_dominated_sort(self):
        self._fronts = self._cache_fronts(self.population)

    def _calculate_crowding_distance(self):
        for front in self._fronts:
            for candidate in front:
                candidate.crowding_distance = 0.0
            if len(front) <= 2:
                for candidate in front:
                    candidate.crowding_distance = float("inf")
                continue
            for objective in range(self._objective_count):
                ordered = sorted(front, key=lambda c: c.fitness[objective])
                span = ordered[-1].fitness[objective] - ordered[0].fitness[objective]
                # Constant objectives carry no diversity information, including at boundaries.
                if span == 0:
                    continue
                ordered[0].crowding_distance = ordered[-1].crowding_distance = float(
                    "inf"
                )
                scale = span if self._normalize_crowding_distance else 1.0
                for index in range(1, len(ordered) - 1):
                    ordered[index].crowding_distance += (
                        ordered[index + 1].fitness[objective]
                        - ordered[index - 1].fitness[objective]
                    ) / scale

    def _prepare_population(self):
        self._fast_non_dominated_sort()
        # After survival, retain crowding computed on union fronts for mating.
        if any(c.crowding_distance is None for c in self.population):
            self._calculate_crowding_distance()

    def _select_survivors(self, candidates):
        self._fronts = self._cache_fronts(candidates)
        self._calculate_crowding_distance()
        indices = {id(candidate): index for index, candidate in enumerate(candidates)}
        selected = []
        for front in self._fronts:
            remaining = self.population_size - len(selected)
            if len(front) > remaining:
                front = sorted(front, key=lambda c: c.crowding_distance, reverse=True)[
                    :remaining
                ]
            selected.extend(indices[id(candidate)] for candidate in front)
            if len(selected) == self.population_size:
                break
        return selected
