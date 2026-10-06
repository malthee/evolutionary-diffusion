"""Shared evaluated-population lifecycle for the unconstrained NSGA variants."""

import random
from abc import abstractmethod
from dataclasses import replace

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from evolutionary.algorithms.algorithm_base import Algorithm
from evolutionary.evolution_base import A, MultiObjectiveFitness, R
from evolutionary.history import (
    SOLUTION_SOURCE_META_KEY,
    SolutionHistoryItem,
    SolutionHistoryKey,
)


class _MultiObjectiveAlgorithm(Algorithm[A, R, MultiObjectiveFitness]):
    def _configure_variation(
        self,
        selector,
        mutator,
        crossover,
        mutation_rate,
        crossover_rate,
        post_sort_callback,
    ):
        for name, rate in (
            ("mutation_rate", mutation_rate),
            ("crossover_rate", crossover_rate),
        ):
            if not np.isfinite(rate) or not 0 <= rate <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        self._selector, self._mutator, self._crossover = selector, mutator, crossover
        self._mutation_rate, self._crossover_rate = (
            float(mutation_rate),
            float(crossover_rate),
        )
        self._post_nd_callback = post_sort_callback
        self._fronts = []
        self._objective_count = None
        self.evaluation_count = 0

    def _create_candidate(self, arguments):
        created = self._solution_creator.create_solution(arguments)
        candidate = self.candidate_type(created.arguments, created.result)
        candidate.fitness = created.fitness
        candidate.meta = dict(created.meta)
        return candidate

    def create_initial_population(self):
        self._fronts = []
        self._objective_count = None
        self.evaluation_count = 0
        self._completed_generations = 0
        self._population = []
        self._statistics.start_time_tracking("creation")
        for index in range(self.population_size):
            self._population.append(
                self._create_candidate(
                    self._initial_arguments[index % len(self._initial_arguments)]
                )
            )
            self._statistics.add_history_item(
                SolutionHistoryItem(
                    SolutionHistoryKey(index, 0, self.ident),
                    False,
                    creation_kind="initial",
                )
            )
        self._statistics.stop_time_tracking("creation")

    def _evaluate_candidates(self, candidates):
        self._statistics.start_time_tracking("evaluation")
        for candidate in candidates:
            if candidate.fitness is None:
                candidate.fitness = self._evaluator.evaluate(candidate.result)
                self.evaluation_count += 1
            fitness = np.asarray(candidate.fitness, dtype=float)
            if fitness.ndim != 1 or not len(fitness) or not np.isfinite(fitness).all():
                raise ValueError("Fitness must be a nonempty, finite objective vector.")
            if self._objective_count is None:
                self._objective_count = len(fitness)
            if len(fitness) != self._objective_count:
                raise ValueError(
                    "All candidates must have the same number of objectives."
                )
            # Lists of Python floats work with existing statistics and JSON export.
            candidate.fitness = fitness.tolist()
        self._statistics.stop_time_tracking("evaluation")

    def _cache_fronts(self, candidates):
        fitness = -np.asarray(
            [candidate.fitness for candidate in candidates], dtype=float
        )
        indices = NonDominatedSorting().do(fitness)
        fronts = [[candidates[int(i)] for i in front] for front in indices]
        for rank, front in enumerate(fronts):
            for candidate in front:
                candidate.rank = rank
        return fronts

    def evaluate_population(self, generation):
        # Children were evaluated before survival; cached parents cost no calls.
        if generation == 0 or any(c.fitness is None for c in self.population):
            self._evaluate_candidates(self.population)
        self._prepare_population()
        self._completed_generations = generation + 1
        self._statistics.update_fitness(self.population)
        self._statistics.start_time_tracking("post_evaluation")
        if self._post_evaluation_callback:
            self._post_evaluation_callback(generation, self)
        if self._post_nd_callback:
            self._post_nd_callback(generation, self)
        self._statistics.stop_time_tracking("post_evaluation")

    def _parent_key(self, parent, parents, generation):
        source = parent.meta.get(SOLUTION_SOURCE_META_KEY)
        return SolutionHistoryKey(
            source.index if source else parents.index(parent),
            generation,
            source.ident if source else self.ident,
        )

    @staticmethod
    def _operator_name(operator):
        return getattr(operator, "last_operator_name", None) or type(operator).__name__

    def _offspring_parents(self, parents):
        for _ in range(self.population_size):
            yield self._selector.select(parents), None

    def perform_generation(self, generation):
        parents = self.population
        offspring, history = [], {}
        self._statistics.start_time_tracking("creation")
        for parent, second in self._offspring_parents(parents):
            parent_key = self._parent_key(parent, parents, generation)
            arguments, second_key = parent.arguments, None
            crossover_name, mutation_name = None, None
            if self._crossover and random.random() < self._crossover_rate:
                if second is None:
                    second = self._selector.select(parents)
                arguments = self._crossover.crossover(arguments, second.arguments)
                second_key = self._parent_key(second, parents, generation)
                crossover_name = self._operator_name(self._crossover)
            if self._mutator and random.random() < self._mutation_rate:
                arguments = self._mutator.mutate(arguments)
                mutation_name = self._operator_name(self._mutator)
            child = self._create_candidate(arguments)
            history[id(child)] = SolutionHistoryItem(
                SolutionHistoryKey(0, generation + 1, self.ident),
                mutation_name is not None,
                parent_key,
                second_key,
                crossover_name=crossover_name,
                mutation_name=mutation_name,
                creation_kind="offspring",
            )
            offspring.append(child)
        self._statistics.stop_time_tracking("creation")
        self._evaluate_candidates(offspring)
        combined = parents + offspring
        indices = self._select_survivors(combined)
        self._population = [combined[int(i)] for i in indices]
        # Survivor indices must describe the selected population, not creation order.
        for index, candidate in enumerate(self.population):
            key = SolutionHistoryKey(index, generation + 1, self.ident)
            if id(candidate) in history:
                item = replace(history[id(candidate)], key=key)
            else:
                item = SolutionHistoryItem(
                    key,
                    False,
                    parent_1=self._parent_key(candidate, parents, generation),
                    creation_kind="elite",
                )
            self._statistics.add_history_item(item)
        self._prepare_population()

    @property
    def fronts(self):
        return [list(front) for front in self._fronts]

    @property
    def pareto_front(self):
        return list(self._fronts[0]) if self._fronts else []

    def best_solution(self):
        """Convenience representative; the Pareto front is the optimization result.

        Equal-weight min/max scaling is used only here, never in selection.
        """
        if not self._fronts:
            raise RuntimeError("Evaluate the population before requesting a solution.")
        front = self.pareto_front
        fitness = np.asarray([candidate.fitness for candidate in front], dtype=float)
        spread = np.ptp(fitness, axis=0)
        scaled = np.divide(
            fitness - fitness.min(axis=0),
            spread,
            out=np.zeros_like(fitness),
            where=spread > 0,
        )
        return front[int(np.argmax(scaled.sum(axis=1)))]

    @abstractmethod
    def _prepare_population(self):
        """Prepare rank/diversity metadata before callbacks and mating."""

    @abstractmethod
    def _select_survivors(self, candidates):
        """Return indices selected from the evaluated parent/child union."""
