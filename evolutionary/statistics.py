from time import time
from typing import List, Sequence, Literal, Generic, Iterable

from evolutionary.evolution_base import SolutionCandidate, Fitness

FitnessList = List[Fitness]
"""Either a list of single fitness values or a list of lists for multi-objective optimization."""
TimeList = List[float]
"""List of time values in seconds."""
Stages = Literal["evaluation", "creation"]
"""Stages for time tracking."""


class StatisticsTracker(Generic[Fitness]):
    """
    Statistics tracker for evolutionary algorithms containing fitness progress and time tracking.
    """

    def __init__(self):
        self._best_fitness: FitnessList = []
        self._avg_fitness: FitnessList = []
        self._worst_fitness: FitnessList = []
        self._evaluation_time: TimeList = []
        self._creation_time: TimeList = []
        self._time_trackers = {}

    def update_fitness(self, population: Iterable[SolutionCandidate]):
        """
        Calculates the average, worst and best fitness of the current population.
        For single-objective optimization, the fitness values are single values.
        For multi-objective optimization, the fitness values are sequences of values for each criterion.
        """

        fitness_values = [candidate.fitness for candidate in population]
        if isinstance(fitness_values[0], Sequence):  # Multi-objective
            zipped_fitness = list(zip(*fitness_values))
            # Track each objective separately
            self._best_fitness.append([max(obj_values) for obj_values in zipped_fitness])
            self._worst_fitness.append([min(obj_values) for obj_values in zipped_fitness])
            self._avg_fitness.append([sum(obj_values) / len(obj_values) for obj_values in zipped_fitness])
        else: # Single-objective
            self._best_fitness.append(max(fitness_values))
            self._worst_fitness.append(min(fitness_values))
            self._avg_fitness.append(sum(fitness_values) / len(fitness_values))

    def start_time_tracking(self, stage: Stages):
        self._time_trackers[stage] = time()

    def stop_time_tracking(self, stage: Stages):
        self._time_trackers[stage] = time() - self._time_trackers[stage]

        if stage == "evaluation":
            self._evaluation_time.append(self._time_trackers[stage])
        elif stage == "creation":
            self._creation_time.append(self._time_trackers[stage])
        else:
            raise ValueError("Invalid stage name")

        self._time_trackers.pop(stage)

    @property
    def best_fitness(self) -> FitnessList:
        """
        Best fitness(es) per generation.
        """
        return self._best_fitness

    @property
    def avg_fitness(self) -> FitnessList:
        """
        Average fitness(es) per generation.
        """
        return self._avg_fitness

    @property
    def worst_fitness(self) -> FitnessList:
        """
        Worst fitness(es) per generation.
        """
        return self._worst_fitness

    @property
    def evaluation_time(self) -> TimeList:
        """
        Time taken to evaluate fitness of solution candidates.
        """
        return self._evaluation_time

    @property
    def creation_time(self) -> TimeList:
        """
        Time taken to create new solution candidates.
        """
        return self._creation_time
