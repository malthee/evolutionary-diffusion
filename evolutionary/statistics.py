from time import time
from typing import List, Sequence, Literal, Generic, Iterable, Optional, Tuple, Dict
from dataclasses import dataclass, replace
from evolutionary.evolution_base import SolutionCandidate, Fitness
from evolutionary.history import SolutionHistoryItem, SolutionHistoryKey

FitnessList = List[Fitness]
"""Either a list of single fitness values or a list of lists for multi-objective optimization."""
TimeList = List[float]
"""List of time values in seconds."""
Stages = Literal["evaluation", "creation", "post_evaluation"]
"""Stages for time tracking."""

class StatisticsTracker(Generic[Fitness]):
    """
    Statistics tracker for evolutionary algorithms containing fitness progress and time tracking,
    along with solution history tracking.
    """

    def __init__(self):
        self._best_fitness: FitnessList = []
        self._avg_fitness: FitnessList = []
        self._worst_fitness: FitnessList = []
        self._evaluation_time: TimeList = []
        self._creation_time: TimeList = []
        self._post_evaluation_time: TimeList = []
        self._time_trackers = {}
        self._solution_history: Dict[SolutionHistoryKey, SolutionHistoryItem] = {}

    def _custom_time_tracking(self, stage: Stages, seconds: float):
        """
        Manually add a time value for a specific stage.
        """
        if stage == "evaluation":
            self._evaluation_time.append(seconds)
        elif stage == "creation":
            self._creation_time.append(seconds)
        elif stage == "post_evaluation":
            self._post_evaluation_time.append(seconds)
        else:
            raise ValueError("Invalid stage name")

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
        """
        Start time tracking for a specific stage. Call stop_time_tracking to stop tracking.
        """
        self._time_trackers[stage] = time()

    def stop_time_tracking(self, stage: Stages):
        """
        Stop time tracking for a specific stage and store the time taken.
        """
        self._time_trackers[stage] = time() - self._time_trackers[stage]

        self._custom_time_tracking(stage, self._time_trackers[stage])

        self._time_trackers.pop(stage)

    def add_history_item(self, item: SolutionHistoryItem) -> None:
        self.solution_history[item.key] = item

    def history_string(self, key: SolutionHistoryKey, depth: int = 3, indent: str = "") -> str:
        """
        Recursively creates a history string of a solution up to a specified depth.
        """
        if depth < 0 or key not in self.solution_history:
            return indent + "No further history\n"

        item = self.solution_history[key]
        result = indent + str(item) + "\n"

        if item.parent_1 is not None:
            result += indent + "└─ Parent 1:\n"
            result += self.history_string(item.parent_1, depth - 1, indent + "    ")
        if item.parent_2 is not None:
            result += indent + "└─ Parent 2:\n"
            result += self.history_string(item.parent_2, depth - 1, indent + "    ")

        return result

    def shift_history_after_removal(self, remove_key: SolutionHistoryKey) -> None:
        """
        After removing a candidate from an island, update the history of that island so the indices are correct.
        """
        keys_to_update = [
            key for key in self._solution_history
            if key.index > remove_key.index and key.generation == remove_key.generation and key.ident == remove_key.ident
        ]

        for old_key in keys_to_update:
            item = self._solution_history.pop(old_key)
            # Create a new history item with the index shifted by 1.
            new_item = replace(item, key=SolutionHistoryKey(index=item.index - 1, generation=item.generation, ident=item.ident))
            self._solution_history[new_item.key] = new_item

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

    @property
    def post_evaluation_time(self) -> TimeList:
        """
        Time taken for post evaluation callback.
        Only applicable if set in the algorithm.
        """
        return self._post_evaluation_time

    @property
    def solution_history(self) -> Dict[SolutionHistoryKey, SolutionHistoryItem]:
        """
        Solution history to see how solutions were created and mutated.
        """
        return self._solution_history
