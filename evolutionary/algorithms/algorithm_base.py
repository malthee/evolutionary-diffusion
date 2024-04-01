from abc import ABC, abstractmethod
from typing import Generic, Callable, List, Optional

from tqdm import tqdm

from evolutionary.evolution_base import A, R, Fitness, SolutionCreator, Evaluator, SolutionCandidate
from evolutionary.statistics import StatisticsTracker, Stages


class Algorithm(ABC, Generic[A, R, Fitness]):
    """
    Base class for evolutionary algorithms.
    When implementing an algorithm, you should subclass this class and implement the abstract methods.
    Additionally, you should weave in the statistics tracker at the appropriate places.
    """

    GenerationCallback = Callable[[int, 'Algorithm[A, R, Fitness]'], None]
    """Callback for a generation event. The first argument is the current generation."""
    FitnessList = List[Fitness]
    """Either a list of single fitness values or a list of lists for multi-objective optimization."""

    def __init__(self, num_generations: int,
                 population_size: int,
                 solution_creator: SolutionCreator[A, R],
                 evaluator: Evaluator[R, Fitness],
                 initial_arguments: List[A],
                 post_evaluation_callback: Optional[GenerationCallback] = None):
        assert num_generations > 0, "Number of generations must be greater than 0"
        assert population_size > 0, "Population size must be greater than 0"
        assert len(initial_arguments) > 0, "Initial arguments must not be empty"
        self._num_generations = num_generations
        self._population_size = population_size
        self._solution_creator = solution_creator
        self._evaluator = evaluator
        self._initial_arguments = initial_arguments
        self._post_evaluation_callback = post_evaluation_callback
        self._statistics = StatisticsTracker()  # Weaved into methods for tracking
        self._population: List[SolutionCandidate[A, R, Fitness]] = []

    def evaluate_population(self, generation: int) -> None:
        """
        Evaluates and sets the fitness of all candidates in the population.
        Tracks statistics and calls the post evaluation callback if set. Works with any fitness type.
        """
        self._statistics.start_time_tracking('evaluation')
        for candidate in self._population:
            if candidate.fitness is None:  # Only evaluate if not already done
                candidate.fitness = self._evaluator.evaluate(candidate.result)
        self._statistics.stop_time_tracking('evaluation')

        if self._post_evaluation_callback:
            self._post_evaluation_callback(generation, self)

        self._statistics.update_fitness(self._population)

    def create_initial_population(self):
        """
        Initializes the population controlled by initial arguments through the solution creator.
        If not enough initial arguments are given, the rest of the population is filled with random choices.
        """
        self._population = []
        i = 0

        self._statistics.start_time_tracking('creation')
        # Init form the args, wrap around if there are not enough
        while len(self._population) < self._population_size:
            self._population.append(self._solution_creator.create_solution(
                self._initial_arguments[i % len(self._initial_arguments)])
            )
            i += 1
        self._statistics.stop_time_tracking('creation')

    @property
    def num_generations(self):
        return self._num_generations

    @property
    def population_size(self):
        return self._population_size

    @property
    def statistics(self):
        return self._statistics

    @property
    def population(self):
        return self._population

    @abstractmethod
    def perform_generation(self, generation: int):
        """
        Run a single generation of the algorithm.
        The population should be evaluated beforehand.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def best_solution(self) -> SolutionCandidate[A, R, Fitness]:
        """
        Returns the best solution candidate from the population. Must be implemented by subclasses.
        """
        pass

    def run(self) -> SolutionCandidate[A, R, Fitness]:
        """
        Runs the algorithm for the specified number of generations and returns the best solution candidate.
        Evaluates the population each generation and tracks the fitness statistics.
        The last generation will only be evaluated, not processed further.
        """
        # (Re)-Initialize statistics each run
        self._statistics = StatisticsTracker()
        self.create_initial_population()

        for generation in tqdm(range(self.num_generations), unit='generation'):
            self.evaluate_population(generation)

            # If this is the last generation, finish here
            if generation == self.num_generations - 1:
                continue

            self.perform_generation(generation)

        return self.best_solution()
