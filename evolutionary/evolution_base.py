"""
Base classes for evolutionary computation in a generational environment.
Crossover and Mutation happens on the argument (A) level, whilst the fitness is evaluated on the result (R) level.
SolutionCandidates are created by a SolutionCreator, their representation is split into arguments (A) and result (R).
"""
from random import randint
from typing import Generic, TypeVar, Optional, List, Callable
from abc import abstractmethod, ABC

A = TypeVar('A')
R = TypeVar('R')
R_covariant = TypeVar('R_covariant', covariant=True)


class Evaluator(Generic[R_covariant], ABC):
    @abstractmethod
    def evaluate(self, result: R_covariant) -> float:
        pass


class Mutator(Generic[A], ABC):
    @abstractmethod
    def mutate(self, argument: A) -> A:
        pass


class Crossover(Generic[A], ABC):
    @abstractmethod
    def crossover(self, argument1: A, argument2: A) -> A:
        pass


class SolutionCandidate(Generic[A, R]):
    def __init__(self, arguments: A, result: R):
        self._arguments = arguments
        self._result = result
        self._fitness: Optional[float] = None

    @property
    def fitness(self) -> Optional[float]:
        return self._fitness

    @fitness.setter
    def fitness(self, value: Optional[float]):
        self._fitness = value

    @property
    def arguments(self) -> A:
        return self._arguments

    @property
    def result(self) -> R:
        return self._result


class SolutionCreator(Generic[A, R], ABC):
    @abstractmethod
    def create_solution(self, argument: A) -> SolutionCandidate[A, R]:
        pass


class Selector(ABC):
    @abstractmethod
    def select(self, candidates: List[SolutionCandidate]) -> SolutionCandidate:
        pass


class Algorithm(ABC, Generic[A, R]):
    """
    Base class for evolutionary algorithms.
    """

    """Callback for a generation event. The first argument is the current generation."""
    GenerationCallback = Callable[[int, 'Algorithm[A, R]'], None]

    FitnessList = List[float]

    def __init__(self, num_generations: int,
                 population_size: int,
                 solution_creator: SolutionCreator[A, R],
                 selector: Selector,
                 mutator: Mutator[A],
                 crossover: Crossover[A],
                 evaluator: Evaluator[R],
                 initial_arguments: List[A],
                 post_evaluation_callback: Optional[GenerationCallback] = None):
        self._num_generations = num_generations
        self._population_size = population_size
        self._solution_creator = solution_creator
        self._selector = selector
        self._mutator = mutator
        self._crossover = crossover
        self._evaluator = evaluator
        self._initial_arguments = initial_arguments
        self._post_evaluation_callback = post_evaluation_callback
        self._avg_fitness: Algorithm.FitnessList = []
        self._worst_fitness: Algorithm.FitnessList = []
        self._best_fitness: Algorithm.FitnessList = []
        self._population: List[SolutionCandidate[A, R]] = []

    def __calculate_fitness_statistics(self) -> None:
        """
        Calculates the average, worst and best fitness of the current population.
        """
        fitness_values = [candidate.fitness for candidate in self._population]
        self._best_fitness.append(max(fitness_values))
        self._worst_fitness.append(min(fitness_values))
        self._avg_fitness.append(sum(fitness_values) / len(fitness_values))

    def _evaluate_population(self, generation: int) -> None:
        """
        Evaluates and sets the fitness of all candidates in the population.
        Tracks statistics and calls the post evaluation callback if set.
        """
        for candidate in self._population:
            candidate.fitness = self._evaluator.evaluate(candidate.result)

        if self._post_evaluation_callback:
            self._post_evaluation_callback(generation, self)

        self.__calculate_fitness_statistics()

    def _create_initial_population(self) -> List[SolutionCandidate[A, R]]:
        """
        Initializes the population controlled by initial arguments through the solution creator.
        If not enough initial arguments are given, the rest of the population is filled with random choices.
        """
        self._population = []
        for arg in self._initial_arguments:
            self._population.append(self._solution_creator.create_solution(arg))

        # If there are not enough initial arguments, fill the rest with random choices
        while len(self._population) < self._population_size:
            self._population.append(self._solution_creator.create_solution(self._initial_arguments[
                                                                               randint(0,
                                                                                       len(self._initial_arguments) - 1)]))

        return self._population

    @property
    def num_generations(self):
        return self._num_generations

    @property
    def avg_fitness(self):
        return self._avg_fitness

    @property
    def worst_fitness(self):
        return self._worst_fitness

    @property
    def best_fitness(self):
        return self._best_fitness

    @property
    def population(self):
        return self._population

    @abstractmethod
    def _run_algorithm(self) -> SolutionCandidate[A, R]:
        """
        Inner method to run the algorithm. Must be implemented by subclasses.
        """
        pass

    def run(self) -> SolutionCandidate[A, R]:
        """
        Runs the algorithm for the specified number of generations and returns the best solution candidate.
        TODO evaluate if can also call the loop?
        """
        # (Re)-Initialize lists to store fitness statistics
        self._best_fitness = []
        self._worst_fitness = []
        self._avg_fitness = []
        self._create_initial_population()
        return self._run_algorithm()
