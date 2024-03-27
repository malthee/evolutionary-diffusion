"""
Base classes for evolutionary computation in a generational environment.
Crossover and Mutation happens on the argument (A) level, whilst the fitness is evaluated on the result (R) level.
SolutionCandidates are created by a SolutionCreator, their representation is split into arguments (A) and result (R).
The fitness type is generic and can be either a single value or a sequence of values in case of multi-objective
optimization.
By default, uses maximization of fitness values, to minimize a fitness value, use the negative value.
"""
from typing import Generic, TypeVar, Optional, List, Sequence, Any
from abc import abstractmethod, ABC

A = TypeVar('A')
R = TypeVar('R')
R_covariant = TypeVar('R_covariant', covariant=True)
SingleObjectiveFitness = float
MultiObjectiveFitness = Sequence[float]
Fitness = TypeVar('Fitness', SingleObjectiveFitness, MultiObjectiveFitness)
"""
Fitness may be a single value or a sequence of values in case of multi-objective optimization.
"""


class Evaluator(Generic[R_covariant, Fitness], ABC):
    @abstractmethod
    def evaluate(self, result: R_covariant) -> Fitness:
        """
        Evaluates the fitness of a result, higher is better (maximization).
        """
        pass


class SingleObjectiveEvaluator(Evaluator[R_covariant, SingleObjectiveFitness], ABC):
    @abstractmethod
    def evaluate(self, result: R_covariant) -> SingleObjectiveFitness:
        pass


class Mutator(Generic[A], ABC):
    @abstractmethod
    def mutate(self, argument: A) -> A:
        pass


class Crossover(Generic[A], ABC):
    @abstractmethod
    def crossover(self, argument1: A, argument2: A) -> A:
        pass


class SolutionCandidate(Generic[A, R, Fitness]):
    def __init__(self, arguments: A, result: R):
        self._arguments = arguments
        self._result = result
        self.fitness: Optional[Fitness] = None

    @property
    def arguments(self) -> A:
        return self._arguments

    @property
    def result(self) -> R:
        return self._result


class SolutionCreator(Generic[A, R], ABC):
    @abstractmethod
    def create_solution(self, argument: A) -> SolutionCandidate[A, R, Any]:
        pass


class Selector(Generic[Fitness], ABC):
    @abstractmethod
    def select(self, candidates: List[SolutionCandidate[Any, Any, Fitness]]) -> SolutionCandidate[Any, Any, Fitness]:
        pass
