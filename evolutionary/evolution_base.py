"""
Base classes for evolutionary computation in a generational environment.
Crossover and Mutation happens on the argument (A) level, whilst the fitness is evaluated on the result (R) level.
SolutionCandidates are created by a SolutionCreator, their representation is split into arguments (A) and result (R).
"""

from typing import Generic, TypeVar, Optional, List
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
