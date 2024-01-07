from random import randint
from typing import List, Callable, Generic, Optional
from evolutionary.evolution_base import (
    SolutionCandidate, SolutionCreator, Evaluator, Mutator, Crossover, Selector, A, R
)


class GeneticAlgorithm(Generic[A, R]):
    def __init__(self,
                 solution_creator: SolutionCreator[A, R],
                 evaluator: Evaluator[R],
                 mutator: Mutator[A],
                 crossover: Crossover[A],
                 selector: Selector,
                 population_size: int,
                 num_generations: int,
                 initial_arguments: List[A],
                 post_evaluation_callback: Optional[Callable[[List[SolutionCandidate[A, R]], int], None]] = None):
        self.solution_creator = solution_creator
        self.evaluator = evaluator
        self.mutator = mutator
        self.crossover = crossover
        self.selector = selector
        self.population_size = population_size
        self.num_generations = num_generations
        self.initial_arguments = initial_arguments
        self.post_evaluation_callback = post_evaluation_callback

    def _create_initial_population(self) -> List[SolutionCandidate[A, R]]:
        population = []
        for arg in self.initial_arguments:
            population.append(self.solution_creator.create_solution(arg))

        # If there are not enough initial arguments, fill the rest with random choices
        while len(population) < self.population_size:
            population.append(self.solution_creator.create_solution(self.initial_arguments[
                                                                        randint(0, len(self.initial_arguments) - 1)]))

        return population

    def _evaluate_population(self, population: List[SolutionCandidate[A, R]]) -> None:
        for candidate in population:
            candidate.fitness = self.evaluator.evaluate(candidate.result)

    def run(self) -> SolutionCandidate[A, R]:
        population = self._create_initial_population()
        for generation in range(self.num_generations):
            self._evaluate_population(population)
            self.post_evaluation_callback(population, generation)

            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.selector.select(population)
                parent2 = self.selector.select(population)
                offspring1_args, offspring2_args = self.crossover.crossover(parent1.arguments, parent2.arguments)
                offspring1_args = self.mutator.mutate(offspring1_args)
                offspring2_args = self.mutator.mutate(offspring2_args)
                new_population.append(self.solution_creator.create_solution(offspring1_args))
                if len(new_population) < self.population_size:
                    new_population.append(self.solution_creator.create_solution(offspring2_args))

            population = new_population

        return max(population, key=lambda candidate: candidate.fitness)
