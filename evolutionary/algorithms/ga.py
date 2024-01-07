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
                 elitism_count: Optional[int] = None,
                 post_evaluation_callback: Optional[Callable[[List[SolutionCandidate[A, R]], int], None]] = None):
        self.best_fitness = None
        self.worst_fitness = None
        self.avg_fitness = None
        self.solution_creator = solution_creator
        self.evaluator = evaluator
        self.mutator = mutator
        self.crossover = crossover
        self.selector = selector
        self.population_size = population_size
        self.num_generations = num_generations
        self.initial_arguments = initial_arguments
        self.elitism_count = elitism_count
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

    def _calculate_fitness_statistics(self, population: List[SolutionCandidate[A, R]]):
        fitness_values = [candidate.fitness for candidate in population]
        self.best_fitness.append(max(fitness_values))
        self.worst_fitness.append(min(fitness_values))
        self.avg_fitness.append(sum(fitness_values) / len(fitness_values))

    def run(self) -> SolutionCandidate[A, R]:
        # Initialize lists to store fitness statistics
        self.best_fitness = []
        self.worst_fitness = []
        self.avg_fitness = []

        population = self._create_initial_population()
        for generation in range(self.num_generations):
            self._evaluate_population(population)
            self._calculate_fitness_statistics(population)
            if self.post_evaluation_callback:
                self.post_evaluation_callback(population, generation)

            # Elitism: Carry over the top individuals if enabled
            elites = sorted(population, key=lambda candidate: candidate.fitness, reverse=True)[:self.elitism_count] \
                if self.elitism_count else []

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1 = self.selector.select(population)
                parent2 = self.selector.select(population)
                offspring_args = self.crossover.crossover(parent1.arguments, parent2.arguments)
                offspring_args = self.mutator.mutate(offspring_args)
                new_population.append(self.solution_creator.create_solution(offspring_args))

            population = new_population
            print(f"Generation {generation + 1} complete.")

        # Final evaluation and statistics
        self._evaluate_population(population)
        self._calculate_fitness_statistics(population)
        if self.post_evaluation_callback:
            self.post_evaluation_callback(population, self.num_generations)

        return max(population, key=lambda candidate: candidate.fitness)
