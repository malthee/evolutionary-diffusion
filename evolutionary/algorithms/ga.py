import random
from typing import List, Optional
from evolutionary.evolution_base import (
    SolutionCandidate, SolutionCreator, Mutator, Crossover, Selector, A, R, SingleObjectiveEvaluator, SingleObjectiveFitness
)
from evolutionary.algorithms.algorithm_base import Algorithm


class GeneticAlgorithm(Algorithm[A, R, SingleObjectiveFitness]):
    def __init__(self,
                 num_generations: int,
                 population_size: int,
                 solution_creator: SolutionCreator[A, R],
                 selector: Selector[SingleObjectiveFitness],
                 mutator: Mutator[A],
                 crossover: Crossover[A],
                 evaluator: SingleObjectiveEvaluator[R],
                 initial_arguments: List[A],
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 post_evaluation_callback: Optional[Algorithm.GenerationCallback] = None,
                 elitism_count: Optional[int] = None,
                 strict_osga: bool = False):
        super().__init__(
            num_generations=num_generations,
            population_size=population_size,
            solution_creator=solution_creator,
            evaluator=evaluator,
            initial_arguments=initial_arguments,
            post_evaluation_callback=post_evaluation_callback
        )
        self._selector = selector
        self._mutator = mutator
        self._mutation_rate = mutation_rate
        self._crossover = crossover
        self._crossover_rate = crossover_rate
        self._strict_osga = strict_osga  # If True, only better offspring are accepted as defined in OSGA
        self.elitism_count = elitism_count

    def perform_generation(self, generation: int):
        # Elitism: Carry over the top individuals if enabled
        elites = sorted(self._population, key=lambda candidate: candidate.fitness, reverse=True)[:self.elitism_count] \
            if self.elitism_count else []
        new_population = elites.copy()

        while len(new_population) < self._population_size:
            parent1 = self._selector.select(self._population)
            parent2 = None

            if random.random() <= self._crossover_rate:
                parent2 = self._selector.select(self._population)
                offspring_args = self._crossover.crossover(parent1.arguments, parent2.arguments)
            else:
                offspring_args = parent1.arguments  # No crossover, just reuse parent

            if random.random() <= self._mutation_rate:
                offspring_args = self._mutator.mutate(offspring_args)

            offspring = self._solution_creator.create_solution(offspring_args)

            # Only add offspring if it is better than the worst parent when OSGA strict enabled
            if self._strict_osga:
                worst_parent_fitness = min(parent1.fitness, parent2.fitness if parent2 else parent1.fitness)
                if offspring.fitness > worst_parent_fitness:
                    new_population.append(offspring)
            else:
                new_population.append(offspring)

        del self._population
        self._population = new_population

    def best_solution(self) -> SolutionCandidate[A, R, SingleObjectiveFitness]:
        return max(self._population, key=lambda candidate: candidate.fitness)
