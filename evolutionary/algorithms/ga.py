from typing import List, Optional
from evolutionary.evolution_base import (
    SolutionCandidate, SolutionCreator, Mutator, Crossover, Selector, A, R, Algorithm,
    SingleObjectiveEvaluator, SingleObjectiveFitness
)


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
                 post_evaluation_callback: Optional[Algorithm.GenerationCallback] = None,
                 elitism_count: Optional[int] = None):
        super().__init__(
            num_generations=num_generations,
            population_size=population_size,
            solution_creator=solution_creator,
            selector=selector,
            mutator=mutator,
            crossover=crossover,
            evaluator=evaluator,
            initial_arguments=initial_arguments,
            post_evaluation_callback=post_evaluation_callback
        )
        self.elitism_count = elitism_count

    def perform_generation(self):
        # Elitism: Carry over the top individuals if enabled
        elites = sorted(self._population, key=lambda candidate: candidate.fitness, reverse=True)[:self.elitism_count] \
            if self.elitism_count else []

        new_population = elites.copy()
        while len(new_population) < self._population_size:
            parent1 = self._selector.select(self._population)
            parent2 = self._selector.select(self._population)
            offspring_args = self._crossover.crossover(parent1.arguments, parent2.arguments)
            offspring_args = self._mutator.mutate(offspring_args)
            offspring = self._solution_creator.create_solution(offspring_args)
            new_population.append(offspring)

        # Test freeing up
        del self._population
        self._population = new_population

    def best_solution(self) -> SolutionCandidate[A, R, SingleObjectiveFitness]:
        return max(self._population, key=lambda candidate: candidate.fitness)
