import random
from typing import List, Optional

from evolutionary.evolution_base import Algorithm, A, R, MultiObjectiveFitness, SolutionCreator, Selector, Mutator, \
    Crossover, MultiObjectiveEvaluator, SolutionCandidate


class NSGASolutionCandidate(SolutionCandidate[A, R, MultiObjectiveFitness]):
    def __init__(self, arguments: A, result: R):
        super().__init__(arguments, result)
        self.domination_count = 0
        self.dominated_solutions = []
        self.rank = None
        self.crowding_distance = 0


class NSGATournamentSelector(Selector[MultiObjectiveFitness]):
    """
    Binary-Tournament selection for NSGA-II based on rank and crowding distance.
    """

    def select(self, candidates: List[NSGASolutionCandidate]) -> NSGASolutionCandidate:
        candidate_a, candidate_b = random.sample(candidates, 2)

        if candidate_a.rank == candidate_b.rank:  # If ranks are equal, decide by crowding distance
            return candidate_a if candidate_a.crowding_distance > candidate_b.crowding_distance else candidate_b
        else:  # Else, decide by rank
            return candidate_a if candidate_a.rank < candidate_b.rank else candidate_b


def _dominates(individual1, individual2):
    """Check if individual1 dominates individual2."""
    better_in_one = False
    for i in range(len(individual1.fitness)):
        if individual1.fitness[i] < individual2.fitness[i]:
            return False
        elif individual1.fitness[i] > individual2.fitness[i]:
            better_in_one = True
    return better_in_one


class NSGA_II(Algorithm[A, R, MultiObjectiveFitness]):
    def __init__(self,
                 num_generations: int,
                 population_size: int,
                 solution_creator: SolutionCreator[A, R],
                 selector: Selector[MultiObjectiveFitness],
                 mutator: Mutator[A],
                 crossover: Crossover[A],
                 evaluator: MultiObjectiveEvaluator[R],
                 initial_arguments: List[A],
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 elitism_count: Optional[int] = None,
                 # Set to true if you are dealing with objectives with different scales
                 normalize_crowding_distance: bool = False,
                 post_evaluation_callback: Optional[Algorithm.GenerationCallback] = None,
                 # Called after fronts are sorted, can access fronts through self.fronts
                 post_non_dominated_sort_callback: Optional[Algorithm.GenerationCallback] = None):
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
        self._elitism_count = elitism_count
        # Normalize objective values, so they contribute equally to crowding distance calculation.
        self._normalize_crowding_distance = normalize_crowding_distance
        self._post_non_dominated_sort_callback = post_non_dominated_sort_callback
        self._population: List[NSGASolutionCandidate] = []  # Override the type to NSGA-II's solution candidate
        self._fronts = [[]]

    def _fast_non_dominated_sort(self):
        self._fronts = [[]]
        for p in self._population:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in self._population:
                if _dominates(p, q):
                    p.dominated_solutions.append(q)
                elif _dominates(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                self._fronts[0].append(p)
        i = 0
        while len(self._fronts[i]) > 0:
            next_front = []
            for p in self._fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            self._fronts.append(next_front)

    def _calculate_crowding_distance(self):
        for front in self._fronts:
            if not front:
                continue
            for p in front:
                p.crowding_distance = 0
            for i in range(len(front[0].fitness)):
                front.sort(key=lambda x: x.fitness[i])
                front[0].crowding_distance = front[-1].crowding_distance = float('inf')
                max_fitness = front[-1].fitness[i]
                min_fitness = front[0].fitness[i]

                if self._normalize_crowding_distance:
                    fitness_range = max_fitness - min_fitness + 1e-10  # Avoid division by zero
                else:
                    fitness_range = 1  # No normalization

                for j in range(1, len(front) - 1):
                    front[j].crowding_distance += ((front[j + 1].fitness[i] - front[j - 1].fitness[i])
                                                   / fitness_range)

    def _sort_and_trim(self):
        # Implement selection based on rank and crowding distance
        self._population.sort(key=lambda x: (x.rank, -x.crowding_distance))
        self._population = self._population[:self._population_size]  # Trim to population size

    def _crossover_and_mutation(self):
        new_population = self._population[:self._elitism_count] if self._elitism_count else []

        while len(new_population) < self._population_size:
            parent1 = self._selector.select(self._population)

            if random.random() <= self._crossover_rate:
                parent2 = self._selector.select(self._population)
                offspring_args = self._crossover.crossover(parent1.arguments, parent2.arguments)
            else:
                offspring_args = parent1.arguments

            if random.random() <= self._mutation_rate:
                offspring_args = self._mutator.mutate(offspring_args)

            offspring = NSGASolutionCandidate(offspring_args,
                                              self._solution_creator.create_solution(offspring_args).result)
            new_population.append(offspring)

        del self._population
        self._population = new_population

    def perform_generation(self, generation: int):
        self._fast_non_dominated_sort()
        if self._post_non_dominated_sort_callback:
            self._post_non_dominated_sort_callback(generation, self)
        self._calculate_crowding_distance()
        self._sort_and_trim()
        self._crossover_and_mutation()

    def best_solution(self) -> NSGASolutionCandidate:
        self._fast_non_dominated_sort()
        # Sort for last generation one more time
        if self._post_non_dominated_sort_callback:
            self._post_non_dominated_sort_callback(self.num_generations - 1, self)

        self._calculate_crowding_distance()

        # When normalization enabled, take best normalized solution
        if self._normalize_crowding_distance:
            # Fitness range (best, worst) for each objective
            normalization_params = [
                (self._fronts[0][0].fitness[i], self._fronts[0][-1].fitness[i])
                for i in range(len(self._fronts[0][0].fitness))
            ]

            # Calculate the sum of normalized fitness values for each solution in the first front
            normalized_fitness_sums = [
                sum(
                    (solution.fitness[i] - min_fitness) / (max_fitness - min_fitness)
                    if max_fitness > min_fitness else 0
                    for i, (min_fitness, max_fitness) in enumerate(normalization_params)
                )
                for solution in self._fronts[0]
            ]

            # Find the solution with the highest sum of normalized fitness values
            best_index = normalized_fitness_sums.index(max(normalized_fitness_sums))
            return self._fronts[0][best_index]
        else:
            # If not normalizing, simply take the solution with the highest sum of fitness values
            return max(self._fronts[0], key=lambda x: sum(x.fitness))

    @property
    def fronts(self):
        return self._fronts
