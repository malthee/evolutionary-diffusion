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


def _dominates(individual1, individual2):
    """Check if individual1 dominates individual2."""
    better_in_one = False
    for i in range(len(individual1.fitness)):
        if individual1.fitness[i] > individual2.fitness[i]:  # For maximization, use >
            return False
        elif individual1.fitness[i] < individual2.fitness[i]:  # For maximization, use <
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
                 normalize_crowding_distance: bool = True,
                 post_evaluation_callback: Optional[Algorithm.GenerationCallback] = None):
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
        # Normalize objective values so they contribute equally to crowding distance calculation.
        self._normalize_crowding_distance = normalize_crowding_distance
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
        new_population = []
        while len(new_population) < self._population_size:
            parent1 = self._selector.select(self._population)
            parent2 = self._selector.select(self._population)
            child_args = self._crossover.crossover(parent1.arguments, parent2.arguments)
            child_args = self._mutator.mutate(child_args)
            child = NSGASolutionCandidate(child_args, self._solution_creator.create_solution(child_args))
            new_population.append(child)
        del self._population
        self._population = new_population

    def _run_algorithm(self) -> SolutionCandidate[A, R, MultiObjectiveFitness]:
        self._create_initial_population()
        for generation in range(self.num_generations):
            print(f"Generation {generation} started.")
            self._evaluate_population(generation)

            # If this is the last generation, finish here
            if generation == self.num_generations - 1:
                break

            self._fast_non_dominated_sort()
            self._calculate_crowding_distance()
            self._sort_and_trim()
            self._crossover_and_mutation()

        # Take the solution with the highest crowding distance in the first front
        return max(self._fronts[0], key=lambda x: x.crowding_distance)
