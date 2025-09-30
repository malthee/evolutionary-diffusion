import random
import numpy as np
from typing import List, Optional

from evolutionary.evolution_base import A, R, MultiObjectiveFitness, SolutionCreator, Selector, Mutator, \
    Crossover, SolutionCandidate
from evolutionary.algorithms.algorithm_base import Algorithm
from evolutionary.evaluators import MultiObjectiveEvaluator
from evolutionary.history import SolutionSourceMeta, SOLUTION_SOURCE_META_KEY
from evolutionary.statistics import SolutionHistoryKey, SolutionHistoryItem


class NSGAIIISolutionCandidate(SolutionCandidate[A, R, MultiObjectiveFitness]):
    """Solution candidate for NSGA-III with additional attributes for reference point association."""
    def __init__(self, arguments: A, result: R):
        super().__init__(arguments, result)
        self.domination_count = 0
        self.dominated_solutions = []
        self.rank = None
        self.reference_point_distance = float('inf')
        self.niche_count = 0


class NSGAIIITournamentSelector(Selector[MultiObjectiveFitness]):
    """
    Binary-Tournament selection for NSGA-III based on rank and reference point distance.
    """

    def select(self, candidates: List[NSGAIIISolutionCandidate]) -> NSGAIIISolutionCandidate:
        candidate_a, candidate_b = random.sample(candidates, 2)

        if candidate_a.rank == candidate_b.rank:  # If ranks are equal, decide by reference point distance
            return candidate_a if candidate_a.reference_point_distance < candidate_b.reference_point_distance else candidate_b
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


def _generate_reference_points(num_objectives: int, divisions: int) -> np.ndarray:
    """
    Generate uniformly distributed reference points on a unit hyperplane using Das-Dennis method.
    
    Args:
        num_objectives: Number of objectives
        divisions: Number of divisions along each objective
        
    Returns:
        Array of reference points, each normalized to sum to 1
    """
    if num_objectives == 1:
        return np.array([[1.0]])
    
    # Generate all combinations that sum to divisions
    # This uses a recursive approach to generate all partitions
    def generate_recursive(n_obj, n_div, current=[]):
        if n_obj == 1:
            yield current + [n_div]
        else:
            for i in range(n_div + 1):
                yield from generate_recursive(n_obj - 1, n_div - i, current + [i])
    
    ref_points = []
    for point in generate_recursive(num_objectives, divisions):
        normalized_point = np.array(point) / divisions
        ref_points.append(normalized_point)
    
    return np.array(ref_points)


class NSGA_III(Algorithm[A, R, MultiObjectiveFitness]):
    """
    NSGA-III (Non-dominated Sorting Genetic Algorithm III) implementation.
    
    NSGA-III is designed for many-objective optimization problems (more than 3 objectives).
    It uses reference points to maintain diversity and ensure good convergence.
    
    Key differences from NSGA-II:
    - Uses predefined reference points instead of crowding distance
    - Better suited for many-objective problems
    - Uses normalization and association with reference points
    """
    
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
                 # Number of divisions for reference point generation (auto-calculated if None)
                 reference_point_divisions: Optional[int] = None,
                 post_evaluation_callback: Optional[Algorithm.GenerationCallback] = None,
                 # Called after fronts are sorted, can access fronts through self.fronts
                 post_non_dominated_sort_callback: Optional[Algorithm.GenerationCallback] = None,
                 ident: Optional[int] = None):
        super().__init__(
            num_generations=num_generations,
            population_size=population_size,
            solution_creator=solution_creator,
            evaluator=evaluator,
            initial_arguments=initial_arguments,
            post_evaluation_callback=post_evaluation_callback,
            ident=ident
        )
        self._selector = selector
        self._mutator = mutator
        self._mutation_rate = mutation_rate
        self._crossover = crossover
        self._crossover_rate = crossover_rate
        self._elitism_count = elitism_count
        self._reference_point_divisions = reference_point_divisions
        self._post_non_dominated_sort_callback = post_non_dominated_sort_callback
        self._population: List[NSGAIIISolutionCandidate] = []
        self._fronts = [[]]
        self._reference_points = None
        self._cached_best_solution = None

    def _initialize_reference_points(self, num_objectives: int):
        """Initialize reference points based on number of objectives."""
        if self._reference_points is None:
            if self._reference_point_divisions is None:
                # Auto-calculate divisions based on population size and objectives
                # Using a heuristic: start with a reasonable number and adjust
                if num_objectives <= 3:
                    divisions = 12
                elif num_objectives <= 5:
                    divisions = 6
                elif num_objectives <= 8:
                    divisions = 4
                else:
                    divisions = 3
            else:
                divisions = self._reference_point_divisions
            
            self._reference_points = _generate_reference_points(num_objectives, divisions)

    def _fast_non_dominated_sort(self):
        """Perform fast non-dominated sorting to create Pareto fronts."""
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

    def _normalize_objectives(self, population: List[NSGAIIISolutionCandidate]) -> np.ndarray:
        """
        Normalize objectives to [0, 1] range for each objective.
        
        Args:
            population: List of solution candidates
            
        Returns:
            Normalized objective array (population_size x num_objectives)
        """
        if not population:
            return np.array([])
        
        # Extract fitness values
        fitness_array = np.array([ind.fitness for ind in population])
        
        # Find ideal point (minimum for each objective, but we maximize so it's maximum)
        ideal_point = np.max(fitness_array, axis=0)
        
        # Find nadir point (maximum for each objective, but we maximize so it's minimum)
        nadir_point = np.min(fitness_array, axis=0)
        
        # Normalize (handle division by zero)
        range_vals = ideal_point - nadir_point
        range_vals = np.where(range_vals == 0, 1, range_vals)
        
        # For maximization, we want to transform so that better solutions are closer to origin
        # Invert the fitness values
        normalized = (ideal_point - fitness_array) / range_vals
        
        return normalized

    def _associate_to_reference_points(self, population: List[NSGAIIISolutionCandidate]):
        """
        Associate each solution to the nearest reference point and calculate distances.
        
        Args:
            population: List of solution candidates to associate
        """
        if not population:
            return
        
        # Normalize objectives
        normalized_objectives = self._normalize_objectives(population)
        
        # Reset niche counts
        for point in population:
            point.niche_count = 0
        
        # For each solution, find closest reference point
        for i, individual in enumerate(population):
            if len(normalized_objectives) > 0:
                obj_normalized = normalized_objectives[i]
                
                # Calculate perpendicular distance to each reference point
                min_distance = float('inf')
                
                for ref_point in self._reference_points:
                    # Calculate perpendicular distance to the reference line
                    # Using the formula: d = ||f - (f·r/||r||²)r||
                    ref_norm = np.linalg.norm(ref_point)
                    if ref_norm > 0:
                        projection = np.dot(obj_normalized, ref_point) / (ref_norm ** 2) * ref_point
                        distance = np.linalg.norm(obj_normalized - projection)
                    else:
                        distance = np.linalg.norm(obj_normalized)
                    
                    if distance < min_distance:
                        min_distance = distance
                
                individual.reference_point_distance = min_distance

    def _niching_selection(self, front: List[NSGAIIISolutionCandidate], 
                          num_to_select: int) -> List[NSGAIIISolutionCandidate]:
        """
        Perform niching-based selection from the last front to fill the population.
        
        Args:
            front: The front from which to select solutions
            num_to_select: Number of solutions to select
            
        Returns:
            Selected solutions
        """
        if num_to_select >= len(front):
            return front
        
        selected = []
        remaining = front.copy()
        
        # Associate remaining solutions to reference points
        self._associate_to_reference_points(remaining)
        
        # Count niche associations for already selected individuals
        niche_counts = {}
        for ref_idx in range(len(self._reference_points)):
            niche_counts[ref_idx] = 0
        
        # Find niche associations for already selected population
        for ind in self._population[:self._population_size - num_to_select]:
            # Find closest reference point
            obj_normalized = self._normalize_objectives([ind])[0]
            min_dist = float('inf')
            closest_ref = 0
            
            for ref_idx, ref_point in enumerate(self._reference_points):
                ref_norm = np.linalg.norm(ref_point)
                if ref_norm > 0:
                    projection = np.dot(obj_normalized, ref_point) / (ref_norm ** 2) * ref_point
                    distance = np.linalg.norm(obj_normalized - projection)
                else:
                    distance = np.linalg.norm(obj_normalized)
                
                if distance < min_dist:
                    min_dist = distance
                    closest_ref = ref_idx
            
            niche_counts[closest_ref] += 1
        
        # Select individuals based on niche preservation
        while len(selected) < num_to_select and remaining:
            # Find reference point with minimum niche count
            min_niche_ref = min(niche_counts.items(), key=lambda x: x[1])[0]
            
            # Find individuals associated with this reference point
            candidates = []
            for ind in remaining:
                obj_normalized = self._normalize_objectives([ind])[0]
                min_dist = float('inf')
                closest_ref = 0
                
                for ref_idx, ref_point in enumerate(self._reference_points):
                    ref_norm = np.linalg.norm(ref_point)
                    if ref_norm > 0:
                        projection = np.dot(obj_normalized, ref_point) / (ref_norm ** 2) * ref_point
                        distance = np.linalg.norm(obj_normalized - projection)
                    else:
                        distance = np.linalg.norm(obj_normalized)
                    
                    if distance < min_dist:
                        min_dist = distance
                        closest_ref = ref_idx
                
                if closest_ref == min_niche_ref:
                    candidates.append((ind, min_dist))
            
            if candidates:
                # Select the one with minimum distance to reference point
                selected_ind = min(candidates, key=lambda x: x[1])[0]
                selected.append(selected_ind)
                remaining.remove(selected_ind)
                niche_counts[min_niche_ref] += 1
            else:
                # If no candidates for this reference point, select randomly
                if remaining:
                    selected_ind = remaining.pop(0)
                    selected.append(selected_ind)
        
        return selected

    def _sort_and_trim(self):
        """
        Sort population by rank and trim to population size using niching.
        """
        # Sort by rank
        self._population.sort(key=lambda x: x.rank)
        
        # If population is within size, we're done
        if len(self._population) <= self._population_size:
            return
        
        # Find the last front that fits completely
        new_population = []
        for front in self._fronts:
            if not front:
                continue
            if len(new_population) + len(front) <= self._population_size:
                new_population.extend(front)
            else:
                # Need to select from this front
                num_to_select = self._population_size - len(new_population)
                selected = self._niching_selection(front, num_to_select)
                new_population.extend(selected)
                break
        
        self._population = new_population

    def _crossover_and_mutation(self, generation: int):
        """Create offspring through crossover and mutation."""
        new_population = self._population[:self._elitism_count] if self._elitism_count else []

        while len(new_population) < self._population_size:
            parent1 = self._selector.select(self._population)
            parent1_source_meta: Optional[SolutionSourceMeta] = parent1.meta.get(SOLUTION_SOURCE_META_KEY)
            parent1_history_key = SolutionHistoryKey(
                index=parent1_source_meta.index if parent1_source_meta else self._population.index(parent1),
                generation=generation,
                ident=parent1_source_meta.ident if parent1_source_meta else self.ident
            )

            mutation_applied = False
            parent2_history_key = None

            if random.random() <= self._crossover_rate:
                parent2 = self._selector.select(self._population)
                parent2_source_meta: Optional[SolutionSourceMeta] = parent2.meta.get(SOLUTION_SOURCE_META_KEY)
                offspring_args = self._crossover.crossover(parent1.arguments, parent2.arguments)
                parent2_history_key = SolutionHistoryKey(
                    index=parent2_source_meta.index if parent2_source_meta else self._population.index(parent2),
                    generation=generation,
                    ident=parent2_source_meta.ident if parent2_source_meta else self.ident
                )
            else:
                offspring_args = parent1.arguments

            if random.random() <= self._mutation_rate:
                offspring_args = self._mutator.mutate(offspring_args)
                mutation_applied = True

            self._statistics.start_time_tracking('creation')
            offspring = NSGAIIISolutionCandidate(offspring_args,
                                                self._solution_creator.create_solution(offspring_args).result)
            self._statistics.stop_time_tracking('creation')

            history_item = SolutionHistoryItem(
                SolutionHistoryKey(index=len(new_population), generation=self.completed_generations, ident=self.ident),
                mutated=mutation_applied, parent_1=parent1_history_key, parent_2=parent2_history_key)

            new_population.append(offspring)
            self._statistics.add_history_item(history_item)

        del self._population
        self._population = new_population

    def perform_generation(self, generation: int):
        """Perform one generation of NSGA-III."""
        self._cached_best_solution = None
        
        # Initialize reference points if not already done
        if self._population and self._population[0].fitness is not None:
            num_objectives = len(self._population[0].fitness)
            self._initialize_reference_points(num_objectives)
        
        # Perform non-dominated sorting
        self._fast_non_dominated_sort()
        
        if self._post_non_dominated_sort_callback:
            self._post_non_dominated_sort_callback(generation, self)
        
        # Associate solutions to reference points
        self._associate_to_reference_points(self._population)
        
        # Sort and trim population
        self._sort_and_trim()
        
        # Create next generation
        self._crossover_and_mutation(generation)

    def best_solution(self) -> NSGAIIISolutionCandidate:
        """
        Return the best solution from the first Pareto front.
        Uses the solution with the highest sum of fitness values.
        """
        if self._cached_best_solution is not None:
            return self._cached_best_solution

        self._fast_non_dominated_sort()
        
        if self._post_non_dominated_sort_callback:
            self._post_non_dominated_sort_callback(self.num_generations - 1, self)

        if self._fronts[0]:
            # Return solution with highest fitness sum from first front
            self._cached_best_solution = max(self._fronts[0], key=lambda x: sum(x.fitness))
        else:
            # Fallback to best from entire population
            self._cached_best_solution = max(self._population, key=lambda x: sum(x.fitness))

        return self._cached_best_solution

    @property
    def fronts(self):
        """Return the Pareto fronts."""
        return self._fronts
    
    @property
    def reference_points(self):
        """Return the reference points used for niching."""
        return self._reference_points
