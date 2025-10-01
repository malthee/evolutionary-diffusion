import random
from typing import  Generic, List, Optional

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

from evolutionary.algorithms.algorithm_base import Algorithm
from evolutionary.evolution_base import (
    A, R, MultiObjectiveFitness,
    SolutionCreator, Evaluator, SolutionCandidate,
    Selector, Mutator, Crossover
)
from evolutionary.history import (
    SolutionHistoryKey, SolutionHistoryItem, SolutionSourceMeta, SOLUTION_SOURCE_META_KEY
)

class _NoConstraintsProblem:
    def has_constraints(self) -> bool:
        return False

_NO_CONSTR_PROBLEM = _NoConstraintsProblem()

# ============== SolutionCandidate specialized for NSGA-III ===================

class NSGAIIISolutionCandidate(SolutionCandidate[A, R, MultiObjectiveFitness]):
    """Augments SolutionCandidate with rank/cache used by NSGA-III."""
    def __init__(self, arguments: A, result: R):
        super().__init__(arguments, result)
        self.rank: Optional[int] = None  # Pareto rank (0 is best/front-0)


# ================================ Selectors ==================================

class NSGAIIIRandomSelector(Selector[MultiObjectiveFitness]):
    """Random mating (vanilla NSGA-III uses random parent selection)."""
    def select(self, candidates: List[NSGAIIISolutionCandidate]
               ) -> NSGAIIISolutionCandidate:
        return candidates[random.randrange(len(candidates))]


class NSGAIIIBinaryRankSelector(Selector[MultiObjectiveFitness]):
    """Binary tournament on precomputed rank (lower is better). Tie -> larger sum(f)."""
    def select(self, candidates: List[NSGAIIISolutionCandidate]
               ) -> NSGAIIISolutionCandidate:
        i, j = random.randrange(len(candidates)), random.randrange(len(candidates))
        ci = candidates[i]; cj = candidates[j]
        ri = ci.rank; rj = cj.rank
        if ri is None or rj is None:
            raise RuntimeError("NSGAIIIBinaryRankSelector requires candidates with 'rank' set by the algorithm.")
        if ri < rj: return ci
        if rj < ri: return cj
        # tie-breaker: maximizing sum of objectives
        return ci if float(np.sum(ci.fitness)) >= float(np.sum(cj.fitness)) else cj


# ================================ NSGA-III ===================================

class NSGA_III(Algorithm[A, R, MultiObjectiveFitness], Generic[A, R]):
    """
    Clean NSGA-III using pymoo's ReferenceDirectionSurvival (pymoo 0.6.1.5).

    - Evaluators maximize; we negate F only when calling pymoo (which minimizes).
    - One child is produced per loop; creation/evaluation stages tracked like NSGA-II.
    - Fronts computed once per phase, cached in `self._fronts`, and ranks stamped on candidates.
    - No explicit extra elitism (μ+λ survival is already elitist).
    """

    GenerationCallback = Algorithm.GenerationCallback

    def __init__(self,
                 num_generations: int,
                 population_size: int,
                 solution_creator: SolutionCreator[A, R],
                 evaluator: Evaluator[R, MultiObjectiveFitness],
                 initial_arguments: List[A],
                 selector: Optional[Selector[MultiObjectiveFitness]] = None,
                 mutator: Optional[Mutator[A]] = None,
                 crossover: Optional[Crossover[A]] = None,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 ref_dirs: Optional[np.ndarray] = None,          # fixed reference directions on the simplex; None -> generate
                 ref_dirs_method: str = "das-dennis",            # generator name for ref dirs (e.g., "das-dennis")
                 n_partitions: int = 12,                         # granularity of ref dirs for das-dennis; pick so #dirs ≥ pop size
                 post_evaluation_callback: Optional[GenerationCallback] = None,
                 post_non_dominated_sort_callback: Optional[GenerationCallback] = None,
                 ident: Optional[int] = None):
        super().__init__(num_generations=num_generations,
                         population_size=population_size,
                         solution_creator=solution_creator,
                         evaluator=evaluator,
                         initial_arguments=initial_arguments,
                         post_evaluation_callback=post_evaluation_callback,
                         ident=ident)
        self._selector = selector or NSGAIIIRandomSelector()
        self._mutator = mutator
        self._crossover = crossover
        self._mutation_rate = float(mutation_rate)
        self._crossover_rate = float(crossover_rate)

        self._ref_dirs = np.asarray(ref_dirs, dtype=float) if ref_dirs is not None else None
        self._ref_dirs_method = ref_dirs_method
        self._n_partitions = int(n_partitions)

        self._survival: Optional[ReferenceDirectionSurvival] = None
        self._fronts: List[List[int]] = []
        self._post_nd_callback = post_non_dominated_sort_callback
        self._cached_best: Optional[NSGAIIISolutionCandidate] = None
        self._population: List[NSGAIIISolutionCandidate] = []  # refine type

    # ------------------------ Public accessors --------------------------------

    @property
    def fronts(self) -> List[List[NSGAIIISolutionCandidate]]:

        front_candidates: List[List[NSGAIIISolutionCandidate]] = [
            [self.population[i] for i in fr] for fr in self._fronts
        ]
        return front_candidates

    # ------------------------ Internal helpers --------------------------------

    def _ensure_survival(self) -> None:
        if self._survival is not None:
            return
        m = len(self.population[0].fitness)  # parents are evaluated by base before perform_generation
        if self._ref_dirs is None:
            self._ref_dirs = get_reference_directions(self._ref_dirs_method, m, n_partitions=self._n_partitions)
        self._ref_dirs = np.asarray(self._ref_dirs, dtype=float)
        self._survival = ReferenceDirectionSurvival(self._ref_dirs)

    def _compute_and_cache_fronts(self, candidates: List[NSGAIIISolutionCandidate], generation: int) -> None:
        # Use pymoo for non-dominated sorting; stamp ranks on candidates
        F = -np.array([np.asarray(c.fitness, dtype=float) for c in candidates])  # negate: maximize -> minimize
        fronts = NonDominatedSorting().do(F, n_stop_if_ranked=len(candidates))
        self._fronts = [list(map(int, fr)) for fr in fronts]

        ranks = np.empty(len(candidates), dtype=int)
        for r, fr in enumerate(self._fronts):
            ranks[np.asarray(fr, dtype=int)] = r
        for i, c in enumerate(candidates):
            c.rank = int(ranks[i])
        if self._post_nd_callback:
            self._statistics.start_time_tracking('post_evaluation')
            self._post_nd_callback(generation, self)
            self._statistics.start_time_tracking('post_evaluation')

    @staticmethod
    def _to_population(cands: List[NSGAIIISolutionCandidate]) -> Population:
        # Create a pymoo Population
        if any(c.fitness is None for c in cands):
            raise ValueError("All candidates must be evaluated before NSGA-III survival.")
        # pymoo minimizes -> negate once here
        F = -np.asarray([np.asarray(c.fitness, dtype=float) for c in cands], dtype=float)  # shape (N, M)
        pop = Population.new("F", F)
        return pop

    # ------------------------- Algorithm methods ------------------------------

    def perform_generation(self, generation: int) -> None:
        parents = self.population  # List[NSGAIIISolutionCandidate]

        # 1) Cache fronts/ranks for current parents (for selectors & callback)
        self._compute_and_cache_fronts(parents, generation)

        # 2) Create λ = N offspring, one child per loop, with history + timings
        offspring: List[NSGAIIISolutionCandidate] = []
        while len(offspring) < self.population_size:
            parent1 = self._selector.select(parents)
            p1_src: Optional[SolutionSourceMeta] = parent1.meta.get(SOLUTION_SOURCE_META_KEY)
            parent1_key = SolutionHistoryKey(
                index=p1_src.index if p1_src else parents.index(parent1),
                generation=generation,
                ident=p1_src.ident if p1_src else self.ident
            )

            parent2_key: Optional[SolutionHistoryKey] = None
            mutated = False

            # crossover (optional)
            if self._crossover and random.random() <= self._crossover_rate:
                parent2 = self._selector.select(parents)
                p2_src: Optional[SolutionSourceMeta] = parent2.meta.get(SOLUTION_SOURCE_META_KEY)
                args = self._crossover.crossover(parent1.arguments, parent2.arguments)
                parent2_key = SolutionHistoryKey(
                    index=p2_src.index if p2_src else parents.index(parent2),
                    generation=generation,
                    ident=p2_src.ident if p2_src else self.ident
                )
            else:
                args = parent1.arguments

            # mutation (optional)
            if self._mutator and random.random() <= self._mutation_rate:
                args = self._mutator.mutate(args)
                mutated = True

            # creation (your creator returns a generic SolutionCandidate -> wrap into NSGAIIISolutionCandidate)
            self._statistics.start_time_tracking('creation')
            created = self._solution_creator.create_solution(args)
            child = NSGAIIISolutionCandidate(args, created.result)
            self._statistics.stop_time_tracking('creation')

            # evaluation (offspring must be evaluated in this generation)
            self._statistics.start_time_tracking('evaluation')
            child.fitness = self._evaluator.evaluate(child.result) if created.fitness is None else created.fitness
            self._statistics.stop_time_tracking('evaluation')

            # history
            hist_key = SolutionHistoryKey(index=len(offspring), generation=self.completed_generations, ident=self.ident)
            self._statistics.add_history_item(SolutionHistoryItem(
                hist_key, mutated=mutated, parent_1=parent1_key, parent_2=parent2_key
            ))

            offspring.append(child)

        # 3) μ+λ environmental selection via NSGA-III
        self._ensure_survival()
        combined: List[NSGAIIISolutionCandidate] = parents + offspring
        pop = self._to_population(combined)
        keep = self._survival.do(
            problem=_NO_CONSTR_PROBLEM,  # dummy problem here as we do not have constraints
            pop=pop,
            n_survive=self.population_size,
            return_indices=True
        )
        self._population = [combined[i] for i in keep]

        # 5) Update fronts one more time and also call post ND callback on last gen
        if generation == self.num_generations - 2:
            self._compute_and_cache_fronts(self._population, generation + 1)

        # 6) Invalidate cached best (recomputed lazily on demand once)
        self._cached_best = None

    def best_solution(self) -> NSGAIIISolutionCandidate:
        if self._cached_best is None:
            if not self._fronts or not self._fronts[0]:
                self._compute_and_cache_fronts(self.population, generation=-1)
            first = [self.population[i] for i in self._fronts[0]]
            self._cached_best = max(first, key=lambda c: float(np.sum(c.fitness)))
        return self._cached_best