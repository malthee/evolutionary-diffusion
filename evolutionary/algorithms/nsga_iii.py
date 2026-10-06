import random
import warnings
from math import comb
from typing import List, Optional

import numpy as np
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival, associate_to_niches
from pymoo.core.population import Population
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory

from evolutionary.algorithms._multi_objective import _MultiObjectiveAlgorithm
from evolutionary.algorithms.algorithm_base import Algorithm
from evolutionary.evolution_base import (
    A,
    Crossover,
    Evaluator,
    MultiObjectiveFitness,
    Mutator,
    R,
    Selector,
    SolutionCandidate,
    SolutionCreator,
)


class _NoConstraintsProblem:
    def has_constraints(self) -> bool:
        return False


_NO_CONSTR_PROBLEM = _NoConstraintsProblem()


class NSGAIIISolutionCandidate(SolutionCandidate[A, R, MultiObjectiveFitness]):
    def __init__(self, arguments: A, result: R):
        super().__init__(arguments, result)
        self.rank: Optional[int] = None
        self.niche: Optional[int] = None
        self.dist_to_niche: Optional[float] = None


class NSGAIIIRandomSelector(Selector[MultiObjectiveFitness]):
    """Original unconstrained NSGA-III uses random mating."""

    def select(
        self, candidates: List[NSGAIIISolutionCandidate]
    ) -> NSGAIIISolutionCandidate:
        return random.choice(candidates)


class NSGAIIIBinaryRankSelector(Selector[MultiObjectiveFitness]):
    """Optional rank tournament, not U-NSGA-III. Equal ranks tie randomly."""

    def select(
        self, candidates: List[NSGAIIISolutionCandidate]
    ) -> NSGAIIISolutionCandidate:
        first, second = random.choices(candidates, k=2)
        if first.rank is None or second.rank is None:
            raise RuntimeError("Rank tournament requires evaluated ranks.")
        if first.rank != second.rank:
            return first if first.rank < second.rank else second
        return random.choice((first, second))


class UNSGAIIITournamentSelector(Selector[MultiObjectiveFitness]):
    """Seada and Deb's niched comparison (2016, Algorithm 2)."""

    def select(
        self, candidates: List[NSGAIIISolutionCandidate]
    ) -> NSGAIIISolutionCandidate:
        first, second = random.choices(candidates, k=2)
        return self._compare(first, second)

    @staticmethod
    def _compare(first, second):
        if any(
            c.rank is None or c.niche is None or c.dist_to_niche is None
            for c in (first, second)
        ):
            raise RuntimeError("Unified tournament requires rank, niche and distance.")
        if first.niche == second.niche:
            if first.rank != second.rank:
                return first if first.rank < second.rank else second
            return first if first.dist_to_niche < second.dist_to_niche else second
        return random.choice((first, second))

    def mating_pool(self, candidates):
        """Paper §3: adjacent tournaments, then repeat on a shuffled population."""
        if len(candidates) % 4:
            raise ValueError(
                "U-NSGA-III mating requires a population divisible by four."
            )
        ordered = list(candidates)
        pool = [self._compare(a, b) for a, b in zip(ordered[::2], ordered[1::2])]
        random.shuffle(ordered)
        pool.extend(self._compare(a, b) for a, b in zip(ordered[::2], ordered[1::2]))
        return pool


class NSGA_III(_MultiObjectiveAlgorithm[A, R]):
    """Reference-direction union survival; objectives are maximized by the framework.

    Pymoo supplies hyperplane normalization and niching. Only its input is negated.
    Custom mating/operators are extensions to the original unconstrained algorithm.
    """

    candidate_type = NSGAIIISolutionCandidate
    default_selector = NSGAIIIRandomSelector

    def __init__(
        self,
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
        ref_dirs: Optional[np.ndarray] = None,
        ref_dirs_method: str = "das-dennis",
        n_partitions: Optional[int] = None,
        post_evaluation_callback: Optional[Algorithm.GenerationCallback] = None,
        post_non_dominated_sort_callback: Optional[Algorithm.GenerationCallback] = None,
        ident: Optional[int] = None,
        *,
        seed: Optional[int] = None,
    ):
        super().__init__(
            num_generations,
            population_size,
            solution_creator,
            evaluator,
            initial_arguments,
            post_evaluation_callback,
            ident,
        )
        self._configure_variation(
            selector if selector is not None else self.default_selector(),
            mutator,
            crossover,
            mutation_rate,
            crossover_rate,
            post_non_dominated_sort_callback,
        )
        if (
            isinstance(self._selector, UNSGAIIITournamentSelector)
            and population_size % 4
        ):
            raise ValueError("U-NSGA-III population_size must be divisible by four.")
        if n_partitions is not None and (
            isinstance(n_partitions, bool)
            or not isinstance(n_partitions, (int, np.integer))
            or n_partitions < 1
        ):
            raise ValueError("n_partitions must be a positive integer or None.")
        if seed is not None and (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer or None.")
        self._seed = int(seed) if seed is not None else None
        self._ref_dirs_method, self._n_partitions = ref_dirs_method, n_partitions
        self._provided_ref_dirs = (
            self._validate_directions(ref_dirs) if ref_dirs is not None else None
        )
        self._ref_dirs = None
        self._survival = None
        self._random_state = None

    @staticmethod
    def _validate_directions(directions):
        directions = np.array(directions, dtype=float, copy=True)
        if (
            directions.ndim != 2
            or not all(directions.shape)
            or not np.isfinite(directions).all()
            or (directions < 0).any()
            or (directions.max(axis=1) <= 0).any()
        ):
            raise ValueError(
                "Reference directions must be finite, nonnegative, nonzero rows."
            )
        # Scale first so finite large coordinates cannot overflow the row sum.
        directions /= directions.max(axis=1, keepdims=True)
        directions /= directions.sum(axis=1, keepdims=True)
        if len(np.unique(directions, axis=0)) != len(directions):
            raise ValueError("Reference directions must be distinct rays.")
        return directions

    @property
    def ref_dirs(self):
        return None if self._ref_dirs is None else self._ref_dirs.copy()

    def create_initial_population(self):
        # Normalization must remember previous generations, never previous runs.
        self._survival = None
        self._ref_dirs = None
        survival_seed = (
            self._seed if self._seed is not None else int(np.random.randint(2**32))
        )
        self._random_state = np.random.default_rng(survival_seed)
        super().create_initial_population()

    def _ensure_survival(self):
        if self._survival is not None:
            return
        objectives = self._objective_count
        if self._provided_ref_dirs is not None:
            directions = self._provided_ref_dirs
        elif objectives == 1:
            directions = np.ones((1, 1))
        elif self._ref_dirs_method == "energy":
            directions = RieszEnergyReferenceDirectionFactory(
                objectives,
                n_points=self.population_size,
            ).do(
                seed=int(self._random_state.integers(2**32)),
            )
        else:
            partitions = self._n_partitions
            if partitions is None:
                partitions = 1
                while (
                    comb(objectives + partitions, partitions + 1)
                    <= self.population_size
                ):
                    partitions += 1
            directions = get_reference_directions(
                self._ref_dirs_method, objectives, n_partitions=partitions
            )
        directions = self._validate_directions(directions)
        if directions.shape[1] != objectives:
            raise ValueError(
                "Reference-direction dimension must match the objective count."
            )
        if len(directions) > self.population_size:
            if isinstance(self._selector, UNSGAIIITournamentSelector):
                raise ValueError(
                    "U-NSGA-III requires population_size >= number of directions."
                )
            warnings.warn(
                "More reference directions than individuals: some niches must remain empty.",
                UserWarning,
                stacklevel=2,
            )
        self._ref_dirs = directions
        self._survival = ReferenceDirectionSurvival(directions)

    @staticmethod
    def _to_population(candidates):
        fitness = np.asarray([c.fitness for c in candidates], dtype=float)
        if fitness.ndim != 2 or not fitness.shape[1] or not np.isfinite(fitness).all():
            raise ValueError(
                "Survival requires finite, equally sized evaluated objective vectors."
            )
        return Population.new("F", -fitness)

    def _prepare_population(self):
        self._ensure_survival()
        self._fronts = self._cache_fronts(self.population)
        fitness = self._to_population(self.population).get("F")
        normalization = self._survival.norm
        if normalization.nadir_point is None:
            first_front = [self.population.index(c) for c in self._fronts[0]]
            normalization.update(fitness, nds=first_front)
        niches, distances, _ = associate_to_niches(
            fitness,
            self._ref_dirs,
            normalization.ideal_point,
            normalization.nadir_point,
        )
        for candidate, niche, distance in zip(self.population, niches, distances):
            candidate.niche, candidate.dist_to_niche = int(niche), float(distance)

    def _select_survivors(self, candidates):
        self._ensure_survival()
        pop = self._to_population(candidates)
        # Pass the generator explicitly: pymoo's implicit default_rng ignores np.random.seed.
        return self._survival.do(
            _NO_CONSTR_PROBLEM,
            pop,
            n_survive=self.population_size,
            return_indices=True,
            random_state=self._random_state,
        )


class U_NSGA_III(NSGA_III[A, R]):
    """NSGA-III survival with within-niche rank/distance mating (Seada and Deb, 2016)."""

    default_selector = UNSGAIIITournamentSelector

    def _offspring_parents(self, parents):
        if not isinstance(self._selector, UNSGAIIITournamentSelector):
            yield from super()._offspring_parents(parents)
            return
        pool = self._selector.mating_pool(parents)
        for first, second in zip(pool[::2], pool[1::2]):
            # Two sibling events preserve the existing single-child operator interface.
            yield first, second
            yield second, first
