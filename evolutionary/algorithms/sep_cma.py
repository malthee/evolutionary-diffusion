"""Bounded separable CMA-ES with a private RNG and resumable ask/tell state."""

import hashlib
import importlib.metadata
import math
import pickle

import numpy as np

from evolutionary.algorithms.algorithm_base import Algorithm
from evolutionary.evolution_base import A, R, SingleObjectiveFitness
from evolutionary.history import SolutionHistoryItem, SolutionHistoryKey

CMA_VERSION = "4.4.4"


class SepCMAOptimizer:
    """One ask/tell trajectory; pending proposals survive an interrupted scorer."""

    def __init__(self, mean, *, seed=42, budget=6400, population=64, sigma=0.15):
        import cma

        if importlib.metadata.version("cma") != CMA_VERSION:
            raise ValueError("Require the frozen pycma version")
        if (
            type(budget) is not int
            or type(population) is not int
            or population < 4
            or budget < population
            or budget % population
        ):
            raise ValueError("Budget must cover a whole number of populations")
        if (
            type(seed) is not int
            or not 0 <= seed < 2**32
            or not math.isfinite(sigma)
            or sigma <= 0
        ):
            raise ValueError("Invalid optimizer seed or sigma")
        mean = np.asarray(mean, dtype=float)
        if (
            mean.ndim != 1
            or len(mean) < 2
            or not np.isfinite(mean).all()
            or (mean < 0).any()
            or (mean > 1).any()
        ):
            raise ValueError("Require an initial mean within normalized bounds")
        # The bound method and its RandomState are serialized together. Never
        # let generator noise or unrelated numpy calls advance optimizer RNG.
        rng = np.random.RandomState(seed)
        self.strategy = cma.CMAEvolutionStrategy(
            mean,
            sigma,
            {
                "CMA_diagonal": True,
                "CMA_diagonal_decoding": 0,
                "CMA_active": False,
                "bounds": [0, 1],
                "popsize": population,
                "seed": np.nan,
                "randn": rng.randn,
                "eval_final_mean": False,
                "signals_filename": "",
                "verbose": -9,
                "verb_disp": 0,
                "verb_log": 0,
            },
        )
        self.budget, self.population, self.completed = budget, population, 0
        self.pending = None

    def ask(self):
        if self.pending is None:
            if self.completed >= self.budget:
                raise StopIteration("The fresh candidate budget is exhausted")
            self.pending = np.asarray(self.strategy.ask(), dtype=np.float64)
            if (
                not np.isfinite(self.pending).all()
                or (self.pending < 0).any()
                or (self.pending > 1).any()
            ):
                raise ValueError("Native boundary transform returned invalid proposals")
        return self.pending.copy()

    def tell(self, objective):
        """Commit a complete population of minimization objectives in ask order."""
        values = np.asarray(objective, dtype=float)
        if (
            self.pending is None
            or values.shape != (self.population,)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Require finite scores for the exact pending population")
        self.strategy.tell(self.pending, values)
        self.completed += self.population
        self.pending = None

    def state(self):
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def restore_owned_state(cls, content, expected_sha256):
        """Load only our own checkpoint bytes after transfer/integrity validation."""
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise ValueError("Optimizer checkpoint checksum differs")
        if importlib.metadata.version("cma") != CMA_VERSION:
            raise ValueError("Optimizer checkpoint dependency version differs")
        state = pickle.loads(content)
        if (
            not isinstance(state, cls)
            or state.strategy.opts["CMA_diagonal"] is not True
            or not 0 <= state.completed <= state.budget
        ):
            raise ValueError("Unexpected optimizer checkpoint configuration")
        return state


class SepCMAES(Algorithm[A, R, SingleObjectiveFitness]):
    """Adapt bounded pycma to creators, maximization fitness and generation callbacks.

    The initial mean and proposals use normalized coordinates in [0, 1]. Supply
    ``decode`` to turn each vector into domain arguments, such as embeddings.
    Each generation evaluates one fresh population, including generation zero.
    """

    def __init__(
        self,
        num_generations,
        population_size,
        solution_creator,
        evaluator,
        initial_mean,
        *,
        decode=None,
        seed=42,
        sigma=0.15,
        post_evaluation_callback=None,
        ident=None,
    ):
        if type(num_generations) is not int or num_generations < 1:
            raise ValueError("Number of generations must be a positive integer")
        super().__init__(
            num_generations,
            population_size,
            solution_creator,
            evaluator,
            [initial_mean],
            post_evaluation_callback,
            ident,
        )
        self._mean = np.asarray(initial_mean, dtype=float).copy()
        self._decode = decode
        self._seed, self._sigma = seed, sigma
        self.optimizer = self._new_optimizer()
        self._best = None

    def _new_optimizer(self):
        return SepCMAOptimizer(
            self._mean,
            seed=self._seed,
            sigma=self._sigma,
            budget=self.num_generations * self.population_size,
            population=self.population_size,
        )

    @property
    def evaluation_count(self):
        return self.optimizer.completed

    def _create_population(self, generation):
        self.statistics.start_time_tracking("creation")
        proposals = self.optimizer.ask()
        arguments = [
            self._decode(vector) if self._decode else vector.copy()
            for vector in proposals
        ]
        create_many = getattr(self._solution_creator, "create_solutions", None)
        self._population = (
            list(create_many(arguments))
            if create_many
            else [self._solution_creator.create_solution(arg) for arg in arguments]
        )
        if len(self.population) != self.population_size:
            raise ValueError("Creator must return one candidate per proposal in order")
        for index in range(self.population_size):
            self.statistics.add_history_item(
                SolutionHistoryItem(
                    SolutionHistoryKey(index, generation, self.ident),
                    mutated=generation > 0,
                    creation_kind="initial" if generation == 0 else "offspring",
                )
            )
        self.statistics.stop_time_tracking("creation")

    def create_initial_population(self):
        self.optimizer = self._new_optimizer()
        self._best = None
        self._create_population(0)

    def evaluate_population(self, generation):
        self.statistics.start_time_tracking("evaluation")
        evaluate_many = getattr(self._evaluator, "evaluate_batch", None)
        results = [candidate.result for candidate in self.population]
        scores = np.asarray(
            list(evaluate_many(results))
            if evaluate_many
            else [self._evaluator.evaluate(result) for result in results],
            dtype=float,
        )
        # The framework maximizes fitness; native pycma minimizes objectives.
        # Validate/commit before publishing fitness or invoking callbacks.
        self.optimizer.tell(-scores)
        for candidate, score in zip(self.population, scores):
            candidate.fitness = float(score)
        self.statistics.stop_time_tracking("evaluation")
        best = max(self.population, key=lambda candidate: candidate.fitness)
        if self._best is None or best.fitness > self._best.fitness:
            self._best = best
        if self._post_evaluation_callback:
            self.statistics.start_time_tracking("post_evaluation")
            self._post_evaluation_callback(generation, self)
            self.statistics.stop_time_tracking("post_evaluation")
        self.statistics.update_fitness(self.population)

    def perform_generation(self, generation):
        self._create_population(generation + 1)

    def best_solution(self):
        if self._best is None:
            raise ValueError("No population has been evaluated")
        return self._best
