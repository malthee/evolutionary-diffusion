import math
import random
from dataclasses import dataclass, replace
from time import perf_counter
from typing import List, Optional

from tqdm import tqdm

from evolutionary.algorithms.algorithm_base import Algorithm
from evolutionary.evolution_base import (
    A,
    Crossover,
    Mutator,
    R,
    Selector,
    SingleObjectiveEvaluator,
    SingleObjectiveFitness,
    SolutionCandidate,
    SolutionCreator,
)
from evolutionary.history import (
    SOLUTION_SOURCE_META_KEY,
    SolutionHistoryItem,
    SolutionHistoryKey,
)
from evolutionary.statistics import (
    EvaluationRecord,
    GenerationSummary,
    StatisticsTracker,
)


@dataclass(frozen=True)
class OffspringSelectionConfig:
    success_ratio: float = 1.0
    comparison_factor: float = 1.0
    max_selection_pressure: float = 10.0

    def __post_init__(self):
        for name in ("success_ratio", "comparison_factor", "max_selection_pressure"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0 <= self.success_ratio <= 1 or not 0 <= self.comparison_factor <= 1:
            raise ValueError("success_ratio and comparison_factor must be in [0, 1]")
        if self.max_selection_pressure < 1:
            raise ValueError("max_selection_pressure must be at least 1")

    def threshold(self, parent_fitness):
        lower, upper = float(min(parent_fitness)), float(max(parent_fitness))
        return float(lower + self.comparison_factor * (upper - lower))


class GeneticAlgorithm(Algorithm[A, R, SingleObjectiveFitness]):
    def __init__(
        self,
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
        strict_osga: bool = False,
        ident: Optional[int] = None,
        *,
        offspring_selection: Optional[OffspringSelectionConfig] = None,
        max_evaluations: Optional[int] = None,
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
        if strict_osga and offspring_selection is not None:
            raise ValueError("strict_osga cannot be combined with offspring_selection")
        if offspring_selection is not None and not isinstance(
            offspring_selection, OffspringSelectionConfig
        ):
            raise TypeError(
                "offspring_selection must be an OffspringSelectionConfig or None"
            )
        for name, value in [
            ("mutation_rate", mutation_rate),
            ("crossover_rate", crossover_rate),
        ]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if elitism_count is not None and (
            isinstance(elitism_count, bool)
            or not isinstance(elitism_count, int)
            or not 0 <= elitism_count <= population_size
        ):
            raise ValueError("elitism_count must be an integer in [0, population_size]")
        if max_evaluations is not None and (
            isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, int)
            or max_evaluations < population_size
        ):
            raise ValueError(
                "max_evaluations must be an integer sufficient for the initial population"
            )
        self._selector, self._mutator, self._crossover = selector, mutator, crossover
        self._mutation_rate, self._crossover_rate = mutation_rate, crossover_rate
        self.elitism_count = elitism_count
        # Legacy strict mode now means strictly better than the better parent, with bounded retries.
        self.offspring_selection = (
            OffspringSelectionConfig() if strict_osga else offspring_selection
        )
        self.max_evaluations = max_evaluations
        self.evaluation_count = 0
        self.termination_reason = None
        self._best_evaluated = None

    def _has_budget(self):
        return (
            self.max_evaluations is None or self.evaluation_count < self.max_evaluations
        )

    def _evaluate(
        self,
        candidate,
        generation,
        kind,
        parents=(),
        crossover=None,
        mutation=None,
        *,
        fresh=False,
    ):
        if candidate.fitness is not None and not fresh:
            candidate.fitness = float(candidate.fitness)
            return None  # Cached fitness, notably elite carryover, costs no evaluation.
        if not self._has_budget():
            raise RuntimeError("Evaluation budget exhausted")
        candidate.fitness = float(self._evaluator.evaluate(candidate.result))
        self.evaluation_count += 1
        fitness = float(candidate.fitness)
        if not math.isfinite(fitness):
            raise ValueError("GA fitness must be finite")
        config = self.offspring_selection if parents else None
        threshold = (
            config.threshold([fitness for _, fitness in parents]) if config else None
        )
        record = EvaluationRecord(
            self.evaluation_count,
            generation,
            self.ident,
            kind,
            tuple(key for key, _ in parents),
            tuple(float(fitness) for _, fitness in parents),
            crossover,
            mutation,
            fitness,
            float(config.comparison_factor) if config else None,
            threshold,
            fitness > threshold if config else None,
        )
        self.statistics.evaluation_records.append(record)
        if (
            self._best_evaluated is None
            or candidate.fitness > self._best_evaluated.fitness
        ):
            self._best_evaluated = candidate
        return record

    def create_initial_population(self):
        self.evaluation_count = 0
        self.termination_reason = None
        self._best_evaluated = None
        super().create_initial_population()
        for key, item in list(self.statistics.solution_history.items()):
            self.statistics.solution_history[key] = replace(
                item, creation_kind="initial"
            )

    def evaluate_population(self, generation):
        # Offspring were evaluated during construction. Initialization is evaluated here.
        start = perf_counter()
        initial_records = []
        for index, candidate in enumerate(self.population):
            record = self._evaluate(candidate, generation, "initial")
            if record:
                key = SolutionHistoryKey(index, generation, self.ident)
                record.survivor_key = key
                item = self.statistics.solution_history[key]
                self.statistics.solution_history[key] = replace(
                    item, evaluation_id=record.evaluation_id
                )
                initial_records.append(record)
            if (
                self._best_evaluated is None
                or candidate.fitness > self._best_evaluated.fitness
            ):
                self._best_evaluated = candidate
        elapsed = perf_counter() - start
        if generation == 0:
            self.statistics._custom_time_tracking("evaluation", elapsed)
            self.statistics.generation_summaries.append(
                GenerationSummary(
                    generation,
                    self.ident,
                    len(initial_records),
                    0,
                    0,
                    0,
                    0.0,
                    self.evaluation_count,
                    True,
                    creation_seconds=self.statistics.creation_time[-1],
                    evaluation_seconds=elapsed,
                )
            )
        if self._post_evaluation_callback:
            self.statistics.start_time_tracking("post_evaluation")
            self._post_evaluation_callback(generation, self)
            self.statistics.stop_time_tracking("post_evaluation")
        self.statistics.update_fitness(self.population)

    def _parent_key(self, parent, generation):
        source = parent.meta.get(SOLUTION_SOURCE_META_KEY)
        return SolutionHistoryKey(
            source.index if source else self.population.index(parent),
            generation,
            source.ident if source else self.ident,
        )

    def _create_offspring(self, generation):
        parent1 = self._selector.select(self.population)
        parents = [(self._parent_key(parent1, generation - 1), parent1.fitness)]
        args = parent1.arguments
        crossover_name = mutation_name = None
        if random.random() < self._crossover_rate:
            parent2 = self._selector.select(self.population)
            parents.append((self._parent_key(parent2, generation - 1), parent2.fitness))
            args = self._crossover.crossover(args, parent2.arguments)
            crossover_name = getattr(
                self._crossover, "last_operator_name", type(self._crossover).__name__
            )
        if random.random() < self._mutation_rate:
            args = self._mutator.mutate(args)
            mutation_name = getattr(
                self._mutator, "last_operator_name", type(self._mutator).__name__
            )
        candidate = self._solution_creator.create_solution(args)
        return candidate, parents, crossover_name, mutation_name

    def perform_generation(self, generation):
        target = generation + 1
        config = self.offspring_selection
        elites = sorted(
            self.population, key=lambda candidate: candidate.fitness, reverse=True
        )[: self.elitism_count or 0]
        elite_keys = [self._parent_key(candidate, generation) for candidate in elites]
        quota = math.ceil(config.success_ratio * self.population_size) if config else 0
        limit = (
            math.floor(config.max_selection_pressure * self.population_size)
            if config
            else self.population_size - len(elites)
        )
        successes, failures, children = [], [], []
        creation_seconds = evaluation_seconds = 0.0
        attempts = 0
        required = (
            self.population_size if config else self.population_size - len(elites)
        )
        # Ordinary GA creates its whole batch before evaluation, as before.
        batch = []
        if not config:
            count = (
                required
                if self.max_evaluations is None
                else min(required, self.max_evaluations - self.evaluation_count)
            )
            start = perf_counter()
            batch = [self._create_offspring(target) for _ in range(count)]
            creation_seconds += perf_counter() - start
        while len(children) < required or (config and len(successes) < quota):
            if not self._has_budget():
                self.termination_reason = "max_evaluations"
                break
            if attempts >= limit:
                self.termination_reason = (
                    "max_selection_pressure" if config else "max_evaluations"
                )
                break
            start = perf_counter()
            if config or attempts >= len(batch):
                child, parents, cross, mutation = self._create_offspring(target)
                creation_seconds += perf_counter() - start
            else:
                child, parents, cross, mutation = batch[attempts]
            start = perf_counter()
            record = self._evaluate(
                child,
                target,
                "offspring",
                parents,
                cross,
                mutation,
                fresh=config is not None,
            )
            evaluation_seconds += perf_counter() - start
            attempts += 1
            entry = (child, parents, cross, mutation, record)
            children.append(entry)
            successful = record.successful if config else False
            (successes if successful else failures).append(entry)
        complete = len(children) >= required and (not config or len(successes) >= quota)
        unsuccessful_survivors = 0
        displaced_evaluation_ids = ()
        if complete:
            selected = (
                successes
                + random.sample(failures, self.population_size - len(successes))
                if config
                else children
            )
            if config:
                # Elitism follows selection and replaces the worst selected offspring, regardless of success.
                selected.sort(key=lambda entry: entry[0].fitness, reverse=True)
                retained_count = self.population_size - len(elites)
                displaced_evaluation_ids = tuple(
                    entry[4].evaluation_id for entry in selected[retained_count:]
                )
                selected = selected[:retained_count]
                unsuccessful_survivors = sum(
                    entry[4] is not None and entry[4].successful is False
                    for entry in selected
                )
            new_population = list(elites)
            for index, parent_key in enumerate(elite_keys):
                previous = self.statistics.solution_history.get(parent_key)
                self.statistics.add_history_item(
                    SolutionHistoryItem(
                        SolutionHistoryKey(index, target, self.ident),
                        False,
                        parent_key,
                        evaluation_id=previous.evaluation_id if previous else None,
                        creation_kind="elite",
                    )
                )
            for child, parents, cross, mutation, record in selected:
                key = SolutionHistoryKey(len(new_population), target, self.ident)
                self.statistics.add_history_item(
                    SolutionHistoryItem(
                        key,
                        mutation is not None,
                        parents[0][0],
                        parents[1][0] if len(parents) > 1 else None,
                        cross,
                        mutation,
                        record.evaluation_id if record else None,
                        "offspring",
                    )
                )
                if record:
                    record.survivor_key = key
                new_population.append(child)
            self._population = new_population
            self.statistics._custom_time_tracking("creation", creation_seconds)
            self.statistics._custom_time_tracking("evaluation", evaluation_seconds)
        evaluated_attempts = sum(entry[4] is not None for entry in children)
        self.statistics.generation_summaries.append(
            GenerationSummary(
                target,
                self.ident,
                evaluated_attempts,
                len(successes) if config else 0,
                unsuccessful_survivors,
                quota,
                evaluated_attempts / self.population_size,
                self.evaluation_count,
                complete,
                None if complete else self.termination_reason,
                creation_seconds,
                evaluation_seconds,
                displaced_evaluation_ids,
            )
        )
        return complete

    def run(self):
        self._statistics = StatisticsTracker()
        self._completed_generations = 0
        self.create_initial_population()
        for generation in tqdm(range(self.num_generations), unit="generation"):
            self.evaluate_population(generation)
            self._completed_generations = generation + 1
            if generation + 1 < self.num_generations and not self.perform_generation(
                generation
            ):
                break
        if self.termination_reason is None:
            self.termination_reason = "num_generations"
            self.statistics.generation_summaries[
                -1
            ].termination_reason = self.termination_reason
        return self.best_solution()

    def best_solution(self) -> SolutionCandidate[A, R, SingleObjectiveFitness]:
        # Includes evaluated children from an interrupted generation.
        if self.termination_reason in ("max_evaluations", "max_selection_pressure"):
            return self._best_evaluated
        return max(self.population, key=lambda candidate: candidate.fitness)
