import math
import random
from copy import copy
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
        candidate_batch_size: int = 1,
        post_evaluation_batch_callback=None,
        reuse_unchanged_offspring: bool = False,
        arguments_equal=None,
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
        if (
            isinstance(candidate_batch_size, bool)
            or not isinstance(candidate_batch_size, int)
            or candidate_batch_size < 1
        ):
            raise ValueError("candidate_batch_size must be a positive integer")
        if not isinstance(reuse_unchanged_offspring, bool):
            raise TypeError("reuse_unchanged_offspring must be boolean")
        if arguments_equal is not None and not callable(arguments_equal):
            raise TypeError("arguments_equal must be callable or None")
        self.reuse_unchanged_offspring = reuse_unchanged_offspring
        self._arguments_equal = arguments_equal or (lambda a, b: a is b)
        self.candidate_batch_size = candidate_batch_size
        self._post_evaluation_batch_callback = post_evaluation_batch_callback
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
        precomputed_fitness=None,
    ):
        reused = kind == "offspring" and getattr(candidate, "_reused_parent", False)
        if candidate.fitness is not None and not fresh and not reused:
            candidate.fitness = float(candidate.fitness)
            return None  # Cached fitness, notably elite carryover, costs no evaluation.
        if not reused:
            if not self._has_budget():
                raise RuntimeError("Evaluation budget exhausted")
            candidate.fitness = float(
                self._evaluator.evaluate(candidate.result)
                if precomputed_fitness is None
                else precomputed_fitness
            )
            self.evaluation_count += 1
        fitness = float(candidate.fitness)
        if not math.isfinite(fitness):
            raise ValueError("GA fitness must be finite")
        config = self.offspring_selection if parents else None
        threshold = (
            config.threshold([fitness for _, fitness in parents]) if config else None
        )
        record = EvaluationRecord(
            candidate._source_evaluation_id if reused else self.evaluation_count,
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
            reused=reused,
        )
        if reused:
            self.statistics.reused_offspring_records.append(record)
        else:
            self.statistics.evaluation_records.append(record)
        if (
            self._best_evaluated is None
            or candidate.fitness > self._best_evaluated.fitness
        ):
            self._best_evaluated = candidate
        return record

    def _create_many(self, arguments):
        create = getattr(self._solution_creator, "create_solutions", None)
        candidates = (
            list(create(arguments))
            if create
            else [self._solution_creator.create_solution(a) for a in arguments]
        )
        if len(candidates) != len(arguments):
            raise ValueError("Creator must return one candidate per argument")
        return candidates

    def _evaluate_many(
        self, candidates, generation, kind, metadata=None, *, fresh=False
    ):
        metadata = metadata or [((), None, None) for _ in candidates]
        pending = [
            i
            for i, candidate in enumerate(candidates)
            if candidate.fitness is None
            or (fresh and not getattr(candidate, "_reused_parent", False))
        ]
        scores = {}
        if pending:
            results = [candidates[i].result for i in pending]
            evaluate = getattr(self._evaluator, "evaluate_batch", None)
            values = (
                list(evaluate(results))
                if evaluate
                else [self._evaluator.evaluate(r) for r in results]
            )
            if len(values) != len(pending) or not all(
                math.isfinite(float(v)) for v in values
            ):
                raise ValueError("Evaluator must return one finite score per candidate")
            scores = dict(zip(pending, values))
        records = [
            self._evaluate(
                c, generation, kind, *m, fresh=fresh, precomputed_fitness=scores.get(i)
            )
            for i, (c, m) in enumerate(zip(candidates, metadata))
        ]
        return records

    def _notify_batch(self, generation, candidates, records):
        evaluated = [
            (c, r)
            for c, r in zip(candidates, records)
            if r is not None and not r.reused
        ]
        if evaluated and self._post_evaluation_batch_callback:
            self._post_evaluation_batch_callback(
                generation, [c for c, _ in evaluated], [r for _, r in evaluated], self
            )

    def create_initial_population(self):
        self.evaluation_count = 0
        self.termination_reason = None
        self._best_evaluated = None
        if self.candidate_batch_size == 1:
            super().create_initial_population()
        else:
            start = perf_counter()
            self._population = []
            for offset in range(0, self.population_size, self.candidate_batch_size):
                args = [
                    self._initial_arguments[i % len(self._initial_arguments)]
                    for i in range(
                        offset,
                        min(offset + self.candidate_batch_size, self.population_size),
                    )
                ]
                self._population.extend(self._create_many(args))
            for index in range(self.population_size):
                self.statistics.add_history_item(
                    SolutionHistoryItem(SolutionHistoryKey(index, 0, self.ident), False)
                )
            self.statistics._custom_time_tracking("creation", perf_counter() - start)
        for key, item in list(self.statistics.solution_history.items()):
            self.statistics.solution_history[key] = replace(
                item, creation_kind="initial"
            )

    def evaluate_population(self, generation):
        # Offspring were evaluated during construction. Initialization is evaluated here.
        start = perf_counter()
        initial_records = []
        for offset in range(0, len(self.population), self.candidate_batch_size):
            candidates = self.population[offset : offset + self.candidate_batch_size]
            records = (
                self._evaluate_many(candidates, generation, "initial")
                if self.candidate_batch_size > 1
                else [self._evaluate(candidates[0], generation, "initial")]
            )
            for index, (candidate, record) in enumerate(
                zip(candidates, records), offset
            ):
                if record:
                    key = SolutionHistoryKey(index, generation, self.ident)
                    record.survivor_key = key
                    self.statistics.solution_history[key] = replace(
                        self.statistics.solution_history[key],
                        evaluation_id=record.evaluation_id,
                    )
                    initial_records.append(record)
                if (
                    self._best_evaluated is None
                    or candidate.fitness > self._best_evaluated.fitness
                ):
                    self._best_evaluated = candidate
            self._notify_batch(generation, candidates, records)
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

    def _create_offspring(self, generation, *, deferred=False):
        parent1 = self._selector.select(self.population)
        parents = [(self._parent_key(parent1, generation - 1), parent1.fitness)]
        args = parent1.arguments
        sources = [parent1]
        crossover_name = mutation_name = None
        if random.random() < self._crossover_rate:
            parent2 = self._selector.select(self.population)
            parents.append((self._parent_key(parent2, generation - 1), parent2.fitness))
            sources.append(parent2)
            args = self._crossover.crossover(args, parent2.arguments)
            crossover_name = getattr(
                self._crossover, "last_operator_name", type(self._crossover).__name__
            )
        if random.random() < self._mutation_rate:
            args = self._mutator.mutate(args)
            mutation_name = getattr(
                self._mutator, "last_operator_name", type(self._mutator).__name__
            )
        if self.reuse_unchanged_offspring:
            for source in sources:
                if self._arguments_equal(args, source.arguments):
                    # Independent metadata; retain the immutable result and its original ID.
                    candidate = copy(source)
                    candidate.meta = dict(source.meta)
                    candidate._reused_parent = True
                    item = self.statistics.solution_history.get(
                        self._parent_key(source, generation - 1)
                    )
                    candidate._source_evaluation_id = (
                        item.evaluation_id if item else None
                    )
                    if candidate._source_evaluation_id is None:
                        raise ValueError(
                            "Reused parent must have an evaluation identity"
                        )
                    return candidate, parents, crossover_name, mutation_name
        candidate = args if deferred else self._solution_creator.create_solution(args)
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
        if not config and self.candidate_batch_size == 1:
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
            if self.candidate_batch_size > 1:
                # Even if every next child succeeds, these attempts are unavoidable.
                # This bound preserves the scalar stopping point without speculative evaluations.
                needed = max(required - len(children), quota - len(successes))
                budget = (
                    self.max_evaluations - self.evaluation_count
                    if self.max_evaluations is not None
                    else limit
                )
                count = min(self.candidate_batch_size, limit - attempts, budget, needed)
                prepared = [
                    self._create_offspring(target, deferred=True) for _ in range(count)
                ]
                cached = {
                    i: entry[0]
                    for i, entry in enumerate(prepared)
                    if self.reuse_unchanged_offspring
                    and isinstance(entry[0], SolutionCandidate)
                    and getattr(entry[0], "_reused_parent", False)
                }
                created = (
                    iter(
                        self._create_many(
                            [
                                entry[0]
                                for i, entry in enumerate(prepared)
                                if i not in cached
                            ]
                        )
                    )
                    if len(cached) < len(prepared)
                    else iter(())
                )
                candidates = [
                    cached[i] if i in cached else next(created)
                    for i in range(len(prepared))
                ]
                metadata = [entry[1:] for entry in prepared]
                creation_seconds += perf_counter() - start
                start = perf_counter()
                records = self._evaluate_many(
                    candidates, target, "offspring", metadata, fresh=config is not None
                )
            else:
                if config or attempts >= len(batch):
                    child, parents, cross, mutation = self._create_offspring(target)
                    creation_seconds += perf_counter() - start
                else:
                    child, parents, cross, mutation = batch[attempts]
                candidates, metadata = [child], [(parents, cross, mutation)]
                start = perf_counter()
                records = [
                    self._evaluate(
                        child,
                        target,
                        "offspring",
                        parents,
                        cross,
                        mutation,
                        fresh=config is not None,
                    )
                ]
            evaluation_seconds += perf_counter() - start
            for child, (parents, cross, mutation), record in zip(
                candidates, metadata, records
            ):
                attempts += 1
                entry = (child, parents, cross, mutation, record)
                children.append(entry)
                successful = record.successful if config else False
                (successes if successful else failures).append(entry)
            self._notify_batch(target, candidates, records)
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
        cached_attempts = sum(
            entry[4] is not None and entry[4].reused for entry in children
        )
        evaluated_attempts = sum(
            entry[4] is not None and not entry[4].reused for entry in children
        )
        self.statistics.generation_summaries.append(
            GenerationSummary(
                target,
                self.ident,
                evaluated_attempts,
                len(successes) if config else 0,
                unsuccessful_survivors,
                quota,
                (evaluated_attempts + cached_attempts) / self.population_size,
                self.evaluation_count,
                complete,
                None if complete else self.termination_reason,
                creation_seconds,
                evaluation_seconds,
                displaced_evaluation_ids,
                cached_attempts=cached_attempts,
                proposed_attempts=attempts,
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
