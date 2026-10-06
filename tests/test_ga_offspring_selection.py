"""Deterministic scalar fixtures: no generative models or optimization experiments."""

import json
import random
from dataclasses import asdict
from itertools import cycle
from types import SimpleNamespace

import numpy as np
import pytest

from evolutionary.algorithms.ga import GeneticAlgorithm, OffspringSelectionConfig
from evolutionary.algorithms.island_model import IslandModel
from evolutionary.evolution_base import SolutionCandidate
from evolutionary.history import SolutionHistoryKey
from evolutionary.operators import MultiCrossover, MultiMutator


class Creator:
    def create_solution(self, argument):
        return SolutionCandidate(argument, argument)


class Evaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, result):
        self.calls += 1
        return result


class SequenceCrossover:
    def __init__(self, values):
        self.values = iter(values)

    def crossover(self, a, b):
        return next(self.values)


class AlternatingSelector:
    def __init__(self):
        self.indices = cycle((0, 1))

    def select(self, population):
        return population[next(self.indices) % len(population)]


def ga(values, initial=(2.0, 10.0), **kwargs):
    defaults = dict(
        num_generations=2,
        population_size=len(initial),
        initial_arguments=list(initial),
        solution_creator=Creator(),
        evaluator=Evaluator(),
        selector=AlternatingSelector(),
        mutator=SimpleNamespace(mutate=lambda a: a),
        crossover=SequenceCrossover(values),
        crossover_rate=1.0,
        mutation_rate=0.0,
    )
    defaults.update(kwargs)
    return GeneticAlgorithm(**defaults)


@pytest.mark.parametrize("factor,threshold", [(0.0, 2.0), (0.5, 6.0), (1.0, 10.0)])
def test_threshold_endpoints_equality_and_mutation_order(factor, threshold):
    algorithm = ga(
        [threshold, threshold],
        offspring_selection=OffspringSelectionConfig(0, factor, 1),
    )
    algorithm.run()
    records = algorithm.statistics.evaluation_records[2:]
    assert [r.threshold for r in records] == [threshold, threshold]
    assert all(r.successful is False for r in records)  # Equality is never success.
    algorithm = ga(
        [threshold, threshold],
        mutation_rate=1.0,
        mutator=SimpleNamespace(mutate=lambda a: a + 1),
        offspring_selection=OffspringSelectionConfig(1, factor, 1),
    )
    algorithm.run()
    assert all(r.successful for r in algorithm.statistics.evaluation_records[2:])


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64])
def test_numpy_evaluator_scores_preserve_classification_counts_and_json(scalar_type):
    # LAION returns np.mean(scores); include improvement, equality and failure.
    algorithm = ga(
        [scalar_type(11), scalar_type(10), scalar_type(1)],
        initial=(scalar_type(2), scalar_type(10), scalar_type(2)),
        offspring_selection=OffspringSelectionConfig(
            scalar_type(0.3), scalar_type(1), scalar_type(1)
        ),
    )
    algorithm.run()
    records = algorithm.statistics.evaluation_records[3:]
    assert [record.successful for record in records] == [True, False, False]
    assert all(type(record.successful) is bool for record in records)
    assert all(type(record.threshold) is float for record in records)
    assert all(type(candidate.fitness) is float for candidate in algorithm.population)
    summary = algorithm.statistics.generation_summaries[-1]
    assert summary.successes == 1 and summary.unsuccessful_survivors == 2
    pair = algorithm.statistics.operator_summary()[0]
    assert pair["successes"] == 1 and pair["success_rate"] == pytest.approx(1 / 3)
    json.dumps(
        {
            "configuration": asdict(algorithm.offspring_selection),
            "evaluations": [
                asdict(record) for record in algorithm.statistics.evaluation_records
            ],
            "generations": [
                asdict(item) for item in algorithm.statistics.generation_summaries
            ],
            "operators": algorithm.statistics.operator_summary(),
            "fitness": algorithm.statistics.best_fitness,
            "lineage": [
                asdict(item) for item in algorithm.statistics.solution_history.values()
            ],
        },
        allow_nan=False,
    )


def test_single_parent_and_skipped_operator_names():
    algorithm = ga(
        [],
        initial=(3.0, 3.0),
        crossover_rate=0.0,
        mutation_rate=1.0,
        mutator=SimpleNamespace(mutate=lambda a: a + 1),
        offspring_selection=OffspringSelectionConfig(1, 0, 1),
    )
    algorithm.run()
    for record in algorithm.statistics.evaluation_records[2:]:
        assert record.threshold == 3.0
        assert len(record.parent_keys) == 1
        assert record.crossover_name is None
        assert record.mutation_name == "SimpleNamespace"


def test_quota_rounding_full_population_then_elitism():
    algorithm = ga(
        [3.0, 3.0, 1.0, 1.0, 3.0],
        initial=(2.0,) * 5,
        elitism_count=1,
        offspring_selection=OffspringSelectionConfig(0.41, 1, 1),
    )
    algorithm.run()
    summary = algorithm.statistics.generation_summaries[-1]
    assert (summary.quota, summary.attempts, summary.successes) == (3, 5, 3)
    assert summary.unsuccessful_survivors == 1
    assert algorithm.evaluation_count == 10  # Full N, not N - elites.
    assert len(algorithm.population) == 5
    elite = algorithm.statistics.solution_history[SolutionHistoryKey(0, 1)]
    assert elite.creation_kind == "elite"
    assert elite.parent_1 == SolutionHistoryKey(0, 0)
    assert elite.evaluation_id == 1


def test_successful_offspring_displaced_by_elite_keeps_classification():
    algorithm = ga(
        [11.0, 12.0],
        elitism_count=1,
        offspring_selection=OffspringSelectionConfig(1, 1, 1),
    )
    algorithm.run()
    first, second = algorithm.statistics.evaluation_records[2:]
    assert algorithm.statistics.generation_summaries[-1].displaced_evaluation_ids == (
        3,
    )
    assert first.successful and first.survivor_key is None
    assert second.successful and second.survivor_key == SolutionHistoryKey(1, 1)
    assert [candidate.fitness for candidate in algorithm.population] == [10.0, 12.0]


def test_rejected_pool_sample_is_uniform_without_replacement(monkeypatch):
    original_sample = random.sample
    sampled = []

    def sample(pool, count):
        sampled.append(([entry[0].fitness for entry in pool], count))
        return original_sample(pool, count)

    monkeypatch.setattr(random, "sample", sample)
    random.seed(7)
    # Fill N=3 only after two successes, retaining a larger failure pool.
    algorithm = ga(
        [0.0, 1.0, 1.5, 3.0, 4.0],
        initial=(2.0,) * 3,
        offspring_selection=OffspringSelectionConfig(0.5, 1, 2),
    )
    algorithm.run()
    assert sampled == [([0.0, 1.0, 1.5], 1)]
    assert len({id(candidate) for candidate in algorithm.population}) == 3
    assert (
        sum(
            record.survivor_key is not None
            for record in algorithm.statistics.evaluation_records[3:]
        )
        == 3
    )


def test_ratio_zero_fills_population_without_quota():
    algorithm = ga([0.0, 0.0], offspring_selection=OffspringSelectionConfig(0, 1, 1))
    algorithm.run()
    summary = algorithm.statistics.generation_summaries[-1]
    assert summary.completed and summary.attempts == 2 and summary.quota == 0
    assert summary.unsuccessful_survivors == 2


def test_pressure_floor_all_failures_and_callbacks():
    calls = []
    algorithm = ga(
        [0.0] * 7,
        num_generations=3,
        offspring_selection=OffspringSelectionConfig(1, 1, 1.9),
        post_evaluation_callback=lambda g, a: calls.append((g, len(a.population))),
    )
    result = algorithm.run()
    assert result.fitness == 10.0
    assert algorithm.termination_reason == "max_selection_pressure"
    assert algorithm.completed_generations == 1
    assert algorithm.evaluation_count == 5  # floor(1.9 * 2) == 3 attempts.
    assert calls == [(0, 2)]
    assert algorithm.statistics.best_fitness == [10.0]
    summary = algorithm.statistics.generation_summaries[-1]
    assert (
        not summary.completed
        and summary.attempts == 3
        and summary.selection_pressure == 1.5
    )
    assert summary.termination_reason == "max_selection_pressure"
    assert (
        len(algorithm.statistics.creation_time)
        == len(algorithm.statistics.evaluation_time)
        == 1
    )
    assert summary.creation_seconds >= 0 and summary.evaluation_seconds >= 0
    assert all(
        record.survivor_key is None
        for record in algorithm.statistics.evaluation_records[2:]
    )
    assert all(key.generation == 0 for key in algorithm.statistics.solution_history)


@pytest.mark.parametrize("osga", [None, OffspringSelectionConfig(1, 1, 5)])
def test_global_limit_returns_best_evaluated_uncommitted_child(osga):
    calls = []
    algorithm = ga(
        [20.0, 30.0],
        max_evaluations=3,
        offspring_selection=osga,
        post_evaluation_callback=lambda g, a: calls.append(g),
    )
    result = algorithm.run()
    assert result.fitness == 20.0
    assert algorithm.evaluation_count == algorithm._evaluator.calls == 3
    assert algorithm.termination_reason == "max_evaluations"
    assert algorithm.completed_generations == 1 and calls == [0]
    assert [c.fitness for c in algorithm.population] == [2.0, 10.0]
    assert algorithm.statistics.evaluation_records[-1].survivor_key is None


def test_budget_exact_initial_and_exact_completed_generation():
    algorithm = ga([], max_evaluations=2)
    algorithm.run()
    assert algorithm.termination_reason == "max_evaluations"
    assert algorithm.statistics.generation_summaries[-1].attempts == 0
    algorithm = ga(
        [11.0, 12.0],
        max_evaluations=4,
        offspring_selection=OffspringSelectionConfig(1, 1, 1),
    )
    algorithm.run()
    assert algorithm.termination_reason == "num_generations"
    assert algorithm.completed_generations == 2
    assert algorithm.statistics.generation_summaries[-1].completed


def test_cached_elites_ordinary_costs_and_completed_callbacks():
    seen = []
    algorithm = ga(
        [11.0, 12.0, 13.0],
        num_generations=4,
        elitism_count=1,
        post_evaluation_callback=lambda g, a: seen.append(g),
    )
    algorithm.run()
    assert algorithm.evaluation_count == 2 + 3
    assert seen == [0, 1, 2, 3]
    assert len(algorithm.statistics.solution_history) == 8
    assert len(algorithm.statistics.evaluation_time) == 4
    assert all(r.successful is None for r in algorithm.statistics.evaluation_records)
    assert (
        algorithm.statistics.solution_history[SolutionHistoryKey(0, 3)].creation_kind
        == "elite"
    )


def test_strict_mode_corrected_bounded_and_conflict_rejected():
    algorithm = ga([10.0] * 20, strict_osga=True)
    algorithm.run()
    assert algorithm.offspring_selection == OffspringSelectionConfig(1, 1, 10)
    assert algorithm.termination_reason == "max_selection_pressure"
    assert algorithm.evaluation_count == 22
    assert not any(
        record.successful for record in algorithm.statistics.evaluation_records[2:]
    )
    with pytest.raises(ValueError, match="combined"):
        ga([], strict_osga=True, offspring_selection=OffspringSelectionConfig())


@pytest.mark.parametrize(
    "values",
    [
        (float("nan"), 1, 10),
        (1, float("inf"), 10),
        (1, 1, float("inf")),
        (-0.1, 1, 10),
        (1.1, 1, 10),
        (1, -0.1, 10),
        (1, 1, 0.9),
    ],
)
def test_invalid_selection_config(values):
    with pytest.raises(ValueError):
        OffspringSelectionConfig(*values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evaluations": 1},
        {"max_evaluations": 2.5},
        {"max_evaluations": True},
        {"mutation_rate": float("nan")},
        {"crossover_rate": 1.1},
        {"elitism_count": 3},
    ],
)
def test_invalid_ga_config(kwargs):
    with pytest.raises(ValueError):
        ga([], **kwargs)


def test_operator_pool_traces_and_summary_counts():
    algorithm = ga(
        [11.0, 12.0],
        elitism_count=1,
        offspring_selection=OffspringSelectionConfig(1, 1, 1),
        crossover=MultiCrossover({"chosen": SequenceCrossover([11.0, 12.0])}),
        mutator=MultiMutator({"identity": SimpleNamespace(mutate=lambda a: a)}),
        mutation_rate=1,
    )
    algorithm.run()
    records = algorithm.statistics.evaluation_records
    assert [r.evaluation_id for r in records] == [1, 2, 3, 4]
    assert records[2].parent_fitness == (2.0, 10.0)
    summary = algorithm.statistics.operator_summary()[0]
    assert (
        summary["crossover_name"] == "chosen" and summary["mutation_name"] == "identity"
    )
    assert summary["attempts"] == summary["successes"] == 2
    assert (
        summary["survivors"] == 1
        and summary["success_rate"] == 1
        and summary["survival_rate"] == 0.5
    )


def test_rerun_resets_records_counts_and_termination():
    algorithm = ga([11.0, 12.0, 13.0, 14.0])
    algorithm.run()
    algorithm.run()
    assert algorithm.evaluation_count == 4
    assert [
        record.evaluation_id for record in algorithm.statistics.evaluation_records
    ] == [1, 2, 3, 4]
    assert len(algorithm.statistics.generation_summaries) == 2


def test_island_model_rejects_unhandled_early_stop_before_creation():
    islands = [ga([], max_evaluations=2), ga([])]
    with pytest.raises(ValueError, match="standalone"):
        IslandModel(islands, 1, 1).run()
    assert not any(island.population for island in islands)


def test_cached_initial_population_and_population_evaluation_do_not_spend_budget():
    class CachedCreator(Creator):
        def create_solution(self, argument):
            candidate = super().create_solution(argument)
            candidate.fitness = argument
            return candidate

    algorithm = ga([], num_generations=1, solution_creator=CachedCreator())
    algorithm.run()
    assert algorithm.evaluation_count == algorithm._evaluator.calls == 0
    assert not algorithm.statistics.evaluation_records
    assert algorithm.statistics.generation_summaries[0].completed


def test_osga_children_always_receive_fresh_evaluation_and_pressure_accounting():
    class CachedCreator(Creator):
        def create_solution(self, argument):
            candidate = super().create_solution(argument)
            candidate.fitness = -99.0  # Not a valid score for this result.
            return candidate

    algorithm = ga(
        [11.0, 12.0],
        solution_creator=CachedCreator(),
        offspring_selection=OffspringSelectionConfig(1, 1, 1),
    )
    algorithm.run()
    assert algorithm.evaluation_count == 2
    assert [record.fitness for record in algorithm.statistics.evaluation_records] == [
        11.0,
        12.0,
    ]
    assert algorithm.statistics.generation_summaries[-1].selection_pressure == 1.0


def test_ordinary_ga_cached_offspring_fill_population_without_index_error():
    class CachedOffspringCreator(Creator):
        def create_solution(self, argument):
            candidate = super().create_solution(argument)
            if argument > 10:
                candidate.fitness = argument
            return candidate

    algorithm = ga(
        [11.0, 12.0], solution_creator=CachedOffspringCreator(), max_evaluations=3
    )
    algorithm.run()
    assert algorithm.completed_generations == 2
    assert algorithm.evaluation_count == 2
    assert algorithm.statistics.generation_summaries[-1].attempts == 0
    assert algorithm.termination_reason == "num_generations"


def test_elitism_only_ordinary_population_and_final_population_best():
    algorithm = ga([], num_generations=3, elitism_count=2, max_evaluations=2)
    algorithm.run()
    assert algorithm.completed_generations == 3 and algorithm.evaluation_count == 2
    algorithm = ga(
        [0.0, 1.0]
    )  # Ordinary behavior: final population, even without elites.
    assert algorithm.run().fitness == 1.0


def test_parent_keys_and_operator_traces_across_completed_generations():
    algorithm = ga(
        [11.0, 12.0, 13.0, 14.0],
        num_generations=3,
        offspring_selection=OffspringSelectionConfig(1, 1, 1),
    )
    algorithm.run()
    for key, item in algorithm.statistics.solution_history.items():
        if key.generation:
            assert item.parent_1 in algorithm.statistics.solution_history
            assert item.parent_1.generation == key.generation - 1
            assert item.crossover_name == "SequenceCrossover"
            assert item.mutation_name is None
    assert (
        algorithm.statistics.generation_summaries[-1].termination_reason
        == "num_generations"
    )


def test_skipped_variation_has_no_operator_names_and_preserves_source_parent():
    algorithm = ga(
        [],
        crossover_rate=0,
        mutation_rate=0,
        offspring_selection=OffspringSelectionConfig(0, 1, 1),
    )
    algorithm.run()
    for record in algorithm.statistics.evaluation_records[2:]:
        assert record.crossover_name is record.mutation_name is None
        assert record.fitness == record.threshold == record.parent_fitness[0]
        assert record.successful is False and record.survivor_key is not None
    for item in algorithm.statistics.solution_history.values():
        if item.creation_kind == "offspring":
            assert item.mutated is False and item.parent_2 is None
