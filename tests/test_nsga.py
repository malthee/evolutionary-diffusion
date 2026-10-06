"""Paper-rule regressions on fixed objective vectors, without model/optimization runs."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory

from evolutionary.algorithms.nsga_ii import NSGA_II, NSGATournamentSelector
from evolutionary.algorithms.nsga_iii import (
    NSGA_III,
    U_NSGA_III,
    NSGAIIIBinaryRankSelector,
    NSGAIIIRandomSelector,
    NSGAIIISolutionCandidate,
    UNSGAIIITournamentSelector,
)
from evolutionary.evolution_base import SolutionCandidate
from evolutionary.history import SolutionHistoryKey
from evolutionary.operators import MultiCrossover, MultiMutator


class Creator:
    def __init__(self, cached=False):
        self.cached = cached

    def create_solution(self, arguments):
        candidate = SolutionCandidate(arguments, arguments)
        candidate.meta["fixture"] = "preserved"
        if self.cached:
            candidate.fitness = np.asarray(arguments)
        return candidate


class Evaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, result):
        self.calls += 1
        return np.asarray(result, dtype=np.float64)


def algorithm(cls=NSGA_III, initial=None, **kwargs):
    initial = initial if initial is not None else [(3, 1), (1, 3), (2, 2), (0, 0)]
    options = dict(
        num_generations=3,
        population_size=len(initial),
        solution_creator=Creator(),
        evaluator=Evaluator(),
        initial_arguments=initial,
        mutator=SimpleNamespace(mutate=lambda args: tuple(-10 for _ in args)),
        crossover=None,
        mutation_rate=1,
        crossover_rate=0,
    )
    if cls is NSGA_II:
        options["selector"] = NSGATournamentSelector()
    else:
        options["seed"] = 7
    options.update(kwargs)
    return cls(**options)


def ready(alg):
    alg.create_initial_population()
    alg.evaluate_population(0)
    return alg


@pytest.mark.parametrize("cls", [NSGA_II, NSGA_III, U_NSGA_III])
def test_elitist_union_callbacks_cost_and_survivor_lineage(cls):
    snapshots = []

    def callback(generation, alg):
        assert alg.completed_generations == generation + 1
        assert len(alg.statistics.best_fitness) == generation + 1
        assert {id(c) for front in alg.fronts for c in front} == {
            id(c) for c in alg.population
        }
        assert alg.best_solution() in alg.pareto_front
        assert all(
            c.rank == rank for rank, front in enumerate(alg.fronts) for c in front
        )
        snapshots.append((generation, [c.arguments for c in alg.population]))

    alg = algorithm(
        cls,
        initial=[(100, 0), (0, 10), (60, 6), (0, 0)],
        post_non_dominated_sort_callback=callback,
    )
    eval_callbacks = []
    alg._post_evaluation_callback = lambda g, a: eval_callbacks.append(g)
    result = alg.run()
    # Scaling selects the compromise despite the larger raw sum at (100, 0).
    assert result.fitness == [60, 6]
    assert eval_callbacks == [0, 1, 2] and [g for g, _ in snapshots] == [0, 1, 2]
    assert all(set(pop) == set(alg._initial_arguments) for _, pop in snapshots)
    assert alg.evaluation_count == alg._evaluator.calls == 12
    assert len(alg.statistics.evaluation_time) == len(alg.statistics.creation_time) == 3
    assert result in alg.pareto_front and isinstance(
        alg.population[0], alg.candidate_type
    )
    assert all(c.meta["fixture"] == "preserved" for c in alg.population)
    history = alg.statistics.solution_history
    assert len(history) == 12
    for key, item in history.items():
        if key.generation == 0:
            assert item.creation_kind == "initial" and item.parent_1 is None
        else:
            assert item.creation_kind == "elite" and item.parent_1 in history
            assert item.parent_1.generation == key.generation - 1
    before = len(snapshots)
    alg.best_solution()
    assert len(snapshots) == before  # Reading a result never invokes callbacks.


@pytest.mark.parametrize(
    "cls,generations", [(NSGA_II, 3), (NSGA_III, 3), (U_NSGA_III, 3), (U_NSGA_III, 1)]
)
def test_cached_fitness_and_rerun_reset(cls, generations):
    alg = algorithm(
        cls, solution_creator=Creator(cached=True), num_generations=generations
    )
    alg.run()
    first_survival = getattr(alg, "_survival", None)
    assert alg.evaluation_count == 0 and alg._evaluator.calls == 0
    assert len(alg.pareto_front) == 3 and alg.population[-1].rank == 1
    alg._initial_arguments = [(30, 10), (10, 30), (20, 20), (0, 0)]
    alg.run()
    assert alg.completed_generations == generations
    assert len(alg.statistics.solution_history) == 4 * generations
    if cls is not NSGA_II:
        assert alg._survival is not first_survival
        np.testing.assert_array_equal(alg._survival.norm.ideal_point, [-30, -30])


def test_maximization_ranks_and_equality():
    alg = ready(algorithm(NSGA_II, initial=[(3, 1), (1, 3), (2, 2), (0, 0), (3, 1)]))
    assert [c.rank for c in alg.population] == [0, 0, 0, 1, 0]


def test_nsga2_crowding_closed_form_scaling_constant_objective_and_split_front():
    values = [(0, 3), (1, 2), (2, 1), (3, 0)]
    for initial in (values, [(1000 * x, y, 5) for x, y in values]):
        alg = ready(algorithm(NSGA_II, initial=initial))
        distances = [c.crowding_distance for c in alg.population]
        np.testing.assert_allclose(distances, [np.inf, 4 / 3, 4 / 3, np.inf])
        alg._population_size = 3
        selected = alg._select_survivors(alg.population)
        assert len(selected) == 3 and {0, 3} <= set(selected)
    identical = ready(algorithm(NSGA_II, initial=[(1, 1)] * 4))
    assert [c.crowding_distance for c in identical.population] == [0] * 4
    with pytest.raises(ValueError, match="elitist union"):
        algorithm(NSGA_II, elitism_count=1)


@pytest.mark.parametrize(
    "selector,fields,expected",
    [
        (NSGATournamentSelector, [(0, 1), (1, np.inf)], 0),
        (NSGATournamentSelector, [(0, 1), (0, 2)], 1),
        (UNSGAIIITournamentSelector, [(0, 0, 10), (1, 0, 0)], 0),
        (UNSGAIIITournamentSelector, [(0, 0, 10), (0, 0, 1)], 1),
        (UNSGAIIITournamentSelector, [(0, 0, 1), (0, 0, 1)], 1),
        (UNSGAIIITournamentSelector, [(0, 0, 0), (1, 1, 10)], 1),
        (NSGAIIIBinaryRankSelector, [(0,), (1,)], 0),
        (NSGAIIIBinaryRankSelector, [(0,), (0,)], 1),
    ],
)
def test_mating_comparison_rules_without_objective_sum(
    monkeypatch, selector, fields, expected
):
    candidates = [NSGAIIISolutionCandidate(None, None) for _ in range(2)]
    names = (
        ("rank", "crowding_distance")
        if selector is NSGATournamentSelector
        else ("rank", "niche", "dist_to_niche")
    )
    for candidate, values in zip(candidates, fields):
        candidate.fitness = [1e6, 1e6] if candidate is candidates[0] else [0, 0]
        for name, value in zip(names, values):
            setattr(candidate, name, value)
    monkeypatch.setattr(random, "choices", lambda population, k: candidates)
    monkeypatch.setattr(random, "choice", lambda population: population[-1])
    assert selector().select(candidates) is candidates[expected]


def test_defaults_and_unprepared_selectors():
    assert isinstance(algorithm()._selector, NSGAIIIRandomSelector)
    assert isinstance(algorithm(U_NSGA_III)._selector, UNSGAIIITournamentSelector)
    for selector in (
        NSGATournamentSelector(),
        NSGAIIIBinaryRankSelector(),
        UNSGAIIITournamentSelector(),
    ):
        with pytest.raises(RuntimeError):
            selector.select([NSGAIIISolutionCandidate(None, None)])


def test_reference_association_closed_form_and_defensive_copy():
    directions = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    alg = ready(algorithm(initial=[(1, 0), (0, 1), (0.5, 0.5)], ref_dirs=directions))
    directions[:] = 9
    assert [c.niche for c in alg.population] == [1, 0, 2]
    np.testing.assert_allclose([c.dist_to_niche for c in alg.population], 0, atol=1e-8)
    copied = alg.ref_dirs
    copied[:] = 9
    np.testing.assert_allclose(alg.ref_dirs, [[1, 0], [0, 1], [0.5, 0.5]])


def test_reference_survival_selects_closest_for_empty_niches():
    alg = ready(algorithm(initial=[(1, 0), (0, 1)], ref_dirs=np.eye(2)))
    children = [alg._create_candidate(args) for args in ((0.95, 0.05), (0.05, 0.95))]
    alg._evaluate_candidates(children)
    assert set(alg._select_survivors(alg.population + children)) == {0, 1}


@pytest.mark.parametrize(
    "initial",
    [[(1, 1)] * 4, [(0, 0), (1, 1), (2, 2), (3, 3)], [(1,), (2,), (3,), (4,)]],
)
def test_degenerate_normalization_and_single_objective(initial):
    alg = algorithm(U_NSGA_III, initial=initial)
    result = alg.run()
    assert all(np.isfinite(c.dist_to_niche) for c in alg.population)
    assert result in alg.pareto_front
    if len(initial[0]) == 1:
        assert result.fitness == [4.0] and alg.ref_dirs.shape == (1, 1)


def test_seeded_survival_and_ideal_point_retained_across_generations():
    def selection(seed):
        alg = ready(algorithm(seed=seed, initial=[(1, 1)] * 4, ref_dirs=[[1, 1]]))
        candidates = [alg._create_candidate((1, 1)) for _ in range(12)]
        alg._evaluate_candidates(candidates)
        return alg._select_survivors(candidates)

    assert selection(4) == selection(4)
    assert selection(4) != selection(5)
    np.random.seed(42)
    implicit = selection(None)
    np.random.seed(42)
    assert implicit == selection(None)
    alg = ready(algorithm())
    before = alg._survival.norm.ideal_point.copy()
    alg.perform_generation(0)
    np.testing.assert_array_equal(before, alg._survival.norm.ideal_point)
    assert alg.best_solution() in alg.pareto_front
    # Boundary estimates remember extremes even when absent in a later population.
    worse = [alg._create_candidate((-10, -10)) for _ in range(8)]
    alg._evaluate_candidates(worse)
    alg._select_survivors(worse)
    np.testing.assert_array_equal(before, alg._survival.norm.ideal_point)


@pytest.mark.parametrize("cls", [NSGA_II, NSGA_III, U_NSGA_III])
def test_selected_child_operator_history_and_parent_preservation(cls):
    crossover = MultiCrossover(
        {"fixture_cross": SimpleNamespace(crossover=lambda a, b: (5, 1))}
    )
    mutator = MultiMutator(
        {"fixture_mutate": SimpleNamespace(mutate=lambda a: (a[0] + 10, a[1] + 10))}
    )
    alg = algorithm(
        cls,
        initial=[(0, 0)] * 4,
        num_generations=2,
        crossover=crossover,
        mutator=mutator,
        crossover_rate=1,
    )
    alg.run()
    assert all(candidate.fitness == [15, 11] for candidate in alg.population)
    assert alg._initial_arguments == [(0, 0)] * 4
    for key, item in alg.statistics.solution_history.items():
        if key.generation:
            assert item.creation_kind == "offspring" and item.mutated
            assert (
                item.crossover_name == "fixture_cross"
                and item.mutation_name == "fixture_mutate"
            )
            assert item.parent_1 in alg.statistics.solution_history
            assert item.parent_2 in alg.statistics.solution_history


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mutation_rate": np.nan},
        {"crossover_rate": np.inf},
        {"mutation_rate": -0.1},
        {"crossover_rate": 1.1},
        {"seed": -1},
        {"seed": 1.5},
        {"n_partitions": 0},
        {"n_partitions": 1.5},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        algorithm(**kwargs)


@pytest.mark.parametrize(
    "directions", [[], [1, 0], [[0, 0]], [[1, -1]], [[np.nan, 1]], [[1, 0], [2, 0]]]
)
def test_invalid_reference_directions(directions):
    with pytest.raises(ValueError, match="Reference directions"):
        algorithm(ref_dirs=directions)


def test_direction_dimension_and_count_validation_and_auto_partitions():
    with pytest.raises(ValueError, match="dimension"):
        ready(algorithm(ref_dirs=[[1, 0, 0]]))
    with pytest.raises(ValueError, match="population_size"):
        ready(algorithm(U_NSGA_III, n_partitions=5))
    with pytest.warns(UserWarning, match="More reference directions"):
        ready(algorithm(n_partitions=5))
    alg = ready(algorithm(initial=[(1, 2, 3)] * 20))
    assert alg.ref_dirs.shape == (15, 3)


def test_generated_energy_directions_use_run_seed_and_population_count(monkeypatch):
    calls = []

    def directions(factory, random_state=None):
        points = random_state.dirichlet(np.ones(factory.n_dim), factory.n_points)
        calls.append((factory.n_dim, factory.n_points))
        return points

    # Exercise the real factory's RNG plumbing; bypass only the energy solver.
    monkeypatch.setattr(RieszEnergyReferenceDirectionFactory, "_do", directions)
    alg = algorithm(
        U_NSGA_III, initial=[(1, 2, 3)] * 4, ref_dirs_method="energy", num_generations=1
    )
    alg.run()
    first = alg.ref_dirs
    alg.run()
    np.testing.assert_array_equal(first, alg.ref_dirs)
    assert calls[0] == calls[1] == (3, 4)
    other = ready(
        algorithm(U_NSGA_III, initial=[(1, 2, 3)] * 4, ref_dirs_method="energy", seed=8)
    )
    assert not np.array_equal(first, other.ref_dirs)


def test_unified_paper_mating_pool_and_sibling_pairs(monkeypatch):
    alg = ready(algorithm(U_NSGA_III, initial=[(i, 8 - i) for i in range(8)]))
    parents = list(alg.population)
    for i, candidate in enumerate(parents):
        candidate.rank, candidate.niche, candidate.dist_to_niche = 0, 0, float(i)
    monkeypatch.setattr(random, "shuffle", lambda candidates: candidates.reverse())
    pool = alg._selector.mating_pool(parents)
    assert pool == [parents[i] for i in (0, 2, 4, 6, 6, 4, 2, 0)]
    pairs = list(alg._offspring_parents(parents))
    assert len(pairs) == alg.population_size
    assert pairs[:2] == [(parents[0], parents[2]), (parents[2], parents[0])]
    assert parents == alg.population  # Shuffling a mating copy never reindexes parents.


@pytest.mark.parametrize("size", [2, 5])
def test_unified_population_requires_paper_batch_size(size):
    with pytest.raises(ValueError, match="divisible by four"):
        algorithm(U_NSGA_III, initial=[(1, 2)] * size)


@pytest.mark.parametrize("fitness", [np.nan, [1, np.inf], [], [[1, 2]], [1, 2, 3]])
def test_invalid_objective_vectors(fitness):
    alg = algorithm(initial=[(1, 2), fitness])
    with pytest.raises(ValueError, match="objective"):
        alg.run()


@pytest.mark.parametrize("cls", [NSGA_II, NSGA_III, U_NSGA_III])
def test_probability_zero_and_offspring_history_after_selection(cls, monkeypatch):
    crossover = SimpleNamespace(
        crossover=lambda *args: pytest.fail("zero probability crossover")
    )
    mutator = SimpleNamespace(
        mutate=lambda *args: pytest.fail("zero probability mutation")
    )
    monkeypatch.setattr(random, "random", lambda: 0)
    alg = algorithm(
        cls, crossover=crossover, mutator=mutator, crossover_rate=0, mutation_rate=0
    )
    alg.run()
    assert alg.evaluation_count == 12
    for key, item in alg.statistics.solution_history.items():
        assert item.crossover_name is None and item.mutation_name is None
        if key.generation:
            assert item.parent_1 in alg.statistics.solution_history
            assert item.parent_1.generation == key.generation - 1
            assert SolutionHistoryKey(key.index, key.generation, alg.ident) == item.key
