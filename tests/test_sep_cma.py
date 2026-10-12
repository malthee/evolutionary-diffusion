"""Core CMA contracts and compatibility with detector-search checkpoints."""

import hashlib
import pickle

import numpy as np
import pytest

from evolutionary.algorithms.sep_cma import SepCMAES, SepCMAOptimizer
from evolutionary.evolution_base import SolutionCandidate
from evolutionary_extensions.experiments.detector_search.optimizer import (
    SearchOptimizer,
    optimizer_source_files,
)


class Creator:
    def create_solution(self, argument):
        return SolutionCandidate(argument, argument.copy())


class Evaluator:
    def evaluate(self, result):
        return -float(np.sum((result - 0.8) ** 2))


def test_algorithm_maximizes_fitness_and_matches_native_ask_tell():
    seen = []
    algorithm = SepCMAES(
        3,
        8,
        Creator(),
        Evaluator(),
        np.full(4, 0.5),
        post_evaluation_callback=lambda generation, state: seen.append(
            (generation, [candidate.arguments.copy() for candidate in state.population])
        ),
    )
    best = algorithm.run()
    reference = SepCMAOptimizer(np.full(4, 0.5), budget=24, population=8)
    all_fitness = []
    for generation, arguments in seen:
        assert np.array_equal(arguments, reference.ask())
        fitness = [Evaluator().evaluate(argument) for argument in arguments]
        all_fitness.extend(fitness)
        reference.tell(-np.asarray(fitness))
    assert [generation for generation, _ in seen] == [0, 1, 2]
    assert best.fitness == max(all_fitness)
    assert algorithm.completed_generations == 3
    assert algorithm.evaluation_count == 24
    assert len(algorithm.statistics.best_fitness) == 3
    assert len(algorithm.statistics.solution_history) == 24
    # A repeated run resets adaptation and statistics without advancing global RNG.
    assert algorithm.run().fitness == best.fitness
    assert algorithm.evaluation_count == 24
    assert len(algorithm.statistics.best_fitness) == 3


def test_decoding_and_batch_contract_preserve_order_and_fitness():
    class BatchCreator(Creator):
        def create_solutions(self, arguments):
            return [self.create_solution(argument) for argument in arguments]

    class BatchEvaluator(Evaluator):
        def evaluate_batch(self, results):
            return [self.evaluate(result) for result in results]

    mean = np.full(4, 0.5)
    scalar = SepCMAES(2, 8, Creator(), Evaluator(), mean, decode=lambda v: 2 * v)
    batched = SepCMAES(
        2, 8, BatchCreator(), BatchEvaluator(), mean, decode=lambda v: 2 * v
    )
    assert scalar.run().fitness == batched.run().fitness
    assert np.array_equal(
        scalar.optimizer.strategy.mean, batched.optimizer.strategy.mean
    )
    assert all(np.array_equal(c.arguments, c.result) for c in batched.population)


def test_legacy_checkpoint_path_and_pending_population_survive_core_extraction():
    optimizer = SearchOptimizer(np.full(4, 0.5), budget=16, population=8)
    pending = optimizer.ask()
    content = pickle.dumps(optimizer)
    assert b"evolutionary_extensions.experiments.detector_search.optimizer" in content
    digest = hashlib.sha256(content).hexdigest()
    restored = SepCMAOptimizer.restore_owned_state(content, digest)
    assert np.array_equal(restored.ask(), pending)
    scores = pending.sum(axis=1)
    optimizer.tell(scores)
    restored.tell(scores)
    assert np.array_equal(restored.ask(), optimizer.ask())


def test_invalid_fitness_does_not_commit_or_call_generation_callback():
    class InvalidEvaluator(Evaluator):
        def evaluate(self, result):
            return np.nan

    callbacks = []
    algorithm = SepCMAES(
        1,
        8,
        Creator(),
        InvalidEvaluator(),
        np.full(4, 0.5),
        post_evaluation_callback=lambda *args: callbacks.append(args),
    )
    with pytest.raises(ValueError, match="finite scores"):
        algorithm.run()
    assert algorithm.evaluation_count == 0
    assert callbacks == []
    assert all(candidate.fitness is None for candidate in algorithm.population)


def test_search_provenance_changes_when_extracted_implementations_change(
    tmp_path, monkeypatch
):
    import evolutionary.algorithms.sep_cma as optimizer_module
    import evolutionary_prompt_embedding.coordinates as coordinates_module

    core = tmp_path / "optimizer.py"
    geometry = tmp_path / "coordinates.py"
    core.write_text("optimizer version one")
    geometry.write_text("geometry version one")
    monkeypatch.setattr(optimizer_module, "__file__", str(core))
    monkeypatch.setattr(coordinates_module, "__file__", str(geometry))
    first = optimizer_source_files()
    assert set(first) == {
        "evolutionary/algorithms/sep_cma.py",
        "evolutionary_prompt_embedding/coordinates.py",
    }
    core.write_text("optimizer version two")
    second = optimizer_source_files()
    assert (
        first["evolutionary/algorithms/sep_cma.py"]
        != second["evolutionary/algorithms/sep_cma.py"]
    )
    assert (
        first["evolutionary_prompt_embedding/coordinates.py"]
        == second["evolutionary_prompt_embedding/coordinates.py"]
    )
    geometry.write_text("geometry version two")
    third = optimizer_source_files()
    assert (
        second["evolutionary_prompt_embedding/coordinates.py"]
        != third["evolutionary_prompt_embedding/coordinates.py"]
    )
