import random
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from evolutionary.algorithms.ga import GeneticAlgorithm, OffspringSelectionConfig
from evolutionary.evolution_base import SolutionCandidate


class Creator:
    def __init__(self):
        self.calls = 0

    def create_solution(self, argument):
        self.calls += 1
        return SolutionCandidate(argument, argument)


class Evaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, result):
        self.calls += 1
        return result


def run(batch, config, *, vary=False, sign=1, equal=None, budget=17):
    random.seed(52)
    creator, evaluator = Creator(), Evaluator()
    observed = []
    algorithm = GeneticAlgorithm(
        8,
        5,
        creator,
        SimpleNamespace(select=lambda p: random.choice(p)),
        SimpleNamespace(mutate=lambda a: a + sign * random.choice([0, 1])),
        SimpleNamespace(crossover=lambda a, b: (a + b) / 2),
        evaluator,
        [sign * float(i) for i in range(5)],
        crossover_rate=0.7 if vary else 0,
        mutation_rate=0.4 if vary else 0,
        elitism_count=1,
        offspring_selection=config,
        max_evaluations=budget,
        candidate_batch_size=batch,
        reuse_unchanged_offspring=True,
        arguments_equal=equal,
        post_evaluation_batch_callback=lambda g, cs, rs, a: observed.extend(rs),
    )
    algorithm.run()
    return algorithm, creator, evaluator, observed, random.getstate()


@pytest.mark.parametrize("batch", [1, 4, 16])
@pytest.mark.parametrize("config", [None, OffspringSelectionConfig(0, 1, 1)])
def test_no_operator_reuses_results_budget_and_lineage(batch, config):
    a, creator, evaluator, observed, _ = run(batch, config)
    assert a.completed_generations == 8
    assert a.evaluation_count == creator.calls == evaluator.calls == 5
    assert len(observed) == 5
    assert a.statistics.reused_offspring_records
    assert all(not r.successful for r in a.statistics.reused_offspring_records)
    assert all(
        1 <= h.evaluation_id <= 5 for h in a.statistics.solution_history.values()
    )
    assert len({id(c.meta) for c in a.population}) == 5
    assert all(
        s.attempts == 0 and s.cached_attempts == s.proposed_attempts
        for s in a.statistics.generation_summaries[1:]
    )


@pytest.mark.parametrize("batch", [1, 4, 16])
def test_cached_proposals_still_bound_pressure_without_success(batch):
    a, creator, evaluator, _, _ = run(batch, OffspringSelectionConfig(1, 1, 2))
    assert a.termination_reason == "max_selection_pressure"
    assert a.evaluation_count == creator.calls == evaluator.calls == 5
    assert a.completed_generations == 1
    summary = a.statistics.generation_summaries[-1]
    assert summary.cached_attempts == summary.proposed_attempts == 10
    assert summary.selection_pressure == 2 and summary.successes == 0
    assert all(r.survivor_key is None for r in a.statistics.reused_offspring_records)


@pytest.mark.parametrize("batch", [4, 16])
@pytest.mark.parametrize(
    "config",
    [None, OffspringSelectionConfig(0.4, 1, 5), OffspringSelectionConfig(0.4, 0.5, 5)],
)
@pytest.mark.parametrize("sign", [1, -1])
def test_reuse_scalar_batch_partial_budget_rng_classification(batch, config, sign):
    def snapshot(batch):
        a, creator, evaluator, observed, state = run(
            batch, config, vary=True, sign=sign, equal=lambda a, b: a == b
        )
        assert a.evaluation_count == creator.calls == evaluator.calls <= 17
        assert [r.evaluation_id for r in observed] == list(
            range(1, a.evaluation_count + 1)
        )
        assert all(not r.reused for r in observed)
        for r in a.statistics.reused_offspring_records:
            if config:
                assert r.successful == (r.fitness > r.threshold)
        return (
            [c.fitness for c in a.population],
            [asdict(r) for r in a.statistics.evaluation_records],
            [asdict(r) for r in a.statistics.reused_offspring_records],
            [asdict(h) for h in a.statistics.solution_history.values()],
            [
                {k: v for k, v in asdict(s).items() if not k.endswith("seconds")}
                for s in a.statistics.generation_summaries
            ],
            a.termination_reason,
            state,
        )

    assert snapshot(1) == snapshot(batch)


def test_exact_equality_is_opt_in_and_does_not_use_similarity():
    a, creator, evaluator, _, _ = run(
        4,
        OffspringSelectionConfig(0, 1, 1),
        vary=True,
        equal=lambda a, b: a == b,
    )
    for r in a.statistics.reused_offspring_records:
        assert r.fitness in r.parent_fitness
    assert a.evaluation_count == creator.calls == evaluator.calls


def test_reuse_configuration_requires_fixed_rendering_and_canonical_scoring():
    from evolutionary_extensions.experiments.prompt_embedding.config import (
        ExperimentConfig,
    )

    with pytest.raises(ValueError, match="canonical"):
        ExperimentConfig(reuse_unchanged_offspring=True)
    assert ExperimentConfig(
        reuse_unchanged_offspring=True, canonical_scoring=True, render_batch_size=4
    ).reuse_unchanged_offspring


@pytest.mark.parametrize("objective", ["maximize", "minimize"])
@pytest.mark.parametrize("algorithm", ["ga", "osga"])
def test_reused_parent_archives_original_ids_without_duplicate_pngs(
    tmp_path, monkeypatch, objective, algorithm
):
    import json
    from pathlib import Path

    import torch
    from PIL import Image
    from safetensors.torch import save_file

    from evolutionary_extensions.experiments.prompt_embedding import (
        ExperimentConfig,
        run_experiment,
    )
    from evolutionary_imaging.image_base import ImageSolutionData
    from evolutionary_prompt_embedding.archive import (
        EmbeddingArchiveReader,
        sha256_file,
    )

    torch.set_num_threads(2)
    for name in ["manual_seed_all", "reset_peak_memory_stats"]:
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ["memory_allocated", "max_memory_allocated"]:
        monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    bounds = tmp_path / "bounds.safetensors"
    save_file(
        {
            "token_min": torch.ones(1, 77, 2048),
            "token_max": torch.ones(1, 77, 2048),
            "pooled_min": torch.ones(1, 1280),
            "pooled_max": torch.ones(1, 1280),
        },
        str(bounds),
    )
    bounds.with_suffix(".json").write_text(json.dumps({"sha256": sha256_file(bounds)}))
    calls = {"render": 0, "score": 0}

    def create(config, args):
        calls["render"] += len(args)
        return [
            SolutionCandidate(a, ImageSolutionData([Image.new("RGB", (16, 16))]))
            for a in args
        ]

    def score(result):
        calls["score"] += 1
        return 2.0

    config = ExperimentConfig(
        output_root=str(tmp_path / "runs"),
        bounds_file=str(bounds),
        cache_dir=str(tmp_path),
        device="cpu",
        population_size=3,
        num_generations=3,
        max_evaluations=9,
        objective=objective,
        algorithm=algorithm,
        seed=42,
        candidate_batch_size=4,
        canonical_scoring=True,
        render_batch_size=4,
        reuse_unchanged_offspring=True,
        success_ratio=0,
    )
    runtime = SimpleNamespace(
        compatible=lambda c: True,
        parity={
            "passed": True,
            "candidate_batch_size": 4,
            "canonical_scoring": True,
            "render_batch_size": 4,
        },
        environment={},
        models={},
        evaluator=SimpleNamespace(evaluate=score),
        create_solutions=create,
    )
    result = run_experiment(config, runtime)
    assert result["completed_generations"] == 3 and result["validation_passed"]
    assert result["evaluation_count"] == calls["render"] == calls["score"] == 3
    assert result["reused_offspring_count"] > 0
    run = Path(result["run_dir"])
    attempts = list(EmbeddingArchiveReader(run / "attempts").iter_records())
    populations = list(EmbeddingArchiveReader(run).iter_records())
    assert len(attempts) == 3 and len(populations) == 9
    assert len(list((run / "attempts/images").glob("*.png"))) == 3
    assert all(1 <= r["metadata"]["evaluation_id"] <= 3 for r in populations)
    cached = json.loads((run / "reused_offspring.json").read_text())
    assert len(cached) == result["reused_offspring_count"]
    assert all(r["reused"] and 1 <= r["evaluation_id"] <= 3 for r in cached)
    if algorithm == "osga":
        assert all(r["successful"] is False for r in cached)
    checkpoint = torch.load(run / "checkpoint.pt", weights_only=False)
    assert len(checkpoint["reused_offspring_records"]) == len(cached)
