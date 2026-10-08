"""Budget/fairness and launch-hold checks use numeric fixtures only."""

from evolutionary_extensions.experiments.prompt_embedding.adapter import (
    PromptEmbeddingExperimentAdapter,
)


import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_prompt_embedding_experiment import Creator, Evaluator

from evolutionary.algorithms.ga import GeneticAlgorithm, OffspringSelectionConfig
from evolutionary.evolutionary_selectors import TournamentSelector
from evolutionary_extensions.execution.campaign import (
    prepare_campaign,
    resolve_campaign_seeds,
    run_campaign,
)
from evolutionary_extensions.experiments.prompt_embedding import (
    ExperimentConfig,
    load_bounds,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "osga,cap,completed,count",
    [(False, 6400, 100, 6301), (True, 6400, 100, 6400), (True, 6397, 99, 6397)],
)
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_population64_hundred_populations_and_exact_fresh_budget(
    osga, cap, completed, count, batch
):
    algorithm = GeneticAlgorithm(
        100,
        64,
        Creator(),
        TournamentSelector(3),
        SimpleNamespace(mutate=lambda a: a + 1),
        SimpleNamespace(crossover=lambda a, b: max(a, b)),
        Evaluator(),
        list(range(64)),
        mutation_rate=1,
        elitism_count=1,
        offspring_selection=OffspringSelectionConfig(0.6, 1, 10) if osga else None,
        max_evaluations=cap,
        candidate_batch_size=batch,
    )
    algorithm.run()
    assert algorithm.evaluation_count == count
    assert len(algorithm.statistics.best_fitness) == completed
    assert len(algorithm.statistics.evaluation_records) == count
    assert len(algorithm.statistics.solution_history) == completed * 64
    assert all(len(p) == 64 for p in [algorithm.population])


def test_strict_failure_stops_at_pressure_without_budget_padding():
    algorithm = GeneticAlgorithm(
        100,
        64,
        Creator(),
        TournamentSelector(3),
        None,
        None,
        Evaluator(),
        [1] * 64,
        mutation_rate=0,
        crossover_rate=0,
        elitism_count=1,
        offspring_selection=OffspringSelectionConfig(0.6, 1, 10),
        max_evaluations=6400,
        candidate_batch_size=4,
    )
    algorithm.run()
    assert algorithm.evaluation_count == 704
    assert len(algorithm.statistics.best_fitness) == 1
    assert algorithm.termination_reason == "max_selection_pressure"
    assert all(
        r.successful is False for r in algorithm.statistics.evaluation_records[64:]
    )


def test_packaged_bounds_and_final_profiles():
    tensors, manifest = load_bounds()
    assert (
        manifest["sha256"]
        == "301f5b3d453a035a6731ef7cf4c8c73511d22d95d02d294fdfc61e7e1e247cd2"
    )
    assert tensors["token_min"].shape == (1, 77, 2048)
    for algorithm, runners, batch in [("ga", 1, 4), ("osga", 1, 4)]:
        config = json.loads((ROOT / f"configs/examples/{algorithm}.json").read_text())
        c = ExperimentConfig(**config["experiment"], **config["deployment"])
        assert (
            c.population_size,
            c.num_generations,
            c.max_evaluations,
            c.inference_steps,
        ) == (64, 100, 6400, 1)
        assert (
            config["execution"]["runner_count"] == runners
            and c.candidate_batch_size == batch
        )
        assert not config["execution"]["enabled"] and not config["drive"]["enabled"]


def test_preparation_freezes_unique_seeds_and_disabled_campaign(tmp_path, monkeypatch):
    import secrets

    values = iter([1, 1, 2, 3])
    monkeypatch.setattr(secrets, "randbits", lambda n: next(values))
    prepared = prepare_campaign(
        ROOT / "configs/examples/osga.json", tmp_path / "campaign"
    )
    resolved = json.loads(prepared.read_text())
    assert resolved["campaign"]["seeds"] == [1, 2, 3]
    assert resolve_campaign_seeds(resolved["campaign"]) == [1, 2, 3]
    assert resolve_campaign_seeds({"seeds": [42, 42, 43]}) == [42, 42, 43]
    with pytest.raises(ValueError, match="disabled"):
        asyncio.run(run_campaign(prepared))
    assert not (prepared.parent / "managed_sessions.json").exists()
    with pytest.raises(FileExistsError):
        prepare_campaign(prepared, prepared.parent)


def test_admission_reserves_current_run_future_zip(tmp_path, monkeypatch):
    from evolutionary_extensions.execution import campaign

    settings = {
        **json.loads((ROOT / "configs/examples/osga.json").read_text())["experiment"],
        "output_root": str(tmp_path),
        "cache_dir": str(tmp_path),
        "minimum_free_gib": 20,
    }
    estimate = PromptEmbeddingExperimentAdapter.estimate_run_bytes(settings)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=20 * 1024**3 + estimate + 1),
    )
    with pytest.raises(OSError, match="backlog"):
        campaign.require_campaign_space(settings, estimated_bytes=estimate)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(free=20 * 1024**3 + 2 * estimate + 1),
    )
    campaign.require_campaign_space(settings, estimated_bytes=estimate)
