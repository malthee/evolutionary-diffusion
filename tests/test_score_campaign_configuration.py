from evolutionary_extensions.experiments.prompt_embedding.adapter import (
    PromptEmbeddingExperimentAdapter,
)

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from evolutionary_extensions.experiments.prompt_embedding.config import (
    ExperimentConfig,
)

from evolutionary_extensions.experiments.prompt_embedding.initialization import (
    initial_arguments,
    load_bounds,
)
from evolutionary_extensions.experiments.prompt_embedding.operator_pool import operators


def bounds():
    b, _ = load_bounds()
    return b


def test_weighted_pool_default_strengths_and_single_operator_ga():
    b = bounds()
    torch.manual_seed(42)
    initial = initial_arguments(b, 2)
    _, _, default = operators(b, initial)
    config = ExperimentConfig(
        crossover_weights={"arithmetic": 4, "slerp": 3},
        mutation_weights={"spherical": 4, "polynomial": 2},
    )
    cross, mut, parameters = operators(b, initial, config)
    assert dict(zip(cross.operators, cross.weights)) == {"arithmetic": 4, "slerp": 3}
    assert dict(zip(mut.operators, mut.weights)) == {"spherical": 4, "polynomial": 2}
    assert parameters == {name: default[name] for name in parameters}
    ga = replace(
        config,
        algorithm="ga",
        population_size=100,
        num_generations=200,
        max_evaluations=19801,
        crossover_weights={"arithmetic": 1},
        mutation_weights={"spherical": 1},
    )
    cross, mut, _ = operators(b, initial, ga)
    assert set(cross.operators) == {"arithmetic"} and set(mut.operators) == {
        "spherical"
    }
    child = mut.mutate(cross.crossover(*initial))
    assert torch.isfinite(child.prompt_embeds).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crossover_weights": {}},
        {"crossover_weights": {"unknown": 1}},
        {"mutation_weights": {"spherical": 0}},
        {"mutation_weights": {"spherical": True}},
        {"operator_parameters": {"spherical": {"participation": 2}}},
        {"operator_parameters": {"gaussian": {"wrong": 1}}},
        {"initialization": "local"},
        {"initial_embeddings_file": "missing"},
    ],
)
def test_invalid_settings_fail_without_models(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_frozen_uniform_and_local_initialization(tmp_path):
    b = bounds()
    torch.manual_seed(42)
    population = initial_arguments(b, 3)
    path = tmp_path / "population.safetensors"
    save_file(
        {
            "prompt_embeds": torch.stack([a.prompt_embeds for a in population]),
            "pooled_prompt_embeds": torch.stack(
                [a.pooled_prompt_embeds for a in population]
            ),
        },
        str(path),
    )
    config = ExperimentConfig(
        initial_embeddings_file=str(path),
        initial_embeddings_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    frozen = initial_arguments(b, 3, config)
    assert all(
        torch.equal(a.prompt_embeds, c.prompt_embeds)
        and torch.equal(a.pooled_prompt_embeds, c.pooled_prompt_embeds)
        for a, c in zip(population, frozen)
    )
    frozen[0].prompt_embeds.add_(1)
    assert not torch.equal(population[0].prompt_embeds, frozen[0].prompt_embeds)
    with pytest.raises(ValueError, match="size"):
        initial_arguments(b, 4, config)
    anchor = tmp_path / "anchor.safetensors"
    save_file(
        {
            "prompt_embeds": population[0].prompt_embeds.unsqueeze(0),
            "pooled_prompt_embeds": population[0].pooled_prompt_embeds.unsqueeze(0),
        },
        str(anchor),
    )
    local = replace(
        config,
        initialization="local",
        initial_embeddings_file=str(anchor),
        initial_embeddings_sha256=hashlib.sha256(anchor.read_bytes()).hexdigest(),
        initial_perturbation_std={"token": 0.25, "pooled": 0.05},
        operator_parameters={
            "gaussian": {
                "prompt_strength": 0.5,
                "pooled_strength": 0.1,
                "prompt_participation": 0.05,
                "pooled_participation": 0.05,
            },
            "spherical": {"angle": 0.03},
        },
    )
    torch.manual_seed(43)
    a = initial_arguments(b, 5, local)
    torch.manual_seed(43)
    c = initial_arguments(b, 5, local)
    assert all(torch.equal(x.prompt_embeds, y.prompt_embeds) for x, y in zip(a, c))
    assert torch.equal(a[0].prompt_embeds, population[0].prompt_embeds)
    assert not torch.equal(a[1].prompt_embeds, a[2].prompt_embeds)
    for item in a:
        for key, value in [
            ("token", item.prompt_embeds),
            ("pooled", item.pooled_prompt_embeds),
        ]:
            assert value.dtype == torch.float16 and torch.isfinite(value).all()
            constant = b[key + "_min"] == b[key + "_max"]
            assert torch.equal(value[constant], b[key + "_min"][constant].half())
    _, _, params = operators(b, a, local)
    assert (
        params["gaussian"]["prompt_strength"] == 0.5
        and params["spherical"]["angle"] == 0.03
    )
    anchor.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        initial_arguments(b, 5, local)


def test_bulk_output_admission_separates_cache_filesystem(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from evolutionary_extensions.execution import campaign

    output = tmp_path / "output"
    cache = tmp_path / "cache"
    output.mkdir()
    cache.mkdir()
    original_stat = type(output).stat
    monkeypatch.setattr(
        type(output),
        "stat",
        lambda path, *args, **kwargs: (
            SimpleNamespace(st_dev=2, st_mode=0o40755)
            if path == output
            else SimpleNamespace(st_dev=1, st_mode=0o40755)
            if path == cache
            else original_stat(path, *args, **kwargs)
        ),
    )
    settings = {
        "output_root": str(output),
        "cache_dir": str(cache),
        "population_size": 100,
        "num_generations": 200,
        "max_evaluations": 19801,
        "minimum_free_gib": 20,
    }
    estimate = PromptEmbeddingExperimentAdapter.estimate_run_bytes(settings)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            free=21 * 1024**3 if path == cache else 20 * 1024**3 + 2 * estimate + 1
        ),
    )
    campaign.require_campaign_space(settings, estimated_bytes=estimate)
    monkeypatch.setattr(
        campaign.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(
            free=19 * 1024**3 if path == cache else 20 * 1024**3 + 2 * estimate + 1
        ),
    )
    with pytest.raises(OSError):
        campaign.require_campaign_space(settings, estimated_bytes=estimate)


def test_score_batch_canonical_calls_preserve_order():
    from evolutionary_extensions.experiments.prompt_embedding.runtime import (
        score_results,
    )

    class Evaluator:
        def evaluate(self, result):
            return result * 2

        def evaluate_batch(self, results):
            return [x * 2 + 0.000001 for x in results]

    values = [3, 1, 4]
    assert score_results(Evaluator(), values, canonical=True) == [6, 2, 8]
    assert score_results(Evaluator(), values, canonical=False) == [
        6.000001,
        2.000001,
        8.000001,
    ]


def test_fixed_renderer_padding_order_and_no_variation_rng():
    import random
    from types import SimpleNamespace

    from evolutionary_extensions.experiments.prompt_embedding.runtime import Runtime

    runtime = Runtime.__new__(Runtime)
    calls = []

    def create(arguments):
        calls.append(list(arguments))
        return [SimpleNamespace(arguments=a) for a in arguments]

    runtime.creator = SimpleNamespace(create_solutions=create)
    config = ExperimentConfig(candidate_batch_size=16, render_batch_size=4)
    state = random.getstate()
    torch_state = torch.get_rng_state()
    values = runtime.create_solutions(config, list(range(17)))
    assert [v.arguments for v in values] == list(range(17))
    assert all(len(c) == 4 for c in calls) and calls[-1] == [16, 16, 16, 16]
    assert (
        runtime.padding_images == 3
        and random.getstate() == state
        and torch.equal(torch_state, torch.get_rng_state())
    )


@pytest.mark.parametrize("position_dependent", [False, True])
def test_canonical_preflight_enforces_shape_and_position(
    tmp_path, monkeypatch, position_dependent
):
    import json
    from types import SimpleNamespace

    from PIL import Image

    from evolutionary.evolution_base import SolutionCandidate
    from evolutionary_extensions.experiments.prompt_embedding.runtime import Runtime
    from evolutionary_imaging.image_base import ImageSolutionData

    runtime = Runtime.__new__(Runtime)

    def create(arguments):
        return [
            SolutionCandidate(
                a,
                ImageSolutionData(
                    [
                        Image.new(
                            "RGB",
                            (8, 8),
                            (
                                len(arguments) + i
                                if position_dependent
                                else len(arguments),
                                0,
                                0,
                            ),
                        )
                    ]
                ),
            )
            for i, a in enumerate(arguments)
        ]

    runtime.creator = SimpleNamespace(create_solutions=create)
    runtime.evaluator = SimpleNamespace(
        evaluate=lambda result: 1.0,
        evaluate_batch=lambda results: [1.000001] * len(results),
    )

    def reference(command, **kwargs):
        Path(command[-1]).write_text(json.dumps([1.0] * 16))

    monkeypatch.setattr(
        "evolutionary_extensions.experiments.prompt_embedding.runtime.subprocess.run",
        reference,
    )
    config = ExperimentConfig(
        cache_dir=str(tmp_path),
        canonical_scoring=True,
        candidate_batch_size=3,
        render_batch_size=4,
    )
    if position_dependent:
        with pytest.raises(RuntimeError):
            runtime.check_parity(config, tmp_path / "parity")
        assert runtime.parity["fixed_positions"] is False
    else:
        parity = runtime.check_parity(config, tmp_path / "parity")
        assert (
            parity["passed"]
            and parity["fixed_positions"]
            and parity["cross_batch_images_identical"]
        )
        assert (
            parity["diagnostic_rng_unchanged"]
            and parity["fixed_scorer_repeated_identical"]
        )


@pytest.mark.parametrize(
    "algorithm,budget,populations,batch",
    [("osga", 15000, 151, 4), ("osga", 14999, 151, 4), ("ga", 19801, 200, 16)],
)
def test_population100_budget_boundaries(algorithm, budget, populations, batch):
    import random
    from types import SimpleNamespace

    from evolutionary.algorithms.ga import GeneticAlgorithm, OffspringSelectionConfig
    from evolutionary.evolution_base import SolutionCandidate
    from evolutionary.evolutionary_selectors import TournamentSelector

    random.seed(42)
    creator = SimpleNamespace(
        create_solution=lambda a: SolutionCandidate(a, a),
        create_solutions=lambda arguments: [SolutionCandidate(a, a) for a in arguments],
    )
    scorer = SimpleNamespace(
        evaluate=lambda a: a, evaluate_batch=lambda arguments: list(arguments)
    )
    ga = GeneticAlgorithm(
        populations,
        100,
        creator,
        TournamentSelector(3),
        SimpleNamespace(mutate=lambda a: a + 200),
        SimpleNamespace(crossover=lambda a, b: (a + b) / 2),
        scorer,
        list(map(float, range(100))),
        mutation_rate=1,
        crossover_rate=0.9,
        elitism_count=1,
        offspring_selection=OffspringSelectionConfig(0.4, 1, 100)
        if algorithm == "osga"
        else None,
        max_evaluations=budget,
        candidate_batch_size=batch,
    )
    ga.run()
    assert ga.evaluation_count == budget
    assert [r.evaluation_id for r in ga.statistics.evaluation_records] == list(
        range(1, budget + 1)
    )
    if algorithm == "ga":
        assert ga.completed_generations == 200
    elif budget == 14999:
        assert ga.statistics.generation_summaries[-1].completed is False
        assert ga.statistics.generation_summaries[-1].attempts == 99
