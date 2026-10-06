"""Construct and export NSGA configurations without executing the run/analysis cells."""

import importlib.metadata
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import nbformat
import numpy as np
import pytest
import torch
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory

from evolutionary.algorithms.algorithm_base import Algorithm
from evolutionary.algorithms.nsga_ii import NSGA_II
from evolutionary.algorithms.nsga_iii import NSGA_III, U_NSGA_III
from evolutionary.history import SolutionHistoryItem, SolutionHistoryKey

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "variant,cls,objectives",
    [
        ("nsga2", NSGA_II, 2),
        ("nsga3", NSGA_III, 3),
        ("unsga3", U_NSGA_III, 1),
        ("unsga3", U_NSGA_III, 2),
        ("unsga3", U_NSGA_III, 10),
    ],
)
@pytest.mark.parametrize("save_embeddings", [False, True])
def test_variant_configuration_and_json_export(
    monkeypatch, tmp_path, variant, cls, objectives, save_embeddings
):
    def forbidden(*args, **kwargs):
        pytest.fail("No optimization or model evaluation is allowed in this test.")

    class Range:
        minimum, maximum = -1.0, 1.0

        def random_tensor_in_range(self):
            return torch.rand(1, 3, 4)

    creator = SimpleNamespace(create_solution=forbidden)
    evaluator = SimpleNamespace(evaluate=forbidden)
    modules = {
        "evolutionary_prompt_embedding.value_ranges": {
            "SDXLTurboEmbeddingRange": Range,
            "SDXLTurboPooledEmbeddingRange": Range,
        },
        "evolutionary_prompt_embedding.image_creation": {
            "SDXLPromptEmbeddingImageCreator": lambda **kwargs: creator
        },
        "evolutionary_imaging.evaluators": {
            "MultiCLIPIQAEvaluator": lambda **kwargs: evaluator
        },
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(Algorithm, "run", forbidden)
    calls = []

    def reference_directions(factory, random_state=None):
        assert factory.n_dim == objectives
        points = random_state.dirichlet(np.ones(factory.n_dim), factory.n_points)
        calls.append(points.copy())
        return points

    monkeypatch.setattr(
        RieszEnergyReferenceDirectionFactory, "_do", reference_directions
    )
    notebook = nbformat.read(ROOT / "notebooks/nsga_notebook.ipynb", as_version=4)
    config = next(
        c.source for c in notebook.cells if "run_configuration = " in c.source
    ).replace('algorithm_variant = "unsga3"', f'algorithm_variant = "{variant}"')
    config = config.replace(
        "algorithm_variant = ",
        f"metrics = metrics[:{objectives}]\nalgorithm_variant = ",
    )
    direction_count = 10 if variant == "unsga3" and objectives == 2 else 20
    config = config.replace(
        "num_reference_directions = 20", f"num_reference_directions = {direction_count}"
    )
    namespace = dict(
        torch=torch,
        Path=Path,
        checkout=ROOT,
        save_images=False,
        save_embeddings=save_embeddings,
        run_dir=tmp_path / "archive",
    )
    exec(compile(config, "NSGA configuration", "exec"), namespace)
    alg = namespace["nsga"]
    assert (namespace["run_dir"] / "manifest.json").exists() is save_embeddings
    assert type(alg) is cls and alg.population_size == 20 and alg.num_generations == 10
    assert len(namespace["metrics"]) == objectives
    assert namespace["run_configuration"]["expected_uncached_evaluations"] == 200
    if variant != "nsga2" and objectives > 1:
        np.testing.assert_array_equal(
            calls[0],
            np.random.default_rng(1).dirichlet(np.ones(objectives), direction_count),
        )
        changed = dict(
            torch=torch,
            Path=Path,
            checkout=ROOT,
            save_images=False,
            save_embeddings=save_embeddings,
            run_dir=tmp_path / "alternate-archive",
        )
        exec(
            compile(
                config.replace(
                    "reference_direction_seed = 1", "reference_direction_seed = 2"
                ),
                "NSGA alternate reference seed",
                "exec",
            ),
            changed,
        )
        assert not np.array_equal(
            alg._provided_ref_dirs, changed["nsga"]._provided_ref_dirs
        )
    if variant != "nsga2":
        assert alg._provided_ref_dirs.shape == (
            (1, 1) if objectives == 1 else (direction_count, objectives)
        )
    # Export a completed fixture population; never call creator/evaluator/run.
    candidate = alg.candidate_type(namespace["init_args"][0], None)
    candidate.fitness = [float(i) for i in range(objectives)]
    alg._population = [candidate]
    alg._fronts = [[candidate]]
    alg._completed_generations = 1
    alg.evaluation_count = 20
    alg.statistics.update_fitness(alg.population)
    alg.statistics.add_history_item(
        SolutionHistoryItem(SolutionHistoryKey(0, 0), False, creation_kind="initial")
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pickle, "dump", lambda obj, file: file.write(b"fixture"))
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: "0.12.0" if package == "evolutionary" else "fixture",
    )
    export = next(c.source for c in notebook.cells if "export_data = " in c.source)
    exec(compile(export, "NSGA export", "exec"), namespace)
    paths = list((tmp_path / "saved_runs").glob("*.json"))
    assert len(paths) == 1 and paths[0].with_suffix(".pkl").exists()
    data = json.loads(paths[0].read_text())
    assert data["configuration"]["algorithm"] == variant
    assert data["configuration"]["seeds"]["diffusion"] == [0]
    assert data["actual_evaluation_count"] == 20 and data["completed_generations"] == 1
    assert data["pareto_fitness"] == [candidate.fitness]
    assert data["lineage"][0]["creation_kind"] == "initial"
    assert data["versions"]["evolutionary"] == importlib.metadata.version(
        "evolutionary"
    )
    assert data["git_revision"]
