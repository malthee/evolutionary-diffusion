"""Validate configuration/export only. Never execute the run or analysis cells."""

import json
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Generic, TypeVar
from unittest.mock import Mock

import nbformat
import numpy as np
import pytest
import torch
from IPython.core.inputtransformer2 import TransformerManager

from evolutionary.algorithms.ga import GeneticAlgorithm, OffspringSelectionConfig
from evolutionary.history import SolutionHistoryItem, SolutionHistoryKey
from evolutionary.statistics import (
    EvaluationRecord,
    GenerationSummary,
    StatisticsTracker,
)

ROOT = Path(__file__).resolve().parents[1]


def notebook(name="ga_osga_notebook.ipynb"):
    return nbformat.read(ROOT / "notebooks" / name, as_version=4)


@pytest.mark.parametrize(
    "name,run_call",
    [
        ("ga_osga_notebook.ipynb", "ga.run()"),
        ("ga_notebook.ipynb", "ga.run()"),
        ("island_ga_notebook.ipynb", "island_model.run()"),
        ("nsga_notebook.ipynb", "nsga.run()"),
    ],
)
def test_format_cleared_outputs_syntax_and_single_run_cell(name, run_call):
    nb = notebook(name)
    nbformat.validate(nb)
    transform = TransformerManager()
    run_cells = []
    for index, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        assert cell.execution_count is None and not cell.outputs
        compile(transform.transform_cell(cell.source), f"cell_{index}", "exec")
        if run_call in cell.source:
            run_cells.append(index)
    assert len(run_cells) == 1
    assert 'pip", "install", "-e"' in nb.cells[2].source
    config = nb.cells[9 if name == "ga_osga_notebook.ipynb" else 8].source
    assert config.index("creator = ") < config.index("torch.manual_seed(seed)")
    assert config.index("evaluator = ") < config.index("torch.manual_seed(seed)")
    assert "fixed_noise_seeds=fixed_noise_seeds" in config


@pytest.mark.parametrize("ordinary", [False, True])
@pytest.mark.parametrize("name", ["ga_osga_notebook.ipynb", "ga_notebook.ipynb"])
def test_configuration_construction_and_export_without_running(
    monkeypatch, tmp_path, ordinary, name
):
    class Range:
        minimum, maximum = -1.0, 1.0

        def random_tensor_in_range(self):
            return torch.rand(1, 3, 4)

    EmbedType, LabelType = TypeVar("EmbedType"), TypeVar("LabelType")

    class Visualizer(Generic[EmbedType, LabelType]):
        def __init__(self, *args, **kwargs):
            pass

    cache = {}

    def load_model(kind):
        if kind not in cache:
            # Real model constructors initialize random layers before loading weights.
            random.random()
            np.random.random()
            cache[kind] = SimpleNamespace(
                model=torch.nn.Linear(4, 4), evaluate=lambda result: np.float64(result)
            )
        return cache[kind]

    fake_evaluators = ModuleType("evolutionary_imaging.evaluators")
    fake_evaluators.AestheticsImageEvaluator = Mock(
        side_effect=lambda: load_model("laion")
    )
    fake_evaluators.MultiCLIPIQAEvaluator = Mock(
        side_effect=lambda **kwargs: load_model("iqa")
    )
    fake_visualizer = ModuleType(
        "evolutionary_prompt_embedding.tensorboard_embed_visualizer"
    )
    fake_visualizer.TensorboardEmbedVisualizer = Visualizer
    fake_visualizer.EmbeddingVariant = object()
    fake_ranges = ModuleType("evolutionary_prompt_embedding.value_ranges")
    fake_ranges.SDXLTurboEmbeddingRange = fake_ranges.SDXLTurboPooledEmbeddingRange = (
        Range
    )
    for module_name, module in [
        (fake_evaluators.__name__, fake_evaluators),
        (fake_visualizer.__name__, fake_visualizer),
        (fake_ranges.__name__, fake_ranges),
    ]:
        monkeypatch.setitem(sys.modules, module_name, module)
    import evolutionary_prompt_embedding.image_creation as image_creation

    # Both creator and evaluator cache their models; only cold setup consumes RNG.
    creator = Mock(side_effect=lambda **kwargs: load_model("diffusion"))
    monkeypatch.setattr(image_creation, "SDXLPromptEmbeddingImageCreator", creator)
    monkeypatch.setattr(
        GeneticAlgorithm, "run", Mock(side_effect=AssertionError("No experiments"))
    )
    config_cell = next(
        c.source
        for c in notebook(name).cells
        if c.cell_type == "code" and "run_configuration = " in c.source
    )
    pooled = name == "ga_osga_notebook.ipynb"
    if ordinary and pooled:
        config_cell = config_cell.replace(
            "offspring_selection = OffspringSelectionConfig(0.6, 1.0, 10.0)",
            "offspring_selection = None",
        )
    elif not ordinary and not pooled:
        config_cell = config_cell.replace(
            "offspring_selection = None",
            "offspring_selection = OffspringSelectionConfig(0.6, 1.0, 10.0)",
        )
    namespace = {"torch": torch, "use_visualizer": False, "save_images": False}
    exec(compile(config_cell, "config", "exec"), namespace)
    algorithm = namespace["ga"]
    assert algorithm.population_size == (100 if pooled else 200)
    assert algorithm.num_generations == 100
    assert algorithm.max_evaluations == (10_000 if pooled else None)
    assert algorithm.elitism_count == 1
    assert algorithm.offspring_selection == (
        None if ordinary else OffspringSelectionConfig(0.6, 1, 10)
    )
    if pooled:
        assert len(namespace["crossover"].operators) == 8
        assert len(namespace["mutator"].operators) == 4
        assert (
            namespace["crossover"].operators["sbx"].index
            == namespace["operator_parameters"]["sbx"]["index"]
        )
    creator.assert_called_once_with(
        batch_size=1, inference_steps=3, fixed_noise_seeds=[0]
    )
    fake_evaluators.AestheticsImageEvaluator.assert_called_once_with()
    assert len(namespace["init_args"]) == algorithm.population_size
    assert (
        namespace["run_configuration"]["evaluator"]
        == "LAION improved aesthetic predictor V2"
    )
    # Repeat configuration reproduces the random initial population; it still does not run GA.
    original_tensor = namespace["init_args"][0].prompt_embeds.clone()
    variation_state = torch.random.get_rng_state().clone()
    first_variation_draw = torch.rand(5)
    first_python_draw, first_numpy_draw = random.random(), np.random.random()
    exec(compile(config_cell, "config", "exec"), namespace)
    assert torch.equal(original_tensor, namespace["init_args"][0].prompt_embeds)
    assert torch.equal(variation_state, torch.random.get_rng_state())
    assert torch.equal(first_variation_draw, torch.rand(5))
    assert first_python_draw == random.random()
    assert first_numpy_draw == np.random.random()
    algorithm = namespace["ga"]
    stats = StatisticsTracker()
    key = SolutionHistoryKey(0, 0)
    stats.add_history_item(
        SolutionHistoryItem(key, False, evaluation_id=1, creation_kind="initial")
    )
    stats.evaluation_records.append(
        EvaluationRecord(
            1, 0, None, "initial", (), (), None, None, 3.0, survivor_key=key
        )
    )
    stats.generation_summaries.append(
        GenerationSummary(0, None, 1, 0, 0, 0, 0, 1, True)
    )
    algorithm._statistics = stats
    algorithm.evaluation_count = 1
    # Exercise a real GA record with LAION's NumPy return type, without running GA.
    record = algorithm._evaluate(
        SimpleNamespace(fitness=None, result=11.0),
        1,
        "offspring",
        parents=((key, np.float32(3.0)),),
        crossover="fixture",
    )
    assert record.successful is (None if ordinary else True)
    algorithm.termination_reason = "fixture"
    algorithm._completed_generations = 1
    namespace.update(save_run_path=str(tmp_path), checkout=ROOT)
    export_cell = next(
        c.source
        for c in notebook(name).cells
        if c.cell_type == "code" and "export_data = " in c.source
    )
    # Keep pickle workflow in notebook; skip pickling our intentionally mocked model objects.
    import pickle

    monkeypatch.setattr(pickle, "dump", lambda obj, file: file.write(b"fixture"))
    exec(compile(export_cell, "export", "exec"), namespace)
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1 and len(list(tmp_path.glob("*.pkl"))) == 1
    data = json.loads(paths[0].read_text())
    assert (
        data["actual_evaluation_count"] == 2 and data["termination_reason"] == "fixture"
    )
    assert data["configuration"]["seeds"]["diffusion"] == [0]
    assert data["evaluations"][0]["survivor_key"] == {
        "index": 0,
        "generation": 0,
        "ident": None,
    }
    assert data["lineage"][0]["creation_kind"] == "initial"
    assert data["generation_summaries"][0]["completed"] is True
    assert "torch" in data["versions"] and data["git_revision"]
    assert "clip" in data["versions"] and "pytorch-lightning" in data["versions"]
    assert "aesthetic-predictor-v2-5" not in data["versions"]
    assert data["evaluations"][1]["successful"] is (None if ordinary else True)
    assert "untracked_sources" in data
    for path, contents in data["untracked_sources"].items():
        assert contents == (ROOT / path).read_text()
    assert not algorithm.run.called
