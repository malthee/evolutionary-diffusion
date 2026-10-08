"""Behavioral validation without model downloads, OAuth grants or GPU experiments."""

import json
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from evolutionary.algorithms.ga import GeneticAlgorithm, OffspringSelectionConfig
from evolutionary.evolution_base import SolutionCandidate
from evolutionary.evolutionary_selectors import TournamentSelector
from evolutionary_extensions.experiments.prompt_embedding import (
    BoundedPNGWriter,
    ExperimentConfig,
    initial_arguments,
    load_bounds,
    operators,
    run_experiment,
)
from evolutionary_prompt_embedding.archive import (
    EmbeddingArchiveReader,
    EmbeddingArchiveWriter,
    sha256_file,
)
from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData


class Creator:
    def create_solution(self, argument):
        return SolutionCandidate(argument, argument)

    def create_solutions(self, arguments):
        return [self.create_solution(a) for a in arguments]


class Evaluator:
    def evaluate(self, result):
        return result

    def evaluate_batch(self, results):
        return list(results)


def trajectory(batch, ratio, cap, pressure=4):
    random.seed(42)
    torch.manual_seed(42)
    observed = []
    algorithm = GeneticAlgorithm(
        6,
        5,
        Creator(),
        TournamentSelector(2),
        SimpleNamespace(mutate=lambda a: a + float(torch.randn(()))),
        SimpleNamespace(crossover=lambda a, b: (a + b) / 2),
        Evaluator(),
        [float(i) for i in range(5)],
        mutation_rate=1,
        elitism_count=1,
        offspring_selection=None
        if ratio is None
        else OffspringSelectionConfig(ratio, 1, pressure),
        max_evaluations=cap,
        candidate_batch_size=batch,
        post_evaluation_batch_callback=lambda g, cs, rs, a: observed.extend(
            r.evaluation_id for r in rs
        ),
    )
    algorithm.run()
    summaries = [
        {k: v for k, v in asdict(s).items() if not k.endswith("seconds")}
        for s in algorithm.statistics.generation_summaries
    ]
    return (
        [asdict(r) for r in algorithm.statistics.evaluation_records],
        [asdict(r) for r in algorithm.statistics.solution_history.values()],
        summaries,
        [c.arguments for c in algorithm.population],
        observed,
        random.getstate(),
        torch.get_rng_state(),
    )


@pytest.mark.parametrize(
    "ratio,cap,pressure",
    [
        (None, 200, 4),
        (None, 17, 4),
        (0, 200, 4),
        (0.6, 200, 4),
        (1, 200, 4),
        (0.6, 17, 4),
        (1, 200, 1),
    ],
)
@pytest.mark.parametrize("batch", [2, 4, 16])
def test_scalar_batched_trajectory_rng_and_budget(batch, ratio, cap, pressure):
    scalar = trajectory(1, ratio, cap, pressure)
    batched = trajectory(batch, ratio, cap, pressure)
    assert scalar[:-1] == batched[:-1]
    assert torch.equal(scalar[-1], batched[-1])
    assert batched[4] == list(range(1, len(batched[0]) + 1))


def test_bad_batches_do_not_commit_records():
    evaluator = SimpleNamespace(
        evaluate=lambda a: a, evaluate_batch=lambda values: [float("nan")] * len(values)
    )
    algorithm = GeneticAlgorithm(
        1, 2, Creator(), None, None, None, evaluator, [1, 2], candidate_batch_size=2
    )
    with pytest.raises(ValueError, match="finite"):
        algorithm.run()
    assert not algorithm.statistics.evaluation_records


def bounds_input(tmp_path):
    values = {
        "token_min": torch.full((1, 77, 2048), -1.0),
        "token_max": torch.full((1, 77, 2048), 1.0),
        "pooled_min": torch.full((1, 1280), -0.5),
        "pooled_max": torch.full((1, 1280), 0.5),
    }
    values["token_min"][:, 0, :] = values["token_max"][:, 0, :] = 0.125
    path = tmp_path / "bounds.safetensors"
    save_file(values, str(path))
    path.with_suffix(".json").write_text(
        json.dumps({"sha256": sha256_file(path), "count": 1528510})
    )
    return path, values


def test_diffusiondb_bounds_seeded_initialization_and_full_pool_repair(tmp_path):
    path, expected = bounds_input(tmp_path)
    bounds, _ = load_bounds(path)
    torch.manual_seed(42)
    initial = initial_arguments(bounds, 2)
    torch.manual_seed(42)
    again = initial_arguments(bounds, 2)
    assert torch.equal(initial[0].prompt_embeds, again[0].prompt_embeds)
    crossover, mutator, parameters = operators(bounds, initial)
    children = [
        operator.crossover(*initial) for operator in crossover.operators.values()
    ]
    children += [operator.mutate(initial[0]) for operator in mutator.operators.values()]
    assert parameters["gaussian"]["prompt_strength"] == 2.0
    assert parameters["gaussian"]["pooled_strength"] == 0.4
    for child in [*initial, *children]:
        for key, tensor in [
            ("token", child.prompt_embeds),
            ("pooled", child.pooled_prompt_embeds),
        ]:
            assert tensor.dtype == torch.float16
            assert torch.isfinite(tensor).all()
            assert torch.all(tensor.float() >= expected[key + "_min"])
            assert torch.all(tensor.float() <= expected[key + "_max"])
        assert torch.all(child.prompt_embeds[:, 0, :] == 0.125)
    path.with_suffix(".json").write_text('{"sha256":"bad"}')
    with pytest.raises(ValueError, match="checksum"):
        load_bounds(path)


def test_multiple_attempt_snapshots_keep_generation_and_unique_ids(tmp_path):
    candidate = SolutionCandidate(
        PooledPromptEmbedData(torch.ones(1, 2, 3), torch.ones(1, 4)), None
    )
    candidate.fitness = 2.0
    writer = EmbeddingArchiveWriter(tmp_path)
    for snapshot in ["1-2", "3-4"]:
        writer.write_generation([candidate], 7, snapshot_id=snapshot)
    records = list(EmbeddingArchiveReader(tmp_path).iter_records())
    assert len({r["record_id"] for r in records}) == 2
    assert [r["generation"] for r in records] == [7, 7]
    with pytest.raises(ValueError, match="Duplicate"):
        writer.write_generation([candidate], 7, snapshot_id="1-2")


def test_bounded_png_writes_and_failure_propagation(tmp_path):
    writer = BoundedPNGWriter(limit=3)
    for i in range(10):
        writer.submit(Image.new("RGB", (8, 8)), tmp_path / f"{i}.png")
    writer.close()
    assert writer.peak_pending <= 3 and len(list(tmp_path.glob("*.png"))) == 10
    writer = BoundedPNGWriter()
    writer.submit(
        SimpleNamespace(
            save=lambda *a, **k: (_ for _ in ()).throw(OSError("write failed"))
        ),
        tmp_path / "bad.png",
    )
    with pytest.raises(OSError, match="write failed"):
        writer.close()


def test_candidate_batch_order_and_fixed_noise_without_global_rng():
    from evolutionary_prompt_embedding.image_creation import (
        SDXLPromptEmbeddingImageCreator,
    )

    calls, noise = [], []

    def generate(**kwargs):
        calls.append(kwargs)
        noise.append(
            [torch.randn(4, generator=generator) for generator in kwargs["generator"]]
        )
        return SimpleNamespace(
            images=[
                Image.new("RGB", (8, 8), (int(a[0, 0]), 0, 0))
                for a in kwargs["prompt_embeds"]
            ]
        )

    class Pipeline:
        device = "cpu"

        def __call__(self, **kwargs):
            return generate(**kwargs)

    creator = SDXLPromptEmbeddingImageCreator(
        3, 1, fixed_noise_seeds=[0], pipeline=Pipeline()
    )
    args = [
        PooledPromptEmbedData(torch.full((1, 77, 2048), float(i)), torch.zeros(1, 1280))
        for i in [3, 1, 2]
    ]
    state = torch.get_rng_state()
    candidates = creator.create_solutions(args)
    assert [c.result.images[0].getpixel((0, 0))[0] for c in candidates] == [3, 1, 2]
    assert torch.equal(torch.get_rng_state(), state)
    assert [g.initial_seed() for g in calls[0]["generator"]] == [0, 0, 0]
    assert calls[0]["num_images_per_prompt"] == 1
    creator.create_solutions(args)
    assert all(torch.equal(noise[0][0], row) for batch in noise for row in batch)
    assert torch.equal(torch.get_rng_state(), state)


def test_aesthetics_batch_precision_empty_and_cache_identity(monkeypatch):
    import evolutionary_imaging.evaluators as module
    from evolutionary_imaging.image_base import ImageSolutionData

    module.clear_model_cache()
    setups = []

    def setup(self, path):
        setups.append(str(self.device))
        return object(), object(), None

    monkeypatch.setattr(module.AestheticsImageEvaluator, "_setup_model", setup)
    cpu = module.AestheticsImageEvaluator(device="cpu", model_path="same")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cuda = module.AestheticsImageEvaluator(device="cuda", model_path="same")
    again = module.AestheticsImageEvaluator(device="cpu", model_path="same")
    assert (
        cpu.model is again.model
        and cpu.model is not cuda.model
        and setups == ["cpu", "cuda:0"]
    )
    cpu.preprocess = lambda image: torch.tensor(
        image.getpixel((0, 0)), dtype=torch.float32
    ).reshape(3, 1, 1)
    cpu.clip_model = SimpleNamespace(encode_image=lambda values: values.flatten(1))
    cpu.model = lambda values: values.sum(1, keepdim=True)
    red = Image.new("RGB", (2, 2), (255, 0, 0))
    green = Image.new("RGB", (2, 2), (0, 255, 0))
    assert cpu.evaluate_batch(
        [ImageSolutionData([red, green]), ImageSolutionData([])]
    ) == [1.0, 0.0]
    assert cpu.evaluate(ImageSolutionData([red])) == 1.0
    module.clear_model_cache()


def test_notebook_template_is_safe_valid_and_parameterized():
    notebook = nbformat.read(
        Path(__file__).resolve().parents[1]
        / "notebooks/prompt_embedding_experiment.ipynb",
        as_version=4,
    )
    nbformat.validate(notebook)
    assert sum("parameters" in c.metadata.get("tags", []) for c in notebook.cells) == 1
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not cell.outputs and cell.execution_count is None
            compile(cell.source, "notebook", "exec")
    assert any("RUN_EXPERIMENT = False" in c.source for c in notebook.cells)


@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("objective", ["maximize", "minimize"])
@pytest.mark.parametrize(
    "algorithm,budget,completed", [("osga", 63, 3), ("osga", 9, 2), ("ga", 63, 3)]
)
def test_cpu_mocked_end_to_end_artifacts_and_rejected_attempts(
    tmp_path, monkeypatch, budget, completed, objective, algorithm, buffered
):
    torch.set_num_threads(2)
    path, _ = bounds_input(tmp_path)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)

    class FakeCreator:
        def create_solutions(self, args):
            from evolutionary_imaging.image_base import ImageSolutionData

            return [
                SolutionCandidate(
                    a,
                    ImageSolutionData(
                        [
                            Image.new(
                                "RGB",
                                (16, 16),
                                (
                                    int(a.pooled_prompt_embeds.float().sum().abs())
                                    % 256,
                                    0,
                                    0,
                                ),
                            )
                        ]
                    ),
                )
                for a in args
            ]

    class FakeEvaluator:
        count = 0

        def evaluate_batch(self, results):
            values = []
            for result in results:
                self.count += 1
                # Alternate rejected and successful attempts; later successes always improve.
                values.append(
                    (1 if objective == "maximize" else -1)
                    * (float(self.count) if self.count % 2 else -1.0)
                )
            return values

    config = ExperimentConfig(
        output_root=str(tmp_path / "runs"),
        bounds_file=str(path),
        cache_dir=str(tmp_path),
        device="cpu",
        population_size=3,
        num_generations=3,
        max_evaluations=budget,
        objective=objective,
        algorithm=algorithm,
        buffered_attempt_writes=buffered,
        seed=42,
        candidate_batch_size=16,
    )
    runtime = SimpleNamespace(
        compatible=lambda c: True,
        parity={"passed": True, "candidate_batch_size": 16},
        environment={},
        models={},
        creator=FakeCreator(),
        evaluator=FakeEvaluator(),
    )
    result = run_experiment(config, runtime)
    assert result["completed_generations"] == completed
    assert result["validation_passed"] is (completed == 3)
    assert result["evaluation_count"] <= budget
    if completed == 2:
        assert result["termination_reason"] == "max_evaluations"
    run = Path(result["run_dir"])
    attempts = list(EmbeddingArchiveReader(run / "attempts").iter_records())
    population = list(EmbeddingArchiveReader(run).iter_records())
    assert (
        len(attempts) == result["evaluation_count"] and len(population) == 3 * completed
    )
    if algorithm == "osga":
        assert any(r["metadata"]["successful"] is False for r in attempts)
    assert {r["metadata"]["experiment_id"] for r in attempts + population} == {
        result["experiment_id"]
    }
    assert (
        len(list((run / "attempts/images").glob("*.png"))) == result["evaluation_count"]
    )
    assert (run / "plots/fitness.pdf").is_file()
    assert (run / result["best_image_path"]).is_file()
    best_evaluations = json.loads((run / "media/best_evaluations.json").read_text())
    assert best_evaluations[-1]["evaluation_id"] == result["best_evaluation_id"]
    assert best_evaluations[-1]["fitness"] == result["best_fitness"]
    sign = 1 if objective == "maximize" else -1
    for record in attempts + population:
        assert record["metadata"]["aesthetic_score"] == sign * record["fitness"]
    rows = [
        json.loads(line)
        for line in (run / "evaluations.jsonl").read_text().splitlines()
    ]
    assert len(rows) == result["evaluation_count"]
    assert all(r["aesthetic_score"] == sign * r["optimization_fitness"] for r in rows)
    assert result["best_evaluated_aesthetic_score"] == sign * max(
        r["fitness"] for r in rows
    )
    progress = json.loads((run / "progress.json").read_text())
    assert progress["generation"] == result["generation_summaries"][-1]["generation"]
    checkpoint = torch.load(
        run / "checkpoint.pt", weights_only=False, map_location="cpu"
    )
    assert (
        checkpoint["resumable"] is False
        and checkpoint["completed_generations"] == completed
    )
    assert all(
        c["prompt_embeds"].device.type == "cpu" for c in checkpoint["population"]
    )
    assert len(checkpoint["fitness_history"]["best"]) == completed


@pytest.mark.parametrize("buffered", [False, True])
def test_deadline_crossing_after_inference_preserves_evaluated_candidates(
    tmp_path, monkeypatch, buffered
):
    from evolutionary_imaging.image_base import ImageSolutionData

    torch.set_num_threads(2)
    path, _ = bounds_input(tmp_path)
    for name in ("manual_seed_all", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: None)
    for name in ("memory_allocated", "max_memory_allocated"):
        monkeypatch.setattr(torch.cuda, name, lambda *a: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", list)
    calls = []

    def guard(config, reserve_bytes=0):
        calls.append(reserve_bytes)
        if len(calls) >= 3:
            raise TimeoutError("deadline crossed after first inference")

    monkeypatch.setattr(ExperimentConfig, "guard", guard)
    runtime = SimpleNamespace(
        compatible=lambda c: True,
        parity={"passed": True, "candidate_batch_size": 16},
        environment={},
        models={},
        creator=SimpleNamespace(
            create_solutions=lambda args: [
                SolutionCandidate(a, ImageSolutionData([Image.new("RGB", (16, 16))]))
                for a in args
            ]
        ),
        evaluator=SimpleNamespace(evaluate_batch=lambda results: [1.0] * len(results)),
    )
    config = ExperimentConfig(
        output_root=str(tmp_path / "runs"),
        bounds_file=str(path),
        cache_dir=str(tmp_path),
        device="cpu",
        run_id="deadline",
        buffered_attempt_writes=buffered,
        population_size=3,
        num_generations=3,
        max_evaluations=63,
    )
    with pytest.raises(TimeoutError, match="deadline crossed"):
        run_experiment(config, runtime)
    run = tmp_path / "runs/deadline"
    assert calls[1] > 0
    assert len(list(EmbeddingArchiveReader(run / "attempts").iter_records())) == 3
    assert len(list((run / "attempts/images").glob("*.png"))) == 3
    assert len(json.loads((run / "evaluations.json").read_text())) == 3
    assert json.loads((run / "failure.json").read_text())["evaluation_count"] == 3
    assert not (run / "experiment_complete.json").exists()


def test_parity_preflight_honors_configured_batch_size(tmp_path, monkeypatch):
    from evolutionary_extensions.experiments.prompt_embedding import Runtime
    from evolutionary_imaging.image_base import ImageSolutionData

    path, _ = bounds_input(tmp_path)
    runtime = Runtime.__new__(Runtime)
    created, scored = [], []

    def create(args):
        created.append(len(args))
        return [
            SolutionCandidate(a, ImageSolutionData([Image.new("RGB", (16, 16))]))
            for a in args
        ]

    def score(results):
        scored.append(len(results))
        return [1.0] * len(results)

    def cpu_reference(command, **kwargs):
        Path(command[-1]).write_text(json.dumps([1.0] * 16))

    runtime.creator = SimpleNamespace(create_solutions=create)
    runtime.evaluator = SimpleNamespace(evaluate_batch=score)
    monkeypatch.setattr(
        "evolutionary_extensions.experiments.prompt_embedding.runtime.subprocess.run",
        cpu_reference,
    )
    config = ExperimentConfig(
        bounds_file=str(path), cache_dir=str(tmp_path), candidate_batch_size=3
    )
    parity = runtime.check_parity(config, tmp_path / "parity")
    assert parity["passed"] and parity["image_count"] == 16
    assert parity["candidate_batch_size"] == 3
    assert created == scored == [3, 3, 3, 3, 3, 1]


@pytest.mark.parametrize(
    "algorithm,budget,completed", [("osga", 63, 3), ("osga", 9, 2), ("ga", 63, 3)]
)
def test_buffered_execution_preserves_trajectory_and_scientific_hashes(
    tmp_path, monkeypatch, algorithm, budget, completed
):
    runs = []
    for buffered in (False, True):
        directory = tmp_path / str(buffered)
        directory.mkdir()
        test_cpu_mocked_end_to_end_artifacts_and_rejected_attempts(
            directory, monkeypatch, budget, completed, "maximize", algorithm, buffered
        )
        run = next((directory / "runs").iterdir())
        result = json.loads((run / "result.json").read_text())
        runs.append(
            (
                result,
                json.loads((run / "evaluations.json").read_text()),
                json.loads((run / "lineage.json").read_text()),
            )
        )
    for key in (
        "evaluation_count",
        "completed_generations",
        "termination_reason",
        "fitness",
        "operator_summary",
        "scientific_hashes",
    ):
        assert runs[0][0][key] == runs[1][0][key]
    assert runs[0][1:] == runs[1][1:]
