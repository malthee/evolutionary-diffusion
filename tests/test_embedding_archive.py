"""Persistence and callback regressions; no model downloads or GPU required."""

import ast
import json
import os
import pickle
import random
import resource
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from evolutionary.evolution_base import SolutionCandidate, SolutionCreator
from evolutionary_prompt_embedding.archive import (
    EmbeddingArchiveReader,
    EmbeddingArchiveWriter,
)
from evolutionary_prompt_embedding.argument_types import (
    PooledPromptEmbedData,
    PromptEmbedData,
)

ROOT = Path(__file__).resolve().parents[1]


def candidate(index=0, pooled=True, dtype=torch.float16, images=0):
    token = torch.arange(24, dtype=dtype).reshape(1, 3, 8) + index
    args = (
        PooledPromptEmbedData(token, torch.arange(4, dtype=dtype).reshape(1, 4))
        if pooled
        else PromptEmbedData(token)
    )
    from evolutionary_imaging.image_base import ImageSolutionData

    result = ImageSolutionData([Image.new("RGB", (4, 4), "red") for _ in range(images)])
    item = SolutionCandidate(args, result)
    item.fitness = 1.23456789012345 + index
    return item


def notebook_callback(name, namespace, island=0):
    nb = json.loads((ROOT / "notebooks" / name).read_text())
    nodes = []
    for cell in nb["cells"]:
        text = "".join(cell["source"])
        if cell["cell_type"] == "code" and (
            "def post_evaluation_callback" in text
            or "class IslandPostEvaluationCallback" in text
        ):
            nodes.extend(
                node
                for node in ast.parse(text).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, "exec"), namespace)  # noqa: S102 -- trusted repository callback fixture
    return (
        namespace["IslandPostEvaluationCallback"](island, "test style")
        if "IslandPostEvaluationCallback" in namespace
        else namespace["post_evaluation_callback"]
    )


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_roundtrip_and_shard_bounds(self):
        for dtype in (torch.float16, torch.float32):
            for pooled in (False, True):
                path = self.root / f"{dtype}-{pooled}"
                population = [candidate(i, pooled, dtype) for i in range(200)]
                writer = EmbeddingArchiveWriter(
                    path, {"model": "test"}, max_shard_bytes=8192
                )
                writer.write_generation(
                    population, 0, metadata=[{"prompt": f"p{i}"} for i in range(200)]
                )
                reader = EmbeddingArchiveReader(path)
                decoded = []
                for records, tensors in reader.iter_batches(batch_size=7):
                    self.assertLessEqual(len(records), 7)
                    for i, record in enumerate(records):
                        original = population[record["candidate_slot"]]
                        self.assertTrue(
                            torch.equal(
                                tensors["prompt_embeds"][i],
                                original.arguments.prompt_embeds,
                            )
                        )
                        self.assertEqual(tensors["prompt_embeds"].dtype, dtype)
                        self.assertEqual(record["fitness"], original.fitness)
                        self.assertEqual(record["image_paths"], [])
                        if pooled:
                            self.assertTrue(
                                torch.equal(
                                    tensors["pooled_prompt_embeds"][i],
                                    original.arguments.pooled_prompt_embeds,
                                )
                            )
                        decoded.append(record)
                self.assertEqual(len(decoded), 200)
                self.assertGreater(len(reader.manifest["shards"]), 1)
                for shard in reader.manifest["shards"]:
                    self.assertLessEqual((path / shard["tensors"]).stat().st_size, 8192)
                # Mutating the original after recording cannot change its snapshot.
                population[0].arguments.prompt_embeds.zero_()
                _, restored = reader.read_record(decoded[0]["record_id"])
                self.assertNotEqual(restored["prompt_embeds"].sum().item(), 0)

    def test_shards_use_cpu_with_a_non_cpu_default_device(self):
        population = [candidate()]
        writer = EmbeddingArchiveWriter(self.root)
        # Simulate a GPU-default caller without requiring a GPU for the test.
        with torch.device("meta"):
            writer.write_generation(population, 0)
        reader = EmbeddingArchiveReader(self.root)
        records, tensors = next(reader.iter_batches())
        self.assertEqual(len(records), 1)
        self.assertEqual(tensors["prompt_embeds"].device.type, "cpu")
        self.assertTrue(
            torch.equal(
                tensors["prompt_embeds"][0], population[0].arguments.prompt_embeds
            )
        )

    def test_interrupted_commit_duplicates_and_pickle(self):
        import evolutionary_prompt_embedding.archive as module

        writer = EmbeddingArchiveWriter(self.root, max_shard_bytes=8192)
        population = [candidate(i, dtype=torch.float32) for i in range(80)]
        original = module.save_file
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated full disk")
            return original(*args, **kwargs)

        with patch.object(module, "save_file", fail_second), self.assertRaises(OSError):
            writer.write_generation(population, 0)
        reader = EmbeddingArchiveReader(self.root)
        self.assertGreater(len(list(reader.iter_records())), 0)
        self.assertLess(len(list(reader.iter_records())), 80)
        self.assertEqual(reader.manifest["snapshots"][0]["status"], "writing")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pickle.loads(pickle.dumps(writer)).write_generation(population, 0)
        writer.write_generation(population[:2], 1)
        self.assertEqual(
            EmbeddingArchiveReader(self.root).manifest["snapshots"][-1]["status"],
            "complete",
        )

    def test_reject_incompatible_and_unsafe_inputs(self):
        writer = EmbeddingArchiveWriter(self.root)
        writer.write_generation([candidate()], 0)
        for population in ([candidate(pooled=False)], [candidate(dtype=torch.float32)]):
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                writer.write_generation(population, 1)
        with self.assertRaises(ValueError):
            writer.write_generation([candidate()], -1)
        with self.assertRaises(ValueError):
            writer.write_generation([candidate()], 2, image_paths=[["../outside.png"]])
        with self.assertRaises(ValueError):
            writer.write_generation([candidate()], 3, metadata=[{"bad": float("nan")}])
        manifest = json.loads((self.root / "manifest.json").read_text())
        self.assertEqual(len(manifest["snapshots"]), 1)
        tensor_path = self.root / manifest["shards"][0]["tensors"]
        with open(tensor_path, "ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            list(EmbeddingArchiveReader(self.root).iter_batches())

    def test_notebook_persistence_switches_and_multiple_images(self):
        from types import SimpleNamespace

        from evolutionary_imaging import processing

        for name in (
            "ga_notebook.ipynb",
            "ga_osga_notebook.ipynb",
            "nsga_notebook.ipynb",
            "island_ga_notebook.ipynb",
        ):
            for save_embeddings in (False, True):
                for save_images in (False, True):
                    path = self.root / f"{name}-{save_embeddings}-{save_images}"
                    writer = EmbeddingArchiveWriter(path) if save_embeddings else None
                    population = [
                        candidate(0, images=2),
                        candidate(1, images=0),
                        candidate(2, images=1),
                    ]
                    if name == "nsga_notebook.ipynb":
                        for item in population:
                            item.fitness = [item.fitness, item.fitness / 2]
                    namespace = {
                        "save_images": save_images,
                        "archive_writer": writer,
                        "save_images_from_generation_grouped": processing.save_images_from_generation_grouped,
                    }
                    callback = notebook_callback(name, namespace)
                    with patch.object(
                        processing, "RESULTS_FOLDER", str(path / "images")
                    ):
                        callback(0, SimpleNamespace(population=population))
                    self.assertEqual(
                        len(list(path.rglob("*.png"))), 3 if save_images else 0
                    )
                    self.assertEqual((path / "manifest.json").exists(), save_embeddings)
                    if writer:
                        records = list(EmbeddingArchiveReader(path).iter_records())
                        self.assertEqual(
                            [len(r["image_paths"]) for r in records],
                            [2, 0, 1] if save_images else [0, 0, 0],
                        )
                        if save_images:
                            self.assertTrue(
                                Path(records[2]["image_paths"][0]).name.startswith(
                                    "id_0_2_" if "island" in name else "2_"
                                )
                            )
                            original_pngs = {
                                file: file.read_bytes() for file in path.rglob("*.png")
                            }
                            population[0].result.images[0] = Image.new(
                                "RGB", (4, 4), "green"
                            )
                            with (
                                patch.object(
                                    processing, "RESULTS_FOLDER", str(path / "images")
                                ),
                                self.assertRaisesRegex(ValueError, "Duplicate"),
                            ):
                                callback(0, SimpleNamespace(population=population))
                            self.assertEqual(
                                original_pngs,
                                {
                                    file: file.read_bytes()
                                    for file in path.rglob("*.png")
                                },
                            )

    @unittest.skipUnless(
        os.getenv("EMBEDDING_LARGE_TEST") == "1",
        "Explicit >4 GiB disk/memory regression",
    )
    def test_larger_than_four_gib_memory_bounded(self):
        writer = EmbeddingArchiveWriter(self.root, max_shard_bytes=65 * 1024 * 1024)
        token = torch.full((1, 1024, 8192), 0.25, dtype=torch.float16)
        population = [SolutionCandidate(PromptEmbedData(token), None) for _ in range(4)]
        writer.write_generation(population, 0)
        peak_after_first = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for generation in range(1, 65):
            writer.write_generation(population, generation)
        reader = EmbeddingArchiveReader(self.root)
        total = sum(
            (self.root / s["tensors"]).stat().st_size for s in reader.manifest["shards"]
        )
        self.assertGreater(total, 4 * 1024**3)
        self.assertEqual(len(list(reader.iter_records())), 260)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys

        unit = 1 if sys.platform == "darwin" else 1024
        self.assertLessEqual((peak - peak_after_first) * unit, 2 * 65 * 1024 * 1024)
        last_id = list(reader.iter_records())[-1]["record_id"]
        _, tensors = reader.read_record(last_id)
        self.assertTrue(torch.equal(tensors["prompt_embeds"], token))
        print(
            f"Large archive: {total / 1024**3:.3f} GiB; additional peak RSS: {(peak - peak_after_first) * unit / 1024**2:.1f} MiB"
        )


class AlgorithmRecordingTests(unittest.TestCase):
    def test_ga_nsga_and_islands_generation_coverage(self):
        from types import SimpleNamespace

        from evolutionary.algorithms.ga import GeneticAlgorithm
        from evolutionary.algorithms.island_model import IslandModel
        from evolutionary.algorithms.nsga_iii import NSGA_III
        from evolutionary.evolutionary_selectors import TournamentSelector

        class Creator(SolutionCreator):
            def create_solution(self, argument):
                return SolutionCandidate(argument, float(argument.prompt_embeds.sum()))

        evaluator = SimpleNamespace(evaluate=lambda value: value)
        mutator = SimpleNamespace(mutate=lambda args: args)
        crossover = SimpleNamespace(crossover=lambda a, b: a)
        random.seed(42)
        with tempfile.TemporaryDirectory() as directory:
            for generations in (1, 3):
                for name in (
                    "ga_notebook.ipynb",
                    "ga_osga_notebook.ipynb",
                    "nsga_notebook.ipynb",
                    "island_ga_notebook.ipynb",
                ):
                    path = Path(directory) / f"{name}-{generations}"
                    writer = EmbeddingArchiveWriter(path)
                    namespace = {"save_images": False, "archive_writer": writer}

                    def make_algorithm(
                        island=None,
                        name=name,
                        namespace=namespace,
                        generations=generations,
                    ):
                        callback = notebook_callback(
                            name, namespace.copy(), island or 0
                        )
                        args = [candidate(i).arguments for i in range(4)]
                        options = {
                            "num_generations": generations,
                            "population_size": 4,
                            "solution_creator": Creator(),
                            "initial_arguments": args,
                            "post_evaluation_callback": callback,
                            "mutator": mutator,
                            "crossover": crossover,
                            "ident": island,
                        }
                        if name == "nsga_notebook.ipynb":
                            return NSGA_III(
                                evaluator=SimpleNamespace(evaluate=lambda v: [v, -v]),
                                ref_dirs=__import__("numpy").array(
                                    [[0, 1], [1, 0], [0.5, 0.5], [0.25, 0.75]]
                                ),
                                **options,
                            )
                        return GeneticAlgorithm(
                            evaluator=evaluator,
                            selector=TournamentSelector(2),
                            **options,
                        )

                    if "island" in name:
                        algorithm = IslandModel(
                            [make_algorithm(0), make_algorithm(1)],
                            migration_interval=1,
                            migration_size=1,
                        )
                    else:
                        algorithm = make_algorithm()
                    algorithm.run()
                    # Calling best_solution again must not create a generation -1 snapshot.
                    if "island" not in name:
                        algorithm.best_solution()
                    manifest = EmbeddingArchiveReader(path).manifest
                    multiplier = 2 if "island" in name else 1
                    self.assertEqual(
                        len(manifest["snapshots"]), generations * multiplier
                    )
                    self.assertEqual(
                        {s["generation"] for s in manifest["snapshots"]},
                        set(range(generations)),
                    )
                    self.assertTrue(
                        all(s["status"] == "complete" for s in manifest["snapshots"])
                    )
                    if "island" in name and generations > 1:
                        self.assertTrue(
                            any(s["expected_count"] != 4 for s in manifest["snapshots"])
                        )

    def test_nsga_variants_archive_evaluated_survivors(self):
        from types import SimpleNamespace

        import numpy as np

        from evolutionary.algorithms.nsga_ii import NSGA_II, NSGATournamentSelector
        from evolutionary.algorithms.nsga_iii import NSGA_III, U_NSGA_III

        class Creator(SolutionCreator):
            def create_solution(self, arguments):
                return SolutionCandidate(
                    arguments, float(arguments.prompt_embeds.sum())
                )

        with tempfile.TemporaryDirectory() as directory:
            for algorithm_type in (NSGA_II, NSGA_III, U_NSGA_III):
                for generations in (1, 3):
                    with self.subTest(
                        variant=algorithm_type.__name__, generations=generations
                    ):
                        writer = EmbeddingArchiveWriter(
                            Path(directory) / f"{algorithm_type.__name__}-{generations}"
                        )
                        callback = notebook_callback(
                            "nsga_notebook.ipynb",
                            {
                                "save_images": False,
                                "archive_writer": writer,
                            },
                        )
                        selection = (
                            {"selector": NSGATournamentSelector()}
                            if algorithm_type is NSGA_II
                            else {
                                "ref_dirs": np.array(
                                    [[0, 1], [1, 0], [0.5, 0.5], [0.25, 0.75]]
                                ),
                                "seed": 42,
                            }
                        )
                        algorithm = algorithm_type(
                            num_generations=generations,
                            population_size=4,
                            solution_creator=Creator(),
                            initial_arguments=[
                                candidate(i).arguments for i in range(4)
                            ],
                            evaluator=SimpleNamespace(
                                evaluate=lambda value: [value, -value]
                            ),
                            mutator=None,
                            crossover=None,
                            mutation_rate=0,
                            crossover_rate=0,
                            post_evaluation_callback=callback,
                            **selection,
                        )
                        algorithm.run()
                        reader = EmbeddingArchiveReader(writer.run_dir)
                        self.assertEqual(
                            [s["generation"] for s in reader.manifest["snapshots"]],
                            list(range(generations)),
                        )
                        self.assertTrue(
                            all(
                                s["committed_count"] == len(algorithm.population) == 4
                                for s in reader.manifest["snapshots"]
                            )
                        )
                        final = [
                            r
                            for r in reader.iter_records()
                            if r["generation"] == generations - 1
                        ]
                        for record, survivor in zip(final, algorithm.population):
                            self.assertEqual(record["fitness"], survivor.fitness)
                            _, tensors = reader.read_record(record["record_id"])
                            self.assertTrue(
                                torch.equal(
                                    tensors["prompt_embeds"],
                                    survivor.arguments.prompt_embeds,
                                )
                            )

    def test_osga_completed_and_early_termination_snapshots(self):
        from types import SimpleNamespace

        from evolutionary.algorithms.ga import (
            GeneticAlgorithm,
            OffspringSelectionConfig,
        )
        from evolutionary.evolutionary_selectors import TournamentSelector

        class Creator(SolutionCreator):
            def create_solution(self, arguments):
                return SolutionCandidate(
                    arguments, float(arguments.prompt_embeds.sum())
                )

        with tempfile.TemporaryDirectory() as directory:
            for reason, selection, budget, completed in (
                ("num_generations", OffspringSelectionConfig(0, 1, 1), 100, 3),
                ("max_selection_pressure", OffspringSelectionConfig(1, 1, 1), None, 1),
                ("max_evaluations", None, 5, 1),
            ):
                with self.subTest(reason=reason):
                    writer = EmbeddingArchiveWriter(Path(directory) / reason)
                    callback = notebook_callback(
                        "ga_osga_notebook.ipynb",
                        {
                            "save_images": False,
                            "archive_writer": writer,
                        },
                    )
                    algorithm = GeneticAlgorithm(
                        num_generations=3,
                        population_size=4,
                        solution_creator=Creator(),
                        initial_arguments=[candidate(i).arguments for i in range(4)],
                        evaluator=SimpleNamespace(evaluate=lambda value: value),
                        selector=TournamentSelector(2),
                        mutator=None,
                        crossover=None,
                        mutation_rate=0,
                        crossover_rate=0,
                        offspring_selection=selection,
                        max_evaluations=budget,
                        post_evaluation_callback=callback,
                    )
                    algorithm.run()
                    self.assertEqual(algorithm.termination_reason, reason)
                    self.assertEqual(algorithm.completed_generations, completed)
                    reader = EmbeddingArchiveReader(writer.run_dir)
                    self.assertEqual(
                        [s["generation"] for s in reader.manifest["snapshots"]],
                        list(range(completed)),
                    )
                    self.assertEqual(len(list(reader.iter_records())), completed * 4)
                    if completed < 3:
                        self.assertFalse(
                            algorithm.statistics.generation_summaries[-1].completed
                        )
                        self.assertGreater(algorithm.evaluation_count, 4)


if __name__ == "__main__":
    unittest.main()
