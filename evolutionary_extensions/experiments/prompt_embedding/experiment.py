"""Lossless GA and OSGA prompt-embedding experiments."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import time
import uuid
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from evolutionary.algorithms.ga import GeneticAlgorithm, OffspringSelectionConfig
from evolutionary.evolutionary_selectors import TournamentSelector
from evolutionary.history import SolutionHistoryKey
from evolutionary_extensions.paths import checkout_root
from evolutionary_prompt_embedding.archive import (
    EmbeddingArchiveReader,
    EmbeddingArchiveWriter,
    _atomic_json,
    sha256_file,
)

from .artifacts import (
    BoundedPNGWriter,
    BufferedAttemptWriter,
    export_figures,
    validate_artifacts,
    write_csv,
)
from .initialization import initial_arguments, load_bounds
from .operator_pool import operators


def run_experiment(config, runtime):
    if config.seed is None:
        import secrets

        config = replace(config, seed=secrets.randbits(32))
    if (
        not runtime.compatible(config)
        or not runtime.parity
        or not runtime.parity["passed"]
        or runtime.parity.get("candidate_batch_size") != config.candidate_batch_size
        or runtime.parity.get("canonical_scoring", False) != config.canonical_scoring
        or runtime.parity.get("render_batch_size") != config.render_batch_size
    ):
        raise RuntimeError(
            "Require a compatible runtime with passed scorer parity for this batch size"
        )
    config.guard()
    root = Path(config.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_id = config.run_id or f"{config.algorithm}-{time.time_ns()}-seed-{config.seed}"
    experiment_id = uuid.uuid4().hex
    run_dir = root / run_id
    run_dir.mkdir(exist_ok=False)
    (run_dir / "inputs").mkdir()
    bounds, bounds_manifest = load_bounds(config.bounds_file, config.bounds_source)
    if config.bounds_file is not None:
        shutil.copyfile(
            Path(config.bounds_file).expanduser(), run_dir / "inputs/bounds.safetensors"
        )
    elif config.bounds_source == "diffusiondb":
        from importlib import resources

        resource = resources.files("evolutionary_prompt_embedding").joinpath(
            "tensors/diffusiondb-full-bounds.safetensors"
        )
        (run_dir / "inputs/bounds.safetensors").write_bytes(resource.read_bytes())
    else:
        from safetensors.torch import save_file

        save_file(bounds, str(run_dir / "inputs/bounds.safetensors"))
        bounds_manifest = {
            **bounds_manifest,
            "sha256": sha256_file(run_dir / "inputs/bounds.safetensors"),
        }
    _atomic_json(run_dir / "inputs/bounds.json", bounds_manifest)
    if config.initial_embeddings_file:
        shutil.copyfile(
            Path(config.initial_embeddings_file).expanduser(),
            run_dir / "inputs/initial_embeddings.safetensors",
        )
        if (
            sha256_file(run_dir / "inputs/initial_embeddings.safetensors")
            != config.initial_embeddings_sha256
        ):
            raise ValueError("Frozen input changed during copy")
    if config.initial_provenance is not None:
        _atomic_json(
            run_dir / "inputs/initial_provenance.json", config.initial_provenance
        )
    if config.initialization == "local":
        shutil.copyfile(
            Path(config.initial_embeddings_file).with_suffix(".png"),
            run_dir / "inputs/anchor.png",
        )
    configuration = asdict(config)
    configuration.update(
        algorithm=config.algorithm,
        elitism_count=1,
        tournament_size=3,
        crossover_rate=0.9,
        mutation_event_probability=0.2,
        width=512,
        height=512,
        guidance_scale=0.0,
        images_per_candidate=1,
        initialization=f"FP32 uniform draws in per-coordinate {bounds_manifest.get('dataset', 'external')} bounds, repaired to FP16",
    )
    _atomic_json(run_dir / "environment.json", runtime.environment)
    _atomic_json(run_dir / "model_manifest.json", runtime.models)
    _atomic_json(run_dir / "parity.json", runtime.parity)
    repo = checkout_root()
    exported_provenance = repo / "deployment_provenance.json"
    provenance = (
        json.loads(exported_provenance.read_text())
        if exported_provenance.is_file()
        else None
    )
    if provenance:
        revision, diff = provenance["base_revision"], provenance["working_tree_diff"]
    else:
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "HEAD"], text=True
        )
    source_files = {}
    for folder in (
        "evolutionary",
        "evolutionary_imaging",
        "evolutionary_model_helpers",
        "evolutionary_prompt_embedding",
        "evolutionary_extensions",
    ):
        for path in (repo / folder).rglob("*.py"):
            relative = path.relative_to(repo)
            target = run_dir / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            source_files[str(relative)] = sha256_file(path)
            if (
                provenance
                and provenance["files"].get(str(relative))
                != source_files[str(relative)]
            ):
                raise ValueError(
                    f"Deployed source differs from its manifest: {relative}"
                )
    _atomic_json(
        run_dir / "source_manifest.json",
        {"git_revision": revision, "working_tree_diff": diff, "files": source_files},
    )
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.reset_peak_memory_stats(config.device)
    initial = initial_arguments(bounds, config.population_size, config)
    crossover, mutator, params = operators(bounds, initial, config)
    configuration.update(
        operator_parameters=params,
        crossover_weights=dict(zip(crossover.operators, crossover.weights)),
        mutation_weights=dict(zip(mutator.operators, mutator.weights)),
    )
    from safetensors.torch import save_file

    save_file(
        {
            "prompt_embeds": torch.stack([a.prompt_embeds for a in initial]),
            "pooled_prompt_embeds": torch.stack(
                [a.pooled_prompt_embeds for a in initial]
            ),
        },
        str(run_dir / "inputs/initial_population.safetensors"),
    )
    configuration["initial_population_sha256"] = sha256_file(
        run_dir / "inputs/initial_population.safetensors"
    )
    configuration["initialization"] = config.initialization
    _atomic_json(run_dir / "config.json", configuration)
    metadata = {
        "experiment_id": experiment_id,
        "algorithm": config.algorithm,
        "model": {"id": config.model_id, "revision": config.model_revision},
        "configuration": configuration,
        "bounds": bounds_manifest,
        "seeds": {"evolution": config.seed, "diffusion": [config.diffusion_seed]},
        "objective_names": ["aesthetics"],
    }
    population_archive = EmbeddingArchiveWriter(
        run_dir, {**metadata, "record_kind": "survivor"}
    )
    attempt_archive = EmbeddingArchiveWriter(
        run_dir / "attempts", {**metadata, "record_kind": "evaluation"}
    )
    attempt_writer = (
        BufferedAttemptWriter(attempt_archive)
        if config.buffered_attempt_writes
        else None
    )
    writer = BoundedPNGWriter()
    initial_padding_images = getattr(runtime, "padding_images", 0)
    started = time.perf_counter()
    core_started_unix = time.time()
    timings, paths, embedding_stats = [], {}, []
    counts = {}
    sign = 1.0 if config.objective == "maximize" else -1.0

    def evaluation_dict(record):
        return {
            **asdict(record),
            "aesthetic_score": sign * record.fitness,
            "optimization_fitness": record.fitness,
        }

    def save_records():
        _atomic_json(
            run_dir / "evaluations.json",
            [evaluation_dict(r) for r in algorithm.statistics.evaluation_records],
        )

        _atomic_json(
            run_dir / "reused_offspring.json",
            [evaluation_dict(r) for r in algorithm.statistics.reused_offspring_records],
        )

    def progress(stage, algorithm, generation):
        count, successes = counts.get(generation, (0, 0))
        cached = [
            record
            for record in algorithm.statistics.reused_offspring_records
            if record.generation == generation
        ]
        proposed = count + len(cached)
        elapsed = time.perf_counter() - started
        _atomic_json(
            run_dir / "progress.json",
            {
                "stage": stage,
                "updated_unix": time.time(),
                "run_id": run_id,
                "seed": config.seed,
                "generation": generation,
                "completed_generations": generation + 1
                if stage == "generation_saved"
                else algorithm.completed_generations,
                "evaluations": algorithm.evaluation_count,
                "reused_offspring_count": len(
                    algorithm.statistics.reused_offspring_records
                ),
                "generation_attempts": proposed,
                "generation_fresh_evaluations": count,
                "generation_cached_attempts": len(cached),
                "generation_successes": successes
                + sum(r.successful is True for r in cached),
                "generation_quota": 0
                if generation == 0 or config.algorithm == "ga"
                else int(np.ceil(config.success_ratio * config.population_size)),
                "selection_pressure": proposed / config.population_size,
                "elapsed_seconds": elapsed,
                "evaluations_per_second": algorithm.evaluation_count / elapsed
                if elapsed
                else None,
                "gpu_allocated_bytes": torch.cuda.memory_allocated(config.device),
                "gpu_peak_bytes": torch.cuda.max_memory_allocated(config.device),
                "cache_disk_free_bytes": shutil.disk_usage(
                    Path(config.cache_dir).expanduser()
                ).free,
                "output_disk_free_bytes": shutil.disk_usage(root).free,
                "pending_images": len(writer.pending),
                "pending_embedding_attempts": attempt_writer.pending_count
                if attempt_writer
                else 0,
            },
        )

    class Creator:
        def create_solution(self, argument):
            return self.create_solutions([argument])[0]

        def create_solutions(self, arguments):
            if attempt_writer:
                attempt_writer.check()
            # Reserve space for the incoming batch, bounded PNG queue and next checkpoint.
            embedding_bytes = 2 * (77 * 2048 + 1280)
            image_bytes = 512 * 512 * 3 + 4096
            config.guard(
                reserve_bytes=len(arguments) * (2 * embedding_bytes + image_bytes)
                + 32 * image_bytes
                + 3 * config.population_size * embedding_bytes
            )
            start = time.perf_counter()
            values = (
                runtime.create_solutions(config, arguments)
                if config.render_batch_size is not None
                else runtime.creator.create_solutions(arguments)
            )
            timings.append(
                {
                    "stage": "render",
                    "count": len(arguments),
                    "seconds": time.perf_counter() - start,
                }
            )
            return values

    class Evaluator:
        def evaluate(self, result):
            return self.evaluate_batch([result])[0]

        def evaluate_batch(self, results):
            start = time.perf_counter()
            # Strict OSGA comparisons must not classify an unchanged image as
            # successful solely because its scorer batch shape changed.
            from .runtime import score_results

            values = score_results(
                runtime.evaluator, results, canonical=config.canonical_scoring
            )
            timings.append(
                {
                    "stage": "score",
                    "count": len(results),
                    "seconds": time.perf_counter() - start,
                }
            )
            return [sign * float(value) for value in values]

    def evaluated(generation, candidates, records, algorithm):
        # The render preflight reserved this space. Persist returned evaluations even
        # if the soft reserve/deadline crossed while inference was in flight.
        labels, image_paths = [], []
        for candidate, record in zip(candidates, records):
            image_path = (
                run_dir
                / "attempts/images"
                / f"evaluation-{record.evaluation_id:07d}.png"
            )
            paths[record.evaluation_id] = image_path
            candidate.meta["evaluation_id"] = record.evaluation_id
            writer.submit(candidate.result.images[0], image_path)
            labels.append(
                {
                    "record_kind": "evaluation",
                    "experiment_id": experiment_id,
                    **evaluation_dict(record),
                }
            )
            image_paths.append([image_path])
            for key, tensor in (
                ("token", candidate.arguments.prompt_embeds),
                ("pooled", candidate.arguments.pooled_prompt_embeds),
            ):
                values = tensor.float()
                if not torch.isfinite(values).all():
                    raise ValueError("Nonfinite embedding")
                embedding_stats.append(
                    {
                        "evaluation_id": record.evaluation_id,
                        "generation": generation,
                        "tensor": key,
                        "min": values.min().item(),
                        "max": values.max().item(),
                        "mean": values.mean().item(),
                        "std": values.std().item(),
                        "norm": values.norm().item(),
                    }
                )
        start = time.perf_counter()
        if attempt_writer:
            attempt_writer.submit(generation, candidates, image_paths, labels)
        else:
            attempt_archive.write_generation(
                candidates,
                generation,
                image_paths=image_paths,
                metadata=labels,
                snapshot_id=f"evaluations-{records[0].evaluation_id}-{records[-1].evaluation_id}",
            )
        timings.append(
            {
                "stage": "attempt_submit" if attempt_writer else "attempt_archive",
                "count": len(records),
                "seconds": time.perf_counter() - start,
            }
        )
        old_count, old_successes = counts.get(generation, (0, 0))
        counts[generation] = (
            old_count + len(records),
            old_successes + sum(r.successful is True for r in records),
        )
        # Persist accepted/rejected records before survivor selection or another batch.
        with (run_dir / "evaluations.jsonl").open("a") as journal:
            for record in records:
                journal.write(
                    json.dumps(evaluation_dict(record), allow_nan=False) + "\n"
                )
        progress("evaluating", algorithm, generation)

    def population(generation, algorithm):
        if attempt_writer:
            drain_start = time.perf_counter()
            attempt_writer.drain()
            timings.append(
                {
                    "stage": "attempt_drain",
                    "count": 0,
                    "seconds": time.perf_counter() - drain_start,
                }
            )
        save_records()
        start = time.perf_counter()
        writer.drain()
        timings.append(
            {"stage": "png_drain", "count": 0, "seconds": time.perf_counter() - start}
        )
        start = time.perf_counter()
        labels = [
            {
                "record_kind": "survivor",
                "experiment_id": experiment_id,
                "optimization_fitness": algorithm.population[i].fitness,
                "aesthetic_score": sign * algorithm.population[i].fitness,
                **asdict(
                    algorithm.statistics.solution_history[
                        SolutionHistoryKey(i, generation, algorithm.ident)
                    ]
                ),
            }
            for i in range(config.population_size)
        ]
        image_paths = [[paths[label["evaluation_id"]]] for label in labels]
        population_archive.write_generation(
            algorithm.population, generation, image_paths=image_paths, metadata=labels
        )
        checkpoint = {
            "format": "prompt-embedding-state-export",
            "resumable": False,
            "completed_generations": generation + 1,
            "configuration": configuration,
            "population": [
                {
                    "prompt_embeds": c.arguments.prompt_embeds.detach().cpu().clone(),
                    "pooled_prompt_embeds": c.arguments.pooled_prompt_embeds.detach()
                    .cpu()
                    .clone(),
                    "fitness": c.fitness,
                    "optimization_fitness": c.fitness,
                    "aesthetic_score": sign * c.fitness,
                    "evaluation_id": label["evaluation_id"],
                }
                for c, label in zip(algorithm.population, labels)
            ],
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "evaluation_count": algorithm.evaluation_count,
            "reused_offspring_records": [
                evaluation_dict(r)
                for r in algorithm.statistics.reused_offspring_records
            ],
            "evaluation_records": [
                evaluation_dict(r) for r in algorithm.statistics.evaluation_records
            ],
            "lineage": [
                asdict(item) for item in algorithm.statistics.solution_history.values()
            ],
            "generation_summaries": [
                asdict(item) for item in algorithm.statistics.generation_summaries
            ],
            "fitness_history": {
                "best": [
                    *algorithm.statistics.best_fitness,
                    max(float(c.fitness) for c in algorithm.population),
                ],
                "average": [
                    *algorithm.statistics.avg_fitness,
                    float(np.mean([float(c.fitness) for c in algorithm.population])),
                ],
                "worst": [
                    *algorithm.statistics.worst_fitness,
                    min(float(c.fitness) for c in algorithm.population),
                ],
            },
            "donor_bank": [
                {
                    "prompt_embeds": a.prompt_embeds,
                    "pooled_prompt_embeds": a.pooled_prompt_embeds,
                }
                for a in initial
            ],
        }
        torch.save(checkpoint, run_dir / "checkpoint.pt.tmp")
        (run_dir / "checkpoint.pt.tmp").replace(run_dir / "checkpoint.pt")
        timings.append(
            {
                "stage": "population_archive_checkpoint",
                "count": config.population_size,
                "seconds": time.perf_counter() - start,
            }
        )
        progress("generation_saved", algorithm, generation)

    algorithm = GeneticAlgorithm(
        config.num_generations,
        config.population_size,
        Creator(),
        TournamentSelector(3),
        mutator,
        crossover,
        Evaluator(),
        initial,
        mutation_rate=0.2,
        crossover_rate=0.9,
        elitism_count=1,
        offspring_selection=OffspringSelectionConfig(
            config.success_ratio,
            config.comparison_factor,
            config.max_selection_pressure,
        )
        if config.algorithm == "osga"
        else None,
        max_evaluations=config.max_evaluations,
        candidate_batch_size=config.candidate_batch_size,
        reuse_unchanged_offspring=config.reuse_unchanged_offspring,
        arguments_equal=lambda a, b: (
            torch.equal(a.prompt_embeds, b.prompt_embeds)
            and torch.equal(a.pooled_prompt_embeds, b.pooled_prompt_embeds)
        ),
        post_evaluation_batch_callback=evaluated,
        post_evaluation_callback=population,
    )
    try:
        best = algorithm.run()
        if attempt_writer:
            drain_start = time.perf_counter()
            attempt_writer.close()
            timings.append(
                {
                    "stage": "attempt_drain",
                    "count": 0,
                    "seconds": time.perf_counter() - drain_start,
                }
            )
        writer.close()
        loop_seconds = time.perf_counter() - started
        timings.append(
            {
                "stage": "variation_and_other_core_overhead",
                "count": 0,
                "seconds": max(0.0, loop_seconds - sum(t["seconds"] for t in timings)),
            }
        )
        statistics = algorithm.statistics
        result = {
            "status": "complete"
            if algorithm.completed_generations == config.num_generations
            and algorithm.termination_reason == "num_generations"
            else "terminated",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "seed": config.seed,
            "population_size": config.population_size,
            "completed_generations": algorithm.completed_generations,
            "evaluation_count": algorithm.evaluation_count,
            "reused_offspring_count": len(statistics.reused_offspring_records),
            "termination_reason": algorithm.termination_reason,
            "algorithm": config.algorithm,
            "objective": config.objective,
            "best_fitness": float(best.fitness),
            "best_aesthetic_score": sign * float(best.fitness),
            "best_evaluated_aesthetic_score": sign
            * max(r.fitness for r in statistics.evaluation_records),
            "unused_evaluation_budget": config.max_evaluations
            - algorithm.evaluation_count,
            "diagnostic_parity_images": runtime.parity.get("count", 16),
            "best_evaluation_id": best.meta["evaluation_id"],
            "best_image_path": paths[best.meta["evaluation_id"]]
            .relative_to(run_dir)
            .as_posix(),
            "loop_seconds": loop_seconds,
            "padding_images_rendered": getattr(runtime, "padding_images", 0)
            - initial_padding_images,
            "canonical_scoring": config.canonical_scoring,
            "render_batch_size": config.render_batch_size,
            "core_started_unix": core_started_unix,
            "core_finished_unix": core_started_unix + loop_seconds,
            "gpu_peak_bytes": torch.cuda.max_memory_allocated(config.device),
            "png_worker_seconds": writer.seconds,
            "attempt_worker_seconds": attempt_writer.worker_seconds
            if attempt_writer
            else 0,
            "peak_pending_embedding_attempts": attempt_writer.peak_pending
            if attempt_writer
            else 0,
            "peak_pending_images": writer.peak_pending,
            "fitness": {
                "best": statistics.best_fitness,
                "average": statistics.avg_fitness,
                "worst": statistics.worst_fitness,
            },
            "generation_summaries": [
                asdict(s) for s in statistics.generation_summaries
            ],
            "operator_summary": statistics.operator_summary(),
        }
        _atomic_json(
            run_dir / "evaluations.json",
            [evaluation_dict(r) for r in statistics.evaluation_records],
        )
        _atomic_json(
            run_dir / "lineage.json",
            [asdict(item) for item in statistics.solution_history.values()],
        )
        result["aesthetic_scores"] = {
            k: [sign * float(x) for x in values]
            for k, values in result["fitness"].items()
        }
        write_csv(run_dir / "embedding_statistics.csv", embedding_stats)
        write_csv(run_dir / "stage_timings.csv", timings)
        write_csv(run_dir / "generation_statistics.csv", result["generation_summaries"])
        write_csv(run_dir / "operator_statistics.csv", result["operator_summary"])
        export_start = time.perf_counter()
        export_figures(run_dir, result, embedding_stats, timings)
        result["export_seconds"] = time.perf_counter() - export_start
        result["validation_passed"] = result["status"] == "complete"
        scientific = {
            "evaluations": sha256_file(run_dir / "evaluations.json"),
            "lineage": sha256_file(run_dir / "lineage.json"),
        }
        tensor_digest, image_digest = hashlib.sha256(), hashlib.sha256()
        for batch, tensors in EmbeddingArchiveReader(run_dir / "attempts").iter_batches(
            batch_size=16
        ):
            for i, record in enumerate(batch):
                for key in sorted(tensors):
                    tensor_digest.update(tensors[key][i].numpy().tobytes())
                image_digest.update(
                    sha256_file(paths[record["metadata"]["evaluation_id"]]).encode()
                )
        scientific.update(
            embeddings=tensor_digest.hexdigest(), images=image_digest.hexdigest()
        )
        result["scientific_hashes"] = scientific
        _atomic_json(run_dir / "result.json", result)
        validate_artifacts(run_dir)
        progress(
            "exports_complete",
            algorithm,
            statistics.generation_summaries[-1].generation,
        )
        return {"run_dir": str(run_dir), **result}
    except BaseException as error:
        save_records()
        write_csv(run_dir / "embedding_statistics.csv", embedding_stats)
        _atomic_json(
            run_dir / "failure.json",
            {
                "type": type(error).__name__,
                "error": str(error),
                "evaluation_count": algorithm.evaluation_count,
                "completed_generations": algorithm.completed_generations,
                "updated_unix": time.time(),
            },
        )
        raise
    finally:
        try:
            if attempt_writer:
                attempt_writer.close()
        finally:
            writer.close()
