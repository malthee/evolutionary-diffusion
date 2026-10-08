"""Validated prompt-embedding experiment settings and pinned model identities."""

from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from evolutionary.algorithms.ga import OffspringSelectionConfig

MODEL_REVISION = "71153311d3dbb46851df1931d3ca6e939de83304"
CLIP_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
PREDICTOR_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"


@dataclass(frozen=True)
class ExperimentConfig:
    output_root: str = "experiment_runs"
    bounds_file: str | None = None
    bounds_source: str = "diffusiondb"
    algorithm: str = "osga"
    objective: str = "maximize"
    cache_dir: str = "~/.cache/evolutionary"
    device: str = "cuda"
    population_size: int = 64
    num_generations: int = 100
    seed: int | None = None
    diffusion_seed: int = 0
    candidate_batch_size: int = 16
    model_id: str = "stabilityai/sdxl-turbo"
    model_revision: str = MODEL_REVISION
    inference_steps: int = 1
    success_ratio: float = 0.6
    comparison_factor: float = 1.0
    max_selection_pressure: float = 10.0
    max_evaluations: int = 6400
    deadline_unix: float | None = None
    minimum_free_gib: float = 20.0
    hard_floor_gib: float = 8.0
    run_id: str = ""
    crossover_weights: dict[str, float] | None = None
    mutation_weights: dict[str, float] | None = None
    operator_parameters: dict | None = None
    canonical_scoring: bool = False
    reuse_unchanged_offspring: bool = False
    buffered_attempt_writes: bool = False
    render_batch_size: int | None = None
    initialization: str = "uniform"
    initial_embeddings_file: str | None = None
    initial_embeddings_sha256: str | None = None
    initial_perturbation_std: dict[str, float] | None = None
    initial_provenance: dict | None = None

    def __post_init__(self):
        if self.algorithm not in {"ga", "osga"}:
            raise ValueError("algorithm must be ga or osga")
        if self.objective not in {"maximize", "minimize"}:
            raise ValueError("objective must be maximize or minimize")
        if self.bounds_source not in {"diffusiondb", "parti"}:
            raise ValueError("bounds_source must be diffusiondb or parti")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be a 32-bit integer or null")
        for name in (
            "population_size",
            "num_generations",
            "candidate_batch_size",
            "max_evaluations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_evaluations < self.population_size:
            raise ValueError("Budget must cover initialization")
        if self.candidate_batch_size > 16:
            raise ValueError("Candidate batches up to 16 are supported")
        if (
            self.inference_steps not in (1, 3)
            or self.model_id != "stabilityai/sdxl-turbo"
        ):
            raise ValueError("SDXL-Turbo supports one or three inference steps")
        if self.hard_floor_gib < 8 or self.minimum_free_gib < max(
            20, self.hard_floor_gib
        ):
            raise ValueError("Require a 20 GiB reserve and an 8 GiB hard floor")
        if not all(
            math.isfinite(getattr(self, k))
            for k in ("minimum_free_gib", "hard_floor_gib")
        ):
            raise ValueError("Disk reserves must be finite")
        if self.deadline_unix is not None and not math.isfinite(self.deadline_unix):
            raise ValueError("Deadline must be finite")
        if self.run_id and (
            Path(self.run_id).name != self.run_id or self.run_id in {".", ".."}
        ):
            raise ValueError("run_id must be a single directory name")
        if not isinstance(self.buffered_attempt_writes, bool):
            raise TypeError("buffered_attempt_writes must be boolean")
        if not isinstance(self.canonical_scoring, bool):
            raise TypeError("canonical_scoring must be boolean")
        if self.render_batch_size is not None and (
            isinstance(self.render_batch_size, bool)
            or not isinstance(self.render_batch_size, int)
            or not 1 <= self.render_batch_size <= 16
        ):
            raise ValueError("render_batch_size must be1..16 or null")
        if not isinstance(self.reuse_unchanged_offspring, bool):
            raise TypeError("reuse_unchanged_offspring must be boolean")
        if self.reuse_unchanged_offspring and (
            not self.canonical_scoring or self.render_batch_size is None
        ):
            raise ValueError(
                "Reuse requires canonical scoring and fixed render_batch_size"
            )
        from .operator_pool import validate_operator_settings

        validate_operator_settings(
            self.crossover_weights, self.mutation_weights, self.operator_parameters
        )
        if self.initial_provenance is not None and not isinstance(
            self.initial_provenance, dict
        ):
            raise TypeError("initial_provenance must be a mapping")
        if self.initialization not in {"uniform", "local"}:
            raise ValueError("initialization must be uniform or local")
        if self.initialization == "local" and not self.initial_embeddings_file:
            raise ValueError("Local initialization requires a frozen anchor file")
        if bool(self.initial_embeddings_file) != bool(self.initial_embeddings_sha256):
            raise ValueError("Embedding input requires its SHA256")
        if self.initial_embeddings_sha256 and (
            len(self.initial_embeddings_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.initial_embeddings_sha256)
        ):
            raise ValueError("Embedding input SHA256 must be lowercase hex")
        if self.initial_perturbation_std is not None and (
            not isinstance(self.initial_perturbation_std, dict)
            or set(self.initial_perturbation_std) != {"token", "pooled"}
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in self.initial_perturbation_std.values()
            )
        ):
            raise ValueError(
                "Local scales require nonnegative finite token/pooled values"
            )
        OffspringSelectionConfig(
            self.success_ratio, self.comparison_factor, self.max_selection_pressure
        )

    def guard(self, reserve_bytes=0):
        if self.deadline_unix is not None and time.time() >= self.deadline_unix:
            raise TimeoutError("Experiment deadline reached")
        for location in (self.output_root, self.cache_dir):
            path = Path(location).expanduser().resolve()
            while not path.exists():
                path = path.parent
            free = shutil.disk_usage(path).free
            if free < self.hard_floor_gib * 1024**3 + reserve_bytes:
                raise OSError(f"Hard disk floor reached on {path}; preserve outputs")
            if free < self.minimum_free_gib * 1024**3 + reserve_bytes:
                raise OSError(f"Disk reserve reached on {path}; stop new work")
