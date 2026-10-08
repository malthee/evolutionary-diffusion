"""Scientific artifact checks and objective-aware exports."""

from __future__ import annotations

import copy
import csv
import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from evolutionary_prompt_embedding.archive import (
    EmbeddingArchiveReader,
    _atomic_json,
)


class BoundedPNGWriter:
    def __init__(self, limit=32):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("Writer queue limit must be positive")
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.pending = deque()
        self.limit = limit
        self.seconds = 0.0
        self.peak_pending = 0

    @staticmethod
    def _save(image, path):
        started = time.perf_counter()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        image.save(temporary, format="PNG")
        temporary.replace(path)
        return time.perf_counter() - started

    def submit(self, image, path):
        if len(self.pending) >= self.limit:
            self.seconds += self.pending.popleft().result()
        self.pending.append(self.pool.submit(self._save, image, Path(path)))
        self.peak_pending = max(self.peak_pending, len(self.pending))

    def drain(self):
        while self.pending:
            self.seconds += self.pending.popleft().result()

    def close(self):
        try:
            self.drain()
        finally:
            self.pool.shutdown(wait=True)


class BufferedAttemptWriter:
    """Bounded immutable CPU snapshots; one archive owner and generation barriers.

    Up to 32 evaluated candidates may be pending in memory. Scientific identity
    stays in metadata, so merging callback batches does not change archive joins.
    The existing archive schema and readers remain unchanged.
    """

    def __init__(self, archive, limit=32):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Attempt queue limit must be a positive integer")
        self.archive = archive
        self.limit = limit
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.buffer = []
        self.jobs = deque()
        self.generation = None
        self.worker_seconds = 0.0
        self.peak_pending = 0
        self.closed = False

    @property
    def pending_count(self):
        return len(self.buffer) + sum(len(rows) for _, rows in self.jobs)

    def check(self):
        while self.jobs and self.jobs[0][0].done():
            # Retain a failed payload until close saves its recovery evidence.
            seconds = self.jobs[0][0].result()
            self.worker_seconds += seconds
            self.jobs.popleft()

    def _write(self, generation, rows):
        started = time.perf_counter()
        first = rows[0][2]["evaluation_id"]
        last = rows[-1][2]["evaluation_id"]
        self.archive.write_generation(
            [row[0] for row in rows],
            generation,
            image_paths=[row[1] for row in rows],
            metadata=[row[2] for row in rows],
            snapshot_id=f"evaluations-{first}-{last}",
        )
        return time.perf_counter() - started

    def _dispatch(self):
        if self.buffer:
            rows = self.buffer
            self.jobs.append(
                (self.pool.submit(self._write, self.generation, rows), rows)
            )
            self.buffer = []

    def submit(self, generation, candidates, image_paths, metadata):
        if self.closed:
            raise RuntimeError("Attempt writer is closed")
        if len(candidates) != len(image_paths) or len(candidates) != len(metadata):
            raise ValueError("Attempt images/metadata must match candidates")
        if len(candidates) > self.limit:
            raise ValueError("Attempt batch exceeds queue limit")
        self.check()
        if self.generation is not None and self.generation != generation:
            self.drain()
        if self.pending_count + len(candidates) > self.limit:
            self.drain()
        self.generation = generation
        for candidate, images, label in zip(candidates, image_paths, metadata):
            argument = SimpleNamespace(
                prompt_embeds=candidate.arguments.prompt_embeds.detach().cpu().clone(),
                pooled_prompt_embeds=candidate.arguments.pooled_prompt_embeds.detach()
                .cpu()
                .clone(),
            )
            self.buffer.append(
                (
                    SimpleNamespace(arguments=argument, fitness=candidate.fitness),
                    list(images),
                    copy.deepcopy(label),
                )
            )
        self.peak_pending = max(self.peak_pending, self.pending_count)
        if len(self.buffer) == self.limit:
            self._dispatch()

    def drain(self):
        self.check()
        self._dispatch()
        while self.jobs:
            self.worker_seconds += self.jobs[0][0].result()
            self.jobs.popleft()

    def close(self):
        if self.closed:
            return
        try:
            self.drain()
        except BaseException as error:
            # Never mistake a failed commit for an archived evaluation. Preserve
            # uncommitted immutable tensors/labels for inspection, not inference resume.
            rows = [row for _, pending in self.jobs for row in pending] + self.buffer
            if rows:
                try:
                    temporary = self.archive.run_dir / "pending_attempts.pt.tmp"
                    torch.save(
                        {"format": "uncommitted-attempts", "rows": rows}, temporary
                    )
                    temporary.replace(self.archive.run_dir / "pending_attempts.pt")
                except (OSError, RuntimeError) as recovery_error:
                    error.add_note(
                        f"Pending-attempt recovery failed: {type(recovery_error).__name__}"
                    )
            raise
        finally:
            self.pool.shutdown(wait=True)
            self.closed = True


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="") as handle:
        if rows:
            output = csv.DictWriter(handle, fieldnames=list(rows[0]))
            output.writeheader()
            output.writerows(rows)


def export_figures(run_dir, result, embeddings, timings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = run_dir / "plots"
    folder.mkdir()

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(folder / f"{name}.{extension}")
        plt.close(fig)

    sign = 1.0 if result.get("objective", "maximize") == "maximize" else -1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, values in result.get("aesthetic_scores", result["fitness"]).items():
        axes[0].plot(values, label=key)
    axes[0].set(xlabel="Completed generation", ylabel="LAION aesthetics score")
    axes[0].legend()
    if result.get("algorithm", "osga") == "osga":
        summaries = result["generation_summaries"]
        generations = [s["generation"] for s in summaries]
        axes[1].plot(
            generations,
            [s["selection_pressure"] for s in summaries],
            label="selection pressure",
        )
        axes[1].plot(
            generations, [s["successes"] for s in summaries], label="successes"
        )
        axes[1].plot(generations, [s["quota"] for s in summaries], label="quota")
        axes[1].set(xlabel="Target generation", ylabel="Pressure / counts")
        axes[1].legend()
    else:
        axes[1].axis("off")
    save(fig, "fitness")
    evaluations = json.loads((run_dir / "evaluations.json").read_text())
    best = np.maximum.accumulate([r["fitness"] for r in evaluations]) * sign
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([r["evaluation_id"] for r in evaluations], best)
    ax.set(
        xlabel="Fresh evaluations (including initialization and rejected offspring)",
        ylabel="Best-so-far raw aesthetic score",
        title=result.get("objective", "maximize"),
    )
    save(fig, "fitness_by_evaluations")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, key in enumerate(("token", "pooled")):
        values = [e for e in embeddings if e["tensor"] == key]
        for metric in ("mean", "std"):
            axes[row, 0].plot(
                [e["evaluation_id"] for e in values],
                [e[metric] for e in values],
                label=metric,
            )
        axes[row, 0].set(
            title=key, xlabel="Evaluation ID", ylabel="Embedding statistic"
        )
        axes[row, 0].legend()
        axes[row, 1].plot(
            [e["evaluation_id"] for e in values], [e["norm"] for e in values]
        )
        axes[row, 1].set(title=key, xlabel="Evaluation ID", ylabel="L2 norm")
    save(fig, "embedding_statistics")
    fig, ax = plt.subplots(figsize=(8, 4))
    stages = sorted({t["stage"] for t in timings})
    ax.bar(
        stages,
        [sum(t["seconds"] for t in timings if t["stage"] == stage) for stage in stages],
    )
    ax.set(ylabel="Wall seconds (PNG work overlaps inference)")
    save(fig, "stage_timings")
    fig, ax = plt.subplots(figsize=(12, 4))
    pairs = result["operator_summary"]
    grouped = {}
    for pair in pairs:
        name = f"{pair['crossover_name']} / {pair['mutation_name']}"
        entry = grouped.setdefault(name, [0, 0, 0])
        for i, key in enumerate(("attempts", "successes", "survivors")):
            entry[i] += pair[key]
    names = list(grouped)
    indices = np.arange(len(names))
    for i, label in enumerate(("attempts", "successes", "survivors")):
        ax.bar(
            indices + i * 0.25, [grouped[n][i] for n in names], width=0.25, label=label
        )
    ax.set_xticks(indices + 0.25, names, rotation=75, ha="right")
    ax.legend()
    save(fig, "operator_outcomes")
    reader = EmbeddingArchiveReader(run_dir)
    records = list(reader.iter_records())
    frames = []
    media = run_dir / "media"
    media.mkdir()
    for generation in range(result["completed_generations"]):
        population = [r for r in records if r["generation"] == generation]
        grid = Image.new("RGB", (5 * 160, ((len(population) + 4) // 5) * 185), "white")
        draw = ImageDraw.Draw(grid)
        for i, record in enumerate(population):
            with Image.open(run_dir / record["image_paths"][0]) as image:
                grid.paste(image.resize((160, 160)), ((i % 5) * 160, (i // 5) * 185))
            draw.text(
                ((i % 5) * 160 + 2, (i // 5) * 185 + 162),
                f"{sign * record['fitness']:.4f} / E{record['metadata']['evaluation_id']}",
                fill="black",
            )
        grid.save(media / f"generation-{generation:03d}.png")
        best = max(population, key=lambda r: r["fitness"])
        with Image.open(run_dir / best["image_paths"][0]) as image:
            frames.append(image.copy())
    frames[0].save(
        media / "best_evolution.gif",
        save_all=True,
        append_images=frames[1:],
        duration=600,
        loop=0,
    )

    attempt_reader = EmbeddingArchiveReader(run_dir / "attempts")
    improving, all_best_frames, best_score = [], [], -np.inf
    for record in attempt_reader.iter_records():
        if record["fitness"] > best_score:
            best_score = record["fitness"]
            improving.append(
                {
                    "evaluation_id": record["metadata"]["evaluation_id"],
                    "generation": record["generation"],
                    "fitness": best_score,
                    "aesthetic_score": sign * best_score,
                    "image_path": "attempts/" + record["image_paths"][0],
                }
            )
            with Image.open(
                attempt_reader.resolve_path(record["image_paths"][0])
            ) as image:
                all_best_frames.append(image.copy())
    _atomic_json(media / "best_evaluations.json", improving)
    all_best_frames[0].save(
        media / "best_evaluation_evolution.gif",
        save_all=True,
        append_images=all_best_frames[1:],
        duration=600,
        loop=0,
    )


def validate_artifacts(run_dir):
    run_dir = Path(run_dir)
    result = json.loads((run_dir / "result.json").read_text())
    records = json.loads((run_dir / "evaluations.json").read_text())
    if [r["evaluation_id"] for r in records] != list(
        range(1, result["evaluation_count"] + 1)
    ):
        raise ValueError("Evaluation IDs/count differ")
    attempt_reader = EmbeddingArchiveReader(run_dir / "attempts")
    attempts = list(attempt_reader.iter_records())
    populations = list(EmbeddingArchiveReader(run_dir).iter_records())
    if (
        len(attempts) != len(records)
        or len(populations)
        != result["completed_generations"] * result["population_size"]
    ):
        raise ValueError("Archive record counts differ")
    by_id = {r["metadata"]["evaluation_id"]: r for r in attempts}
    if len(by_id) != len(records):
        raise ValueError("Duplicate/missing attempt identity")
    for record in records:
        sign = 1.0 if result.get("objective", "maximize") == "maximize" else -1.0
        if (
            "aesthetic_score" in record
            and record["aesthetic_score"] != sign * record["fitness"]
        ):
            raise ValueError("Raw aesthetic score differs from signed fitness")
        if float(by_id[record["evaluation_id"]]["fitness"]) != record["fitness"]:
            raise ValueError("Archived fitness differs")
    for population in populations:
        original = by_id.get(population["metadata"]["evaluation_id"])
        if not original or population["fitness"] != original["fitness"]:
            raise ValueError("Survivor identity/fitness differs from evaluation")
        if [str((run_dir / p).resolve()) for p in population["image_paths"]] != [
            str(attempt_reader.resolve_path(p)) for p in original["image_paths"]
        ]:
            raise ValueError(
                "Survivor does not reference its canonical evaluated image"
            )
    if len(list((run_dir / "attempts/images").glob("*.png"))) != len(records):
        raise ValueError("Canonical PNG count differs from evaluations")
    for reader in (attempt_reader, EmbeddingArchiveReader(run_dir)):
        for batch, tensors in reader.iter_batches(batch_size=16):
            if not all(torch.isfinite(t).all() for t in tensors.values()):
                raise ValueError("Archive contains nonfinite tensors")
            for record in batch:
                if not np.isfinite(record["fitness"]):
                    raise ValueError("Archive contains nonfinite fitness")
                for relative in record["image_paths"]:
                    with Image.open(reader.resolve_path(relative)) as image:
                        image.verify()
    return {
        "evaluations": len(attempts),
        "survivor_records": len(populations),
        "passed": True,
    }
