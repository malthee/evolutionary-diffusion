"""A managed target kernel runs only after the audited benchmark prerequisite."""

import argparse
import json
import time
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.config import (
    DETECTORS,
    digest,
    load_config,
)
from evolutionary_extensions.experiments.detector_evaluation.journal import (
    validate_complete,
)
from evolutionary_extensions.experiments.detector_evaluation.runner import (
    assets_for,
    cuda_preflight,
    guard_disk,
    smoke_pool,
    status,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .loop import freeze_finalists, search
from .optimizer import EmbeddingCoordinates, SearchOptimizer, optimizer_source_files
from .protocol import search_targets
from .rendering import SearchRenderer
from .scoring import (
    SearchAeroblade,
    SearchClassifier,
    package_versions,
    scoring_procedure,
)
from .store import SearchStore


def verify_prerequisite(config):
    """Verify the prior phase before CUDA initialization or model loading."""
    completion = json.loads(Path(config["benchmark_completion"]).read_text())
    sha = completion.get("archive_sha256", "")
    if (
        completion.get("status") != "complete"
        or completion.get("primary_scores") != 5000
        or completion.get("compute_state") != "Stopped"
        or completion.get("archive_files_verified", 0) <= 0
        or len(sha) != 64
        or any(c not in "0123456789abcdef" for c in sha)
    ):
        raise ValueError(
            "Require the audited, verified-archive benchmark completion receipt"
        )
    if (
        config.get("budget") != 6400
        or config.get("population") != 64
        or config.get("generation_batch") != 4
        or config.get("algorithm") != "sep-cma-es"
        or config.get("seed") != 42
        or config.get("sigma") != 0.15
    ):
        raise ValueError("Search differs from the resolved authorized recipe")
    benchmark = load_config(config["benchmark_config"])
    root = Path(benchmark["output"])
    rows = json.loads((root / "images.json").read_text())
    scores = json.loads((root / "reports/scores.json").read_text())
    if validate_complete(rows, scores, DETECTORS) != 5000:
        raise ValueError("Benchmark primary scores are incomplete")
    proofs = [
        root / "parity" / f"{name}.json" for name in DETECTORS if name != "aeroblade"
    ]
    aero = sorted((root / "parity").glob("aeroblade-*.json"))
    if len(aero) != 3 or any(
        json.loads(p.read_text()).get("passed") is not True for p in [*proofs, *aero]
    ):
        raise ValueError(
            "Require all seven successful native/offline benchmark parity receipts"
        )
    return benchmark, rows, scores, completion


def calibrate(scorer, target, store, rows, benchmark_root):
    from evolutionary_extensions.experiments.detector_evaluation.analysis import (
        threshold_at_fpr,
    )

    if target not in {"ssp", "aeroblade"}:
        return
    codecs = ("png", "jpeg95") if target == "aeroblade" else ("png",)
    for codec in codecs:
        pool = [r for r in rows if r["group"] == "calibration" and r["codec"] == codec]
        if len(pool) != 200:
            raise ValueError("Require 200 disjoint calibration photographs")
        values = []
        for row in pool:
            receipt = store.root / "calibration" / f"{row['image_id']}-{codec}.json"
            if receipt.exists():
                saved = json.loads(receipt.read_text())
                if (
                    saved["identity"] != store.identity
                    or saved["image_sha256"] != row["sha256"]
                ):
                    raise ValueError("Adapted calibration receipt changed")
                if (
                    hashes(Path(benchmark_root) / row["path"])["sha256"]
                    != row["sha256"]
                ):
                    raise ValueError("Calibration pixels changed")
            else:
                intent = receipt.with_suffix(".attempt.json")
                previous = (
                    json.loads(intent.read_text()).get("attempts", 0)
                    if intent.exists()
                    else 0
                )
                atomic_json(
                    intent,
                    {
                        "identity": store.identity,
                        "state": "started",
                        "attempts": previous + 1,
                    },
                )
                record = scorer.score([row], image_root=benchmark_root)[0]
                saved = {
                    "identity": store.identity,
                    "image_sha256": row["sha256"],
                    **record,
                }
                atomic_json(receipt, saved)
                atomic_json(
                    intent,
                    {
                        "identity": store.identity,
                        "state": "complete",
                        "attempts": previous + 1,
                    },
                )
            values.append(saved["synthetic_score"])
        atomic_json(
            store.root / "calibration" / f"threshold-{codec}.json",
            {
                "identity": store.identity,
                "samples": 200,
                "threshold": threshold_at_fpr(values),
                "comparison": "strict >",
                "procedure": "common SSP patch pool"
                if target == "ssp"
                else "independently reset reconstruction seed 1",
            },
        )


def guard_resources(runtime, root, cuda):
    """Retain allocator peaks across recovery; completed scores are already durable."""
    path = Path(root) / "memory.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    measured = {
        "peak_allocated_gpu_bytes": int(cuda.max_memory_allocated()),
        "peak_reserved_gpu_bytes": int(cuda.max_memory_reserved()),
    }
    peaks = {key: max(value, prior.get(key, 0)) for key, value in measured.items()}
    passed = max(peaks.values()) < 64 * 1024**3
    atomic_json(
        path,
        {
            **peaks,
            "passed": passed,
            "checks": prior.get("checks", 0) + 1,
            "checked_unix": time.time(),
            "scope": "CUDA allocator peaks before and after persisted optimizer batches; earlier process peaks retained",
        },
    )
    if not passed:
        raise RuntimeError("Search exceeds 64 GiB peak allocated/reserved GPU memory")
    guard_disk(runtime)


def run_target(config_path, target, *, calibration_only=False):
    import torch
    from safetensors.torch import load_file

    from evolutionary_extensions.experiments.prompt_embedding.initialization import (
        load_bounds,
    )
    from evolutionary_prompt_embedding.argument_types import PooledPromptEmbedData

    if target not in DETECTORS:
        raise ValueError("Unknown detector target")
    config = json.loads(Path(config_path).read_text())
    targets = search_targets(config)
    if calibration_only:
        if target != "aeroblade" or target in targets:
            raise ValueError(
                "Calibration-only stage requires a non-optimized AEROBLADE evaluator"
            )
    elif target not in targets:
        raise ValueError("Target excluded from the frozen optimization selection")
    benchmark, rows, _, completion = verify_prerequisite(config)
    from .generator_validation import verify_generator_parity

    generator_proof = verify_generator_parity(config, target)
    root = Path(config["output"]).resolve() / target
    root.mkdir(parents=True, exist_ok=True)
    runtime = {**benchmark, "output": str(root)}
    guard_disk(runtime)
    cuda_preflight(root)
    assets = assets_for(benchmark, target)
    assets_for(benchmark, "generate")
    bounds, manifest = load_bounds()
    frozen_bounds = load_file(
        str(Path(benchmark["output"]) / "coordinate_bounds.safetensors")
    )
    if set(bounds) != set(frozen_bounds) or any(
        not torch.equal(bounds[k].cpu(), frozen_bounds[k]) for k in bounds
    ):
        raise ValueError("Search and benchmark coordinate bounds differ")
    geometry = EmbeddingCoordinates(bounds)
    embeddings = load_file(str(Path(benchmark["output"]) / "embeddings.safetensors"))
    initial = PooledPromptEmbedData(
        embeddings["prompt_embeds"][:1], embeddings["pooled_prompt_embeds"][:1]
    )
    recipe = {
        "target": target,
        "budget": config["budget"],
        "population": config["population"],
        "algorithm": config["algorithm"],
        "seed": config["seed"],
        "sigma": config["sigma"],
        "generator_parity_sha256": digest(generator_proof),
        "generation_batch": 4,
        "diffusion_seed": 0,
        "model_revision": benchmark["model_revision"],
        "patch_seed": benchmark["patch_seed"],
        "bounds": manifest,
        "geometry": geometry.identity,
        "initial_embedding_source_sha256": hashes(
            Path(benchmark["output"]) / "embeddings.safetensors"
        )["sha256"],
        "initial_embedding_index": 0,
        "benchmark_archive_sha256": completion["archive_sha256"],
        "assets": assets["detectors"][target],
        "source_files": {
            **optimizer_source_files(),
            **{p.name: hashes(p)["sha256"] for p in Path(__file__).parent.glob("*.py")},
        },
        "packages": package_versions(),
    }
    store = SearchStore(root, recipe)
    optimizer = (
        None
        if calibration_only
        else store.restore()
        or SearchOptimizer(
            geometry.encode(initial),
            seed=config["seed"],
            budget=config["budget"],
            population=config["population"],
            sigma=config["sigma"],
        )
    )
    if optimizer is not None and optimizer.completed == optimizer.budget:
        finalists = freeze_finalists(store)
        atomic_json(
            root / "search_complete.json",
            {
                "status": "complete",
                "candidates": optimizer.completed,
                "selected_finalists": finalists["selected"],
                "queries": store.query_counts(),
                "heldout_pending": True,
            },
        )
        status(root, "search_complete", candidates=optimizer.completed, recovered=True)
        return
    tick = time.perf_counter()
    scorer = (
        SearchAeroblade(runtime, assets)
        if target == "aeroblade"
        else SearchClassifier(target, runtime, assets)
    )
    pool = smoke_pool(rows)
    if target == "aeroblade":
        pool = pool[::4]
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    attempt = diagnostics / f"{len(list(diagnostics.glob('*.json'))):04d}.json"
    atomic_json(
        attempt,
        {
            "state": "started",
            "candidate_budget": 0,
            "expected_logical_smoke_queries": len(pool) * 4,
            "expected_component_calls": len(pool)
            * 4
            * (3 if target == "aeroblade" else 1),
        },
    )
    scorer.validate_native(pool, benchmark["output"])
    atomic_json(
        attempt,
        {
            "state": "complete",
            "candidate_budget": 0,
            "logical_smoke_queries": len(pool) * 4,
            "component_calls": len(pool) * 4 * (3 if target == "aeroblade" else 1),
        },
    )
    scoring_batch = (
        1 if target == "aeroblade" else scorer.profile(pool, benchmark["output"])
    )
    procedure = {
        "search_identity": store.identity,
        "procedure": scoring_procedure(
            target, recipe["assets"], recipe["packages"], scoring_batch
        ),
    }
    procedure_path = root / "scoring-procedure.json"
    if procedure_path.exists() and json.loads(procedure_path.read_text()) != procedure:
        raise ValueError("Search effective scoring procedure changed")
    atomic_json(procedure_path, procedure)
    calibrate(scorer, target, store, rows, benchmark["output"])
    if calibration_only:
        guard_resources(runtime, root, torch.cuda)
        atomic_json(
            root / "calibration_complete.json",
            {
                "status": "complete",
                "optimizer_candidates": 0,
                "calibration_score_identities": 400,
                "identity": store.identity,
            },
        )
        return
    renderer = SearchRenderer(root, target, geometry, assets)
    if (
        max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
        >= 64 * 1024**3
    ):
        raise RuntimeError(
            "Combined generation/scoring exceeds the 64 GiB memory limit"
        )
    atomic_json(
        root / "loading.json",
        {
            "seconds": time.perf_counter() - tick,
            "peak_gpu_bytes": max(
                torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
            ),
            "scoring_batch": scoring_batch,
            "generation_batch": 4,
        },
    )
    search(
        optimizer,
        store,
        renderer,
        scorer.score
        if target == "aeroblade"
        else lambda images: scorer.score(images, batch=scoring_batch),
        batch=4,
        guard=lambda: guard_resources(runtime, root, torch.cuda),
        progress=lambda **details: status(root, target, **details),
    )
    finalists = freeze_finalists(store)
    atomic_json(
        root / "search_complete.json",
        {
            "status": "complete",
            "candidates": optimizer.completed,
            "selected_finalists": finalists["selected"],
            "queries": store.query_counts(),
            "heldout_pending": True,
        },
    )
    status(
        root, "search_complete", candidates=optimizer.completed, heldout_pending=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=DETECTORS)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_target(args.config, args.target)


if __name__ == "__main__":
    main()
