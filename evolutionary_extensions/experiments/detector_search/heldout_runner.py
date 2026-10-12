"""Managed held-out generation and detector stages after frozen search selection."""

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

from evolutionary_extensions.experiments.detector_evaluation.analysis import (
    threshold_at_fpr,
)
from evolutionary_extensions.experiments.detector_evaluation.config import (
    DETECTORS,
    digest,
)
from evolutionary_extensions.experiments.detector_evaluation.data import (
    patch_coordinates,
)
from evolutionary_extensions.experiments.detector_evaluation.runner import (
    assets_for,
    cuda_preflight,
    guard_disk,
    smoke_pool,
    status,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes

from .heldout import HeldoutJournal, score_matrix, validate_value
from .heldout_rendering import (
    NoiseRenderer,
    checked_path,
    prepare_inputs,
    render_variants,
)
from .protocol import search_targets
from .reuse import reuse_search_finalists
from .runner import verify_prerequisite
from .scoring import (
    HeldoutClassifier,
    SearchAeroblade,
    package_versions,
    scoring_procedure,
)


def load_phase(config_path):
    config = json.loads(Path(config_path).read_text())
    benchmark, rows, scores, completion = verify_prerequisite(config)
    search_root = Path(config["output"]).resolve()
    marker = json.loads((search_root / "searches_complete.json").read_text())
    if (
        marker.get("candidates") != 6400 * len(search_targets(config))
        or marker.get("status") != "searches_complete"
    ):
        raise ValueError(
            "Require all selected completed optimizer budgets before held-out execution"
        )
    return config, benchmark, rows, scores, completion


def validate_variants(rows, plan):
    expected = {
        (r["subject"], r["diffusion_seed"], r["codec"]) for r in plan["variants"]
    }
    actual = [(r["subject"], r["diffusion_seed"], r["codec"]) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError(
            "Held-out image factorial has missing, duplicate or extra variants"
        )
    subjects = {s["subject"]: s for s in plan["subjects"]}
    for row in rows:
        subject = subjects[row["subject"]]
        if row["group"] != subject["group"] or row["target"] != subject["target"]:
            raise ValueError("Held-out group/target provenance differs")
        if (
            row["diffusion_seed"] == 0
            and row["codec"] == "png"
            and row["sha256"] != subject["noise0_sha256"]
        ):
            raise ValueError("Held-out selection PNG was regenerated or changed")
    return len(rows)


def run_generation(config_path):
    config, benchmark, rows, _, _ = load_phase(config_path)
    from .generator_validation import verify_generator_parity

    verify_generator_parity(config, "generate")
    root, recipe = prepare_inputs(config["output"], benchmark["output"], rows)
    runtime = {**benchmark, "output": str(root)}
    guard_disk(runtime)
    assets = assets_for(benchmark, "generate")
    renderer = None

    def render(subjects, seed):
        nonlocal renderer
        if renderer is None:
            cuda_preflight(root / "generation_preflight")
            renderer = NoiseRenderer(root, recipe, assets)
        return renderer(subjects, seed)

    images = render_variants(
        root,
        recipe,
        render,
        guard=lambda: guard_disk(runtime),
        progress=lambda **k: status(root, k.pop("stage"), **k),
    )
    plan = json.loads((Path(config["output"]) / "heldout_plan.json").read_text())
    count = validate_variants(images, plan)
    atomic_json(
        root / "generation_complete.json",
        {
            "status": "complete",
            "variants": count,
            "plan_sha256": digest(plan),
            "finished_unix": time.time(),
        },
    )


def calibration_rows(root, benchmark_root, rows, seed):
    pool = [r for r in rows if r["group"] == "calibration" and r["codec"] == "png"]
    if len(pool) != 200:
        raise ValueError("Require the frozen 200 separate calibration photographs")
    copied = []
    for row in pool:
        source = checked_path(benchmark_root, row["path"], row["sha256"])
        relative = Path("calibration_images") / f"{row['image_id']}.png"
        destination = Path(root) / relative
        if destination.exists():
            checked_path(root, relative, row["sha256"])
        else:
            destination.parent.mkdir(exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        copied.append(
            {
                **row,
                "path": relative.as_posix(),
                "subject": row["image_id"],
                "diffusion_seed": None,
                "target": None,
                "scorer_context": {
                    "patch_coordinates": [
                        list(c) for c in patch_coordinates(row["image_id"], seed=seed)
                    ]
                },
            }
        )
    return copied


def adapted_aeroblade_thresholds(
    search_root, benchmark_rows, assets, packages, benchmark_root
):
    """Reuse the already measured matching independent-seed calibration only."""
    folder = Path(search_root) / "aeroblade"
    frozen = json.loads((folder / "recipe.json").read_text())
    recipe = frozen["recipe"]
    source_files = {
        p.name: hashes(p)["sha256"] for p in Path(__file__).parent.glob("*.py")
    }
    if (
        recipe["target"] != "aeroblade"
        or recipe["assets"] != assets["detectors"]["aeroblade"]
        or recipe["source_files"] != source_files
        or recipe["packages"] != packages
        or frozen["identity"] != digest(recipe)
    ):
        raise ValueError("Adapted AEROBLADE calibration scorer configuration changed")
    thresholds, receipts = {}, []
    for codec in ("png", "jpeg95"):
        pool = [
            r
            for r in benchmark_rows
            if r["group"] == "calibration" and r["codec"] == codec
        ]
        if len(pool) != 200:
            raise ValueError("Incomplete adapted AEROBLADE calibration pool")
        values = []
        for row in pool:
            checked_path(benchmark_root, row["path"], row["sha256"])
            path = folder / "calibration" / f"{row['image_id']}-{codec}.json"
            record = json.loads(path.read_text())
            if (
                record["identity"] != frozen["identity"]
                or record["image_sha256"] != row["sha256"]
                or record.get("reconstruction_seed") != 1
                or record.get("reconstruction_seed_scope")
                != "independently reset per image"
            ):
                raise ValueError("Adapted AEROBLADE calibration receipt differs")
            validate_value("aeroblade", record)
            values.append(record["synthetic_score"])
            receipts.append(
                {
                    "image_id": row["image_id"],
                    "codec": codec,
                    "receipt_sha256": hashes(path)["sha256"],
                    "source_identity": frozen["identity"],
                    "raw": record["raw"],
                    "synthetic_score": record["synthetic_score"],
                }
            )
        threshold = threshold_at_fpr(values)
        saved = json.loads(
            (folder / "calibration" / f"threshold-{codec}.json").read_text()
        )
        if (
            saved["identity"] != frozen["identity"]
            or saved["threshold"] != threshold
            or saved["samples"] != 200
        ):
            raise ValueError("Adapted calibration threshold differs from saved scores")
        thresholds[f"aeroblade/{codec}"] = threshold
    return thresholds, receipts


def run_detector(config_path, target):
    if target not in DETECTORS:
        raise ValueError("Unknown held-out detector")
    config, benchmark, baseline_rows, _, _ = load_phase(config_path)
    import torch

    search_root = Path(config["output"])
    root = search_root / "heldout"
    plan = json.loads((search_root / "heldout_plan.json").read_text())
    marker = json.loads((root / "generation_complete.json").read_text())
    rows = json.loads((root / "images.json").read_text())
    if (
        marker.get("status") != "complete"
        or marker["plan_sha256"] != digest(plan)
        or marker["variants"] != validate_variants(rows, plan)
    ):
        raise ValueError("Held-out generation has not completed its frozen factorial")
    assets = assets_for(benchmark, target)
    packages = package_versions()
    thresholds = json.loads((Path(benchmark["output"]) / "thresholds.json").read_text())
    if target == "aeroblade":
        adapted, receipts = adapted_aeroblade_thresholds(
            search_root, baseline_rows, assets, packages, benchmark["output"]
        )
        thresholds.update(adapted)
        atomic_json(
            root / "calibration/aeroblade-reused.json",
            {
                "thresholds": adapted,
                "reused_score_identities": 400,
                "fresh_queries": 0,
                "receipts": receipts,
            },
        )
    runtime = {**benchmark, "output": str(root)}
    summaries = []
    for seed in (42, 43, 44, 45) if target == "ssp" else (42,):
        folder = root / "scorers" / target / f"patch{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        selected = rows if seed == 42 else [r for r in rows if r["diffusion_seed"] == 0]
        measured = [
            {
                **r,
                "scorer_context": {
                    "patch_coordinates": [
                        list(c) for c in patch_coordinates(r["subject"], seed=seed)
                    ]
                },
            }
            if target == "ssp"
            else r
            for r in selected
        ]
        if target == "ssp" and seed != 42:
            measured += calibration_rows(root, benchmark["output"], baseline_rows, seed)
        # Scalar inference is an explicitly validated fallback. A future faster
        # batch requires its own frozen continuous/native parity profile.
        identity = {
            "asset": assets["detectors"][target],
            "packages": packages,
            "source_files": {
                p.name: hashes(p)["sha256"] for p in Path(__file__).parent.glob("*.py")
            },
            "batch": 1,
            "tf32": False,
            "precision": "native FP16 AE, FP32 LPIPS, FP16 error reduction"
            if target == "aeroblade"
            else "FP32",
            "ssp_patch_seed": seed if target == "ssp" else None,
            "ssp_patch_identity": "frozen subject" if target == "ssp" else None,
            "reconstruction_seed": 1 if target == "aeroblade" else None,
            "reconstruction_seed_scope": "independently reset per image"
            if target == "aeroblade"
            else None,
            "effective_procedure": scoring_procedure(
                target, assets["detectors"][target], packages, 1
            ),
        }
        frozen = folder / "recipe.json"
        if frozen.exists() and json.loads(frozen.read_text()) != identity:
            raise ValueError(
                "Held-out stage configuration changed; use another output root"
            )
        atomic_json(frozen, identity)
        journal = HeldoutJournal(root, target, identity, measured)
        reuse_search_finalists(journal, search_root, identity["effective_procedure"])
        pending = [r for r in measured if journal.existing(r) is None]
        scorer = None
        scoring_batch = 1
        tick = time.perf_counter()
        timing = {
            "loading_and_native_validation_seconds": 0.0,
            "scoring_and_preprocessing_seconds": 0.0,
            "peak_allocated_gpu_bytes": 0,
            "peak_reserved_gpu_bytes": 0,
        }
        if pending:
            guard_disk(runtime)
            cuda_preflight(folder)
            scorer_config = {**benchmark, "output": str(folder), "patch_seed": seed}
            scorer = (
                SearchAeroblade(scorer_config, assets)
                if target == "aeroblade"
                else HeldoutClassifier(target, scorer_config, assets)
            )
            pool = smoke_pool(baseline_rows)
            if target == "aeroblade":
                pool = pool[::4]
            diagnostic = folder / "diagnostics" / f"{time.time_ns()}.json"
            count = len(pool) * 4
            atomic_json(
                diagnostic,
                {
                    "state": "started",
                    "logical_queries": count,
                    "component_calls": count * (3 if target == "aeroblade" else 1),
                    "optimizer_candidates": 0,
                },
            )
            scorer.validate_native(pool, benchmark["output"])
            timing["loading_and_native_validation_seconds"] = time.perf_counter() - tick
            atomic_json(
                diagnostic,
                {
                    "state": "complete",
                    "logical_queries": count,
                    "component_calls": count * (3 if target == "aeroblade" else 1),
                    "optimizer_candidates": 0,
                },
            )
            if (
                max(
                    torch.cuda.max_memory_allocated(),
                    torch.cuda.max_memory_reserved(),
                )
                >= 64 * 1024**3
            ):
                raise RuntimeError("Held-out detector exceeds 64 GiB peak GPU memory")

        def score(
            pool,
            scorer=scorer,
            folder=folder,
            scoring_batch=scoring_batch,
            timing=timing,
        ):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            values = (
                scorer.score(pool, image_root=root)
                if target == "aeroblade"
                else scorer.score(pool, image_root=root, batch=scoring_batch)
            )
            torch.cuda.synchronize()
            timing["scoring_and_preprocessing_seconds"] += time.perf_counter() - tick
            timing["peak_allocated_gpu_bytes"] = max(
                timing["peak_allocated_gpu_bytes"], torch.cuda.max_memory_allocated()
            )
            timing["peak_reserved_gpu_bytes"] = max(
                timing["peak_reserved_gpu_bytes"], torch.cuda.max_memory_reserved()
            )
            for value in values:
                if "patch_path" in value:
                    value["patch_path"] = str(
                        (folder / value["patch_path"]).relative_to(root)
                    )
            return values

        def progress(
            patch_seed=seed,
            folder=folder,
            timing=timing,
            scoring_batch=scoring_batch,
            **details,
        ):
            atomic_json(
                folder / "timing.json",
                {**timing, "validated_scoring_batch": scoring_batch},
            )
            status(
                root, f"heldout_{target}", patch_seed=patch_seed, **details, **timing
            )
            # The evaluated batch is committed before a resource failure stops
            # further work. Recovery retains those finite completed results.
            if (
                max(
                    timing["peak_allocated_gpu_bytes"],
                    timing["peak_reserved_gpu_bytes"],
                )
                >= 64 * 1024**3
            ):
                raise RuntimeError("Held-out scoring exceeds 64 GiB peak GPU memory")

        records = score_matrix(
            journal,
            score,
            guard=lambda: guard_disk(runtime),
            batch=scoring_batch,
            progress=progress,
        )
        summary = {
            "identity": journal.identity,
            "patch_seed": seed if target == "ssp" else None,
            "measurements": len(records),
            "matrix_measurements": sum(
                r["row"]["group"] != "calibration" for r in records
            ),
            "calibration_measurements": sum(
                r["row"]["group"] == "calibration" for r in records
            ),
            "queries": journal.counts(),
            "reused_search_measurements": sum(
                (r.get("reused_from") or {}).get("phase") == "search" for r in records
            ),
            "threshold_png": thresholds[f"{target}/png"],
            "secondary_threshold_jpeg95": thresholds[f"{target}/jpeg95"],
            "comparison": "strict >",
            "jpeg75_threshold": "unchanged PNG threshold; no JPEG75 calibration",
        }
        if target == "ssp" and seed != 42:
            values = [
                r["value"]["synthetic_score"]
                for r in records
                if r["row"]["group"] == "calibration"
            ]
            summary["png_fpr_drift_false_positives"] = sum(
                v > thresholds["ssp/png"] for v in values
            )
        summaries.append(summary)
        del score, progress, scorer
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        root / f"{target}_complete.json",
        {
            "status": "complete",
            "detector": target,
            "summaries": summaries,
            "finished_unix": time.time(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", *DETECTORS))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    if args.stage == "generate":
        run_generation(args.config)
    else:
        run_detector(args.config, args.stage)


if __name__ == "__main__":
    main()
