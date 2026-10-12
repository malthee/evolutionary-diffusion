"""Read-only optimizer trajectories and separately counted physical query intents."""

import argparse
import math
from pathlib import Path

import numpy as np

from evolutionary_extensions.experiments.detector_evaluation.config import (
    DETECTORS,
    digest,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes
from evolutionary_extensions.persistence.tables import write_csv

from .heldout import validate_value
from .heldout_rendering import checked_path
from .reporting import read_json


def target_trajectory(folder, *, budget=6400, population=64):
    """Bind population curves to every score and query; never restore pickle state."""
    folder = Path(folder).resolve()
    frozen = read_json(folder / "recipe.json")
    identity, recipe = frozen["identity"], frozen["recipe"]
    target = recipe["target"]
    if (
        target not in DETECTORS
        or identity != digest(recipe)
        or recipe["budget"] != budget
        or recipe["population"] != population
        or budget % population
    ):
        raise ValueError("Search trajectory recipe differs")
    marker = read_json(folder / "search_complete.json")
    if marker.get("status") != "complete" or marker.get("candidates") != budget:
        raise ValueError("Require the complete fresh-candidate budget")
    scores = sorted((folder / "scores").glob("*.json"))
    if [p.name for p in scores] != [f"{i:05d}.json" for i in range(budget)]:
        raise ValueError("Missing or extra candidate score receipts")
    source_hashes, verified, values = {}, set(), []

    def source(path):
        source_hashes[str(path.relative_to(folder))] = hashes(path)["sha256"]

    def image(record):
        key = (record["path"], record["sha256"])
        if key not in verified:
            checked_path(folder, *key)
            verified.add(key)

    for index, path in enumerate(scores):
        record = read_json(path)
        if record["identity"] != identity or record["index"] != index:
            raise ValueError("Candidate score identity/order differs")
        image(record["image"])
        if (
            record["image"]["candidate_index"] != index
            or record["image"]["codec"] != "png"
            or not math.isfinite(record["objective"])
        ):
            raise ValueError("Candidate codec or objective differs")
        validate_value(target, record)
        if "patch_path" in record:
            image({"path": record["patch_path"], "sha256": record["patch_sha256"]})
        source(path)
        values.append(record)
    paths = sorted((folder / "generations").glob("*.json"))
    endpoints = list(range(population, budget + 1, population))
    if [p.name for p in paths] != [f"{i:05d}.json" for i in endpoints]:
        raise ValueError("Missing or extra optimizer generations")
    curves, best = [], math.inf
    for number, (path, end) in enumerate(zip(paths, endpoints), 1):
        row = read_json(path)
        actual = [r["objective"] for r in values[end - population : end]]
        if (
            row["completed_candidates"] != end
            or row["objectives"] != actual
            or row["best"] != min(actual)
        ):
            raise ValueError("Generation objectives differ from candidate receipts")
        best = min(best, min(actual))
        q25, median, q75 = np.quantile(actual, [0.25, 0.5, 0.75])
        curves.append(
            {
                "target": target,
                "generation": number,
                "candidates": end,
                "population_minimum": min(actual),
                "best_so_far": best,
                "population_q25": float(q25),
                "population_median": float(median),
                "population_q75": float(q75),
            }
        )
        source(path)
    attempts = sorted((folder / "queries").glob("*.json"))
    if [p.name for p in attempts] != [f"{i:06d}.json" for i in range(len(attempts))]:
        raise ValueError("Noncontiguous fitness query ledger")
    attempted = components = uncertain = 0
    scored, query_coverage = set(), set()
    persisted_seconds = 0.0
    for path in attempts:
        row = read_json(path)
        indices, images = row["indices"], row["images"]
        n = len(indices)
        if (
            row["identity"] != identity
            or row["kind"] != "fitness"
            or n == 0
            or len(images) != n
            or len(set(indices)) != n
            or row["state"] not in ("started", "complete")
            or row["logical_queries"] != n
            or row["component_calls"] != n * (3 if target == "aeroblade" else 1)
            or any(type(i) is not int or not 0 <= i < budget for i in indices)
            or images != [values[i]["image"] for i in indices]
        ):
            raise ValueError("Fitness query intent differs from candidate evidence")
        query_coverage.update(indices)
        attempted += n
        components += row["component_calls"]
        uncertain += n if row["state"] == "started" else 0
        if row["state"] == "complete":
            if scored.intersection(indices) or len(row["records"]) != n:
                raise ValueError("Completed fitness query repeated or truncated")
            for index, value in zip(indices, row["records"]):
                saved = values[index]
                if value != {
                    k: v
                    for k, v in saved.items()
                    if k not in ("identity", "index", "image")
                }:
                    raise ValueError("Physical query result differs from saved score")
            scored.update(indices)
            seconds = row["finished_unix"] - row["started_unix"]
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("Invalid persisted query duration")
            persisted_seconds += seconds
        source(path)
    counts = {
        "attempted_queries": attempted,
        "component_calls": components,
        "unfinished_queries": uncertain,
    }
    if marker["queries"]["fitness"] != counts or query_coverage != set(range(budget)):
        raise ValueError("Fitness query summary differs from physical ledger")
    diagnostics, profiling, calibration = [], None, []
    for path in sorted((folder / "diagnostics").glob("*.json")):
        row = read_json(path)
        if row["state"] not in ("started", "complete") or row["candidate_budget"] != 0:
            raise ValueError("Native diagnostic budget/state differs")
        diagnostics.append(row)
        source(path)
    path = folder / "profiles/classifier.json"
    if path.exists():
        profiling = read_json(path)
        if (
            profiling["state"] != "complete"
            or profiling["logical_queries_attempted"] < 0
        ):
            raise ValueError("Scoring profile is incomplete")
        source(path)
    for path in sorted((folder / "calibration").glob("*.attempt.json")):
        row = read_json(path)
        if (
            row["identity"] != identity
            or row["state"] not in ("started", "complete")
            or type(row["attempts"]) is not int
            or row["attempts"] < 1
        ):
            raise ValueError("Calibration query intent differs")
        calibration.append(row)
        source(path)
    generation_paths = sorted((folder / "generation").glob("batch-*.json"))
    if [p.name for p in generation_paths] != [
        f"batch-{i:05d}.json" for i in range(0, budget, 4)
    ]:
        raise ValueError("Missing or extra saved rendering batches")
    for first, path in zip(range(0, budget, 4), generation_paths):
        saved = read_json(path)
        indices = list(range(first, first + 4))
        if saved["indices"] != indices or saved["images"] != [
            values[i]["image"] for i in indices
        ]:
            raise ValueError("Saved rendering images differ from candidate scores")
        image({"path": saved["embedding_path"], "sha256": saved["embedding_sha256"]})
        source(path)
    rendering = []
    paths = sorted((folder / "generation_attempts").glob("*.json"))
    if [p.name for p in paths] != [f"{i:06d}.json" for i in range(len(paths))]:
        raise ValueError("Noncontiguous rendering attempt ledger")
    for path in paths:
        row = read_json(path)
        indices = row["indices"]
        if (
            row["state"] not in ("started", "complete")
            or row["physical_images"] != 4
            or len(indices) != 4
            or indices != list(range(indices[0], indices[0] + 4))
            or indices[0] % 4
            or not 0 <= indices[0] <= budget - 4
        ):
            raise ValueError("Physical rendering intent differs")
        if row["state"] == "complete":
            receipt = folder / "generation" / row["receipt"]
            if not receipt.resolve().is_relative_to((folder / "generation").resolve()):
                raise ValueError("Unsafe generation receipt path")
            saved = read_json(receipt)
            if (
                saved["indices"] != indices
                or saved["proposal_sha256"] != row["proposal_sha256"]
                or saved["images"] != [values[i]["image"] for i in indices]
                or saved["render_seconds"] != row["render_seconds"]
                or not math.isfinite(row["render_seconds"])
                or row["render_seconds"] < 0
            ):
                raise ValueError("Rendering result differs from score evidence")
            image(
                {"path": saved["embedding_path"], "sha256": saved["embedding_sha256"]}
            )
            source(receipt)
        rendering.append(row)
        source(path)
    covered = {i for r in rendering for i in r["indices"]}
    if covered != set(range(budget)):
        raise ValueError("Rendering attempt ledger does not cover every candidate")
    loading = read_json(folder / "loading.json")
    if (
        not math.isfinite(loading["seconds"])
        or loading["seconds"] < 0
        or not 0 <= loading["peak_gpu_bytes"] < 64 * 1024**3
    ):
        raise ValueError("Invalid setup timing or peak GPU memory")
    memory = read_json(folder / "memory.json")
    if (
        memory.get("passed") is not True
        or memory.get("checks", 0) < 2 * budget // 4
        or any(
            type(memory.get(key)) is not int or not 0 <= memory[key] < 64 * 1024**3
            for key in ("peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes")
        )
    ):
        raise ValueError("Invalid full-search GPU memory evidence")
    for name in (
        "loading.json",
        "memory.json",
        "recipe.json",
        "search_complete.json",
        "parity.json",
    ):
        source(folder / name)
    if read_json(folder / "parity.json").get("passed") is not True:
        raise ValueError("Native CUDA parity unresolved")
    audit = {
        "target": target,
        "candidate_identities": budget,
        "fitness": counts,
        "completed_fitness_queries": len(scored),
        "persisted_complete_fitness_seconds": persisted_seconds,
        "diagnostics": diagnostics,
        "profile": profiling,
        "calibration_attempted_queries": sum(r["attempts"] for r in calibration),
        "calibration_uncertain_queries": sum(
            r["state"] == "started" for r in calibration
        ),
        "render_image_intents": sum(r["physical_images"] for r in rendering),
        "uncertain_render_image_intents": sum(
            r["physical_images"] for r in rendering if r["state"] == "started"
        ),
        "completed_render_seconds": sum(
            r["render_seconds"] for r in rendering if r["state"] == "complete"
        ),
        "setup": loading,
        "search_memory": memory,
        "setup_scope": "Latest loading.json receipt; previous recovery setup durations are separate lifecycle evidence",
        "source_receipt_hashes": source_hashes,
        "counting_note": "Started/failed intents are attempted upper bounds; completion cannot establish how many component kernels ran before interruption. Component calls count image-component evaluations, not GPU kernel launches or forward batches. Diagnostic/profile/calibration calls are outside the candidate budget. Persisted query seconds include preprocessing and writes; setup includes native validation/profile/calibration and generator loading. Search CUDA allocated/reserved peaks are checked before and after persisted batches, including the final batch; peaks from earlier recovery processes are retained. Other GPU processes and allocations outside the CUDA allocator are not included.",
    }
    return curves, audit


def calibration_only_audit(folder):
    """Account for an evaluator's calibration without inventing candidates."""
    folder = Path(folder)
    complete = read_json(folder / "calibration_complete.json")
    recipe = read_json(folder / "recipe.json")
    identity = recipe["identity"]
    if (
        complete.get("status") != "complete"
        or complete.get("optimizer_candidates") != 0
        or complete.get("calibration_score_identities") != 400
        or complete.get("identity") != identity
        or identity != digest(recipe["recipe"])
    ):
        raise ValueError("Incomplete calibration-only stage")
    if (
        any((folder / "scores").glob("*.json"))
        or (folder / "search_complete.json").exists()
    ):
        raise ValueError("Excluded evaluator contains optimization candidates")
    intents = sorted((folder / "calibration").glob("*.attempt.json"))
    if len(intents) != 400:
        raise ValueError("Calibration-only query coverage differs")
    counts = {codec: 0 for codec in ("png", "jpeg95")}
    attempted = uncertain = 0
    sources = {}
    for intent in intents:
        item = read_json(intent)
        score = intent.with_name(intent.name.replace(".attempt.json", ".json"))
        value = read_json(score)
        codec = score.stem.rsplit("-", 1)[-1]
        if (
            codec not in counts
            or item.get("identity") != identity
            or value.get("identity") != identity
            or item.get("state") not in ("started", "complete")
            or type(item.get("attempts")) is not int
            or item["attempts"] < 1
        ):
            raise ValueError("Calibration-only query identity differs")
        validate_value("aeroblade", value)
        counts[codec] += 1
        attempted += item["attempts"]
        uncertain += item["state"] == "started"
        sources[str(intent.relative_to(folder))] = hashes(intent)["sha256"]
        sources[str(score.relative_to(folder))] = hashes(score)["sha256"]
    if (
        counts != {"png": 200, "jpeg95": 200}
        or read_json(folder / "parity.json").get("passed") is not True
    ):
        raise ValueError("Calibration coverage or native parity differs")
    diagnostics = [
        read_json(p) for p in sorted((folder / "diagnostics").glob("*.json"))
    ]
    if any(
        d.get("candidate_budget") != 0 or d.get("state") not in ("started", "complete")
        for d in diagnostics
    ):
        raise ValueError("Calibration native diagnostic budget differs")
    memory = read_json(folder / "memory.json")
    if memory.get("passed") is not True or any(
        type(memory.get(key)) is not int or not 0 <= memory[key] < 64 * 1024**3
        for key in ("peak_allocated_bytes", "peak_reserved_bytes")
    ):
        raise ValueError("Invalid calibration-only GPU memory evidence")
    return {
        "target": "aeroblade",
        "candidate_identities": 0,
        "calibration_score_identities": 400,
        "calibration_attempted_queries": attempted,
        "calibration_uncertain_queries": uncertain,
        "component_calls_upper_bound": attempted * 3,
        "diagnostics": diagnostics,
        "source_files": sources,
        "memory": memory,
    }


def export_trajectories(search_root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .generator_validation import audit_generator_validation

    search_root = Path(search_root).resolve()
    generator_audit = audit_generator_validation(search_root)
    from .protocol import search_targets

    manifest_path = search_root / "managed_manifest.json"
    targets = search_targets(read_json(manifest_path) if manifest_path.exists() else {})
    curves, audits = [], []
    for target in targets:
        rows, audit = target_trajectory(search_root / target)
        curves.extend(rows)
        audits.append(audit)
    root = search_root / "reports"
    root.mkdir(exist_ok=True)
    calibration_audit = (
        calibration_only_audit(search_root / "aeroblade")
        if "aeroblade" not in targets
        else None
    )
    if calibration_audit is not None:
        atomic_json(root / "calibration_only_query_audit.json", calibration_audit)
    atomic_json(root / "optimizer_curves.json", curves)
    atomic_json(root / "search_query_audit.json", audits)
    atomic_json(root / "generator_query_audit.json", generator_audit)
    write_csv(root / "optimizer_curves.csv", curves)
    figure, axes = plt.subplots(
        len(targets), 1, figsize=(10, 3 * len(targets)), squeeze=False
    )
    for ax, target in zip(axes.flat, targets):
        rows = [r for r in curves if r["target"] == target]
        x = [r["candidates"] for r in rows]
        ax.fill_between(
            x,
            [r["population_q25"] for r in rows],
            [r["population_q75"] for r in rows],
            alpha=0.2,
            label="Population IQR (not CI)",
        )
        ax.plot(x, [r["population_median"] for r in rows], label="Population median")
        ax.plot(x, [r["best_so_far"] for r in rows], label="Best so far")
        ax.set(
            title=target,
            ylabel="Minimized objective",
            xlabel="Fresh candidate identities",
        )
        ax.legend(loc="best")
    figure.suptitle(
        "One sep-CMA-ES trajectory per detector · raw objective scales remain separate"
    )
    figure.tight_layout()
    files = {}
    for suffix in ("png", "pdf", "svg"):
        path = root / f"optimizer_curves.{suffix}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        files[path.name] = hashes(path)["sha256"]
    plt.close(figure)
    for name in (
        "optimizer_curves.json",
        "optimizer_curves.csv",
        "search_query_audit.json",
        "generator_query_audit.json",
    ):
        files[name] = hashes(root / name)["sha256"]
    if calibration_audit is not None:
        files["calibration_only_query_audit.json"] = hashes(
            root / "calibration_only_query_audit.json"
        )["sha256"]
    atomic_json(
        root / "trajectory_manifest.json",
        {
            "status": "trajectory_exports_complete",
            "candidate_identities": sum(a["candidate_identities"] for a in audits),
            "files": files,
            "visual_audit_pending": True,
        },
    )
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    export_trajectories(parser.parse_args().root)


if __name__ == "__main__":
    main()
