import json

import pytest

from evolutionary_extensions.experiments.detector_evaluation.config import digest
from evolutionary_extensions.experiments.detector_search.store import SearchStore
from evolutionary_extensions.experiments.detector_search.trajectories import (
    target_trajectory,
)
from evolutionary_extensions.persistence.packaging import atomic_json, hashes


@pytest.fixture
def completed(tmp_path):
    target = "ssp"
    store = SearchStore(tmp_path, {"target": target, "budget": 8, "population": 4})
    pixels = tmp_path / "pixels.png"
    pixels.write_bytes(b"owned source hash fixture")
    embed = tmp_path / "embedding.safetensors"
    embed.write_bytes(b"owned FP16 source hash fixture")
    all_scores = []
    for first in (0, 4):
        indices = list(range(first, first + 4))
        images = [
            {
                "candidate_index": i,
                "image_id": f"ssp-{i:05d}",
                "path": pixels.name,
                "sha256": hashes(pixels)["sha256"],
                "codec": "png",
            }
            for i in indices
        ]
        intent = store.begin_queries(indices, images)
        records = [
            {"raw": 0.5, "synthetic_score": 0.5, "objective": float(8 - i)}
            for i in indices
        ]
        store.finish_queries(intent, records)
        all_scores.extend(records)
        receipt = f"batch-{first:05d}.json"
        row = {
            "indices": indices,
            "proposal_sha256": digest(indices),
            "images": images,
            "embedding_path": embed.name,
            "embedding_sha256": hashes(embed)["sha256"],
            "render_seconds": 0.2,
        }
        atomic_json(tmp_path / "generation" / receipt, row)
        atomic_json(
            tmp_path / "generation_attempts" / f"{first // 4:06d}.json",
            {
                "indices": indices,
                "proposal_sha256": row["proposal_sha256"],
                "physical_images": 4,
                "state": "complete",
                "receipt": receipt,
                "render_seconds": 0.2,
            },
        )
        objectives = [r["objective"] for r in records]
        atomic_json(
            tmp_path / "generations" / f"{first + 4:05d}.json",
            {
                "completed_candidates": first + 4,
                "objectives": objectives,
                "best": min(objectives),
            },
        )
    atomic_json(
        tmp_path / "search_complete.json",
        {"status": "complete", "candidates": 8, "queries": store.query_counts()},
    )
    atomic_json(tmp_path / "loading.json", {"seconds": 1.2, "peak_gpu_bytes": 1024})
    atomic_json(
        tmp_path / "memory.json",
        {
            "passed": True,
            "checks": 4,
            "peak_allocated_gpu_bytes": 1024,
            "peak_reserved_gpu_bytes": 2048,
        },
    )
    atomic_json(tmp_path / "parity.json", {"passed": True})
    return tmp_path


def test_curves_and_query_costs_are_recomputed_from_owned_receipts(completed):
    before = {str(p): hashes(p)["sha256"] for p in completed.rglob("*") if p.is_file()}
    curves, audit = target_trajectory(completed, budget=8, population=4)
    assert [r["best_so_far"] for r in curves] == [5, 1]
    assert [r["population_median"] for r in curves] == [6.5, 2.5]
    assert (
        audit["candidate_identities"] == 8 and audit["completed_fitness_queries"] == 8
    )
    assert audit["fitness"] == {
        "attempted_queries": 8,
        "component_calls": 8,
        "unfinished_queries": 0,
    }
    assert (
        audit["render_image_intents"] == 8 and audit["completed_render_seconds"] == 0.4
    )
    assert audit["profile"] is None and audit["calibration_attempted_queries"] == 0
    assert all(hashes(p)["sha256"] == v for p, v in before.items())


def test_generation_and_query_tampering_are_rejected(completed):
    path = completed / "generations/00004.json"
    original = path.read_bytes()
    value = json.loads(original)
    value["objectives"][0] = 9
    atomic_json(path, value)
    with pytest.raises(ValueError, match="Generation objectives"):
        target_trajectory(completed, budget=8, population=4)
    path.write_bytes(original)
    path = completed / "queries/000000.json"
    original = path.read_bytes()
    value = json.loads(original)
    value["records"][0]["raw"] = 0.7
    atomic_json(path, value)
    with pytest.raises(ValueError, match="Physical query result"):
        target_trajectory(completed, budget=8, population=4)
    path.write_bytes(original)
    (completed / "generation/batch-00000.json").unlink()
    with pytest.raises(ValueError, match="Missing or extra saved rendering"):
        target_trajectory(completed, budget=8, population=4)


def test_uncertain_interrupted_query_does_not_inflate_candidate_budget(completed):
    previous = json.loads((completed / "queries/000000.json").read_text())
    previous["state"] = "started"
    previous.pop("records")
    previous.pop("finished_unix")
    atomic_json(completed / "queries/000002.json", previous)
    path = completed / "search_complete.json"
    marker = json.loads(path.read_text())
    marker["queries"]["fitness"] = {
        "attempted_queries": 12,
        "component_calls": 12,
        "unfinished_queries": 4,
    }
    atomic_json(path, marker)
    _, audit = target_trajectory(completed, budget=8, population=4)
    assert (
        audit["candidate_identities"] == 8 and audit["completed_fitness_queries"] == 8
    )
    assert (
        audit["fitness"]["attempted_queries"] == 12
        and audit["fitness"]["unfinished_queries"] == 4
    )
    assert "upper bounds" in audit["counting_note"]


def test_curve_exports_preserve_values_and_all_file_hashes(completed, monkeypatch):
    from evolutionary_extensions.experiments.detector_evaluation.config import DETECTORS
    from evolutionary_extensions.experiments.detector_search.trajectories import (
        export_trajectories,
    )

    curves, audit = target_trajectory(completed, budget=8, population=4)
    monkeypatch.setattr(
        "evolutionary_extensions.experiments.detector_search.generator_validation.audit_generator_validation",
        lambda _: {
            "fixture": "CPU export only; no CUDA validation claim",
            "candidate_budget_used": 0,
        },
    )
    monkeypatch.setattr(
        "evolutionary_extensions.experiments.detector_search.trajectories.target_trajectory",
        lambda folder: (
            [{**r, "target": folder.name} for r in curves],
            {**audit, "target": folder.name},
        ),
    )
    exported = export_trajectories(completed)
    assert [r["target"] for r in exported] == list(DETECTORS)
    saved = json.loads((completed / "reports/optimizer_curves.json").read_text())
    assert len(saved) == 10
    assert saved[0]["population_median"] == 6.5
    manifest = json.loads((completed / "reports/trajectory_manifest.json").read_text())
    assert manifest["visual_audit_pending"]
    assert set(manifest["files"]) == {
        "optimizer_curves.json",
        "optimizer_curves.csv",
        "search_query_audit.json",
        "generator_query_audit.json",
        "optimizer_curves.png",
        "optimizer_curves.pdf",
        "optimizer_curves.svg",
    }
    assert all(
        hashes(completed / "reports" / name)["sha256"] == sha
        for name, sha in manifest["files"].items()
    )


def test_uncertain_duplicates_cannot_hide_candidates_without_query_provenance(
    completed,
):
    row = json.loads((completed / "queries/000000.json").read_text())
    row["state"] = "started"
    row.pop("records")
    row.pop("finished_unix")
    atomic_json(completed / "queries/000001.json", row)
    path = completed / "search_complete.json"
    marker = json.loads(path.read_text())
    marker["queries"]["fitness"]["unfinished_queries"] = 4
    atomic_json(path, marker)
    with pytest.raises(ValueError, match="Fitness query summary"):
        target_trajectory(completed, budget=8, population=4)


@pytest.mark.parametrize("key", ["peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes"])
def test_search_peak_above_limit_rejects_completion(completed, key):
    memory = json.loads((completed / "memory.json").read_text())
    memory[key] = 64 * 1024**3
    atomic_json(completed / "memory.json", memory)
    with pytest.raises(ValueError, match="GPU memory"):
        target_trajectory(completed, budget=8, population=4)


def test_calibration_only_query_audit_has_zero_candidates_and_full_codec_coverage(
    tmp_path,
):
    from evolutionary_extensions.experiments.detector_evaluation.config import digest
    from evolutionary_extensions.experiments.detector_search.trajectories import (
        calibration_only_audit,
    )
    from evolutionary_extensions.persistence.packaging import atomic_json

    recipe = {"target": "aeroblade", "assets": "frozen"}
    identity = digest(recipe)
    atomic_json(tmp_path / "recipe.json", {"identity": identity, "recipe": recipe})
    atomic_json(
        tmp_path / "calibration_complete.json",
        {
            "status": "complete",
            "optimizer_candidates": 0,
            "calibration_score_identities": 400,
            "identity": identity,
        },
    )
    atomic_json(tmp_path / "parity.json", {"passed": True})
    atomic_json(
        tmp_path / "memory.json",
        {"passed": True, "peak_allocated_bytes": 1024, "peak_reserved_bytes": 2048},
    )
    for codec in ("png", "jpeg95"):
        for i in range(200):
            name = f"cal-{i:03d}-{codec}"
            atomic_json(
                tmp_path / "calibration" / f"{name}.json",
                {
                    "identity": identity,
                    "raw": 0.1,
                    "synthetic_score": -0.1,
                    "reconstruction_errors": [0.1, 0.2, 0.3],
                },
            )
            atomic_json(
                tmp_path / "calibration" / f"{name}.attempt.json",
                {"identity": identity, "state": "complete", "attempts": 1},
            )
    audit = calibration_only_audit(tmp_path)
    assert (
        audit["candidate_identities"] == 0
        and audit["calibration_attempted_queries"] == 400
    )
    assert audit["component_calls_upper_bound"] == 1200
    atomic_json(tmp_path / "memory.json", {"passed": True})
    with pytest.raises(ValueError, match="GPU memory"):
        calibration_only_audit(tmp_path)
    atomic_json(tmp_path / "scores/00000.json", {})
    with pytest.raises(ValueError, match="Excluded"):
        calibration_only_audit(tmp_path)
